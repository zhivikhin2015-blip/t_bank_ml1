"""Общие метрики: те же определения и порядок равенств, что в reviewed baseline.

Основные ranking-метрики усредняются по клиент-месяцам с покупкой внутри каталога.
Bootstrap парный: выбирает клиентов, сохраняет месяцы и кратность выбора.
"""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CASE_KEYS = ['customerID', 'prediction_date']
PAIR_KEYS = CASE_KEYS + ['ISIN']

def _tie(customer, date, asset, seed):
    payload = json.dumps([seed, str(customer), pd.Timestamp(date).strftime('%Y-%m-%d'), str(asset)],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b'FAR-tie').digest(), 'big')

def prepare_ranking(data, seed=42):
    """Хеши ключей вычисляем один раз; ответы не участвуют в их расчёте."""
    result=data.copy()
    result['_tie']=np.array([_tie(c,d,a,seed) for c,d,a in
        result[PAIR_KEYS].itertuples(index=False,name=None)],dtype=np.uint64)
    return result

def top5(data, probabilities=None, criteria=None, seed=42):
    """Оценки/критерии ↓; воспроизводимый хеш ключа ↑; ISIN ↑ при коллизии."""
    ranked = data.copy()
    if '_tie' not in ranked:
        ranked['_tie'] = np.array([_tie(c,d,a,seed) for c,d,a in
            ranked[PAIR_KEYS].itertuples(index=False, name=None)], dtype=np.uint64)
    if probabilities is not None:
        scores = probabilities if isinstance(probabilities,pd.Series) else pd.Series(probabilities,index=data.index)
        ranked['_score'] = scores.reindex(data.index)
        assert ranked['_score'].notna().all()
        criteria = ['_score']
    if not criteria: raise ValueError('Нужны вероятности или критерии ранжирования')
    ranked = ranked.sort_values(CASE_KEYS + criteria + ['_tie','ISIN'],
                               ascending=[True,True]+[False]*len(criteria)+[True,True])
    result = ranked.groupby(CASE_KEYS,sort=False).head(5).copy()
    result['_rank'] = result.groupby(CASE_KEYS,sort=False).cumcount()+1
    return result

def case_scores(data, top):
    """Одна строка на клиент-месяц; нулевые покупки исключены из primary метрик."""
    result = data.groupby(CASE_KEYS)['label'].sum().rename('positive_assets').to_frame()
    result['hits'] = top.groupby(CASE_KEYS)['label'].sum().reindex(result.index,fill_value=0)
    eligible = result['positive_assets']>0
    result['Recall@5'] = (result['hits']/result['positive_assets']).where(eligible)
    result['Precision@5'] = (result['hits']/5).where(eligible)
    result['Precision@5_all'] = result['hits']/5
    gains = top.assign(_gain=top['label']/np.log2(top['_rank']+1))
    dcg = gains.groupby(CASE_KEYS)['_gain'].sum().reindex(result.index,fill_value=0)
    ideal = np.r_[0.,np.cumsum(1/np.log2(np.arange(1,6)+1))]
    idcg = ideal[result['positive_assets'].clip(upper=5).to_numpy(dtype=int)]
    result['NDCG@5'] = np.divide(dcg,idcg,out=np.full(len(dcg),np.nan),where=idcg>0)
    return result

def evaluate(data, probabilities=None, criteria=None):
    """Вероятностные метрики по всем парам, ranking-метрики по eligible случаям."""
    top = top5(data,probabilities,criteria)
    cases = case_scores(data,top)
    result = {metric:float(cases[metric].mean()) for metric in
              ['Recall@5','Precision@5','NDCG@5','Precision@5_all']}
    result.update(candidate_pairs=len(data),positive_pairs=int(data['label'].sum()),
                  total_user_months=len(cases),eligible_user_months=int((cases['positive_assets']>0).sum()),
                  eligible_unique_clients=int(cases[cases['positive_assets']>0].index.get_level_values('customerID').nunique()))
    if probabilities is None:
        result.update(LogLoss=None,**{'ROC-AUC':None})
    else:
        p = np.asarray(probabilities,dtype=float)
        assert len(p)==len(data) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        p=np.clip(p,1e-15,1-1e-15); y=data['label'].to_numpy(dtype=int)
        result['LogLoss']=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
        result['ROC-AUC']=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
    return result

def recall_cases(data, probabilities=None, criteria=None):
    return case_scores(data,top5(data,probabilities,criteria))['Recall@5']

def paired_bootstrap(data, selected_probabilities, references, client_ids, repeats=5000, seed=20261013):
    """references: имя -> {probabilities: ...} или {criteria: [...]}.
    Делим сумму различий случаев на число случаев, не усредняем средние клиентов.
    """
    selected=recall_cases(data,selected_probabilities)
    eligible=selected.notna()
    counts=eligible.groupby(level='customerID').sum().reindex(client_ids,fill_value=0).to_numpy()
    names=list(references)
    sums=np.column_stack([
        (selected[eligible]-recall_cases(data,**references[name])[eligible])
        .groupby(level='customerID').sum().reindex(client_ids,fill_value=0).to_numpy()
        for name in names])
    rng=np.random.default_rng(seed)
    draws=np.full((repeats,len(names)),np.nan)
    for i in range(repeats):
        draw=rng.integers(0,len(client_ids),size=len(client_ids))
        denominator=counts[draw].sum()
        if denominator: draws[i]=sums[draw].sum(axis=0)/denominator
    rows=[]
    for j,name in enumerate(names):
        valid=np.isfinite(draws[:,j]); values=draws[valid,j]
        low,high=np.quantile(values,[.025,.975]) if valid.any() else (np.nan,np.nan)
        rows.append({'reference':name,'difference':float(sums[:,j].sum()/counts.sum()),
                     'ci_low':float(low),'ci_high':float(high),
                     'contains_zero':bool(low<=0<=high) if valid.any() else None,
                     'valid_draws':int(valid.sum()),'empty_draws':int((~valid).sum())})
    return pd.DataFrame(rows)
