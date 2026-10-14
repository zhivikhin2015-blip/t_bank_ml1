from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
sys.path.insert(0,str(Path.cwd()/'outputs'))
from automatic_feature_metrics import evaluate,top5,recall_cases,paired_bootstrap,prepare_ranking
f=pd.DataFrame({'customerID':['A']*6+['B']*6,'prediction_date':[pd.Timestamp('2022-01-01')]*12,
 'ISIN':list('abcdef')*2,'label':[1,0,1,0,0,1]+[0]*6})
p=np.tile(np.arange(6,0,-1)/7,2)
e=evaluate(f,p)
assert np.isclose(e['Recall@5'],2/3) and np.isclose(e['Precision@5'],.4)
assert np.isclose(e['Precision@5_all'],.2)
assert np.isclose(e['NDCG@5'],1.5/(1+1/np.log2(3)+.5))
s=f.sample(frac=1,random_state=123)
a=top5(f,p);b=top5(s,pd.Series(p,index=f.index))
assert a[['customerID','ISIN','_rank']].equals(b[['customerID','ISIN','_rank']])
# Проверяем cluster bootstrap независимым явным копированием случаев с кратностью.
data=pd.concat([f.assign(prediction_date='2022-01-01'),f.assign(prediction_date='2022-02-01')],ignore_index=True)
data['prediction_date']=pd.to_datetime(data['prediction_date'])
pa=np.tile(p,2);pb=pa[::-1]
result=paired_bootstrap(data,pa,{'reference':{'probabilities':pb}},['A','B','C'],repeats=100,seed=1234)
delta=recall_cases(data,pa)-recall_cases(data,pb)
rng=np.random.default_rng(1234);brute=[];empty=0
for _ in range(100):
    picked=rng.choice(['A','B','C'],size=3,replace=True)
    chunks=[delta[delta.index.get_level_values('customerID')==client].dropna() for client in picked]
    values=pd.concat(chunks)
    if values.empty: empty+=1
    else: brute.append(values.mean())
row=result.iloc[0]
assert row['empty_draws']==empty
assert np.allclose([row['ci_low'],row['ci_high']],np.quantile(brute,[.025,.975]))
# Артефакты проверочного запуска: стандартный JSON и совпадение модели/метаданных.
import joblib
path=Path('work/automatic_features_preflight')
r=json.loads((path/'automatic_features_results.json').read_text(),parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
saved=joblib.load(path/'automatic_features_model.joblib')
assert saved['metadata']==r and saved['features']==r['features']
assert r['test_evaluated'] is False
assert len(r['features'])==10 and r['budget']['boosting_fits_per_full_run']==6
assert hashlib.sha256(Path('/Users/annazhivikhina/Desktop/manual_baseline.ipynb').read_bytes()).hexdigest()=='42d48c5e0549478e4d0ae1d4e6e8f70de6130ac105459ebbd062c4c765496b1c'
report={'metric_fixture':True,'row_order_invariance':True,'bootstrap_explicit_multiplicity':True,
 'bootstrap_empty_draws':True,'standard_json_nulls':True,'model_metadata_consistency':True,
 'test_evaluated':False,'source_notebook_unchanged':True}
Path('work/automatic_features_preflight/checks.json').write_text(json.dumps(report,indent=2))
print('Independent metrics, bootstrap, and artifacts checks passed.')
