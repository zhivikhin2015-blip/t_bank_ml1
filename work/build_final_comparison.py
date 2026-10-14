from pathlib import Path
import json,hashlib,textwrap,ast,sys
root=Path.cwd();path=root/'outputs/03_final_comparison.ipynb'
mode=sys.argv[1] if len(sys.argv)>1 else 'pending'
assert mode in ['pending','approved_complete_months','keep_reserved_period']
if path.exists():
    digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    backup=root/f'work/03_final_comparison_before_completion_{digest}.ipynb'
    if not backup.exists():backup.write_bytes(path.read_bytes())
cells=[]
def add(kind,text):
    c={'cell_type':kind,'metadata':{},'id':f'final-{len(cells):02d}','source':textwrap.dedent(text).strip()+'\n'}
    if kind=='code':ast.parse(c['source']);c.update(execution_count=None,outputs=[])
    cells.append(c)
def md(t):add('markdown',t)
def code(t):add('code',t)
md('''
# Итоговое сравнение ручного baseline и его автоматического расширения

## 1. Задача и зафиксированные решения
На первое число месяца по прошлым покупкам рекомендуем пять активов. Одна строка —
клиент, актив, дата прогноза. `label=1` — хотя бы одна покупка за целевой календарный месяц;
0 — покупка не записана, это не явное отрицательное предпочтение. Повторные покупки допустимы.

Ручная модель и автоматическое расширение выбраны на validation и сохранены.
Здесь **нет обучения, нового отбора признаков или выбора настроек**. Загружаем готовые модели
и проверяем их на более позднем периоде. История обновляется перед каждым прогнозом,
модели фиксированы. Все методы используют тех же 500 клиентов, 50 активов и одинаковые ответы.
Контроль ручных пяти признаков 100/7 — дополнительное заранее заданное сравнение.

**Граница данных.** Репозиторий FAR-Trans указывает сбор до 29 ноября 2022,
поэтому сохранённая исключающая граница — 30 ноября. Для полного ноября нужна граница
не раньше 1 декабря. README и статья не подтверждают этот точный день;
максимальная дата сделки не доказывает полноту. Источники:
[репозиторий](https://researchdata.gla.ac.uk/1658/),
[README](https://researchdata.gla.ac.uk/1658/2/README.txt),
[статья](https://arxiv.org/abs/2407.08692).

Первоначально зарезервирован test июль — ноябрь 2022. Его границы в предыдущих артефактах
не меняются. В этом файле выбранный период итоговой оценки задаётся явно **до просмотра
результатов**. Без согласованного периода либо подтверждения полной границы реальные метрики
не считаются, а все разделы показывают статус ожидания. Ноябрь не исключается молча.
''')
md('''
## 2. Загружаем модели и проверяем условия
Читаем собственные локальные JSON и `joblib`. Проверяем метаданные, параметры, версии,
хеш CSV и хеши моделей. Идентификаторы и дата не подаются в модель.
`threadpool_limits` ограничивает вычисления четырьмя потоками.
''')
code('''
from pathlib import Path
import os
import json
import hashlib
import platform
os.environ.setdefault('LOKY_MAX_CPU_COUNT','4')
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

project_dir=Path.cwd()
if project_dir.name=='outputs':project_dir=project_dir.parent
output_dir=project_dir/'outputs'
data_path=project_dir/'work/data/FAR-Trans/transactions.csv'
artifact_paths={
    'manual_results':output_dir/'manual_baseline_reviewed_results.json',
    'automatic_results':output_dir/'automatic_features_reviewed_results.json',
    'manual_model':output_dir/'manual_baseline_reviewed_model.joblib',
    'automatic_model':output_dir/'automatic_features_reviewed_model.joblib',
    'manual_control':output_dir/'manual_control_100_7_reviewed_model.joblib'
}
artifact_hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in artifact_paths.items()}
manual_record=json.loads(artifact_paths['manual_results'].read_text())
automatic_record=json.loads(artifact_paths['automatic_results'].read_text())
manual_saved=joblib.load(artifact_paths['manual_model'])
automatic_saved=joblib.load(artifact_paths['automatic_model'])
control_saved=joblib.load(artifact_paths['manual_control'])
assert manual_saved['metadata']==manual_record and automatic_saved['metadata']==automatic_record
for key in ['client_ids','asset_ids','train_months','validation_months','test_months',
            'transactions_sha256','label_definition','metric_definitions','tie_break','observation','history_protocol']:
    assert manual_record[key]==automatic_record[key],key
assert manual_saved['features']==manual_record['features']
assert automatic_saved['features']==automatic_record['features']
assert control_saved['features']==manual_record['features']
for key in ['transactions_sha256','client_ids','asset_ids','train_months','validation_months']:
    assert control_saved['metadata'][key]==manual_record[key]
assert control_saved['metadata']['reference_results_sha256']==artifact_hashes['manual_results']
for saved,meta in [(manual_saved,manual_record),(automatic_saved,automatic_record)]:
    for key,value in meta['parameters'].items():assert saved['model'].get_params()[key]==value
    assert list(saved['model'].feature_names_in_)==saved['features']
assert (control_saved['model'].max_iter,control_saved['model'].max_leaf_nodes)==(100,7)
assert control_saved['model'].random_state==42 and not control_saved['model'].early_stopping
assert control_saved['model'].learning_rate==.1
assert sklearn.__version__==manual_record['versions']['scikit_learn']==automatic_record['versions']['scikit_learn']
transactions_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest()
assert transactions_sha256==manual_record['transactions_sha256']
client_ids=manual_record['client_ids'];asset_ids=manual_record['asset_ids']
assert len(set(client_ids))==500 and len(set(asset_ids))==50
manual_features=manual_record['features']
candidate_plan=automatic_record['candidate_plan']
all_features=manual_features+[entry['name'] for entry in candidate_plan]
assert set(automatic_saved['features']).issubset(all_features)
seed=manual_record['tie_break']['seed']
assert seed==42
checks={'source_artifacts_consistent':True,'csv_hash_matches':True,'models_and_parameters_frozen':True}
print('Ручная модель:',len(manual_saved['features']),'признаков,',manual_record['parameters'])
print('Автоматическая:',len(automatic_saved['features']),'признаков,',automatic_record['parameters'])
print('Обучений в финальном сравнении: 0.')
''')
md('''
## 3. Явно задаём период до оценки
Таблица ниже сравнивает только календарные границы, а не сделки и метрики test.
`pending` — решение о периоде не получено; `approved_complete_months` — пользователь
явно согласовал оценку июля–октября с сохранением ноября закрытым;
`keep_reserved_period` — сохраняем весь исходный период и ждём подтверждения полноты.
Параметр не подбирается по результатам. Проверка календаря не доказывает полноту наблюдений
для каждого клиента или актива.
''')
code('''
evaluation_policy='__MODE__'
observation_start=pd.Timestamp(manual_record['observation']['start_inclusive'])
observation_end_exclusive=pd.Timestamp(manual_record['observation']['end_exclusive'])
reserved_dates=pd.date_range(*manual_record['test_months'],freq='MS')
def calendar_supported(date):
    date=pd.Timestamp(date)
    return bool(date>=observation_start and date+pd.DateOffset(months=1)<=observation_end_exclusive)
calendar_table=pd.DataFrame({
    'prediction_date':reserved_dates,
    'required_end_exclusive':[date+pd.DateOffset(months=1) for date in reserved_dates],
    'documented_month_supported':[calendar_supported(date) for date in reserved_dates]
})
print(calendar_table.to_string(index=False))
if evaluation_policy=='approved_complete_months':
    evaluation_dates=pd.date_range('2022-07-01','2022-10-01',freq='MS')
    period_reason='User explicitly approved July–October before seeing outcomes; November remains closed.'
elif evaluation_policy=='keep_reserved_period':
    evaluation_dates=reserved_dates
    period_reason='Original July–November retained; evaluation waits for documented complete boundary.'
else:
    evaluation_dates=pd.DatetimeIndex([])
    period_reason='Awaiting explicit choice of final evaluation period.'

test_gate_open=bool(len(evaluation_dates) and all(calendar_supported(date) for date in evaluation_dates))
evaluation_start=evaluation_dates.min() if len(evaluation_dates) else None
evaluation_end_exclusive=evaluation_dates.max()+pd.DateOffset(months=1) if len(evaluation_dates) else None
excluded_reserved_dates=[str(date.date()) for date in reserved_dates if date not in evaluation_dates]
status=('ready_for_fixed_model_evaluation' if test_gate_open else 'awaiting_period_resolution')
print('Статус:',status)
print('Причина:',period_reason)
if test_gate_open:print('Оценка:',str(evaluation_start.date()),'—',str((evaluation_end_exclusive-pd.Timedelta(days=1)).date()))
else:print('ОСТАНОВКА ПЕРЕД ОЦЕНКОЙ: реальные данные и метрики test не открываются.')

protocol={
    'evaluation_policy':evaluation_policy,'period_reason':period_reason,
    'original_reserved_test_months':manual_record['test_months'],
    'evaluation_dates':[str(date.date()) for date in evaluation_dates],
    'source_artifact_sha256':artifact_hashes,'transactions_sha256':transactions_sha256,
    'models_frozen':True,'new_training_fits':0,
    'primary_comparison':'automatic_selected - manual_selected',
    'secondary_comparisons':['automatic_selected - pair_then_popularity','automatic_selected - manual_100_7_control'],
    'comparison_methods':['manual_selected','automatic_selected','manual_100_7_control','pair_then_popularity','popularity'],
    'metric_definitions':manual_record['metric_definitions'],'tie_break':manual_record['tie_break'],
    'bootstrap':{'seed':20261013,'repeats':5000,'unit':'client','replacement':True,'pool_size':500,'draw_size':500},
    'period_selected_before_outcomes':evaluation_policy in ['approved_complete_months','keep_reserved_period'],
    'test_evaluated':False
}
protocol_path=output_dir/'final_comparison_protocol.json'
if protocol_path.exists():
    old_protocol=json.loads(protocol_path.read_text())
    assert old_protocol['source_artifact_sha256']==artifact_hashes
    if old_protocol.get('test_evaluated'):
        for key in ['evaluation_policy','evaluation_dates','primary_comparison','secondary_comparisons','metric_definitions','tie_break','bootstrap']:
            assert old_protocol[key]==protocol[key], 'Нельзя менять протокол после просмотра test: '+key
        protocol['test_evaluated']=True
protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False))
checks['period_gate_before_test_access']=True
'''.replace('__MODE__',mode))
md('''
## 4. Одна функция признаков и ответа
Считаем прежние пять ручных признаков по календарным окнам и кандидаты по дневным.
Модель получает только свой сохранённый список столбцов. Общая таблица нужна для одинаковых
кандидатов и групповой диагностики. Популярность — покупки всех клиентов, история клиента
включает активы вне каталога. Ответ — уникальная покупка пары за целевой месяц.
''')
code('''
group_columns={'user':['customerID'],'item':['ISIN'],'user_item':['customerID','ISIN']}
pair_keys=['customerID','ISIN']
case_keys=['customerID','prediction_date']
candidate_pairs=pd.DataFrame({'customerID':client_ids}).merge(pd.DataFrame({'ISIN':asset_ids}),how='cross')
def build_month(buy_history,pairs,cutoff):
    cutoff=pd.Timestamp(cutoff);end=cutoff+pd.DateOffset(months=1)
    if not cutoff.is_month_start or cutoff!=cutoff.normalize():raise ValueError('Нужно первое число месяца')
    if not calendar_supported(cutoff):raise ValueError('Целевой месяц не входит полностью в документированный период')
    result=pairs.copy()
    for entry in candidate_plan:
        recent=buy_history[(buy_history['timestamp']>=cutoff-pd.Timedelta(days=entry['days']))
                           &(buy_history['timestamp']<cutoff)]
        columns=group_columns[entry['group']];name=entry['name']
        counts=recent.groupby(columns).size().reset_index(name=name)
        result=result.merge(counts,on=columns,how='left',validate='many_to_one')
        result[name]=result[name].fillna(0).astype(int)
    for months in [1,3]:
        recent=buy_history[(buy_history['timestamp']>=cutoff-pd.DateOffset(months=months))
                           &(buy_history['timestamp']<cutoff)]
        for group in (['user','user_item','item'] if months==1 else ['user','user_item']):
            name=f'{group}_buy_count_{months}m';columns=group_columns[group]
            counts=recent.groupby(columns).size().reset_index(name=name)
            result=result.merge(counts,on=columns,how='left',validate='many_to_one')
            result[name]=result[name].fillna(0).astype(int)
    future=buy_history[(buy_history['timestamp']>=cutoff)&(buy_history['timestamp']<end)]
    labels=future[pair_keys].drop_duplicates().assign(label=1)
    result=result.merge(labels,on=pair_keys,how='left',validate='one_to_one')
    result['label']=result['label'].fillna(0).astype(int);result['prediction_date']=cutoff
    return result[pair_keys+['prediction_date']+all_features+['label']]
''')
md('''
## 5. Общие метрики и bootstrap
Используем те же определения: основные Recall, Precision и NDCG — средние по клиент–месяцам
с покупкой внутри каталога; `Precision@5_all` включает все случаи. LogLoss и AUC — по всем парам.
У правил ранжирования они неприменимы. Равенства разрешаются тем же хешем ключа и seed.
Bootstrap выбирает клиентов, сохраняет все месяцы и кратность выбора. Выборка без учитываемых
случаев даёт пропуск, исключается из процентилей и отдельно подсчитывается.
''')
metric_source=(root/'outputs/automatic_feature_metrics.py').read_text()
metric_source=metric_source.replace("float(sums[:,j].sum()/counts.sum())", "float(sums[:,j].sum()/counts.sum()) if counts.sum() else np.nan")
code(metric_source)
md('''
## 6. Проверки на искусственных примерах
Этот раздел работает даже при закрытом test. Он проверяет метрики, пустые случаи, порядок
равенств и временную границу. Событие до cutoff учитывается, на cutoff и после — нет.
Синтетические данные не содержат реальных покупок test.
''')
code('''
fixture_date=pd.Timestamp('2022-07-01')
fixture_pairs=pd.DataFrame({'customerID':['A'],'ISIN':['x']})
fixture_history=pd.DataFrame({'customerID':['A']*3,'ISIN':['x']*3,
    'timestamp':[fixture_date-pd.Timedelta(nanoseconds=1),fixture_date,fixture_date+pd.Timedelta(nanoseconds=1)]})
one_event=build_month(fixture_history.iloc[:1],fixture_pairs,fixture_date)
with_future=build_month(fixture_history,fixture_pairs,fixture_date)
pd.testing.assert_frame_equal(one_event[all_features],with_future[all_features])
assert one_event[all_features].eq(1).all().all()
assert build_month(fixture_history.iloc[1:],fixture_pairs,fixture_date)[all_features].eq(0).all().all()
try:build_month(fixture_history,fixture_pairs,'2022-11-01')
except ValueError:pass
else:raise AssertionError('Полный ноябрь не подтверждён текущей границей')
fixture=pd.DataFrame({'customerID':['A']*6+['B']*6,'prediction_date':[fixture_date]*12,
                     'ISIN':list('abcdef')*2,'label':[1,0,1,0,0,1]+[0]*6})
fixture=prepare_ranking(fixture)
p=np.tile(np.arange(6,0,-1)/7,2)
measured=evaluate(fixture,p)
assert np.isclose(measured['Recall@5'],2/3) and np.isclose(measured['Precision@5'],.4)
assert np.isclose(measured['Precision@5_all'],.2)
assert np.isclose(measured['NDCG@5'],1.5/(1+1/np.log2(3)+.5))
assert evaluate(fixture[fixture['customerID']=='B'],p[6:])['ROC-AUC'] is None
shuffled=fixture.sample(frac=1,random_state=123)
columns=['customerID','prediction_date','ISIN','_rank']
pd.testing.assert_frame_equal(top5(fixture,p)[columns].reset_index(drop=True),
                              top5(shuffled,pd.Series(p,index=fixture.index))[columns].reset_index(drop=True))
changed=fixture.assign(label=1-fixture['label'])
pd.testing.assert_frame_equal(top5(fixture,p)[columns].reset_index(drop=True),top5(changed,p)[columns].reset_index(drop=True))
# Явная кратность: A имеет два случая, B один. Выбор A,A,B даёт 5 случаев, не 3 клиента.
assert np.isclose((.6+.6+.8)/(2+2+1),.4)
empty=paired_bootstrap(fixture[fixture['customerID']=='B'],p[6:],
                      {'same':{'probabilities':p[6:]}},['B'],repeats=10,seed=123)
assert empty.iloc[0]['empty_draws']==10 and pd.isna(empty.iloc[0]['ci_low'])
checks.update(cutoff_boundary_fixture=True, metric_fixture_with_zero_buys=True,
              one_class_auc=True, row_order_invariance_fixture=True, label_invariance_fixture=True,
              bootstrap_multiplicity_fixture=True, empty_bootstrap_draw=True,
              unsupported_november_guard=True)
print('Синтетические проверки пройдены.')
''')
md('''
## 7. Строим только согласованный test
Реальные сделки загружаются **только при открытом календарном шлюзе**. Для признаков нужны
три месяца перед первым прогнозом и история до конца выбранного периода. Последующие месяцы
используют уже наблюдённые прошлые покупки, но ни одна модель не переобучается.
При согласованном июле–октябре ноябрьские события исключаются из анализа.
''')
code('''
test=None;all_buys=None
if test_gate_open:
    transactions=pd.read_csv(data_path,dtype={'customerID':'string','ISIN':'string'})
    transactions['timestamp']=pd.to_datetime(transactions['timestamp'],errors='raise')
    history_start=evaluation_start-pd.DateOffset(months=3)
    all_buys=transactions[(transactions['transactionType']=='Buy')
                          &(transactions['timestamp']>=history_start)
                          &(transactions['timestamp']<evaluation_end_exclusive)].copy()
    del transactions
    test=pd.concat([build_month(all_buys,candidate_pairs,date) for date in evaluation_dates],ignore_index=True)
    assert not test.duplicated(pair_keys+['prediction_date']).any()
    assert test.groupby(case_keys).size().eq(50).all()
    assert len(test)==500*50*len(evaluation_dates)
    assert not test[all_features].isna().any().any() and test['label'].isin([0,1]).all()
    for group in ['user','item','user_item']:
        for short,long in zip([7,14,30],[14,30,60]):
            assert (test[f'{group}_buy_count_{short}d']<=test[f'{group}_buy_count_{long}d']).all()
    for days in [7,14,30,60]:
        assert (test[f'user_item_buy_count_{days}d']<=test[f'user_buy_count_{days}d']).all()
        assert (test[f'user_item_buy_count_{days}d']<=test[f'item_buy_count_{days}d']).all()
    test=prepare_ranking(test)
    checks.update(unique_test_pair_months=True,fifty_candidates_per_case=True,counter_bounds=True)
    print('Подготовлено строк:',len(test),'| месяцев:',len(evaluation_dates))
else:print('Test не построен: требуется решение о полном периоде.')
''')
md('''
## 8. Прогнозы и общее/месячное сравнение
Готовые модели выдают вероятности; выбираем пять наибольших для каждого клиента и месяца.
Сравниваем выбранные модели, ручной контроль 100/7 и два заранее заданных правила.
Правила сортируют по прошлым покупкам пары и/или популярности. По результатам test
не выбираем новый набор признаков, настройки или нового победителя для последующей доработки.
''')
code('''
method_specs={};top_lists={};overall=pd.DataFrame();monthly=pd.DataFrame();denominators=pd.DataFrame()
metric_names=['Recall@5','Precision@5','NDCG@5','Precision@5_all','LogLoss','ROC-AUC']
if test is not None:
    with threadpool_limits(limits=4):
        for name,saved in [('manual_selected',manual_saved),('automatic_selected',automatic_saved),('manual_100_7_control',control_saved)]:
            probabilities=saved['model'].predict_proba(test[saved['features']])[:,1]
            method_specs[name]={'probabilities':probabilities}
    method_specs['pair_then_popularity']={'criteria':['user_item_buy_count_3m','item_buy_count_1m']}
    method_specs['popularity']={'criteria':['item_buy_count_1m']}
    top_lists={name:top5(test,**spec) for name,spec in method_specs.items()}
    overall=pd.DataFrame([{'method':name,**evaluate(test,**spec)} for name,spec in method_specs.items()])
    monthly_rows=[]
    for date,month in test.groupby('prediction_date'):
        for name,spec in method_specs.items():
            sliced=({'criteria':spec['criteria']} if 'criteria' in spec else
                    {'probabilities':pd.Series(spec['probabilities'],index=test.index).reindex(month.index)})
            monthly_rows.append({'month':str(date.date()),'method':name,**evaluate(month,**sliced)})
    monthly=pd.DataFrame(monthly_rows)
    count_fields=['candidate_pairs','positive_pairs','total_user_months','eligible_user_months','eligible_unique_clients']
    count_rows=[{'period':'evaluated_test',**{key:int(overall.iloc[0][key]) for key in count_fields}}]
    for date in evaluation_dates:
        row=monthly[monthly['month']==str(date.date())].iloc[0]
        count_rows.append({'period':str(date.date()),**{key:int(row[key]) for key in count_fields}})
    denominators=pd.DataFrame(count_rows)
    print('Общее сравнение:')
    print(overall[['method']+metric_names].round(5).to_string(index=False,na_rep='—'))
    print('\\nМесячное сравнение:')
    print(monthly[['month','method']+metric_names].round(5).to_string(index=False,na_rep='—'))
    print('\\nЗнаменатели:')
    print(denominators.to_string(index=False))
else:print('Метрики test не вычислены.')
''')
md('''
## 9. Покрытие каталога и группы покупок
Покрытие — уникальные тройки клиент–актив–месяц с Buy внутри каталога / такие же покупки
этих клиентов по всем активам. Несколько сделок пары за месяц дают одну покупку.
Повторная положительная пара имеет Buy за три календарных месяца перед прогнозом;
новая — не имеет, но может покупаться ранее в жизни.

Групповой Recall считаем по **исходным top-5**, не фильтруя кандидатов по будущим ответам.
Знаменатель — все положительные активы группы внутри каталога в случае; среднее только
по случаям с положительной покупкой группы. Групповые средние не складываются в общий Recall.
''')
code('''
coverage=pd.DataFrame();purchase_diagnostics=pd.DataFrame()
def coverage_row(period,triples):
    inside=int(triples['ISIN'].isin(asset_ids).sum())
    return {'period':period,'catalog_unique_buys':inside,'all_unique_buys':len(triples),
            'coverage':inside/len(triples) if len(triples) else None}
def group_results(data,top,period,method):
    rows=[]
    for group,repeated in [('new_3m',False),('repeat_3m',True)]:
        positives=data[(data['label']==1)&((data['user_item_buy_count_3m']>0)==repeated)]
        totals=positives.groupby(case_keys).size()
        hits=top[(top['label']==1)&((top['user_item_buy_count_3m']>0)==repeated)]
        found=hits.groupby(case_keys).size().reindex(totals.index,fill_value=0)
        rows.append({'period':period,'method':method,'group':group,'Recall@5':float((found/totals).mean()),
                     'positive_pairs':len(positives),'eligible_user_months':len(totals),
                     'eligible_unique_clients':int(totals.index.get_level_values('customerID').nunique())})
    return rows
if test is not None:
    selected_buys=all_buys[(all_buys['customerID'].isin(client_ids))&(all_buys['timestamp']>=evaluation_start)].copy()
    selected_buys['prediction_date']=selected_buys['timestamp'].dt.to_period('M').dt.to_timestamp()
    triples=selected_buys[['customerID','ISIN','prediction_date']].drop_duplicates()
    coverage_rows=[coverage_row('evaluated_test',triples)]
    coverage_rows += [coverage_row(str(date.date()),triples[triples['prediction_date']==date]) for date in evaluation_dates]
    coverage=pd.DataFrame(coverage_rows)
    assert coverage_rows[0]['catalog_unique_buys']==int(test['label'].sum())
    purchase_rows=[]
    for name,top in top_lists.items():
        purchase_rows += group_results(test,top,'evaluated_test',name)
        for date,month in test.groupby('prediction_date'):
            purchase_rows += group_results(month,top[top['prediction_date']==date],str(date.date()),name)
    purchase_diagnostics=pd.DataFrame(purchase_rows)
    print('Покрытие:');print(coverage.round(5).to_string(index=False,na_rep='—'))
    print('\\nОбщая диагностика:')
    print(purchase_diagnostics[purchase_diagnostics['period']=='evaluated_test'].round(5).to_string(index=False,na_rep='—'))
    print('\\nМесячная диагностика:')
    print(purchase_diagnostics[purchase_diagnostics['period']!='evaluated_test'].round(5).to_string(index=False,na_rep='—'))
else:print('Покрытие и группы test не исследованы.')
''')
md('''
## 10. Парный bootstrap и итоговые проверки
5000 раз выбираем из всех 500 клиентов с возвращением. Одни выборки используются для
сравнений выбранной автоматической модели с ручной, с правилом пары и с ручным контролем 100/7.
Сохраняем месяцы и кратность; пересчитываем среднее по учитываемым случаям, а не среднее
клиентских средних. Первое сравнение — основное, остальные диагностические.
Интервалы описывают неопределённость по клиентам этого фиксированного периода; они не охватывают
все риски смены рынка или ограничения происхождения FAR-Trans.
''')
code('''
bootstrap=pd.DataFrame();bootstrap_seed=20261013;bootstrap_repeats=5000
if test is not None:
    bootstrap=paired_bootstrap(test,method_specs['automatic_selected']['probabilities'],
        references={
            'manual_selected':method_specs['manual_selected'],
            'pair_then_popularity':method_specs['pair_then_popularity'],
            'manual_100_7_control':method_specs['manual_100_7_control']
        },client_ids=client_ids,repeats=bootstrap_repeats,seed=bootstrap_seed)
    print(bootstrap.round(6).to_string(index=False,na_rep='—'))
    shuffled=test.sample(frac=1,random_state=123);changed=shuffled.assign(label=1-shuffled['label'])
    columns=['customerID','prediction_date','ISIN','_rank']
    for name,spec in method_specs.items():
        top=top_lists[name]
        assert top.groupby(case_keys).size().eq(5).all()
        other=spec.copy()
        if 'probabilities' in other:
            keyed=pd.Series(other['probabilities'],index=pd.MultiIndex.from_frame(test[['customerID','prediction_date','ISIN']]))
            other['probabilities']=pd.Series(keyed.reindex(pd.MultiIndex.from_frame(shuffled[['customerID','prediction_date','ISIN']])).to_numpy(),index=shuffled.index)
        pd.testing.assert_frame_equal(top[columns].reset_index(drop=True),top5(shuffled,**other)[columns].reset_index(drop=True))
        pd.testing.assert_frame_equal(top[columns].reset_index(drop=True),top5(changed,**other)[columns].reset_index(drop=True))
        overall_row=overall[overall['method']==name].iloc[0];month_rows=monthly[monthly['method']==name]
        for metric in ['Recall@5','Precision@5','NDCG@5']:
            eligible_rows=month_rows['eligible_user_months']>0
            if eligible_rows.any():assert np.isclose(overall_row[metric],np.average(month_rows.loc[eligible_rows,metric],weights=month_rows.loc[eligible_rows,'eligible_user_months']))
        assert np.isclose(overall_row['Precision@5_all'],np.average(month_rows['Precision@5_all'],weights=month_rows['total_user_months']))
        if pd.notna(overall_row['LogLoss']):assert np.isclose(overall_row['LogLoss'],np.average(month_rows['LogLoss'],weights=month_rows['candidate_pairs']))
        total_groups=purchase_diagnostics[(purchase_diagnostics['method']==name)&(purchase_diagnostics['period']=='evaluated_test')]
        assert total_groups['positive_pairs'].sum()==int(test['label'].sum())
    # Независимая проверка первых десяти реальных bootstrap-выборок.
    deltas=[recall_cases(test,**method_specs['automatic_selected'])-recall_cases(test,**method_specs[name])
            for name in ['manual_selected','pair_then_popularity','manual_100_7_control']]
    values_by_client=[{client:delta[delta.index.get_level_values('customerID')==client].dropna().to_numpy()
                      for client in client_ids} for delta in deltas]
    rng=np.random.default_rng(bootstrap_seed)
    eligible_cases=recall_cases(test,**method_specs['automatic_selected']).notna()
    counts=eligible_cases.groupby(level='customerID').sum().reindex(client_ids,fill_value=0).to_numpy()
    sums=np.column_stack([delta.dropna().groupby(level='customerID').sum().reindex(client_ids,fill_value=0).to_numpy()
                          for delta in deltas])
    reconstructed=np.full((bootstrap_repeats,len(deltas)),np.nan)
    for iteration in range(bootstrap_repeats):
        draw=rng.integers(0,len(client_ids),size=len(client_ids))
        denominator=counts[draw].sum()
        if denominator:reconstructed[iteration]=sums[draw].sum(axis=0)/denominator
        if iteration<10:
            explicit=[np.concatenate([values[client_ids[index]] for index in draw]).mean()
                      if sum(len(values[client_ids[index]]) for index in draw) else np.nan
                      for values in values_by_client]
            assert np.allclose(reconstructed[iteration],explicit,equal_nan=True)
    for column,row in enumerate(bootstrap.to_dict('records')):
        values=reconstructed[:,column];valid=np.isfinite(values)
        expected=np.quantile(values[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
        assert np.allclose([row['ci_low'],row['ci_high']],expected,equal_nan=True)
        assert row['empty_draws']==int((~valid).sum())
    checks.update(five_recommendations_each=True,key_aligned_row_invariance=True,
                  fixed_predictions_label_invariance=True,monthly_overall_weights=True,group_denominators=True,
                  bootstrap_real_draws_reconstructed=True,coverage_label_consistency=True)
# Ни веса, ни исходные артефакты не должны измениться.
assert artifact_hashes=={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in artifact_paths.items()}
checks['input_artifacts_unchanged']=True
print('Выполненные проверки:',', '.join(checks))
''')
md('''
## 11. Выводы из фактических результатов
Выводы формируются из таблиц. Recall — доля найденных купленных активов среди учитываемых
случаев, не доля правильных рекомендаций. LogLoss оценивает вероятности, NDCG — позиции
попаданий в списке. Разные метрики могут измениться в разных направлениях.
Для групп без отдельных интервалов сообщаем только наблюдаемую разницу.
''')
code('''
conclusions=[]
test_evaluated=test is not None and not overall.empty
full_reserved_test_evaluated=bool(test_evaluated and list(evaluation_dates)==list(reserved_dates))
if test_evaluated:
    scores=overall.set_index('method');auto=scores.loc['automatic_selected'];manual=scores.loc['manual_selected']
    conclusions=[
        f"Оценён явно согласованный период: {str(evaluation_start.date())} — {str((evaluation_end_exclusive-pd.Timedelta(days=1)).date())}.",
        f"Первоначальный test июль–ноябрь не менялся в старых артефактах; полная первоначальная проверка выполнена: {full_reserved_test_evaluated}.",
        f"Recall ручной модели {manual['Recall@5']:.4f}, автоматической {auto['Recall@5']:.4f}; разница {auto['Recall@5']-manual['Recall@5']:+.4f}.",
        f"ΔLogLoss={auto['LogLoss']-manual['LogLoss']:+.5f} (меньше лучше), ΔPrecision={auto['Precision@5']-manual['Precision@5']:+.4f}, ΔNDCG={auto['NDCG@5']-manual['NDCG@5']:+.4f}.",
        f"Случаев для основных метрик: {int(auto['eligible_user_months'])} из {int(auto['total_user_months'])}; клиентов {int(auto['eligible_unique_clients'])}.",
        f"Каталог покрывает {coverage.iloc[0]['coverage']:.2%} уникальных покупок выбранных клиентов в оценённом периоде.",
        "Обучения, нового отбора и изменения настроек после просмотра test не было."
    ]
    for row in bootstrap.to_dict('records'):
        note=('интервал не определён: нет учитываемых случаев' if row['valid_draws']==0 else
              'интервал включает ноль, устойчивое преимущество не подтверждено' if row['contains_zero'] else
              'интервал не включает ноль; вывод относится к этим клиентам и периоду')
        conclusions.append(f"Автоматическая минус {row['reference']}: Δ={row['difference']:+.4f}, 95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]; {note}.")
    for group in ['new_3m','repeat_3m']:
        rows=purchase_diagnostics[(purchase_diagnostics['period']=='evaluated_test')&(purchase_diagnostics['group']==group)].set_index('method')
        a=rows.loc['automatic_selected','Recall@5'];m=rows.loc['manual_selected','Recall@5']
        conclusions.append(f"{group}: ручной Recall {m:.4f}, автоматический {a:.4f}; наблюдаемая разница {a-m:+.4f}, отдельный интервал группы не рассчитан.")
    conclusions.append('Ноябрьские ответы не исследованы, если месяц исключён согласованным решением.')
else:
    conclusions=[
        'Код итогового сравнения полностью подготовлен и проверен на искусственных данных.',
        'Реальные метрики test не вычислены: решение о полном периоде не принято либо граница не подтверждена.',
        'Нельзя представлять результаты validation как независимые результаты test.'
    ]
conclusions += [
    'Это автоматическое расширение ручной базы в заданном пространстве, не полностью автономное проектирование.',
    'Ограничения реконструированных событий FAR-Trans и фиксированного каталога сохраняются.',
    'По этим метрикам не доказывается прибыльность, финансовая пригодность или причинный эффект рекомендаций.'
]
print('\\n'.join(conclusions))
''')
md('''
## 12. Сохраняем протокол и результаты
JSON содержит исходные хеши, выбранный период, определения, таблицы, bootstrap, проверки
и ограничения. Неопределённые метрики — `null`. Если шлюз закрыт, результаты test — `null`,
статус ожидания записан явно. Старые модели и результаты остаются без изменений.
''')
code('''
def clean_json(value):
    if isinstance(value,dict):return {key:clean_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [clean_json(item) for item in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value
protocol['test_evaluated']=test_evaluated
protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2,allow_nan=False))
results={
    'status':'completed_on_approved_period' if test_evaluated else 'awaiting_period_resolution',
    'protocol':protocol,'test_evaluated':test_evaluated,'full_reserved_test_evaluated':full_reserved_test_evaluated,
    'excluded_reserved_dates':excluded_reserved_dates,'excluded_months_not_inspected':True,
    'observation':manual_record['observation'],
    'observation_rechecked_sources':{
        'checked_on':'2026-10-13','repository':'https://researchdata.gla.ac.uk/1658/',
        'readme':'https://researchdata.gla.ac.uk/1658/2/README.txt',
        'paper':'https://arxiv.org/abs/2407.08692',
        'author_code':'https://github.com/JavierSanzCruza/far-trans',
        'full_november_confirmed':False,'max_timestamp_used_as_proof':False},
    'model_parameters':{'manual':manual_record['parameters'],'automatic':automatic_record['parameters']},
    'model_features':{'manual':manual_saved['features'],'automatic':automatic_saved['features']},
    'metric_definitions':manual_record['metric_definitions'],'tie_break':manual_record['tie_break'],
    'overall':overall.to_dict('records') if test_evaluated else None,
    'monthly':monthly.to_dict('records') if test_evaluated else None,
    'denominators':denominators.to_dict('records') if test_evaluated else None,
    'catalog_coverage':coverage.to_dict('records') if test_evaluated else None,
    'repeat_definition':automatic_record['repeat_definition'],'new_definition':automatic_record['new_definition'],
    'purchase_diagnostics':purchase_diagnostics.to_dict('records') if test_evaluated else None,
    'bootstrap':{'seed':bootstrap_seed,'repeats':bootstrap_repeats,'paired_client_sampling':True,
                 'pool_size':500,'draw_size':500,'all_months_and_multiplicity_retained':True,
                 'empty_draws_excluded_and_counted':True,
                 'results':bootstrap.to_dict('records') if test_evaluated else None},
    'checks':checks,'conclusions':conclusions,'new_training_fits':0,
    'previous_experiment_budget':{'manual':manual_record['experiment_budget'],'automatic':automatic_record['experiment_budget']},
    'limits':automatic_record['limits']+[
        'If approved, July–October is an explicit pre-outcome period amendment; original November remains unevaluated.',
        'Client bootstrap describes this fixed time period and does not cover all future market regimes.'
    ],
    'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                'scikit_learn':sklearn.__version__,'joblib':joblib.__version__}
}
results=clean_json(results)
(output_dir/'final_comparison_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Сохранены final_comparison_protocol.json и final_comparison_results.json.')
print('test_evaluated:',test_evaluated,'| весь исходный test оценён:',full_reserved_test_evaluated)
''')
md('''
## Что важно уметь объяснить
1. Человек задаёт пространство кандидатов; программа генерирует и отбирает внутри него.
2. Train обучает, validation выбирает, согласованный test проверяет зафиксированные решения.
3. Признаки используют прошлое; будущие покупки нужны только для ответов.
4. Повторные сделки пары за месяц дают один положительный ответ; счётчики истории считают сделки.
5. Recall, Precision, NDCG и LogLoss оценивают разные свойства результата.
6. Bootstrap сохраняет месяцы и повторный выбор клиента, а не усредняет клиентские средние.
7. Календарная граница, фиксированный каталог и обработка авторов ограничивают выводы.
8. Период проверки нельзя менять после просмотра качества; исключение ноября должно быть явным.
''')
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python (.venv)','language':'python','name':'python3'},
    'language_info':{'name':'python'},'stage':'final_comparison','evaluation_policy':mode,'test_evaluated':False},
    'nbformat':4,'nbformat_minor':5}
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
print('Built final notebook:',len(cells),'cells; policy:',mode)
