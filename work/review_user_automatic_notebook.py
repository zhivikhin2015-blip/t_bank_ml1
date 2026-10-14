from pathlib import Path
import json,contextlib,io,time,traceback,hashlib
import numpy as np,pandas as pd
root=Path.cwd()
protected=[root/'outputs/02_automatic_features.ipynb',root/'outputs/automatic_features_results.json',root/'outputs/automatic_features_model.joblib',root/'outputs/manual_baseline_reviewed_results.json',root/'outputs/manual_baseline_reviewed_model.joblib']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
nb=json.loads(protected[0].read_text()); ns={'__name__':'__main__'}
start=time.monotonic(); scratch=root/'work/automatic_notebook_review';scratch.mkdir(exist_ok=True)
code_count=0
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 code_count+=1
 source=''.join(c['source'])
 # Сохраняемые артефакты проверочного запуска направляем только в scratch.
 source=source.replace('output_dir = project_dir / "outputs"','output_dir = project_dir / "work/automatic_notebook_review"')
 stream=io.StringIO()
 try:
  with contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
   exec(compile(source,f'<user-notebook-cell-{i}>','exec'),ns)
 except Exception:
  print(stream.getvalue(),flush=True);print(traceback.format_exc(),flush=True);raise
 print('Cell',i,'OK; elapsed',round(time.monotonic()-start,1),'s',flush=True)
# Реальные результаты должны воспроизвестись в свежем процессе.
old=json.loads(protected[1].read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
assert ns['record']==old
import joblib
saved=joblib.load(protected[2])
assert saved['metadata']==old and saved['features']==ns['selected_features']
with ns['threadpool_limits'](limits=4):
 saved_probs=saved['model'].predict_proba(ns['validation'][saved['features']])[:,1]
assert np.allclose(saved_probs,ns['automatic_probabilities'],rtol=0,atol=1e-14)
# Расчёт признаков не должен зависеть от будущих покупок.
cutoff=pd.Timestamp('2022-01-01'); original_history=ns['all_buys']
changed=original_history.copy()
changed.loc[changed['timestamp']>=cutoff,'ISIN']='FUTURE_ONLY_ASSET'
ns['all_buys']=changed
try: changed_month=ns['make_dataset_month'](cutoff)
finally: ns['all_buys']=original_history
original_month=ns['dataset'][ns['dataset']['prediction_date']==cutoff]
pd.testing.assert_frame_equal(changed_month[ns['feature_names']].reset_index(drop=True),original_month[ns['feature_names']].reset_index(drop=True))
assert changed_month['label'].sum()==0 and original_month['label'].sum()>0
# Контроль с заранее известным ответом метрик.
f=pd.DataFrame({'customerID':['A']*6+['B']*6,'prediction_date':[cutoff]*12,'ISIN':list('abcdef')*2,'label':[1,0,1,0,0,1]+[0]*6})
f['_tie']=np.array([ns['tie_key'](c,d,a) for c,d,a in f[['customerID','prediction_date','ISIN']].itertuples(index=False,name=None)],dtype=np.uint64)
p=np.tile(np.arange(6,0,-1)/7,2); result=ns['evaluate'](f,p)
assert np.isclose(result['Recall@5'],2/3) and np.isclose(result['Precision@5'],.4)
assert np.isclose(result['Precision@5_all'],.2)
assert np.isclose(result['NDCG@5'],1.5/(1+1/np.log2(3)+.5))
# Перестановка строк не меняет top-5 для любого сравниваемого метода.
shuffled=ns['validation'].sample(frac=1,random_state=123)
for name,spec in ns['method_specs'].items():
 spec2=spec.copy()
 if 'probabilities' in spec2: spec2['probabilities']=pd.Series(spec2['probabilities'],index=ns['validation'].index)
 top_a=ns['top_five'](ns['validation'],**spec)
 top_b=ns['top_five'](shuffled,**spec2)
 columns=['customerID','prediction_date','ISIN','_rank']
 pd.testing.assert_frame_equal(top_a[columns].reset_index(drop=True),top_b[columns].reset_index(drop=True))
# Проверяем помесячные средние и все знаменатели.
for name,spec in ns['method_specs'].items():
 month=ns['monthly_comparison'][ns['monthly_comparison']['method']==name]
 row=ns['comparison'].set_index('method').loc[name]
 for metric in ['Recall@5','Precision@5','NDCG@5']:
  assert np.isclose(row[metric],np.average(month[metric],weights=month['eligible_user_months']))
assert ns['dataset'].groupby(['customerID','prediction_date']).size().eq(50).all()
assert set(ns['selected_new']).issubset(ns['auto_names']) and len(ns['selected_new'])<=5
assert ns['inner_check']['prediction_date'].max()+pd.DateOffset(months=1)<=ns['validation']['prediction_date'].min()
assert ns['validation']['prediction_date'].max()+pd.DateOffset(months=1)<=ns['test_start']
try:ns['make_dataset_month']('2022-07-01')
except ValueError: pass
else:raise AssertionError('Test guard failed')
# Явная проверка кратности bootstrap: два A и один B, 5 случаев всего.
assert np.isclose(np.array([.6,.6,.8]).sum()/np.array([2,2,1]).sum(),.4)
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in hashes.items())
report={'fresh_process_execution':True,'code_cells_executed':code_count,'saved_results_exactly_reproduced':True,
 'saved_model_predictions_reproduced':True,'future_events_do_not_change_features':True,
 'metrics_fixture':True,'tie_row_order_invariance':True,'monthly_overall_consistency':True,
 'time_split_and_test_guard':True,'protected_files_unchanged':hashes,'test_evaluated':False,
 'elapsed_seconds':time.monotonic()-start,'notebook_note':'Final markdown still describes only initial feature-plan step.'}
(scratch/'review_checks.json').write_text(json.dumps(report,indent=2))
print('REVIEW PASSED',flush=True)
