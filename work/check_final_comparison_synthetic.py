"""Проверка полного финального пути только на вымышленных событиях.
CSV реальных сделок не парсится. Опубликованные результаты не изменяются.
"""
from pathlib import Path
import json,contextlib,io,time
import pandas as pd,numpy as np
root=Path.cwd();target=root/'work/final_comparison_synthetic_check';target.mkdir(exist_ok=True)
for name in ['manual_baseline_reviewed_results.json','automatic_features_reviewed_results.json',
             'manual_baseline_reviewed_model.joblib','automatic_features_reviewed_model.joblib',
             'manual_control_100_7_reviewed_model.joblib']:
    destination=target/name
    if not destination.exists():destination.symlink_to(root/'outputs'/name)
metadata=json.loads((root/'outputs/manual_baseline_reviewed_results.json').read_text())
rng=np.random.default_rng(123);events=[]
for date in pd.date_range('2022-04-01','2022-10-01',freq='MS'):
    for user in rng.choice(metadata['client_ids'],size=180,replace=False):
        for asset in rng.choice(metadata['asset_ids'],size=int(rng.integers(1,4)),replace=False):
            events.append({'customerID':user,'ISIN':asset,'timestamp':date+pd.Timedelta(days=15),'transactionType':'Buy'})
fake=pd.DataFrame(events).astype({'customerID':'string','ISIN':'string'})
real_read_csv=pd.read_csv
ns={'__name__':'__main__'};start=time.monotonic()
nb=json.loads((root/'outputs/03_final_comparison.ipynb').read_text())
try:
    for i,c in enumerate(nb['cells']):
        if c['cell_type']!='code':continue
        text=''.join(c['source'])
        text=text.replace("output_dir=project_dir/'outputs'", "output_dir=project_dir/'work/final_comparison_synthetic_check'")
        text=text.replace("evaluation_policy='pending'", "evaluation_policy='approved_complete_months'")
        if i==12:
            # Единственное чтение событий в финальном пути подменено искусственными данными.
            pd.read_csv=lambda *args,**kwargs:fake.copy()
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            exec(compile(text,f'<synthetic-final-cell-{i}>','exec'),ns)
        print('Synthetic cell',i,'OK',flush=True)
finally:pd.read_csv=real_read_csv
assert ns['results']['test_evaluated'] and len(ns['test'])==100000
assert ns['results']['full_reserved_test_evaluated'] is False
assert ns['results']['excluded_reserved_dates']==['2022-11-01']
assert ns['results']['new_training_fits']==0
report={'synthetic_events_only':True,'real_transactions_csv_not_parsed':True,
        'full_final_evaluation_path_verified':True,'published_notebook_remains_pending':True,
        'checks':ns['checks'],'elapsed_seconds':time.monotonic()-start}
(target/'synthetic_checks.json').write_text(json.dumps(report,indent=2))
actual=json.loads((root/'outputs/final_comparison_results.json').read_text())
assert actual['test_evaluated'] is False and actual['overall'] is None
print('SYNTHETIC END-TO-END CHECK PASSED',flush=True)
