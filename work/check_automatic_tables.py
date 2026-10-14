from pathlib import Path
import pandas as pd, json, time
project_dir=Path.cwd()
baseline=json.loads((project_dir/'outputs/manual_baseline_reviewed_results.json').read_text())
feature_plan=[{'name':f'{g}_buy_count_{d}d','group':g,'days':d} for g in ['user','item','user_item'] for d in [7,14,30,60]]
candidate_pairs=pd.DataFrame({'customerID':baseline['client_ids']}).merge(pd.DataFrame({'ISIN':baseline['asset_ids']}),how='cross')
group_columns={'user':['customerID'],'item':['ISIN'],'user_item':['customerID','ISIN']}
def generate_features(history,pairs,cutoff,plan):
    result=pairs.copy()
    for feature in plan:
        start=cutoff-pd.Timedelta(days=feature['days'])
        recent_buys=history[(history['timestamp']>=start)&(history['timestamp']<cutoff)]
        columns=group_columns[feature['group']]
        name=feature['name']
        counts=recent_buys.groupby(columns).size().reset_index(name=name)
        result=result.merge(counts,on=columns,how='left',validate='many_to_one')
        result[name]=result[name].fillna(0).astype(int)
    result['prediction_date']=cutoff
    return result
start=time.monotonic()
exec(compile(Path('work/automatic_dataset_step.py').read_text(),'<next-user-cell>','exec'),globals())
reference={}
nb=json.loads(Path('outputs/manual_baseline_reviewed.ipynb').read_text())
for i in [2,4,6]: exec(compile(''.join(nb['cells'][i]['source']),f'<manual-reference-{i}>','exec'),reference)
for features in [manual_features+['label']]:
    pd.testing.assert_frame_equal(dataset[features],reference['dataset'][features],check_dtype=False)
# Independent direct count for a populated user-item-day window, all kinds and dates
for date in [pd.Timestamp('2020-04-01'),pd.Timestamp('2022-06-01')]:
    month=dataset[dataset['prediction_date']==date]
    recent=all_buys[(all_buys['timestamp']>=date-pd.Timedelta(days=60))&(all_buys['timestamp']<date)]
    covered=recent[recent['customerID'].isin(baseline['client_ids'])&recent['ISIN'].isin(baseline['asset_ids'])]
    row=covered.iloc[0]
    example=month[(month['customerID']==row['customerID'])&(month['ISIN']==row['ISIN'])].iloc[0]
    for spec in feature_plan:
        w=all_buys[(all_buys['timestamp']>=date-pd.Timedelta(days=spec['days']))&(all_buys['timestamp']<date)]
        for column in group_columns[spec['group']]: w=w[w[column]==row[column]]
        assert example[spec['name']]==len(w)
try: make_dataset_month('2022-07-01')
except ValueError: pass
else: raise AssertionError('Test should be closed')
print('PASS: manual features and labels identical; 12 feature counts verified; test guard; elapsed',round(time.monotonic()-start,1),'s')
