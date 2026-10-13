"""Aggregate data diagnostics, explicitly excluding November 2022 outcomes."""
import argparse,hashlib,io,json,pathlib,zipfile
import numpy as np,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);args=p.parse_args();root=args.source
manual=json.loads((root/'outputs/manual_baseline_reviewed_results.json').read_text())
csv=root/'work/data/FAR-Trans/transactions.csv'
if csv.exists():transactions=pd.read_csv(csv,dtype={'customerID':'string','ISIN':'string'})
else:
 with zipfile.ZipFile(root/'FAR-Trans.zip') as z:raw=z.read('FAR-Trans/transactions.csv')
 assert hashlib.sha256(raw).hexdigest()=='358871fa1311a802799df86e56c3d633513566965b01412d058e0f8fbe6cd54b'
 transactions=pd.read_csv(io.BytesIO(raw),dtype={'customerID':'string','ISIN':'string'})
transactions['timestamp']=pd.to_datetime(transactions.timestamp,errors='raise')
transactions=transactions[transactions.timestamp<pd.Timestamp('2022-11-01')].copy()
buys=transactions[transactions.transactionType=='Buy'].copy()
selection=buys[(buys.timestamp>='2019-10-01')&(buys.timestamp<'2020-04-01')]
selected=selection.customerID.drop_duplicates().sample(n=500,random_state=42).tolist();assets=selection.ISIN.value_counts().head(50).index.tolist()
assert selected==manual['client_ids'] and assets==manual['asset_ids']
record={'november_outcomes_excluded':True,'diagnostic_end_exclusive':'2022-11-01','rows_before_end':len(transactions),'clients_before_end':int(transactions.customerID.nunique()),'assets_before_end':int(transactions.ISIN.nunique()),'exact_duplicate_rows':int(transactions.duplicated().sum()),'duplicate_transaction_id_rows':int(transactions.duplicated('transactionID').sum()),'missing_key_cells':int(transactions[['customerID','ISIN','timestamp','transactionType']].isna().sum().sum()),'transaction_types':transactions.transactionType.value_counts().to_dict(),'selection_eligible_clients':int(selection.customerID.nunique()),'selection_eligible_assets':int(selection.ISIN.nunique()),'cohort_order_reproduced':True,'catalog_order_reproduced':True,'calendar_day_timestamp_only':bool(transactions.timestamp.eq(transactions.timestamp.dt.normalize()).all()),'periods':[]}
for name,start,end in [('train','2020-04-01','2022-01-01'),('validation','2022-01-01','2022-07-01'),('test','2022-07-01','2022-11-01')]:
 b=buys[(buys.timestamp>=start)&(buys.timestamp<end)&(buys.customerID.isin(selected))].copy();b['month']=b.timestamp.dt.to_period('M').astype(str)
 triples=b[['customerID','ISIN','month']].drop_duplicates();inside=triples[triples.ISIN.isin(assets)]
 repeated_rows=len(b)-len(triples)
 record['periods'].append({'period':name,'buy_transaction_rows':len(b),'unique_positive_triples_all_catalogs':len(triples),'in_catalog_positive_triples':len(inside),'coverage':len(inside)/len(triples),'more_than_one_transaction_per_pair_month_extra_rows':repeated_rows,'positive_clients_in_catalog':int(inside.customerID.nunique())})
dup=transactions[transactions.duplicated('transactionID',keep=False)]
groups=dup.groupby('transactionID')
record['transaction_id_diagnostics']={'groups_with_repeated_id':len(groups),'max_rows_per_id':int(groups.size().max()),'repeated_id_groups_with_multiple_clients':int(groups.customerID.nunique().gt(1).sum()),'repeated_id_groups_with_multiple_assets':int(groups.ISIN.nunique().gt(1).sum()),'repeated_id_groups_with_multiple_dates':int(groups.timestamp.nunique().gt(1).sum()),'repeated_id_groups_with_multiple_types':int(groups.transactionType.nunique().gt(1).sum()),'duplicate_customer_transaction_id_rows':int(transactions.duplicated(['customerID','transactionID']).sum())}
# A source field is historical provenance, not the hash of the currently executed code.
claimed=manual['source_notebook_sha256'];matches=[]
for f in root.rglob('*.ipynb'):
 if '.git' in f.parts:continue
 if hashlib.sha256(f.read_bytes()).hexdigest()==claimed:matches.append(str(f.relative_to(root)))
record['manual_historical_source_hash']={'claimed_sha256':claimed,'matching_files':matches,'current_reviewed_sha256':hashlib.sha256((root/'outputs/manual_baseline_reviewed.ipynb').read_bytes()).hexdigest(),'matches_current_reviewed':hashlib.sha256((root/'outputs/manual_baseline_reviewed.ipynb').read_bytes()).hexdigest()==claimed}
# Quantify record-date metadata; these tables are not used by the frozen models.
record['metadata_availability']={}
for name,key in [('customer_information.csv','customerID'),('asset_information.csv','ISIN')]:
 f=pd.read_csv(root/'work/data/FAR-Trans'/name,dtype={key:'string'});f.timestamp=pd.to_datetime(f.timestamp)
 chosen=f[f[key].isin(selected if key=='customerID' else assets)]
 record['metadata_availability'][name]={'selected_entities':len(selected if key=='customerID' else assets),'entities_with_record_before_first_prediction':int(chosen.loc[chosen.timestamp<'2020-04-01',key].nunique()),'record_count_selected_entities':len(chosen),'earliest_record':str(chosen.timestamp.min().date())}
args.out.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(record,ensure_ascii=False,indent=2))
