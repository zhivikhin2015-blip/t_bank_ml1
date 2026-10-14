from pathlib import Path
import json, numpy as np, joblib, hashlib
root=Path.cwd(); out=root/'outputs'
def strict_load(path):
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
r=strict_load(out/'manual_baseline_reviewed_results.json')
old=strict_load(out/'manual_baseline_results.json')
nb=strict_load(out/'manual_baseline_reviewed.ipynb')
assert r['client_ids']==old['client_ids'] and r['asset_ids']==old['asset_ids']
assert r['features']==old['features'] and r['transactions_sha256']==old['transactions_sha256']
for key in ['train_months','validation_months','test_months']:
    assert r[key]==old[key]
current={v['method_id']:v for v in r['comparison']}
for previous, method in zip(old['comparison'],['constant','repeat_1m','boost_2','boost_3','boost_selected']):
    for metric in ['LogLoss','ROC-AUC']:
        assert np.isclose(previous[metric],current[method][metric],rtol=0,atol=1e-12)
for method in ['popularity','pair_3m','pair_3m_popularity']:
    assert current[method]['LogLoss'] is None and current[method]['ROC-AUC'] is None
assert all(row['LogLoss'] is None and row['ROC-AUC'] is None
           for row in r['monthly_results'] if row['method_id'] in ['popularity','pair_3m','pair_3m_popularity'])
assert r['test_evaluated'] is False and r['test_table_built'] is False
assert r['bootstrap']['repeats']==5000
assert r['denominators'][0]['eligible_user_months']==481
assert r['denominators'][0]['eligible_unique_clients']==170
# These are observed denominator checks, not optimization targets.
assert all(not any(o.get('output_type')=='error' for o in cell.get('outputs',[])) for cell in nb['cells'])
code_cells=[cell for cell in nb['cells'] if cell['cell_type']=='code']
assert [c['execution_count'] for c in code_cells]==list(range(1,len(code_cells)+1))
assert all(not any(o.get('name')=='stderr' for o in c['outputs']) for c in code_cells)
saved=joblib.load(out/'manual_baseline_reviewed_model.joblib')
assert saved['metadata']==r and saved['features']==r['features']
assert saved['model'].max_iter==r['parameters']['max_iter']
assert saved['model'].max_leaf_nodes==r['parameters']['max_leaf_nodes']
execution=strict_load(root/'work/manual_baseline_reviewed_execution_check.json')
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
           for path,digest in execution['protected_files_unchanged'].items())
execution['artifact_checks']={'json_standard_nulls':True,'saved_model_metadata_matches_json':True,
    'cohort_features_protocol_unchanged':True,'existing_probability_metrics_unchanged':True,
    'all_code_cells_executed_no_errors_or_warnings':True,'original_files_unchanged':True}
(root/'work/manual_baseline_reviewed_execution_check.json').write_text(json.dumps(execution,indent=2))
print('Artifact validation passed; probability metrics unchanged to 1e-12.')
