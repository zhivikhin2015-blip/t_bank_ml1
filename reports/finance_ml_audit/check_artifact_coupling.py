"""Check cache guards after exact manual reproduction; no new training."""
import argparse,ast,contextlib,hashlib,io,json,os,pathlib,shutil,tempfile,traceback
import joblib
from run_audit import prepare
p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=pathlib.Path);p.add_argument('--manual-run',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path);args=p.parse_args();source=args.source.resolve();manual_run=args.manual_run.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
root=prepare(source,pathlib.Path(tempfile.mkdtemp(prefix='finance-coupling-',dir='/private/tmp')),'automatic')
for name in ['manual_baseline_reviewed_results.json','manual_baseline_reviewed_model.joblib']:shutil.copy2(manual_run/'outputs'/name,root/'outputs'/name)
nb=json.loads((source/'outputs/02_automatic_features_reviewed.ipynb').read_text());ns={'__name__':'__main__'};old=pathlib.Path.cwd();error=None
try:
 os.chdir(root)
 with contextlib.redirect_stdout(io.StringIO()):
  try:
   for index in [2,3,4,6]:
    tree=ast.parse(''.join(nb['cells'][index]['source']));assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='fit' for n in ast.walk(tree));exec(compile(tree,f'<automatic-cell-{index}>','exec'),ns)
  except AssertionError:error=traceback.format_exc()
finally:os.chdir(old)
manual_hash=hashlib.sha256((root/'outputs/manual_baseline_reviewed_results.json').read_bytes()).hexdigest();cached=joblib.load(root/'outputs/manual_control_100_7_reviewed_model.joblib')['metadata']['reference_results_sha256']
assert manual_hash!=cached and error and '<automatic-cell-6>' in error
record={'no_training':True,'exact_manual_reproduction_already_verified':True,'automatic_stops_at_cell':6,'cached_manual_reference_sha256':cached,'reproduced_manual_results_sha256':manual_hash,'expected_guard_failure':True,'traceback':error,'recommendation':'Use separate semantic protocol signatures and byte provenance; regenerate dependent cached metadata when reproducing full pipeline.'}
(out/'artifact_coupling_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2));print({k:v for k,v in record.items() if k!='traceback'})
