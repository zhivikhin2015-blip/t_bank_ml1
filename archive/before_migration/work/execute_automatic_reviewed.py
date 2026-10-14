from pathlib import Path
import json,hashlib,ast,contextlib,io,time,traceback
from IPython.core.formatters import DisplayFormatter
root=Path.cwd()
protected=[Path('/Users/annazhivikhina/Desktop/02_automatic_features.ipynb'),
 root/'outputs/02_automatic_features.ipynb',root/'outputs/automatic_features_results.json',
 root/'outputs/automatic_features_model.joblib',root/'outputs/manual_baseline_reviewed.ipynb',
 root/'outputs/manual_baseline_reviewed_results.json',root/'outputs/manual_baseline_reviewed_model.joblib',
 root/'outputs/manual_baseline_results.json',root/'outputs/manual_baseline_model.joblib']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
path=root/'outputs/02_automatic_features_reviewed.ipynb';nb=json.loads(path.read_text())
ns={'__name__':'__main__'};start=time.monotonic();count=0;formatter=DisplayFormatter()
for i,cell in enumerate(nb['cells']):
 if cell['cell_type']!='code':continue
 count+=1;out=io.StringIO();err=io.StringIO();value=None
 source=''.join(cell['source']);tree=ast.parse(source)
 try:
  with contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):
   if tree.body and isinstance(tree.body[-1],ast.Expr):
    exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),f'<reviewed-cell-{i}>','exec'),ns)
    value=eval(compile(ast.Expression(tree.body[-1].value),f'<reviewed-cell-{i}>','eval'),ns)
   else:exec(compile(tree,f'<reviewed-cell-{i}>','exec'),ns)
 except Exception:
  print(out.getvalue());print(err.getvalue());print(traceback.format_exc());raise
 cell['execution_count']=count;cell['outputs']=[]
 for name,stream in [('stdout',out),('stderr',err)]:
  if stream.getvalue():cell['outputs'].append({'output_type':'stream','name':name,'text':stream.getvalue()})
 if value is not None:
  bundle,metadata=formatter.format(value)
  cell['outputs'].append({'output_type':'execute_result','execution_count':count,'data':bundle,'metadata':metadata})
 path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
 print('Cell',i,'OK;',round(time.monotonic()-start,1),'s',flush=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
record=json.loads((root/'outputs/automatic_features_reviewed_results.json').read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
assert record==ns['record'] and record['test_evaluated'] is False
import joblib
model=joblib.load(root/'outputs/automatic_features_reviewed_model.joblib')
assert model['metadata']==record and model['features']==record['features']
report={'fresh_process':True,'code_cells_executed':count,'elapsed_seconds':time.monotonic()-start,
 'checks':ns['checks'],'protected_files_unchanged':hashes,'standard_json_nulls':True,
 'model_json_consistency':True,'test_evaluated':False,'manual_control_fits':ns['manual_control_fits']}
(root/'work/automatic_reviewed_execution_checks.json').write_text(json.dumps(report,indent=2))
print(ns['comparison'][['method']+ns['metric_names']].round(6).to_string(index=False,na_rep='—'))
print(ns['bootstrap'].round(6).to_string(index=False))
print(ns['purchase_diagnostics'][ns['purchase_diagnostics']['period']=='validation'].round(6).to_string(index=False))
print('ALL REVIEW CHECKS PASSED',flush=True)
