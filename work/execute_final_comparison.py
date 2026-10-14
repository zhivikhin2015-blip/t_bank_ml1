from pathlib import Path
import json,hashlib,ast,io,contextlib,traceback,time
from IPython.core.formatters import DisplayFormatter
root=Path.cwd();path=root/'outputs/03_final_comparison.ipynb'
protected=[root/'outputs'/name for name in ['manual_baseline_reviewed_results.json','manual_baseline_reviewed_model.joblib',
 'automatic_features_reviewed_results.json','automatic_features_reviewed_model.joblib','manual_control_100_7_reviewed_model.joblib',
 '02_automatic_features_reviewed.ipynb']]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
nb=json.loads(path.read_text());ns={'__name__':'__main__'};formatter=DisplayFormatter();count=0;start=time.monotonic()
for i,c in enumerate(nb['cells']):
 if c['cell_type']!='code':continue
 count+=1;tree=ast.parse(''.join(c['source']));stdout=io.StringIO();stderr=io.StringIO();value=None
 # Проверяем, что в финальном ноутбуке нет вызовов обучения.
 assert not any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='fit' for node in ast.walk(tree))
 try:
  with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stderr):
   if tree.body and isinstance(tree.body[-1],ast.Expr):
    exec(compile(ast.Module(body=tree.body[:-1],type_ignores=[]),f'<final-cell-{i}>','exec'),ns)
    value=eval(compile(ast.Expression(tree.body[-1].value),f'<final-cell-{i}>','eval'),ns)
   else:exec(compile(tree,f'<final-cell-{i}>','exec'),ns)
 except Exception:
  print(stdout.getvalue());print(stderr.getvalue());print(traceback.format_exc());raise
 c['execution_count']=count;c['outputs']=[]
 for name,stream in [('stdout',stdout),('stderr',stderr)]:
  if stream.getvalue():c['outputs'].append({'output_type':'stream','name':name,'text':stream.getvalue()})
 if value is not None:
  bundle,metadata=formatter.format(value)
  c['outputs'].append({'output_type':'execute_result','execution_count':count,'data':bundle,'metadata':metadata})
 nb['metadata']['test_evaluated']=ns.get('test_evaluated',False)
 path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
 print('Cell',i,'OK;',round(time.monotonic()-start,1),'s',flush=True)
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
results=json.loads((root/'outputs/final_comparison_results.json').read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
assert results==ns['results']
if results['status']=='awaiting_period_resolution':
 assert ns['test'] is None and ns['all_buys'] is None
 for key in ['overall','monthly','denominators','catalog_coverage','purchase_diagnostics']:
  assert results[key] is None
 assert not results['test_evaluated']
report={'fresh_process':True,'code_cells_executed':count,'elapsed_seconds':time.monotonic()-start,
        'checks':ns['checks'],'test_evaluated':results['test_evaluated'],'new_training_fits':0,
        'status':results['status'],'protected_files_unchanged':hashes}
(root/'work/final_comparison_execution_checks.json').write_text(json.dumps(report,indent=2))
print('FINAL NOTEBOOK CHECKS PASSED;',results['status'])
