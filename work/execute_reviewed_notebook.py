from pathlib import Path
import contextlib, io, json, hashlib, traceback, time
root = Path.cwd()
protected = [(Path.cwd() / 'archive/source_inputs/Desktop/manual_baseline.ipynb'),
             root/'outputs/manual_baseline_results.json', root/'outputs/manual_baseline_model.joblib',
             root/'outputs/manual_baseline.ipynb', root/'outputs/01_far_trans_data.ipynb']
hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
path = root/'outputs/manual_baseline_reviewed.ipynb'
nb = json.loads(path.read_text())
namespace = {'__name__': '__main__'}
start = time.time()
count = 0
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code': continue
    count += 1
    out, err = io.StringIO(), io.StringIO()
    failed = None
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try: exec(compile(''.join(cell['source']), f'reviewed-cell-{i}', 'exec'), namespace)
        except Exception:
            failed = traceback.format_exc()
    cell['execution_count'] = count
    cell['outputs'] = []
    for name, stream in [('stdout', out), ('stderr', err)]:
        if stream.getvalue(): cell['outputs'].append({'output_type': 'stream', 'name': name, 'text': stream.getvalue()})
    print(f'Cell {i}: {"FAILED" if failed else "OK"}, elapsed {time.time()-start:.1f}s', flush=True)
    if failed:
        print(failed, flush=True)
        raise SystemExit(1)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
report = {'fresh_process_execution': True, 'code_cells_executed': count,
          'elapsed_seconds': time.time()-start, 'protected_files_unchanged': hashes,
          'checks': namespace['checks'], 'test_evaluated': False}
(root/'work/manual_baseline_reviewed_execution_check.json').write_text(json.dumps(report, indent=2))
print(namespace['manual_comparison'].drop(columns='AUC_note').round(6).to_string(index=False))
print(namespace['bootstrap_results'].round(6).to_string(index=False))
print('ALL CHECKS PASSED')
