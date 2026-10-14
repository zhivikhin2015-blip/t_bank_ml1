from pathlib import Path
import json, time
start = time.monotonic()
# Повторяем ранее подготовленный код и независимые проверки таблиц.
exec(compile(Path('work/check_automatic_tables.py').read_text(),'<setup-and-check>','exec'),globals())
steps=json.loads(Path('work/automatic_finish_steps.json').read_text())
for i,step in enumerate(steps):
    code=step['code']
    if i == 6:
        code=code.replace('output_dir = project_dir / "outputs"', 'output_dir = project_dir / "work/automatic_features_preflight"\noutput_dir.mkdir(exist_ok=True)')
    print('START',step['title'],flush=True)
    exec(compile(code,f'<finish-step-{i}>','exec'),globals())
    print('DONE',i+1,'elapsed',round(time.monotonic()-start,1),'s',flush=True)
assert all(name in auto_names for name in selected_new)
assert len(selected_new)<=5
assert set(manual_features).issubset(selected_features)
assert sum(record['budget'][key] for key in ['selection_model_fits','all_features_control_fits','tuning_model_fits'])==6
assert not record['test_evaluated']
print('AUTOMATIC PIPELINE CHECK PASSED',flush=True)
