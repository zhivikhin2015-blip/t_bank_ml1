"""Read-only integrity checks. Optional final-notebook run keeps test closed."""
from pathlib import Path
import argparse
import ast
import hashlib
import io
import json
import contextlib
import os
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-final-runtime', action='store_true',
                        help='Load fixed models and run the pending notebook without writing files or reading events')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'migration_manifest.json').read_text())
    files_checked = 0
    for entry in manifest['project_files']:
        path = root / entry['path']
        if entry['kind'] == 'file':
            assert digest(path) == entry['destination_sha256'], f'File changed: {path}'
            files_checked += 1
        else:
            assert path.is_symlink() and path.exists(), f'Broken link: {path}'
            assert path.resolve().is_relative_to(root), f'External link: {path}'
    for entry in manifest['external_inputs']:
        assert digest(root / entry['path']) == entry['sha256'], entry['path']
        files_checked += 1
    for entry in manifest['path_changes']:
        assert digest(root / entry['original_copy']) == entry['original_sha256']
        assert digest(root / entry['path']) == entry['migrated_sha256']
        files_checked += 1
    notebooks = list((root / 'outputs').glob('*.ipynb'))
    for path in notebooks:
        notebook = json.loads(path.read_text())
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
    for path in root.rglob('*.py'):
        if '.venv' not in path.parts:
            ast.parse(path.read_text())
    final = json.loads((root / 'outputs/final_comparison_results.json').read_text())
    assert final['test_evaluated'] is False and final['overall'] is None
    report = {'verified_files_including_archived_originals': files_checked,
              'notebooks_syntax_valid': len(notebooks), 'internal_links_valid': True,
              'real_test_evaluated': False, 'files_written_by_check': 0}
    if args.check_final_runtime:
        import pandas as pd
        notebook = json.loads((root / 'outputs/03_final_comparison.ipynb').read_text())
        assert any("evaluation_policy='pending'" in ''.join(c['source']) for c in notebook['cells'])
        namespace = {'__name__': '__main__'}
        virtual_writes = {}
        allowed_writes = {root / 'outputs/final_comparison_protocol.json',
                          root / 'outputs/final_comparison_results.json'}

        def capture_write(path, data, *unused_args, **unused_kwargs):
            assert path in allowed_writes, f'Unexpected write: {path}'
            virtual_writes[str(path)] = json.loads(data)
            return len(data)

        def forbid_event_read(*unused_args, **unused_kwargs):
            raise AssertionError('Reading transaction events is forbidden in the migration check')

        previous_cwd = Path.cwd()
        try:
            os.chdir(root)
            with patch.object(Path, 'write_text', capture_write), patch.object(pd, 'read_csv', forbid_event_read):
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    for i, cell in enumerate(notebook['cells']):
                        if cell['cell_type'] != 'code':
                            continue
                        code = ''.join(cell['source'])
                        tree = ast.parse(code)
                        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                       and n.func.attr == 'fit' for n in ast.walk(tree))
                        exec(compile(tree, f'final-cell-{i}', 'exec'), namespace)
        finally:
            os.chdir(previous_cwd)
        assert namespace['test_gate_open'] is False
        assert namespace['results']['test_evaluated'] is False
        assert namespace['results']['overall'] is None
        assert namespace['results']['new_training_fits'] == 0
        assert len(virtual_writes) == 2
        report.update({'final_notebook_pending_runtime_valid': True,
                       'saved_models_loaded_and_consistent': True,
                       'real_transaction_events_read': 0, 'new_training_fits': 0})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
