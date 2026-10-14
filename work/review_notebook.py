"""Repair the saved teaching notebook and execute its cells in a fresh namespace."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import uuid

root = Path(__file__).resolve().parent.parent
path = root / "outputs/01_far_trans_data.ipynb"
original = path.read_bytes()
notebook = json.loads(original)
backup = root / "work" / f"01_far_trans_data_before_review_{uuid.uuid4().hex[:8]}.ipynb"
backup.write_bytes(original)

has_march = any(
    cell["cell_type"] == "code"
    and re.search(r"^march\s*=", "".join(cell["source"]), re.MULTILINE)
    for cell in notebook["cells"]
)
if not has_march:
    position = next(
        i for i, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code" and re.search(r"^april\s*=", "".join(cell["source"]), re.MULTILINE)
    )
    source = '''# История за март: доступна для прогноза на 1 апреля.
client_dates = pd.to_datetime(client_history["timestamp"])

march = client_history[
    (client_dates >= "2020-03-01")
    & (client_dates < "2020-04-01")
]

print(march[["timestamp", "transactionType", "ISIN"]])
'''
    notebook["cells"].insert(position, {
        "cell_type": "code", "id": f"march-{uuid.uuid4().hex[:8]}", "metadata": {},
        "source": source.splitlines(keepends=True), "execution_count": None, "outputs": [],
    })

renamed = 0
for cell in notebook["cells"]:
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if "first_customer =" in source and "assets = pd.read_csv" in source:
        # This older demonstration used the same variable name as the training row.
        new_source = re.sub(r"\bexample\b", "transaction_example", source)
        renamed += int(new_source != source)
        cell["source"] = new_source.splitlines(keepends=True)

namespace = {}
counter = 0
for index, cell in enumerate(notebook["cells"], start=1):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if not source.strip():
        cell["execution_count"] = None
        cell["outputs"] = []
        continue
    counter += 1
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(source, f"<cell-{index}>", "exec"), namespace)
    cell["execution_count"] = counter
    output = stdout.getvalue()
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": output.splitlines(keepends=True)}] if output else []

if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(original).digest():
    raise RuntimeError("Файл изменился во время проверки; исправления не записаны.")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("Восстановлена мартовская ячейка:", not has_march)
print("Устранено пересечение имён:", renamed)
print("Ячеек выполнено с чистого состояния:", counter)
print("pandas:", namespace["pd"].__version__)
print("Итоговые примеры:")
print(namespace["examples"][["ISIN", "user_buy_count_1m", "user_item_buy_count_1m", "label"]].to_string(index=False))
print("Первый обучающий пример после всех ячеек:", namespace["example"].shape)
print("Резервная копия:", backup)
