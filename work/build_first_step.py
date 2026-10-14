"""Build and execute a small data-inspection notebook; do not train models."""
import contextlib
import hashlib
import io
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
data_dir = root / "work" / "data" / "FAR-Trans"
cells = []


def markdown(source):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "metadata": {}, "source": source.splitlines(keepends=True), "execution_count": None, "outputs": []})


markdown("""# Шаг 1. Понимаем FAR-Trans

Сегодняшняя цель: прочитать таблицу, понять одну запись и сформулировать, что будет предсказывать модель.

Источник: [Университет Глазго](https://researchdata.gla.ac.uk/1658/).
Публикация: [Sanz-Cruzado, Droukas, McCreadie, 2024](https://arxiv.org/abs/2407.08692).

Используем `pandas` — библиотеку для чтения и обработки таблиц. Код ниже уже выполнен; его можно запускать повторно. Путь в следующей ячейке относится к этому компьютеру: при переносе ноутбука его нужно заменить.
""")
code(f'''import pandas as pd

data_dir = {str(data_dir)!r}

transactions = pd.read_csv(
    data_dir + "/transactions.csv",
    dtype={{"customerID": "string", "ISIN": "string"}},
)

print(transactions.head(3).to_string(index=False))
''')
markdown("""`transactions` — переменная, содержащая таблицу.

- `customerID`: обезличенный идентификатор клиента. Храним как текст, чтобы сохранить начальные нули.
- `ISIN`: код финансового актива.
- `transactionType`: покупка (`Buy`) или продажа (`Sell`).
- `timestamp`: дата сделки; точного времени внутри дня здесь нет.
- `totalValue`: сумма сделки в евро в обработанном авторами наборе.
- `units`: количество единиц актива.
- `channel`: канал совершения сделки.

`head(3)` показывает первые три строки, а `to_string(index=False)` выводит их без дополнительного номера строки.

Одна строка описывает состоявшуюся сделку. Это ещё не обучающий пример «клиент — возможный актив — момент прогноза».
""")
code(r'''transactions["timestamp"] = pd.to_datetime(transactions["timestamp"], errors="raise")

print("Строк:", len(transactions))
print("Клиентов:", transactions["customerID"].nunique())
print("Активов со сделками:", transactions["ISIN"].nunique())
print("Первая дата:", transactions["timestamp"].min().date())
print("Последняя дата:", transactions["timestamp"].max().date())
print("\nВиды сделок:")
print(transactions["transactionType"].value_counts().to_string())
''')
markdown("""`pd.to_datetime` превращает текстовые даты в даты Python/pandas. `nunique()` считает разные значения, `value_counts()` — сколько раз встречается каждое значение.

Точные числа получены из скачанного файла. Они немного отличаются от таблицы в статье (388 049 записей и 321 актив); версия файлов и их проверка важны для воспроизводимости.
""")
code(r'''assets = pd.read_csv(data_dir + "/asset_information.csv")

first_customer = transactions.iloc[0]["customerID"]
first_asset = transactions.iloc[0]["ISIN"]

asset_example = assets[assets["ISIN"] == first_asset]
print("Информация об активе:")
print(asset_example[["ISIN", "assetName", "assetCategory", "timestamp"]].to_string(index=False))

example = transactions[
    (transactions["customerID"] == first_customer)
    & (transactions["ISIN"] == first_asset)
    & (transactions["timestamp"] >= "2020-03-01")
    & (transactions["timestamp"] < "2020-05-01")
].sort_values("timestamp")

print("\nСделки одного клиента с этим активом:")
print(example[["timestamp", "transactionType", "totalValue", "units"]].to_string(index=False))
''')
markdown("""В последней ячейке `==` проверяет равенство для каждой строки таблицы. Выражение в квадратных скобках оставляет подходящие строки. Символ `&` объединяет несколько условий; каждое условие заключено в скобки. `sort_values` сортирует строки по дате. `iloc[0]` выбирает первую строку по позиции.

## Предлагаемая постановка — пока без обучения

На первое число месяца для пары «клиент — доступный актив» оцениваем, купит ли клиент этот актив в начинающемся календарном месяце. Например, прогноз от 1 апреля относится к периоду **[1 апреля, 1 мая)**.

- История: только сделки и информация, доступные **до 1 апреля**.
- Ответ `label=1`: в апреле есть хотя бы одна покупка этого актива клиентом.
- Ответ `label=0`: в апреле в данных нет такой покупки. Это не означает неприязнь, отказ от показа или непригодность инвестиции.

В показанном примере покупка 27 марта — история. Покупка 24 апреля даёт будущий ответ 1. Продажу 6 апреля нельзя использовать как признак прогноза от 1 апреля.

**Выбор на обсуждение:** месячный горизонт позволяет понятно разделить прошлое и будущее. В статье авторы используют другой протокол (шесть месяцев и новые для пользователя активы). Мы допускаем повторные покупки и не будем напрямую сравнивать собственные числа с таблицей статьи.

Следующим шагом определим клиентов и активы-кандидаты по информации, доступной на дату прогноза. Затем построим несколько примеров с ответами 0 и 1. `Buy`/`Sell` из таблицы нельзя просто переименовать в такие ответы.

## Вопрос на понимание

Для прогноза от 1 апреля можем ли мы использовать продажу 6 апреля? Объясни одной фразой.

## Что важно для честного эксперимента

- Авторы добавили часть покупок, реконструируя начальные владения по последующим продажам, и скорректировали некоторые даты. Это обработанный исследовательский набор, а не неизменённый журнал. При подготовке эксперимента нужно решить, как ограничить влияние восстановленных событий; полностью идентифицировать их без флага происхождения может быть невозможно. [Раздел 3.3.1 статьи](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).
- `transactionID` встречается у разных клиентов. Это не основание удалять строки по этому полю: единица наблюдения требует отдельного решения. Полностью одинаковых строк в файле нет.
- Метаданные клиентов и активов меняются со временем. Для каждого прогноза выбираем запись, доступную на тот момент, а не последнюю во всей таблице.
- Временной горизонт последнего прогноза должен целиком помещаться в наблюдаемый период.
- Пока построение модели и сравнение признаков не выполнялись.
""")

namespace = {}
execution_count = 0
for cell in cells:
    if cell["cell_type"] != "code":
        continue
    execution_count += 1
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile("".join(cell["source"]), "<notebook-cell>", "exec"), namespace)
    cell["execution_count"] = execution_count
    cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": captured.getvalue().splitlines(keepends=True)}]

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
output = root / "outputs" / "01_far_trans_data.ipynb"
output.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")

t = namespace["transactions"]
sha256 = hashlib.sha256((root / "work/data/FAR-Trans.zip").read_bytes()).hexdigest()
manifest = {"dataset": "FAR-Trans", "download_date": "2026-10-12", "source": "https://researchdata.gla.ac.uk/1658/1/FAR-Trans.zip", "archive_sha256": sha256, "rows": len(t), "customers": int(t.customerID.nunique()), "transacted_assets": int(t.ISIN.nunique()), "actions": {key: int(value) for key, value in t.transactionType.value_counts().items()}, "first_date": str(t.timestamp.min().date()), "last_date": str(t.timestamp.max().date()), "pandas_version": namespace["pd"].__version__}
(root / "work/data/FAR-Trans/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("Выполнены все", execution_count, "ячейки:", output)
