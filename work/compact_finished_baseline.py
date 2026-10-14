"""Consolidate the completed manual baseline, preserving its data and scores."""
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/'outputs/01_far_trans_data.ipynb'
original_bytes = TARGET.read_bytes()
original = json.loads(original_bytes)
learning_path = ROOT/'outputs/01_far_trans_learning.ipynb'
if learning_path.exists() and learning_path.read_bytes() != original_bytes:
    learning_path = ROOT/'outputs'/(
        '01_far_trans_learning_' + hashlib.sha256(original_bytes).hexdigest()[:8] + '.ipynb'
    )
learning_path.write_bytes(original_bytes)
cells = []


def add(kind, text):
    cell = {'cell_type': kind, 'id': f'baseline-{len(cells):02d}',
            'metadata': {}, 'source': text.strip().splitlines(keepends=True)}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


def extract_function(name):
    for cell in original['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        for node in ast.parse(source).body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return ast.get_source_segment(source, node)
    raise ValueError(f'Missing {name}')


add('markdown', '''
# Ручной baseline FAR-Trans

## 1. Задача и протокол

На первое число месяца предсказываем, купит ли клиент конкретный актив в начинающемся
календарном месяце. Одна строка — **клиент, актив, дата прогноза**, а не отдельная сделка.
`label=1` означает хотя бы одну покупку за месяц; `label=0` — отсутствие записанной покупки.
Повторные покупки допускаются. Признаки используют только события **до даты прогноза**.

Ручной baseline зафиксирован на validation; test зарезервирован для сравнения
с автоматическим подходом. Учебные примеры и промежуточные версии сохранены
в отдельном ноутбуке `01_far_trans_learning.ipynb`.

[Данные](https://researchdata.gla.ac.uk/1658/) · [Публикация](https://arxiv.org/abs/2407.08692).

**Границы выводов:** 500 существующих клиентов и фиксированные 50 активов;
поздняя доступность активов не моделируется. Нет показов и явных отрицательных предпочтений.
Авторы реконструировали часть событий по последующим продажам и изменили некоторые даты
без флага происхождения строк. Поэтому фильтры нашего кода не устраняют все ограничения
исходного набора. Начальный массовый блок покупок 2018 года в признаках не используется.
Цены и метаданные не подаются моделям. Наш месячный протокол отличается от протокола статьи.
[Обработка авторами, §3.3.1](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).
''')

add('markdown', '''
## 2. Загрузка и выбор участников

Клиентов и активы выбираем по покупкам за полгода **до первого прогноза**:
500 случайных клиентов и 50 активов с наибольшим числом покупок. Состав фиксируем для всех месяцев.
Seed=42 воспроизводит выбор. Идентификаторы читаем как текст, чтобы сохранить начальные нули.
На другом компьютере замени только `project_dir`.
''')

add('code', '''
from pathlib import Path
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

project_dir = Path.cwd()
if project_dir.name == 'outputs':
    project_dir = project_dir.parent
data_dir = project_dir / 'work' / 'data' / 'FAR-Trans'
output_dir = project_dir / 'outputs'

seed = 42
first_prediction = pd.Timestamp('2020-04-01')
last_prediction = pd.Timestamp('2022-11-01')
validation_start = pd.Timestamp('2022-01-01')
test_start = pd.Timestamp('2022-07-01')

transactions = pd.read_csv(
    data_dir / 'transactions.csv',
    dtype={'customerID': 'string', 'ISIN': 'string'}
)
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'], errors='raise')
all_buys = transactions[transactions['transactionType'] == 'Buy'].copy()

selection_buys = all_buys[
    (all_buys['timestamp'] >= '2019-10-01')
    & (all_buys['timestamp'] < first_prediction)
]
large_client_ids = (
    selection_buys['customerID'].drop_duplicates()
    .sample(n=500, random_state=seed).tolist()
)
large_asset_ids = selection_buys['ISIN'].value_counts().head(50).index.tolist()
large_buys = all_buys[all_buys['customerID'].isin(large_client_ids)].copy()
candidate_pairs = pd.DataFrame({'customerID': large_client_ids}).merge(
    pd.DataFrame({'ISIN': large_asset_ids}), how='cross'
)

manual_features = [
    'user_buy_count_1m', 'user_item_buy_count_1m', 'item_buy_count_1m',
    'user_buy_count_3m', 'user_item_buy_count_3m'
]
print('Сделок:', len(transactions), '| Клиентов:', len(large_client_ids),
      '| Активов:', len(large_asset_ids), '| Пар в месяц:', len(candidate_pairs))
''')

add('markdown', '''
## 3. Одна функция расчёта признаков и ответа

| Признак | Что считаем строго до даты прогноза |
|---|---|
| `user_buy_count_1m` | все покупки клиента за один месяц |
| `user_item_buy_count_1m` | покупки этого актива клиентом за один месяц |
| `item_buy_count_1m` | покупки актива всеми клиентами за один месяц |
| `user_buy_count_3m` | все покупки клиента за три месяца |
| `user_item_buy_count_3m` | покупки этого актива клиентом за три месяца |

Месячные признаки описывают недавнюю активность; трёхмесячные — более длительную историю.
Популярность различает активы, которые клиент раньше не покупал. Считаем покупки,
а не уникальных покупателей. История включает активы вне выбранных 50.
`groupby().size()` считает строки группами; `left` сохраняет все пары;
пропуски в подсчётах заменяем нулями. Ответ определяется отдельно по следующему месяцу.
''')

add('code', '''
def make_month_dataset(prediction_date):
    cutoff = pd.Timestamp(prediction_date)
    end = cutoff + pd.DateOffset(months=1)
    if not cutoff.is_month_start or cutoff != cutoff.normalize():
        raise ValueError('Дата прогноза должна быть первым числом месяца')
    if end > transactions['timestamp'].max().normalize() + pd.Timedelta(days=1):
        raise ValueError('В данных нет полного месяца для ответа')

    result = candidate_pairs.copy()

    # Одинаковый вид агрегатов для двух вручную выбранных окон.
    for months in [1, 3]:
        start = cutoff - pd.DateOffset(months=months)
        past = large_buys[
            (large_buys['timestamp'] >= start)
            & (large_buys['timestamp'] < cutoff)
        ]
        user_counts = past.groupby('customerID').size().reset_index(
            name=f'user_buy_count_{months}m'
        )
        pair_counts = past.groupby(['customerID', 'ISIN']).size().reset_index(
            name=f'user_item_buy_count_{months}m'
        )
        result = result.merge(
            user_counts, on='customerID', how='left', validate='many_to_one'
        )
        result = result.merge(
            pair_counts, on=['customerID', 'ISIN'], how='left', validate='one_to_one'
        )

    past_all = all_buys[
        (all_buys['timestamp'] >= cutoff - pd.DateOffset(months=1))
        & (all_buys['timestamp'] < cutoff)
    ]
    item_counts = past_all.groupby('ISIN').size().reset_index(name='item_buy_count_1m')
    result = result.merge(item_counts, on='ISIN', how='left', validate='many_to_one')

    future = large_buys[
        (large_buys['timestamp'] >= cutoff)
        & (large_buys['timestamp'] < end)
    ]
    labels = future[['customerID', 'ISIN']].drop_duplicates().copy()
    labels['label'] = 1
    result = result.merge(
        labels, on=['customerID', 'ISIN'], how='left', validate='one_to_one'
    )
    numeric_columns = manual_features + ['label']
    result[numeric_columns] = result[numeric_columns].fillna(0).astype(int)
    result['prediction_date'] = cutoff

    return result[['customerID', 'ISIN', 'prediction_date'] + manual_features + ['label']]
''')

add('markdown', '''
## 4. Общая таблица и временное разделение

Данные с пятью признаками собираем один раз. Для всех сравнений используем те же пары и ответы.

| Часть | Месяцы покупок | Назначение |
|---|---|---|
| Train | апрель 2020 — декабрь 2021 | обучение |
| Validation | январь — июнь 2022 | выбор признаков и настроек |
| Test | июль — ноябрь 2022 | итоговая проверка после автоматического подхода |

Проверки ниже контролируют уникальность строк, пропуски и временные границы.
Test готов с пятью признаками, но метрики на нём не вычисляются.
''')

add('code', '''
monthly_tables = [
    make_month_dataset(date)
    for date in pd.date_range(first_prediction, last_prediction, freq='MS')
]
dataset = pd.concat(monthly_tables, ignore_index=True)
del monthly_tables

train = dataset[dataset['prediction_date'] < validation_start].copy()
validation = dataset[
    (dataset['prediction_date'] >= validation_start)
    & (dataset['prediction_date'] < test_start)
].copy()
test = dataset[dataset['prediction_date'] >= test_start].copy()

assert not dataset.duplicated(['customerID', 'ISIN', 'prediction_date']).any()
assert not dataset[manual_features].isna().any().any()
assert dataset['label'].isin([0, 1]).all()
assert (dataset['user_buy_count_3m'] >= dataset['user_buy_count_1m']).all()
assert (dataset['user_item_buy_count_3m'] >= dataset['user_item_buy_count_1m']).all()
assert (dataset['user_item_buy_count_3m'] <= dataset['user_buy_count_3m']).all()
assert train['prediction_date'].max() + pd.DateOffset(months=1) <= validation['prediction_date'].min()
assert validation['prediction_date'].max() + pd.DateOffset(months=1) <= test['prediction_date'].min()

print('Всего:', len(dataset), '| Train:', len(train),
      '| Validation:', len(validation), '| Test:', len(test))
''')

add('markdown', '''
## 5. Метрики и простые прогнозы

- **LogLoss:** ошибка вероятностей, меньше — лучше; уверенный неправильный прогноз штрафуется сильнее.
- **ROC-AUC:** порядок купленных и некупленных пар во всей выборке, больше — лучше; не заменяет ранжирование внутри клиента.
- **Recall@5:** средняя доля купленных активов, попавших в пять рекомендаций, больше — лучше.

Recall считаем по случаям «клиент — месяц» с покупкой хотя бы одного из выбранных 50 активов.
Повторные покупки одного актива дают один положительный ответ. При равных оценках используем
случайный порядок с seed=42, одинаковый для сравниваемых моделей.

Два простых прогноза: общая доля покупок в train и отдельная доля для пар с прошлой покупкой/без неё.
Оба оценивают вероятности только по train.
''')

metric_functions = '\n\n'.join(extract_function(name) for name in
                               ['binary_logloss', 'recall_at_5', 'evaluate_predictions'])
metric_functions += '''

baseline_probability = train['label'].mean()
validation_predictions = np.full(len(validation), baseline_probability)

repeat_rates = train.groupby(train['user_item_buy_count_1m'] > 0)['label'].mean()
repeat_predictions = (
    (validation['user_item_buy_count_1m'] > 0)
    .map(repeat_rates).fillna(baseline_probability).to_numpy()
)
print(f'Постоянная вероятность: {baseline_probability:.2%}')
'''
add('code', metric_functions)

add('markdown', '''
## 6. Бустинг и четыре настройки

Сначала сравниваем два и три признака при 100 деревьях и 7 листьях.
Затем для пяти признаков проверяем ровно четыре настройки: 100/200 деревьев × 7/15 листьев.
Контрольная настройка 100/7 позволяет выделить пользу трёхмесячной истории.

Правило выбора: **выше Recall@5**, затем ниже LogLoss, затем меньше деревьев и листьев.
Настройки учатся только на train; сравнение — на validation. Это ограниченный подбор.
Встроенное случайное разделение отключено (`early_stopping=False`).
`threadpool_limits` ограничивает нагрузку четырьмя потоками. В модель передаём только
список численных признаков; идентификаторы, дата и `label` в него не входят.
''')

add('code', '''
def fit_boosting(feature_names, trees=100, leaves=7):
    fitted_model = HistGradientBoostingClassifier(
        max_iter=trees,
        max_leaf_nodes=leaves,
        early_stopping=False,
        random_state=seed
    )
    with threadpool_limits(limits=4):
        fitted_model.fit(train[feature_names], train['label'])
        probabilities = fitted_model.predict_proba(validation[feature_names])[:, 1]
    return fitted_model, probabilities

two_features = manual_features[:2]
three_features = manual_features[:3]
model_two, predictions_two = fit_boosting(two_features)
model_three, predictions_three = fit_boosting(three_features)
''')

add('code', '''
settings_to_check = [(100, 7), (100, 15), (200, 7), (200, 15)]
trained_models = {}
stored_predictions = {}
tuning_rows = []

for trees, leaves in settings_to_check:
    fitted_model, probabilities = fit_boosting(manual_features, trees, leaves)
    trained_models[(trees, leaves)] = fitted_model
    stored_predictions[(trees, leaves)] = probabilities
    tuning_rows.append({
        'max_iter': trees,
        'max_leaf_nodes': leaves,
        **evaluate_predictions(validation, probabilities)
    })

tuning_results = pd.DataFrame(tuning_rows).sort_values(
    ['Recall@5', 'LogLoss', 'max_iter', 'max_leaf_nodes'],
    ascending=[False, True, True, True]
).reset_index(drop=True)
best_settings = tuning_results.iloc[0]
best_trees = int(best_settings['max_iter'])
best_leaves = int(best_settings['max_leaf_nodes'])
manual_model = trained_models[(best_trees, best_leaves)]
manual_predictions = stored_predictions[(best_trees, best_leaves)]

print(tuning_results.round(5).to_string(index=False))
print('Выбрано:', best_trees, 'деревьев,', best_leaves, 'листьев')
''')

add('markdown', '''
## 7. Результаты, выводы и сохранение

Общая таблица сравнивает все подходы на одинаковом validation.
Популярность улучшает выбор активов; трёхмесячная история улучшает обе метрики при прежних настройках.
Модель 200/7 выбрана по Recall@5, хотя 100/7 имеет немного лучший LogLoss.

**Ручной baseline зафиксирован:** пять признаков, ограниченный подбор четырёх настроек,
сохранённая модель и результаты. Это не доказательство лучшего возможного качества.
Следующий этап — автоматическое предложение/отбор признаков на тех же данных, участниках,
временных границах и с тем же бюджетом подбора. Test оценим после выбора обоих решений.

Для проверки воспроизводимости: сохранить ноутбук, затем **Restart → Run All** с ядром `.venv`.
Последняя ячейка сохраняет модель, условия эксперимента, версии библиотек и хеш исходных сделок.
''')

add('code', '''
versions_to_compare = [
    ('Постоянный baseline', validation_predictions),
    ('Правило прошлой покупки', repeat_predictions),
    ('Бустинг: 2 признака', predictions_two),
    ('Бустинг: 3 признака', predictions_three),
    ('Бустинг: 5 признаков, выбранные настройки', manual_predictions)
]
comparison_rows = [
    {'Модель': name, **evaluate_predictions(validation, probabilities)}
    for name, probabilities in versions_to_compare
]
manual_comparison = pd.DataFrame(comparison_rows)
print(manual_comparison.round(5).to_string(index=False))

positive_user_months = validation.groupby(
    ['customerID', 'prediction_date']
)['label'].sum()
print('Для Recall@5:', int((positive_user_months > 0).sum()),
      'из', len(positive_user_months), 'случаев клиент — месяц')
''')

add('code', '''
baseline_record = {
    'status': 'manual_baseline_frozen',
    'features': manual_features,
    'parameters': {
        'max_iter': best_trees,
        'max_leaf_nodes': best_leaves,
        'learning_rate': float(manual_model.learning_rate),
        'early_stopping': False,
        'random_state': seed
    },
    'selection_rule': 'max Recall@5, then min LogLoss, then simpler model',
    'train_months': ['2020-04-01', '2021-12-01'],
    'validation_months': ['2022-01-01', '2022-06-01'],
    'test_months': ['2022-07-01', '2022-11-01'],
    'client_ids': large_client_ids,
    'asset_ids': large_asset_ids,
    'cohort_random_state': seed,
    'comparison': manual_comparison.to_dict(orient='records'),
    'tuning': tuning_results.to_dict(orient='records'),
    'test_evaluated': False,
    'transactions_sha256': hashlib.sha256(
        (data_dir / 'transactions.csv').read_bytes()
    ).hexdigest(),
    'versions': {
        'python': platform.python_version(),
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit_learn': sklearn.__version__
    }
}

output_dir.mkdir(exist_ok=True)
joblib.dump(
    {'model': manual_model, 'features': manual_features, 'metadata': baseline_record},
    output_dir / 'manual_baseline_model.joblib'
)
(output_dir / 'manual_baseline_results.json').write_text(
    json.dumps(baseline_record, ensure_ascii=False, indent=2), encoding='utf-8'
)
print('Ручной baseline и результаты сохранены. Test не оценивался.')
''')

compact = {'cells': cells, 'metadata': original['metadata'],
           'nbformat': original['nbformat'], 'nbformat_minor': original['nbformat_minor']}


def execute(nb, label):
    ns = {'__name__': '__main__'}
    count = 0
    print(f'Executing {label} from scratch...', flush=True)
    with threadpool_limits(limits=4):
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            if not source.strip():
                continue
            count += 1
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exec(compile(source,f'{label}_cell_{i+1}','exec'),ns)
            cell['execution_count'] = count
            cell['outputs'] = ([{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue()}]
                               if output.getvalue() else [])
    return ns, count


def dataframe_hash(frame, columns):
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def snapshot(ns, old):
    split_names = ['train_v3', 'validation_v3', 'test_v3'] if old else ['train','validation','test']
    cols = ['customerID','ISIN','prediction_date']+ns['manual_features']+['label']
    return {
        'split_rows': [len(ns[name]) for name in split_names],
        'split_hashes': [dataframe_hash(ns[name], cols) for name in split_names],
        'clients': ns['large_client_ids'],
        'assets': ns['large_asset_ids'],
        'features': ns['manual_features'],
        'chosen': [ns['best_trees'],ns['best_leaves']],
        'comparison': ns['manual_comparison'].copy(),
        'tuning': ns['tuning_results'].copy(),
        'probability_hash': hashlib.sha256(ns['manual_predictions'].tobytes()).hexdigest()
    }


if __name__ == '__main__':
    original_ns, original_count = execute(original, 'learning')
    before = snapshot(original_ns, True)
    del original_ns
    gc.collect()

    compact_ns, compact_count = execute(compact, 'compact')
    after = snapshot(compact_ns, False)
    for key in ['split_rows','split_hashes','clients','assets','features','chosen','probability_hash']:
        assert before[key] == after[key], f'Changed {key}'
    pd.testing.assert_frame_equal(before['comparison'],after['comparison'],atol=1e-12,rtol=0)
    pd.testing.assert_frame_equal(before['tuning'],after['tuning'],atol=1e-12,rtol=0)
    assert after['split_rows'] == [525000,150000,125000]
    assert after['chosen'] == [200,7]
    assert compact_ns['baseline_record']['test_evaluated'] is False
    assert TARGET.read_bytes() == original_bytes, 'Source changed during check; preserve newer edits'
    assert learning_path.read_bytes() == original_bytes
    TARGET.write_text(json.dumps(compact,ensure_ascii=False,indent=1)+'\n')

    report = {
        'original_cells':len(original['cells']), 'compact_cells':len(compact['cells']),
        'original_code_cells_executed':original_count,
        'compact_code_cells_executed':compact_count,
        'same_all_split_rows_and_values':True,'same_selected_clients_and_assets':True,
        'same_all_model_metrics':True,'same_four_setting_results':True,
        'same_chosen_model_validation_predictions':True,'test_evaluated':False,
        'learning_copy':str(learning_path),'working_file':str(TARGET)
    }
    (ROOT/'work/finished_baseline_compaction_check.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    )
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    print(after['comparison'].round(5).to_string(index=False),flush=True)
