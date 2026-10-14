from pathlib import Path
import json, hashlib, textwrap

root = Path.cwd()
source = Path('/Users/annazhivikhina/Desktop/manual_baseline.ipynb')
original = json.loads(source.read_text())
cells = []
def add(kind, text):
    cell = {'cell_type': kind, 'metadata': {}, 'source': textwrap.dedent(text).strip()+'\n', 'id': f'reviewed-{len(cells):02d}'}
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
def md(text): add('markdown', text)
def code(text): add('code', text)

md('''
# Ручной baseline FAR-Trans: проверенная версия

## 1. Задача и границы исследования
На первое число месяца прогнозируем покупку конкретного актива за начинающийся календарный месяц.
Одна строка — **клиент, актив, дата прогноза**. `label=1` — хотя бы одна записанная покупка;
`label=0` — покупка не записана, это не явное отрицательное предпочтение. Повторные покупки допускаются.
Пять признаков выбраны вручную, без добавления новых признаков в этой доработке.

500 клиентов и 50 активов выбираем по истории до первого прогноза и фиксируем.
Признаки используют события строго до даты прогноза. Историю перед каждым месячным прогнозом
обновляем, обученную модель на validation не переобучаем. Все методы сравниваем на одних парах и ответах.

**Ограничения источника.** Авторы реконструировали начальные владения по последующим продажам,
переносили некоторые даты к доступным ценам, добавляли продажи и оценивали стоимость сделок.
Флага происхождения каждой строки нет. Фильтрация по timestamp не устраняет возможное влияние
такой обработки будущих событий на исторические данные. Начальные покупки января 2018 года не входят
в наши окна признаков. Цены и метаданные в модель не подаются.
[Данные и лицензия CC BY 4.0](https://researchdata.gla.ac.uk/1658/) ·
[Обработка авторами, §3.3.1](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).
Наш месячный протокол с повторными покупками отличается от протокола статьи.

**Период наблюдения.** Репозиторий указывает 2 января 2018 — 29 ноября 2022;
статья описывает конец периода только как ноябрь 2022. Консервативно включаем весь указанный день
29 ноября и задаём исключающую границу `2022-11-30`. Максимальная дата сделки не служит доказательством
полноты. Проверка календарных границ не подтверждает отсутствие пропущенных событий или полноту
наблюдений для каждого клиента и актива.

Зарезервированный test остаётся июль — ноябрь 2022. В этой копии его таблица не строится,
метрики и распределения не изучаются. Полнота **целого ноября** не подтверждена указанной границей:
перед будущей итоговой проверкой потребуется уточнить документацию, не менять протокол молча.
''')
md('''
## 2. Загрузка и выбор участников
`pandas` нужен для таблиц и агрегатов, `numpy` — для численных расчётов;
`scikit-learn` — для прежнего бустинга, `joblib` — для сохранения модели.
`threadpool_limits` ограничивает вычисления четырьмя потоками.
Пути заданы относительно рабочего каталога проекта; если ноутбук открыт из `outputs`,
код находит проект уровнем выше. На другом компьютере положи CSV в `work/data/FAR-Trans`.
Идентификаторы читаем как текст, сохраняя начальные нули.
''')
load = original['cells'][2]['source']
load = ''.join(load) if isinstance(load,list) else load
load = load.replace("project_dir = Path('/Users/annazhivikhina/Documents/Codex/2026-10-11/t-lab-recsys-ml-train-validation')", "project_dir = Path.cwd()\nif project_dir.name == 'outputs':\n    project_dir = project_dir.parent\nif not (project_dir / 'work/data/FAR-Trans/transactions.csv').is_file():\n    raise FileNotFoundError('Запускай из папки проекта или укажи project_dir с CSV')")
load = load.replace('import platform', 'import platform\nimport os\nos.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # лимит joblib для этого процесса')
load = load.replace("transactions = pd.read_csv(", "observation_start = pd.Timestamp('2018-01-02')\nobservation_end_exclusive = pd.Timestamp('2022-11-30')\nobservation_source = 'https://researchdata.gla.ac.uk/1658/'\ntransactions_sha256 = hashlib.sha256((data_dir / 'transactions.csv').read_bytes()).hexdigest()\n\ntransactions = pd.read_csv(")
load = load.replace("all_buys = transactions[transactions['transactionType'] == 'Buy'].copy()", "# Сразу закрываем события test для дальнейшего анализа.\ntransactions = transactions[transactions['timestamp'] < test_start].copy()\nall_buys = transactions[transactions['transactionType'] == 'Buy'].copy()")
load = load.replace("print('Сделок:', len(transactions), '| Клиентов:', len(large_client_ids),", "print('Клиентов:', len(large_client_ids),")
load += "\ncase_keys = ['customerID', 'prediction_date']\npair_keys = case_keys + ['ISIN']\n"
code(load)
md(''.join(original['cells'][3]['source']))
builder = ''.join(original['cells'][4]['source'])
builder = builder.replace("    if end > transactions['timestamp'].max().normalize() + pd.Timedelta(days=1):\n        raise ValueError('В данных нет полного месяца для ответа')", "    if not calendar_month_supported(cutoff):\n        raise ValueError('Целевой месяц выходит за документированный период')\n    if end > test_start:\n        raise ValueError('Test закрыт: в этой версии строим только train и validation')")
code('''
def calendar_month_supported(prediction_date):
    cutoff = pd.Timestamp(prediction_date)
    return bool(cutoff >= observation_start
                and cutoff + pd.DateOffset(months=1) <= observation_end_exclusive)

''' + builder)
md('''
## 4. Временное разделение
| Часть | Целевые месяцы | Назначение |
|---|---|---|
| Train | апрель 2020 — декабрь 2021 | обучение |
| Validation | январь — июнь 2022 | сравнение признаков и настроек |
| Test (закрыт) | июль — ноябрь 2022 | будущая итоговая проверка |

Строим только train и validation. Состав кандидатов одинаков во всех месяцах,
признаки и ответы рассчитываются заново для каждой даты. Проверяем уникальность и согласованность счётчиков.
''')
code('''
monthly_tables = [make_month_dataset(date) for date in pd.date_range(
    first_prediction, test_start - pd.DateOffset(months=1), freq='MS')]
dataset = pd.concat(monthly_tables, ignore_index=True)
del monthly_tables
train = dataset[dataset['prediction_date'] < validation_start].copy()
validation = dataset[dataset['prediction_date'] >= validation_start].copy()
assert not dataset.duplicated(pair_keys).any()
assert not dataset[manual_features].isna().any().any()
assert dataset['label'].isin([0, 1]).all()
assert (dataset['user_buy_count_3m'] >= dataset['user_buy_count_1m']).all()
assert (dataset['user_item_buy_count_3m'] >= dataset['user_item_buy_count_1m']).all()
assert (dataset['user_item_buy_count_3m'] <= dataset['user_buy_count_3m']).all()
assert train['prediction_date'].max() + pd.DateOffset(months=1) <= validation['prediction_date'].min()
assert validation['prediction_date'].max() + pd.DateOffset(months=1) <= test_start
assert dataset.groupby(case_keys).size().eq(50).all()
print('Train:', len(train), '| Validation:', len(validation), '| Test: закрыт, не построен')
''')
md('''
## 5. Единое ранжирование и определения метрик
Для равных оценок используем хеш **ключа пары и seed=42**. Он задаёт воспроизводимый порядок,
не зависит от позиции строки, ответов или Python `hash()`. При коллизии хеша последний критерий — ISIN.
Один порядок используется для всех методов, месяцев и настройки модели.

Основные Recall@5, Precision@5 и NDCG@5 — средние по случаям «клиент — месяц»
с хотя бы одной покупкой **внутри каталога**. Для случая с R положительными активами и H попаданиями:
Recall = H/R; Precision = H/5; DCG = сумма `label / log2(rank+1)` в top-5;
IDCG = сумма скидок для первых `min(R,5)` мест, NDCG = DCG/IDCG.
Случаи с R=0 исключаются из всех трёх основных средних. Отдельный `Precision@5_all` включает их.
Итоговый результат — среднее всех учитываемых случаев, а не простое среднее шести месяцев.

LogLoss оценивает вероятности на **всех парах**, меньше — лучше. ROC-AUC оценивает порядок
положительных и отрицательных пар во всём срезе; не заменяет ранжирование внутри клиента.
Если класс один, AUC не определён. Для правил ранжирования обе вероятностные метрики неприменимы.
''')
code('''
def tie_key(customer, date, asset):
    payload = json.dumps([seed, str(customer), pd.Timestamp(date).strftime('%Y-%m-%d'), str(asset)],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b'FAR-tie').digest(), 'big')

def add_ties(data):
    result = data.copy()
    result['_tie'] = np.array([tie_key(c, d, a) for c, d, a in
                              result[pair_keys].itertuples(index=False, name=None)], dtype=np.uint64)
    return result

validation = add_ties(validation)

def top_five(data, criteria):
    # criteria: список столбцов, каждый сортируем по убыванию.
    ranked = data.sort_values(case_keys + criteria + ['_tie', 'ISIN'],
                              ascending=[True, True] + [False]*len(criteria) + [True, True])
    top = ranked.groupby(case_keys, sort=False).head(5).copy()
    top['_rank'] = top.groupby(case_keys, sort=False).cumcount() + 1
    return top

def probability_top(data, probabilities):
    ranked = data.copy()
    # Series связывает оценки с исходными индексами, даже после перестановки строк.
    scores = (probabilities if isinstance(probabilities, pd.Series)
              else pd.Series(probabilities, index=data.index))
    ranked['_score'] = scores.reindex(data.index)
    assert ranked['_score'].notna().all()
    return top_five(ranked, ['_score'])

def case_metrics(data, top):
    result = data.groupby(case_keys)['label'].sum().rename('positive_assets').to_frame()
    result['hits'] = top.groupby(case_keys)['label'].sum().reindex(result.index, fill_value=0)
    eligible = result['positive_assets'] > 0
    result['Recall@5'] = (result['hits'] / result['positive_assets']).where(eligible)
    result['Precision@5'] = (result['hits'] / 5).where(eligible)
    result['Precision@5_all'] = result['hits'] / 5
    gains = top.assign(_gain=top['label'] / np.log2(top['_rank'] + 1))
    dcg = gains.groupby(case_keys)['_gain'].sum().reindex(result.index, fill_value=0)
    ideal_cumulative = np.r_[0., np.cumsum(1 / np.log2(np.arange(1, 6) + 1))]
    idcg = ideal_cumulative[result['positive_assets'].clip(upper=5).to_numpy(dtype=int)]
    result['NDCG@5'] = np.divide(dcg, idcg, out=np.full(len(dcg), np.nan), where=idcg>0)
    return result

def probability_metrics(labels, probabilities):
    if probabilities is None:
        return {'LogLoss': None, 'ROC-AUC': None, 'AUC_note': 'Правило ранжирования не выдаёт вероятности'}
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    assert len(p) == len(y) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    p = np.clip(p, 1e-15, 1 - 1e-15)
    loss = float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())
    single_class = len(np.unique(y)) < 2
    return {'LogLoss': loss,
            'ROC-AUC': None if single_class else float(roc_auc_score(y, p)),
            'AUC_note': 'ROC-AUC не определён: один класс' if single_class else None}

def evaluate_method(data, top, probabilities=None):
    cases = case_metrics(data, top)
    return {name: float(cases[name].mean()) for name in
            ['Recall@5', 'Precision@5', 'NDCG@5', 'Precision@5_all']} | probability_metrics(
                data['label'], probabilities)

def evaluate_predictions(data, probabilities):
    return evaluate_method(data, probability_top(data, probabilities), probabilities)
''')
md('''
## 6. Прежние вероятностные baseline и бустинг
Постоянный baseline — общая доля положительных пар в train. Правило прошлой покупки —
две train-доли для пар с покупкой за прошлый месяц и без неё. Оба оценивают вероятность по train.
Бустинг последовательно строит деревья, исправляющие ошибки общего прогноза.
Сначала сравниваем 2 и 3 признака при 100/7, затем 5 признаков при четырёх прежних настройках.
В модель входят только численные признаки; дата, идентификаторы и ответы не входят.
Встроенное случайное разделение отключено (`early_stopping=False`).

Выбор: выше Recall@5, затем ниже LogLoss, затем меньше итераций и листьев.
Новые диагностики не меняют это правило. Всего **шесть обучений**, не только четыре настройки сетки.
''')
code('''
baseline_probability = float(train['label'].mean())
validation_predictions = np.full(len(validation), baseline_probability)
repeat_rates = train.groupby(train['user_item_buy_count_1m'] > 0)['label'].mean()
repeat_predictions = ((validation['user_item_buy_count_1m'] > 0).map(repeat_rates)
                      .fillna(baseline_probability).to_numpy())
print(f'Постоянная train-вероятность: {baseline_probability:.4%}')
''' + ''.join(original['cells'][10]['source']))
code(''.join(original['cells'][11]['source']))
md('''
## 7. Полное сравнение на validation
Добавляем три правила без обучения: популярность, покупки пары за 3 месяца,
покупки пары за 3 месяца → популярность. Последнее сортирует по двум критериям
лексикографически: популярность влияет только при равенстве первого счётчика.
Для этих правил LogLoss и ROC-AUC пусты, поскольку счётчики не являются вероятностями.
Контроль 5 признаков 100/7 отделяет пользу признаков от подбора настроек.
''')
code('''
methods = {
    'constant': {'name': 'Постоянный baseline', 'probabilities': validation_predictions},
    'repeat_1m': {'name': 'Правило прошлой покупки (1м)', 'probabilities': repeat_predictions},
    'boost_2': {'name': 'Бустинг: 2 признака, 100/7', 'probabilities': predictions_two},
    'boost_3': {'name': 'Бустинг: 3 признака, 100/7', 'probabilities': predictions_three},
    'boost_5_control': {'name': 'Бустинг: 5 признаков, 100/7', 'probabilities': stored_predictions[(100, 7)]},
    'boost_selected': {'name': f'Бустинг: 5 признаков, {best_trees}/{best_leaves}', 'probabilities': manual_predictions},
    'popularity': {'name': 'Популярность за 1м', 'criteria': ['item_buy_count_1m']},
    'pair_3m': {'name': 'Покупки пары за 3м', 'criteria': ['user_item_buy_count_3m']},
    'pair_3m_popularity': {'name': 'Покупки пары за 3м → популярность',
                          'criteria': ['user_item_buy_count_3m', 'item_buy_count_1m']}
}
top_lists, case_tables, comparison_rows, monthly_rows = {}, {}, [], []
for method_id, spec in methods.items():
    probabilities = spec.get('probabilities')
    top = (top_five(validation, spec['criteria']) if probabilities is None
           else probability_top(validation, probabilities))
    top_lists[method_id] = top
    case_tables[method_id] = case_metrics(validation, top)
    comparison_rows.append({'method_id': method_id, 'method': spec['name'],
                            **evaluate_method(validation, top, probabilities)})
    for date, month in validation.groupby('prediction_date', sort=True):
        month_top = top[top['prediction_date'] == date]
        month_prob = None if probabilities is None else pd.Series(
            probabilities, index=validation.index).reindex(month.index).to_numpy()
        monthly_rows.append({'method_id': method_id, 'method': spec['name'],
                             'month': date.strftime('%Y-%m-%d'),
                             **evaluate_method(month, month_top, month_prob)})
manual_comparison = pd.DataFrame(comparison_rows)
monthly_results = pd.DataFrame(monthly_rows)
print(manual_comparison.drop(columns='AUC_note').round(5).to_string(index=False, na_rep='—'))
print('\\nМесячные результаты (все методы):')
print(monthly_results.drop(columns='AUC_note').round(5).to_string(index=False, na_rep='—'))

def evaluation_counts(data):
    cases = data.groupby(case_keys)['label'].sum()
    eligible = cases[cases > 0]
    return {'candidate_pairs': len(data), 'positive_pairs': int(data['label'].sum()),
            'total_user_months': len(cases), 'eligible_user_months': len(eligible),
            'eligible_unique_clients': int(eligible.index.get_level_values('customerID').nunique())}

count_rows = [{'period': 'validation', **evaluation_counts(validation)}]
count_rows += [{'period': date.strftime('%Y-%m-%d'), **evaluation_counts(month)}
               for date, month in validation.groupby('prediction_date')]
evaluation_denominators = pd.DataFrame(count_rows)
print('\\nЗнаменатели для всех методов:')
print(evaluation_denominators.to_string(index=False, na_rep='—'))
''')
md('''
## 8. Покрытие каталога и новые/повторные покупки
Покрытие — доля уникальных покупок выбранных клиентов, входящих в наши 50 активов.
Считаем уникальные тройки клиент–актив–месяц; повторные сделки одной пары в месяце дают одну покупку.

**Повторная** положительная пара: есть Buy за 3 календарных месяца перед прогнозом.
**Новая**: Buy в этом окне нет; это не означает первую покупку в жизни.
Для диагностики сохраняем исходные top-5 методов. Ответы и группа используются только при оценке:
не фильтруем кандидатов и не перестраиваем рекомендации. Recall группы усредняется
по случаям с хотя бы одним положительным активом этой группы внутри каталога.
''')
code('''
validation_buys = large_buys[(large_buys['timestamp'] >= validation_start)
                             & (large_buys['timestamp'] < test_start)].copy()
validation_buys['prediction_date'] = validation_buys['timestamp'].dt.to_period('M').dt.to_timestamp()
unique_buys = validation_buys[pair_keys].drop_duplicates()
in_catalog = unique_buys['ISIN'].isin(large_asset_ids)
def coverage_row(period, triples):
    numerator = int(triples['ISIN'].isin(large_asset_ids).sum())
    denominator = len(triples)
    return {'period': period, 'catalog_unique_buys': numerator,
            'all_unique_buys': denominator,
            'coverage': numerator / denominator if denominator else None}
coverage_rows = [coverage_row('validation', unique_buys)]
coverage_rows += [coverage_row(date.strftime('%Y-%m-%d'),
                              unique_buys[unique_buys['prediction_date'] == date])
                 for date in sorted(validation['prediction_date'].unique())]
catalog_coverage = pd.DataFrame(coverage_rows)
assert coverage_rows[0]['catalog_unique_buys'] == int(validation['label'].sum())
print(catalog_coverage.round(5).to_string(index=False, na_rep='—'))

def stratum_results(data, top, period, method_id):
    rows = []
    for group, repeated in [('new_3m', False), ('repeat_3m', True)]:
        positives = data[(data['label'] == 1) & ((data['user_item_buy_count_3m'] > 0) == repeated)]
        totals = positives.groupby(case_keys).size()
        hits = top[(top['label'] == 1) & ((top['user_item_buy_count_3m'] > 0) == repeated)]
        hits_per_case = hits.groupby(case_keys).size().reindex(totals.index, fill_value=0)
        rows.append({'period': period, 'method_id': method_id, 'group': group,
                     'Recall@5': float((hits_per_case / totals).mean()),
                     'positive_pairs': len(positives), 'eligible_user_months': len(totals),
                     'eligible_unique_clients': int(totals.index.get_level_values('customerID').nunique())})
    return rows

strata_rows = []
for method_id, top in top_lists.items():
    strata_rows += stratum_results(validation, top, 'validation', method_id)
    for date, month in validation.groupby('prediction_date'):
        strata_rows += stratum_results(month, top[top['prediction_date'] == date],
                                       date.strftime('%Y-%m-%d'), method_id)
purchase_diagnostics = pd.DataFrame(strata_rows)
print('\\nДиагностика всего validation:')
print(purchase_diagnostics[purchase_diagnostics['period']=='validation'].round(5).to_string(index=False, na_rep='—'))
''')
md('''
## 9. Парный bootstrap по клиентам
Сэмплируем всех 500 клиентов с возвращением 5000 раз. Выбранный клиент приносит все шесть месяцев;
повторный выбор приносит их повторно. Одни выборки используются для обеих разниц.
В каждой выборке делим **сумму разниц Recall всех учитываемых случаев** на число этих случаев,
а не усредняем средние клиентов. Для скорости суммируем вклад клиента, сохраняя его число случаев.
Выборка без учитываемых случаев даёт пропуск и исключается из процентилей с отдельным счётчиком.

95% процентильные интервалы **диагностические**: выбор модели на этом же validation не позволяет
считать их независимым подтверждением преимущества. Интервал, включающий ноль, не доказывает
устойчивого преимущества. Bootstrap не отражает все риски смены периода или ограничений источника.
''')
code('''
bootstrap_seed = 20261012
bootstrap_repeats = 5000
bootstrap_comparisons = ['pair_3m_popularity', 'boost_5_control']
selected_cases = case_tables['boost_selected']
eligible_cases = selected_cases['positive_assets'] > 0
client_case_counts = eligible_cases.groupby(level='customerID').sum().reindex(large_client_ids, fill_value=0).to_numpy()
client_delta_sums = np.column_stack([
    (selected_cases.loc[eligible_cases, 'Recall@5'] - case_tables[other].loc[eligible_cases, 'Recall@5'])
    .groupby(level='customerID').sum().reindex(large_client_ids, fill_value=0).to_numpy()
    for other in bootstrap_comparisons])

def draw_delta(draw, delta_sums, case_counts):
    denominator = int(case_counts[draw].sum())
    if denominator == 0:
        return np.full(delta_sums.shape[1], np.nan)
    return delta_sums[draw].sum(axis=0) / denominator

rng = np.random.default_rng(bootstrap_seed)
bootstrap_deltas = np.empty((bootstrap_repeats, len(bootstrap_comparisons)))
for iteration in range(bootstrap_repeats):
    draw = rng.integers(0, len(large_client_ids), size=len(large_client_ids))
    bootstrap_deltas[iteration] = draw_delta(draw, client_delta_sums, client_case_counts)
bootstrap_rows = []
for column, other in enumerate(bootstrap_comparisons):
    values = bootstrap_deltas[:, column]
    valid = np.isfinite(values)
    low, high = np.quantile(values[valid], [0.025, 0.975]) if valid.any() else (np.nan, np.nan)
    point = float((selected_cases.loc[eligible_cases, 'Recall@5']
                   - case_tables[other].loc[eligible_cases, 'Recall@5']).mean())
    bootstrap_rows.append({'comparison': f'boost_selected - {other}',
                           'point_difference': point, 'ci_95_low': float(low), 'ci_95_high': float(high),
                           'contains_zero': bool(low <= 0 <= high) if valid.any() else None,
                           'valid_draws': int(valid.sum()), 'zero_eligible_draws': int((~valid).sum()),
                           'interpretation': 'Диагностика на validation, использованном для выбора модели'})
bootstrap_results = pd.DataFrame(bootstrap_rows)
print(bootstrap_results.round(6).to_string(index=False, na_rep='—'))
''')
md('''
## 10. Проверки расчётов
Небольшие искусственные примеры проверяют смысл метрик и кратность bootstrap.
На реальном validation проверяем одинаковые кандидаты, пятиэлементные списки,
порядок при перестановке строк, согласованность покрытия и знаменателей.
Эти проверки не оценивают test.
''')
code('''
# Известный ответ: один клиент купил 3 актива, top-5 нашёл 2;
# второй не купил ничего, он исключён из основных средних.
fixture = pd.DataFrame({'customerID': ['A']*6 + ['B']*6,
                        'prediction_date': [pd.Timestamp('2022-01-01')]*12,
                        'ISIN': list('abcdef')*2,
                        'label': [1,0,1,0,0,1] + [0]*6,
                        'user_item_buy_count_3m': [1,0,0,0,0,1]+[0]*6,
                        '_score': list(range(6,0,-1))*2})
fixture = add_ties(fixture)
fixture_top = top_five(fixture, ['_score'])
fixture_metrics = evaluate_method(fixture, fixture_top)
assert np.isclose(fixture_metrics['Recall@5'], 2/3)
assert np.isclose(fixture_metrics['Precision@5'], 2/5)
assert np.isclose(fixture_metrics['Precision@5_all'], 1/5)
assert np.isclose(fixture_metrics['NDCG@5'], 1.5 / (1+1/np.log2(3)+0.5))
fixture_strata = {r['group']: r for r in stratum_results(fixture, fixture_top, 'fixture', 'fixture')}
assert fixture_strata['new_3m']['Recall@5'] == 1
assert fixture_strata['repeat_3m']['Recall@5'] == 0.5
assert probability_metrics([0,0], [0.1,0.2])['ROC-AUC'] is None
assert probability_metrics([0,1], None)['LogLoss'] is None
# Клиент A: 2 случая, сумма разниц 0.6; B: 1 случай, 0.8.
# Выбор A,A,B даёт (0.6+0.6+0.8)/(2+2+1)=0.4.
assert np.allclose(draw_delta(np.array([0,0,1]), np.array([[0.6],[0.8]]), np.array([2,1])), [0.4])
assert np.isnan(draw_delta(np.array([0]), np.array([[0.]]), np.array([0]))).all()

def recommendation_keys(top):
    return list(top.sort_values(case_keys + ['_rank'])[pair_keys + ['_rank']]
                .itertuples(index=False, name=None))

shuffled = validation.sample(frac=1, random_state=123)
for method_id, spec in methods.items():
    top = top_lists[method_id]
    assert top.groupby(case_keys).size().eq(5).all()
    assert len(top) == 15000 and not top.duplicated(pair_keys).any()
    assert case_tables[method_id].index.equals(selected_cases.index)
    probs = spec.get('probabilities')
    other_top = (top_five(shuffled, spec['criteria']) if probs is None else
                 probability_top(shuffled, pd.Series(probs, index=validation.index)))
    assert recommendation_keys(top) == recommendation_keys(other_top)
    # Изменение будущих ответов не меняет сами рекомендации.
    changed_labels = shuffled.assign(label=1-shuffled['label'])
    changed_top = (top_five(changed_labels, spec['criteria']) if probs is None else
                   probability_top(changed_labels, pd.Series(probs, index=validation.index)))
    assert recommendation_keys(top) == recommendation_keys(changed_top)
    monthly = monthly_results[monthly_results['method_id'] == method_id].merge(
        evaluation_denominators, left_on='month', right_on='period', validate='one_to_one')
    for metric in ['Recall@5', 'Precision@5', 'NDCG@5']:
        weighted = np.average(monthly[metric], weights=monthly['eligible_user_months'])
        actual = manual_comparison.set_index('method_id').loc[method_id, metric]
        assert np.isclose(weighted, actual)
for row in coverage_rows[1:]:
    assert row['catalog_unique_buys'] == int(validation.loc[
        validation['prediction_date'] == row['period'], 'label'].sum())
for method_id in methods:
    overall = purchase_diagnostics[(purchase_diagnostics['method_id']==method_id)
                                  & (purchase_diagnostics['period']=='validation')]
    assert overall['positive_pairs'].sum() == int(validation['label'].sum())
assert calendar_month_supported('2022-06-01')
assert not calendar_month_supported('2022-11-01')
try:
    make_month_dataset('2022-07-01')
except ValueError as error:
    assert 'Test закрыт' in str(error)
else:
    raise AssertionError('Нельзя строить закрытый test')
checks = {'metric_fixture': True, 'stratum_original_top5': True, 'one_class_auc': True,
          'ranking_only_null_metrics': True, 'bootstrap_multiplicity': True,
          'bootstrap_empty_draw': True, 'tie_row_order_invariance_all_methods': True,
          'future_labels_do_not_change_top5': True, 'monthly_overall_consistency': True,
          'coverage_label_consistency': True, 'test_guard': True}
print('Проверки пройдены:', ', '.join(checks))
''')
md('''
## 11. Выводы и бюджет следующего сравнения
Числа ниже берутся из рассчитанных таблиц. Recall описывает найденные покупки среди учитываемых
случаев, Precision — долю попаданий в пятёрке, LogLoss — ошибку вероятностей.
Учитываем только фиксированный каталог; покрытие показывает, какую часть покупок он включает.
Нет доказательства качества рекомендаций для остальных активов, новых клиентов или прибыльности.

Для будущего автоматического подхода отдельно фиксируем бюджет **генерации и отбора признаков**
и бюджет **настройки модели**, включая все проверенные наборы, а не только финальную сетку.
Здесь есть 3 набора признаков, 6 обучений бустинга (2 сравнения наборов + 4 настройки пяти признаков),
3 новых правила ранжирования и bootstrap без обучения. Более ранние эксперименты не задокументированы,
их число неизвестно. Стоимость ручного проектирования и отбора признаков также не измерялась.
Это ограниченный ручной baseline, не полностью настроенное или лучшее возможное решение.
''')
code('''
results_by_id = manual_comparison.set_index('method_id')
selected = results_by_id.loc['boost_selected']
control = results_by_id.loc['boost_5_control']
combo = results_by_id.loc['pair_3m_popularity']
denoms = evaluation_denominators.iloc[0]
coverage = catalog_coverage.iloc[0]['coverage']
conclusions = [
    f"Выбрано {best_trees}/{best_leaves} по прежнему правилу, с новым разрешением равенств.",
    f"Recall@5 выбранной модели: {selected['Recall@5']:.4f}; это средняя доля найденных купленных активов, не доля правильных рекомендаций.",
    f"Оценка по {int(denoms['eligible_user_months'])} из {int(denoms['total_user_months'])} случаев и {int(denoms['eligible_unique_clients'])} уникальным клиентам.",
    f"Precision@5: {selected['Precision@5']:.4f}; Precision@5_all: {selected['Precision@5_all']:.4f}; NDCG@5: {selected['NDCG@5']:.4f}.",
    f"Фиксированный каталог покрывает {coverage:.2%} уникальных покупок выбранных клиентов на validation.",
    f"При 100/7 добавление популярности (2→3 признака) меняет Recall на {results_by_id.loc['boost_3','Recall@5']-results_by_id.loc['boost_2','Recall@5']:+.4f}, LogLoss на {results_by_id.loc['boost_3','LogLoss']-results_by_id.loc['boost_2','LogLoss']:+.5f}.",
    f"При 100/7 добавление трёхмесячной истории (3→5) меняет Recall на {control['Recall@5']-results_by_id.loc['boost_3','Recall@5']:+.4f}, LogLoss на {control['LogLoss']-results_by_id.loc['boost_3','LogLoss']:+.5f}.",
    f"Подбор настроек при пяти признаках меняет Recall на {selected['Recall@5']-control['Recall@5']:+.4f}, LogLoss на {selected['LogLoss']-control['LogLoss']:+.5f}.",
    f"Простое правило пары + популярность: Recall {combo['Recall@5']:.4f}; разница выбранной модели {selected['Recall@5']-combo['Recall@5']:+.4f}.",
    "Изменение разрешения равенств может изменить результаты относительно прежней проверки: новый порядок привязан к ключу, прежний — к позиции строки. Вероятности и LogLoss от этого не меняются.",
    "Bootstrap — диагностика на том же validation, где выбрана модель; test не оценён."
]
for row in bootstrap_rows:
    message = ('интервал включает ноль, устойчивое преимущество не подтверждено'
               if row['contains_zero'] else 'интервал не включает ноль, но оценка зависима от выбора модели на validation')
    conclusions.append(f"{row['comparison']}: Δ={row['point_difference']:+.4f}, 95% CI [{row['ci_95_low']:+.4f}, {row['ci_95_high']:+.4f}]; {message}.")
print('\\n'.join(conclusions))
''')
md('''
## 12. Сохранение и воспроизводимость
Создаём только новые файлы с суффиксом `reviewed`, исходный ноутбук и результаты не меняем.
JSON хранит определения, знаменатели, месячные и общие результаты, диагностики, ограничения,
версии, хеш CSV, состав участников и `test_evaluated=False`. Пропуски сохраняем как стандартный `null`.
Для повторного запуска: **Restart → Run All**, окружение с pandas, numpy, scikit-learn, joblib, threadpoolctl.
Ниже зафиксированы версии фактического запуска; для точного повторения используй их.
''')
code('''
metric_definitions = {
    'primary_population': 'user-months with at least one in-catalog positive',
    'Recall@5': 'mean(hits / in-catalog positive assets) over eligible user-months',
    'Precision@5': 'mean(hits / 5) over eligible user-months',
    'NDCG@5': 'mean(binary DCG / IDCG(min(R,5))) over eligible user-months; discount=1/log2(rank+1)',
    'Precision@5_all': 'mean(hits / 5) over all user-months, including zero positives',
    'LogLoss': 'mean binary negative log-likelihood over all candidate pairs; clip [1e-15,1-1e-15]',
    'ROC-AUC': 'global pair AUC over all candidates; null for one class or ranking-only method',
    'overall_aggregation': 'direct macro average over eligible user-months, not unweighted month averages'
}
limits = [
    'Fixed cohort of existing clients and fixed catalog; asset availability not modeled.',
    'Coverage denominator includes out-of-catalog buys; ranking metrics do not.',
    'Zero means no recorded buy, not an explicit dislike; no exposure data.',
    'Source reconstructed events and date changes may use future information; timestamp filters do not fully remove this limitation.',
    'No independent advantage claim from validation selection and diagnostic bootstrap.',
    'Calendar checks do not establish event completeness for each client and asset.',
    'November test month is not fully supported by repository collection boundary; documentation clarification required before final evaluation.',
    'Earlier experimental count and manual feature design/selection cost unknown; monthly repeats protocol differs from paper.',
    'No profitability, suitability or causal recommendation-effect assessment.'
]
baseline_record = {
    'status': 'manual_baseline_reviewed_validation_frozen', 'features': manual_features,
    'parameters': {'max_iter': best_trees, 'max_leaf_nodes': best_leaves,
                   'learning_rate': float(manual_model.learning_rate), 'early_stopping': False, 'random_state': seed},
    'selection_rule': 'max Recall@5, then min LogLoss, then min max_iter, then min max_leaf_nodes',
    'train_months': ['2020-04-01', '2021-12-01'], 'validation_months': ['2022-01-01', '2022-06-01'],
    'test_months': ['2022-07-01', '2022-11-01'], 'test_evaluated': False, 'test_table_built': False,
    'history_protocol': 'rolling history before monthly cutoff; fixed train-fitted model during validation; repeats allowed',
    'client_ids': large_client_ids, 'asset_ids': large_asset_ids, 'cohort_random_state': seed,
    'cohort_selection_window': ['2019-10-01', '2020-04-01 exclusive'],
    'label_definition': 'at least one Buy in target calendar month; pair-month duplicates count once',
    'metric_definitions': metric_definitions, 'denominators': evaluation_denominators.to_dict('records'),
    'comparison': manual_comparison.to_dict('records'), 'monthly_results': monthly_results.to_dict('records'),
    'tuning': tuning_results.to_dict('records'),
    'ranking_rules': {key: value['criteria'] for key, value in methods.items() if 'criteria' in value},
    'tie_break': {'seed': seed, 'algorithm': 'BLAKE2b 64-bit, person=FAR-tie, unsigned big-endian',
                  'payload': 'UTF-8 compact JSON [seed, customerID, YYYY-MM-DD, ISIN]',
                  'order': 'criteria descending; digest ascending; ISIN ascending on collision',
                  'row_order_independent': True, 'same_for_all_methods': True},
    'repeat_definition': 'positive pair with Buy in preceding 3 calendar months strictly before cutoff',
    'new_definition': 'positive pair without Buy in same 3-month window, not lifetime first purchase',
    'purchase_diagnostics': purchase_diagnostics.to_dict('records'),
    'catalog_coverage': {'definition': 'unique client-asset-month Buy triples inside catalog / all assets for cohort',
                         'results': catalog_coverage.to_dict('records')},
    'bootstrap': {'seed': bootstrap_seed, 'repeats': bootstrap_repeats, 'unit': 'client',
                  'pool_size': len(large_client_ids), 'draw_size': len(large_client_ids), 'replacement': True,
                  'paired_draws_shared_across_comparisons': True,
                  'estimand': 'macro Recall difference over eligible user-months with client multiplicity',
                  'interval': '2.5% and 97.5% percentile',
                  'empty_draw_handling': 'null delta, excluded from quantiles, counted explicitly',
                  'diagnostic_only': True, 'results': bootstrap_results.to_dict('records')},
    'observation': {'start_inclusive': str(observation_start.date()),
                    'end_exclusive': str(observation_end_exclusive.date()), 'source': observation_source,
                    'source_checked_on': '2026-10-12',
                    'reason': 'Repository collection period ends 29 November 2022; include that day, conservative exclusive next day',
                    'article_source': 'https://arxiv.org/html/2407.08692v1',
                    'max_timestamp_used_as_completeness_proof': False,
                    'full_november_supported': False, 'event_completeness_proven': False},
    'experiment_budget': {'boosting_fits_in_this_notebook': 6, 'counting_unit': 'six fits per complete execution; verification reruns repeat the same configurations', 'feature_sets_evaluated': [2,3,5],
                         'five_feature_settings': settings_to_check,
                         'additional_ranking_only_rules': 3, 'bootstrap_draws': bootstrap_repeats,
                         'earlier_experiments': None, 'manual_feature_generation_selection_cost': None,
                         'future_budget': 'record feature generation/selection and model tuning separately, including all trials'},
    'limits': limits, 'conclusions': conclusions, 'checks': checks,
    'transactions_sha256': transactions_sha256,
    'source_notebook_sha256': '__SOURCE_SHA__',
    'versions': {'python': platform.python_version(), 'pandas': pd.__version__, 'numpy': np.__version__,
                 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__},
    'output_directory': str(output_dir.resolve())
}

def json_safe(value):
    if isinstance(value, dict): return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value

baseline_record = json_safe(baseline_record)
output_dir.mkdir(exist_ok=True)
joblib.dump({'model': manual_model, 'features': manual_features, 'metadata': baseline_record},
            output_dir / 'manual_baseline_reviewed_model.joblib')
(output_dir / 'manual_baseline_reviewed_results.json').write_text(
    json.dumps(baseline_record, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
print('Сохранены manual_baseline_reviewed_model.joblib и manual_baseline_reviewed_results.json. Test не оценивался.')
'''.replace('__SOURCE_SHA__', hashlib.sha256(source.read_bytes()).hexdigest()))
notebook = {'cells': cells, 'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
notebook['metadata']['reviewed_execution'] = {'mode': 'sequential fresh Python process', 'test_evaluated': False}
path = root / 'outputs/manual_baseline_reviewed.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
print('Created', path, 'cells:', len(cells))
