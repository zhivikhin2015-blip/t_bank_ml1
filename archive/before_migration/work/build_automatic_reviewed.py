from pathlib import Path
import json,hashlib,textwrap,ast
root=Path.cwd()
source_path=Path('/Users/annazhivikhina/Desktop/02_automatic_features.ipynb')
source=json.loads(source_path.read_text())
source_sha=hashlib.sha256(source_path.read_bytes()).hexdigest()
cells=[]
def add(kind,text):
    cell={'cell_type':kind,'metadata':{},'id':f'automatic-reviewed-{len(cells):02d}',
          'source':textwrap.dedent(text).strip()+'\n'}
    if kind=='code':
        ast.parse(cell['source'])
        cell.update(execution_count=None,outputs=[])
    cells.append(cell)
def md(text):add('markdown',text)
def code(text):add('code',text)
def old(i):return ''.join(source['cells'][i]['source'])

md('''
# Автоматическое расширение ручного baseline и отбор признаков в заданном пространстве поиска

## 1. Что автоматизируем
Человек задаёт группы (`user`, `item`, `user_item`), агрегат (число покупок), окна
(7, 14, 30, 60 дней), критерий отбора и лимит. **Программа создаёт 12 вариантов и выбирает
до пяти новых признаков внутри этих рамок. Пять ручных признаков сохраняются.**
Это расширение исходного baseline; полного отказа от ручного проектирования нет.

Одна строка — клиент, актив и первое число целевого месяца. Ответ 1 означает хотя бы одну
записанную покупку за этот месяц, 0 — её отсутствие; это не явное отрицательное предпочтение.
Повторные покупки допускаются. Признаки используют только события строго до прогноза.
Перед каждым месяцем история обновляется, обученная модель остаётся фиксированной.

500 клиентов и 50 активов взяты из ручного baseline, выбраны по истории до первого прогноза.
Train: апрель 2020 — декабрь 2021; validation: январь — июнь 2022.
Test: июль — ноябрь 2022, **закрыт: не строится и не оценивается**.

**Границы источника.** FAR-Trans содержит реконструированные события и переносы дат по доступности
цен; обработка авторов могла использовать будущие события, а происхождение строк не размечено.
Фильтры timestamp не устраняют этот риск полностью. Январский блок реконструкций 2018 года
не попадает в наши окна. Цены и метаданные модели не используют.
[Обработка FAR-Trans, §3.3.1](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).

[Репозиторий](https://researchdata.gla.ac.uk/1658/) указывает конец сбора 29 ноября 2022.
Сохраняем консервативную исключающую границу ручного baseline `2022-11-30`.
Она проверяет календарные границы, но не доказывает отсутствие пропущенных событий.
Полнота всего ноября для будущего test требует уточнения документации; максимальная дата сделки
не разрешает этот вопрос. Границы зарезервированного test не меняем.
''')
md('''
## 2. План и небольшой пример
В **первой ячейке этого раздела** создаются только описания вариантов, без прогнозов.
Первый `for` выбирает группу, второй — окно; `append()` сохраняет комбинацию.
Дальнейшие разделы считают признаки, обучают модели и оценивают качество.
Окно в 30 дней отличается от календарного месяца.
''')
code(old(2))
code(old(3))
code(old(4))
md('''
## 3. Проверка исходных артефактов и общий генератор
До любого обучения сравниваем хеш CSV, состав участников, даты, определения пяти ручных
признаков и метрик в ручных и прежних автоматических артефактах. `joblib` открывает наши локальные
сохранённые модели. Хеши исходных файлов сохраняются для воспроизводимости.
`groupby().size()` считает записанные сделки; `merge(..., how="left")` сохраняет все пары,
а отсутствующие подсчёты заменяются нулями. Популярность считается по всем клиентам в истории.
''')
s=old(5)
insertion='''
import hashlib
import joblib

artifact_paths = {
    'source_notebook': Path('__SOURCE_PATH__'),
    'manual_results': project_dir / 'outputs/manual_baseline_reviewed_results.json',
    'manual_model': project_dir / 'outputs/manual_baseline_reviewed_model.joblib',
    'previous_automatic_results': project_dir / 'outputs/automatic_features_results.json',
    'previous_automatic_model': project_dir / 'outputs/automatic_features_model.joblib'
}
# При повторном запуске отдельно проверяем уже подготовленный ручной контроль до обучения.
prechecked_control_path = project_dir / 'outputs/manual_control_100_7_reviewed_model.joblib'
if prechecked_control_path.exists():
    prechecked_control = joblib.load(prechecked_control_path)
    assert prechecked_control['features'] == baseline['features']
    for key in ['transactions_sha256','client_ids','asset_ids','train_months','validation_months']:
        assert prechecked_control['metadata'][key] == baseline[key]
    assert prechecked_control['metadata']['reference_results_sha256'] == hashlib.sha256(artifact_paths['manual_results'].read_bytes()).hexdigest()
    params = prechecked_control['model'].get_params()
    assert (params['max_iter'],params['max_leaf_nodes'],params['random_state']) == (100,7,42)
    assert not params['early_stopping'] and params['learning_rate'] == .1
    artifact_paths['cached_manual_control'] = prechecked_control_path
source_artifact_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in artifact_paths.items()}
manual_saved = joblib.load(artifact_paths['manual_model'])
previous_auto = json.loads(artifact_paths['previous_automatic_results'].read_text())
previous_auto_saved = joblib.load(artifact_paths['previous_automatic_model'])
assert manual_saved['metadata'] == baseline
assert previous_auto_saved['metadata'] == previous_auto
expected_manual_features = [
    'user_buy_count_1m', 'user_item_buy_count_1m', 'item_buy_count_1m',
    'user_buy_count_3m', 'user_item_buy_count_3m'
]
assert baseline['features'] == manual_saved['features'] == previous_auto['manual_features'] == expected_manual_features
assert previous_auto_saved['features'] == previous_auto['features']
for key in ['client_ids', 'asset_ids', 'train_months', 'validation_months', 'test_months',
            'cohort_random_state', 'cohort_selection_window', 'metric_definitions', 'tie_break', 'observation', 'label_definition', 'history_protocol']:
    assert baseline[key] == previous_auto[key], key
assert baseline['train_months'] == ['2020-04-01', '2021-12-01']
assert baseline['validation_months'] == ['2022-01-01', '2022-06-01']
assert baseline['test_months'] == ['2022-07-01', '2022-11-01']
assert len(set(baseline['client_ids'])) == 500 and len(set(baseline['asset_ids'])) == 50
assert feature_plan == previous_auto['candidate_plan']
transactions_sha256 = hashlib.sha256((project_dir/'work/data/FAR-Trans/transactions.csv').read_bytes()).hexdigest()
assert transactions_sha256 == baseline['transactions_sha256'] == previous_auto['transactions_sha256']
assert baseline['tie_break']['seed'] == 42
checks = {'input_artifact_consistency_before_training': True, 'csv_hash_before_training': True}
print('Исходные артефакты согласованы, хеш CSV совпадает.')
'''.replace('__SOURCE_PATH__',str(source_path))
pos=s.index('candidate_pairs =')
s=s[:pos]+insertion+'\n'+s[pos:]
code(s)
md('''
## 4. Таблица всех месяцев и проверки счётчиков
Сохраняем прежние 5 признаков с календарными окнами и 12 кандидатов с дневными окнами.
Ответ рассчитывается отдельно по целевому месяцу и не передаётся модели как признак.
Окна дневных подсчётов вложены: при увеличении окна число покупок не уменьшается.
Покупки пары не могут превышать ни все покупки клиента, ни все покупки актива.
''')
s=old(6)+'''
assert len(train) == 500*50*21 and len(validation) == 500*50*6
assert [str(train['prediction_date'].min().date()), str(train['prediction_date'].max().date())] == baseline['train_months']
assert [str(validation['prediction_date'].min().date()), str(validation['prediction_date'].max().date())] == baseline['validation_months']
assert set(candidate_pairs['customerID']) == set(baseline['client_ids'])
assert set(candidate_pairs['ISIN']) == set(baseline['asset_ids'])
for group in groups:
    for smaller, larger in zip(windows[:-1], windows[1:]):
        assert (dataset[f'{group}_buy_count_{smaller}d'] <= dataset[f'{group}_buy_count_{larger}d']).all()
for days in windows:
    assert (dataset[f'user_item_buy_count_{days}d'] <= dataset[f'user_buy_count_{days}d']).all()
    assert (dataset[f'user_item_buy_count_{days}d'] <= dataset[f'item_buy_count_{days}d']).all()
for months in [1,3]:
    assert (dataset[f'user_item_buy_count_{months}m'] <= dataset[f'user_buy_count_{months}m']).all()
assert (dataset['user_buy_count_1m'] <= dataset['user_buy_count_3m']).all()
assert (dataset['user_item_buy_count_1m'] <= dataset['user_item_buy_count_3m']).all()
checks.update(unique_pair_months=True, fifty_candidates_per_case=True, counter_monotonicity=True,
              pair_count_bounds=True, development_time_boundaries=True)
'''
code(s)
md('''
## 5. Внутренний временной отбор
Вспомогательная модель учится на апреле 2020 — июне 2021. Важность признаков оценивается
на июле — декабре 2021, внутри train. Основной validation ещё не участвует в выборе признаков.
Вспомогательная модель имеет прежние настройки 100/7; лимит потоков снижает нагрузку.
''')
s=old(7)+'''
assert seed == baseline['cohort_random_state'] == previous_auto['feature_selection']['seed']
assert [str(inner_train['prediction_date'].min().date()), str(inner_train['prediction_date'].max().date())] == previous_auto['feature_selection']['inner_train']
assert [str(inner_check['prediction_date'].min().date()), str(inner_check['prediction_date'].max().date())] == previous_auto['feature_selection']['inner_check']
assert inner_check['prediction_date'].max() + pd.DateOffset(months=1) <= validation['prediction_date'].min()
'''
code(s)
md('''
## 6. Permutation importance — эвристика отбора
Перемешиваем один столбец и измеряем ухудшение `neg_log_loss` на внутреннем отложенном периоде.
Сохраняем до пяти новых кандидатов с положительной средней важностью; ручные пять остаются.
Три перестановки и seed сохраняем. При равной важности сортируем названия признаков.

**Ограничения метода:** соседние окна и ручные признаки коррелируют и могут заменять друг друга.
Низкая важность не доказывает отсутствие полезной информации. Положительная средняя при трёх
перестановках не является тестом статистической значимости. Перестановка по отдельным строкам
разрушает структуру агрегатов: пользовательский признак перестаёт быть постоянным внутри
клиент–месяца, а признак актива — внутри актив–месяца. Отбор по LogLoss не гарантирует улучшения
Recall или NDCG. Алгоритм здесь не меняем; структурные перестановки или другие способы отбора
могут быть отдельными будущими экспериментами.
[Документация метода](https://scikit-learn.org/stable/modules/permutation_importance.html).
''')
s=old(8)
pos=s.index('selector_model =')
s=s[:pos]+'''assert max_new_features == previous_auto['feature_selection']['max_new_features']
assert permutation_repeats == previous_auto['feature_selection']['repeats']
assert previous_auto['feature_selection']['scoring'] == 'neg_log_loss'

'''+s[pos:]
s+='''
selection_reproduced = selected_new == previous_auto['selected_new_features']
print('Прежний отбор воспроизведён:', selection_reproduced)
checks['internal_split_before_feature_selection'] = True
'''
code(s)
md('''
## 7. Метрики и одинаковый порядок равенств
Основные Recall@5, Precision@5 и NDCG@5 — средние по клиент–месяцам с хотя бы одной покупкой
внутри каталога из 50 активов. Если R — число положительных активов случая, а H — попадания
в top-5: Recall=H/R, Precision=H/5. NDCG использует бинарные ответы, скидку `1/log2(rank+1)`
и идеальный DCG по первым `min(R,5)` местам. Случаи без покупок исключены из этих трёх средних.
`Precision@5_all` включает все случаи.

LogLoss и ROC-AUC считаются по всем кандидатным парам; один класс даёт неопределённый AUC.
Правило ранжирования не выдаёт вероятности, его LogLoss и AUC неприменимы.
Итоговые метрики — прямое среднее учитываемых случаев, не простое среднее шести месяцев.

Равенства разрешает прежний BLAKE2b-хеш ключа (seed, клиент, дата, актив), затем ISIN.
Он не зависит от позиции строки и будущих ответов.
''')
s=old(9)
s=s.replace('''    result["eligible_user_months"] = int(
        (cases["positive_assets"] > 0).sum()
    )''','''    eligible = cases['positive_assets'] > 0
    result.update(
        candidate_pairs=len(data), positive_pairs=int(data['label'].sum()),
        total_user_months=len(cases), eligible_user_months=int(eligible.sum()),
        eligible_unique_clients=int(cases[eligible].index.get_level_values('customerID').nunique())
    )''')
s=s.replace('result.update({"LogLoss": None, "ROC-AUC": None})',"result.update({'LogLoss': None, 'ROC-AUC': None, 'AUC_note': 'Правило ранжирования не выдаёт вероятности'})")
s=s.replace('''    return result
''','''    result.setdefault('AUC_note', None if data['label'].nunique() > 1 else 'ROC-AUC не определён: один класс')
    return result
''')
code(s)
md('''
## 8. Проверки на искусственных данных до обучения итоговых моделей
Проверяем точную границу окна: событие перед cutoff учитывается; ровно на cutoff или после него
не влияет на признаки. Второй пример проверяет метрики, включая клиента без покупок.
Test-месяцы запрещены функцией построения данных.
''')
code('''
fixture_date = pd.Timestamp('2022-01-01')
fixture_pairs = pd.DataFrame({'customerID':['A'], 'ISIN':['x']})
fixture_history = pd.DataFrame({
    'customerID':['A']*3, 'ISIN':['x']*3,
    'timestamp':[fixture_date-pd.Timedelta(nanoseconds=1), fixture_date,
                 fixture_date+pd.Timedelta(nanoseconds=1)]
})
before_only = generate_features(fixture_history.iloc[:1], fixture_pairs, fixture_date, feature_plan)
with_future = generate_features(fixture_history, fixture_pairs, fixture_date, feature_plan)
pd.testing.assert_frame_equal(before_only, with_future)
assert with_future[auto_names].eq(1).all().all()
assert generate_features(fixture_history.iloc[1:], fixture_pairs, fixture_date, feature_plan)[auto_names].eq(0).all().all()
# Та же граница для календарных ручных окон.
old_history = all_buys
calendar_fixture = fixture_history.assign(transactionType='Buy', customerID=baseline['client_ids'][0], ISIN=baseline['asset_ids'][0])
all_buys = calendar_fixture.iloc[:1].copy()
try: calendar_before = make_dataset_month(fixture_date)
finally: all_buys = old_history
all_buys = calendar_fixture.copy()
try: calendar_after = make_dataset_month(fixture_date)
finally: all_buys = old_history
pd.testing.assert_frame_equal(calendar_before[feature_names], calendar_after[feature_names])
assert calendar_before.iloc[0][manual_features].eq(1).all()

fixture = pd.DataFrame({
    'customerID':['A']*6+['B']*6, 'prediction_date':[fixture_date]*12,
    'ISIN':list('abcdef')*2, 'label':[1,0,1,0,0,1]+[0]*6
})
fixture['_tie'] = np.array([tie_key(c,d,a) for c,d,a in
    fixture[['customerID','prediction_date','ISIN']].itertuples(index=False,name=None)],dtype=np.uint64)
fixture_probabilities = np.tile(np.arange(6,0,-1)/7,2)
fixture_metrics = evaluate(fixture, fixture_probabilities)
assert np.isclose(fixture_metrics['Recall@5'], 2/3)
assert np.isclose(fixture_metrics['Precision@5'], .4)
assert np.isclose(fixture_metrics['Precision@5_all'], .2)
assert np.isclose(fixture_metrics['NDCG@5'], 1.5/(1+1/np.log2(3)+.5))
assert fixture_metrics['eligible_user_months'] == 1
assert evaluate(fixture[fixture['customerID']=='B'], fixture_probabilities[6:])['ROC-AUC'] is None
for test_date in pd.date_range(baseline['test_months'][0], baseline['test_months'][1], freq='MS'):
    try: make_dataset_month(test_date)
    except ValueError: pass
    else: raise AssertionError('Test должен оставаться закрытым')
checks.update(cutoff_boundary_daily=True, cutoff_boundary_calendar=True, metric_fixture_including_zero_buys=True,
              undefined_auc_handling=True, test_months_cannot_be_built=True)
print('Граница времени, метрики на примере и закрытый test проверены.')
''')
md('''
## 9. Контроль со всеми кандидатами и прежняя сетка настроек
Обучаем контроль с 17 признаками при 100/7 и четыре модели отобранного набора.
Все учатся на полном train. Правило выбора: выше Recall, затем ниже LogLoss, затем меньше итераций
и листьев. Контроль с 17 признаками не участвует в выборе победителя. Проверки и диагностики
также не меняют выбор. Вариант отобранного набора 100/7 уже есть в сетке, повторно его не обучаем.
''')
s=old(10)+'''
settings_reproduced = best_key == (previous_auto['parameters']['max_iter'], previous_auto['parameters']['max_leaf_nodes'])
print('Прежние выбранные настройки воспроизведены:', settings_reproduced)
'''
code(s)
md('''
## 10. Ручной контроль 100/7 и полное сравнение
Для парного bootstrap нужны прогнозы ручных пяти признаков при 100/7. Если отдельный проверенный
артефакт этого контроля уже есть, используем его; иначе обучаем ровно один дополнительный контроль,
с прежним seed и `early_stopping=False`, и сохраняем отдельно. Его общие и все месячные метрики
должны совпасть с ручным baseline. Месячные строки ручного контроля показываем из прежнего JSON
только после этой проверки. Выбранная ручная модель загружается без переобучения.
''')
code('''
manual_control_path = project_dir / 'outputs/manual_control_100_7_reviewed_model.joblib'
manual_control_fits = 0
if manual_control_path.exists():
    cached_control = joblib.load(manual_control_path)
    assert cached_control['features'] == manual_features
    for key in ['transactions_sha256','client_ids','asset_ids','train_months','validation_months']:
        assert cached_control['metadata'][key] == baseline[key]
    assert cached_control['model'].max_iter == 100 and cached_control['model'].max_leaf_nodes == 7
    assert cached_control['model'].random_state == seed and not cached_control['model'].early_stopping
    assert cached_control['model'].learning_rate == .1
    manual_control_model = cached_control['model']
    with threadpool_limits(limits=4):
        manual_control_probabilities = manual_control_model.predict_proba(validation[manual_features])[:,1]
    manual_control_source = 'verified_cached_artifact'
else:
    manual_control_model, manual_control_probabilities = fit_version(manual_features, 100, 7)
    manual_control_fits = 1
    manual_control_source = 'retrained_original_control'

with threadpool_limits(limits=4):
    manual_probabilities = manual_saved['model'].predict_proba(validation[manual_saved['features']])[:,1]
metric_names = ['Recall@5','Precision@5','NDCG@5','Precision@5_all','LogLoss','ROC-AUC']
old_control = next(row for row in baseline['comparison'] if row['method_id']=='boost_5_control')
old_selected = next(row for row in baseline['comparison'] if row['method_id']=='boost_selected')
for probabilities, stored in [(manual_control_probabilities,old_control),(manual_probabilities,old_selected)]:
    measured = evaluate(validation, probabilities)
    for metric in metric_names:
        assert np.isclose(measured[metric],stored[metric],rtol=0,atol=1e-12), metric

method_specs = {
    'manual_5_100_7': {'probabilities':manual_control_probabilities},
    'manual_selected': {'probabilities':manual_probabilities},
    'all_17_100_7': {'probabilities':all_features_probabilities},
    'selected_100_7': {'probabilities':predictions[(100,7)]},
    'automatic_selected': {'probabilities':automatic_probabilities},
    'pair_then_popularity': {'criteria':['user_item_buy_count_3m','item_buy_count_1m']}
}
rule_criteria = method_specs['pair_then_popularity']['criteria']
comparison = pd.DataFrame([{'method':name, **evaluate(validation,**spec)}
                           for name,spec in method_specs.items()])
print(comparison[['method']+metric_names].round(5).to_string(index=False,na_rep='—'))
if best_key == (100,7):
    assert automatic_model is models[(100,7)]
    print('selected_100_7 и automatic_selected — один обученный вариант в двух ролях, не два независимых эксперимента.')

monthly_rows = []
for date, month in validation.groupby('prediction_date'):
    for name, spec in method_specs.items():
        month_spec = ({'criteria':spec['criteria']} if 'criteria' in spec else
                      {'probabilities':pd.Series(spec['probabilities'],index=validation.index).reindex(month.index)})
        measured = evaluate(month,**month_spec)
        if name in ['manual_5_100_7','manual_selected']:
            previous_method = 'boost_5_control' if name=='manual_5_100_7' else 'boost_selected'
            stored = next(row for row in baseline['monthly_results']
                          if row['method_id']==previous_method and row['month']==str(date.date()))
            for metric in metric_names:
                assert np.isclose(measured[metric],stored[metric],rtol=0,atol=1e-12), (name,date,metric)
            if name=='manual_5_100_7':
                # Выводим сохранённые метрики после проверки на текущих данных.
                measured.update({metric:stored[metric] for metric in metric_names})
        monthly_rows.append({'month':str(date.date()),'method':name,**measured})
monthly_comparison = pd.DataFrame(monthly_rows)
print('\\nМесячные результаты всех шести методов:')
print(monthly_comparison[['month','method']+metric_names].round(5).to_string(index=False,na_rep='—'))
count_fields = ['candidate_pairs','positive_pairs','total_user_months','eligible_user_months','eligible_unique_clients']
denominator_rows = [{'period':'validation', **{key:int(comparison.iloc[0][key]) for key in count_fields}}]
for date in sorted(validation['prediction_date'].unique()):
    row = monthly_comparison[(monthly_comparison['month']==str(pd.Timestamp(date).date()))].iloc[0]
    denominator_rows.append({'period':str(pd.Timestamp(date).date()), **{key:int(row[key]) for key in count_fields}})
denominators = pd.DataFrame(denominator_rows)
assert denominator_rows == baseline['denominators']
print('\\nОбщие знаменатели для всех методов:')
print(denominators.to_string(index=False))
checks.update(manual_overall_and_monthly_metrics_reproduced=True, denominator_consistency=True)
''')
md('''
## 11. Покрытие фиксированного каталога
Числитель — уникальные покупки клиент–актив–месяц внутри 50 активов; знаменатель — такие же
уникальные покупки выбранных 500 клиентов по всем активам. Повторные сделки пары за месяц
считаются один раз. Считаем покрытие заново и проверяем общие и месячные значения ручного baseline.
''')
code('''
validation_buys = all_buys[(all_buys['customerID'].isin(baseline['client_ids']))
                         & (all_buys['timestamp'] >= '2022-01-01')
                         & (all_buys['timestamp'] < test_start)].copy()
validation_buys['prediction_date'] = validation_buys['timestamp'].dt.to_period('M').dt.to_timestamp()
triples = validation_buys[['customerID','ISIN','prediction_date']].drop_duplicates()
def coverage_row(period, data):
    inside = int(data['ISIN'].isin(baseline['asset_ids']).sum())
    return {'period':period,'catalog_unique_buys':inside,'all_unique_buys':len(data),
            'coverage':inside/len(data) if len(data) else None}
coverage_rows = [coverage_row('validation',triples)]
for date in sorted(validation['prediction_date'].unique()):
    coverage_rows.append(coverage_row(str(pd.Timestamp(date).date()),triples[triples['prediction_date']==date]))
for measured, stored in zip(coverage_rows,baseline['catalog_coverage']['results']):
    for key in ['period','catalog_unique_buys','all_unique_buys']: assert measured[key]==stored[key]
    assert np.isclose(measured['coverage'],stored['coverage'],rtol=0,atol=1e-12)
assert coverage_rows[0]['catalog_unique_buys']==int(validation['label'].sum())
coverage_table = pd.DataFrame(coverage_rows)
print(coverage_table.round(5).to_string(index=False))
checks['catalog_coverage_reproduced'] = True
''')
md('''
## 12. Диагностика новых и повторных покупок
Повторная положительная пара — была покупка за **предыдущие три календарных месяца** строго
до прогноза. Новая — покупки в этом окне не было, это не обязательно первая покупка за всю жизнь.
Используем исходный top-5 каждого метода. Будущие ответы и группы служат только оценке;
кандидаты не фильтруются и рекомендации не перестраиваются.

Для группы Recall = найденные положительные активы группы / все положительные активы группы
внутри каталога, внутри каждого случая; усредняем только случаи с положительной покупкой группы.
Число случаев и знаменатели групп различаются: **нельзя восстановить общий Recall простым
усреднением групповых Recall**. Для популярности добавляем только диагностику, без обучения
и участия в выборе модели. Без отдельных интервалов говорим о наблюдаемых различиях групп.
''')
code('''
diagnostic_specs = {
    'manual_selected':method_specs['manual_selected'],
    'automatic_selected':method_specs['automatic_selected'],
    'pair_then_popularity':method_specs['pair_then_popularity'],
    'popularity':{'criteria':['item_buy_count_1m']}
}
top_lists = {name:top_five(validation,**spec) for name,spec in method_specs.items()}
diagnostic_tops = {name:top_five(validation,**spec) for name,spec in diagnostic_specs.items()}
def group_diagnostic(data, top, period, method):
    rows=[]
    for group,repeated in [('new_3m',False),('repeat_3m',True)]:
        positive = data[(data['label']==1)&((data['user_item_buy_count_3m']>0)==repeated)]
        totals = positive.groupby(case_keys).size()
        hits = top[(top['label']==1)&((top['user_item_buy_count_3m']>0)==repeated)]
        found = hits.groupby(case_keys).size().reindex(totals.index,fill_value=0)
        rows.append({'period':period,'method':method,'group':group,
                     'Recall@5':float((found/totals).mean()),'positive_pairs':len(positive),
                     'eligible_user_months':len(totals),
                     'eligible_unique_clients':int(totals.index.get_level_values('customerID').nunique())})
    return rows
purchase_rows=[]
for name,top in diagnostic_tops.items():
    purchase_rows += group_diagnostic(validation,top,'validation',name)
    for date,month in validation.groupby('prediction_date'):
        purchase_rows += group_diagnostic(month,top[top['prediction_date']==date],str(date.date()),name)
purchase_diagnostics = pd.DataFrame(purchase_rows)
print('Общая диагностика:')
print(purchase_diagnostics[purchase_diagnostics['period']=='validation'].round(5).to_string(index=False,na_rep='—'))
print('\\nМесячная диагностика:')
print(purchase_diagnostics[purchase_diagnostics['period']!='validation'].round(5).to_string(index=False,na_rep='—'))
''')
md('''
## 13. Парный bootstrap: три сравнения на одних выборках
Сохраняем seed=20261013 и 5000 повторов. Из всех 500 клиентов выбираем 500 с возвращением;
каждый выбранный клиент приносит все validation-месяцы и их повторения при кратном выборе.
Делим сумму различий Recall учитываемых случаев на их число, не усредняем средние клиентов.
Одни выборки используются для всех трёх сравнений:
1) выбранная автоматическая минус выбранная ручная;
2) выбранная автоматическая минус правило пары и популярности;
3) отобранный набор 100/7 минус ручные пять 100/7.

Интервалы диагностические: validation использован для выбора настроек и в процессе разработки.
Они не заменяют независимую итоговую проверку. Пустая выборка даёт пропуск, исключается
из процентилей и учитывается отдельным счётчиком.
''')
code('''
def recall_cases(data, probabilities=None, criteria=None):
    return case_scores(data,top_five(data,probabilities,criteria))['Recall@5']
clients = baseline['client_ids']
bootstrap_seed = 20261013
bootstrap_repeats = 5000
assert bootstrap_seed == previous_auto['bootstrap']['seed']
assert bootstrap_repeats == previous_auto['bootstrap']['repeats']
bootstrap_specs = [
    ('automatic_selected - manual_selected',method_specs['automatic_selected'],method_specs['manual_selected']),
    ('automatic_selected - pair_then_popularity',method_specs['automatic_selected'],method_specs['pair_then_popularity']),
    ('selected_100_7 - manual_5_100_7',method_specs['selected_100_7'],method_specs['manual_5_100_7'])
]
reference_cases = recall_cases(validation,**method_specs['automatic_selected'])
eligible = reference_cases.notna()
counts = eligible.groupby(level='customerID').sum().reindex(clients,fill_value=0).to_numpy()
case_deltas=[]
for name,left,right in bootstrap_specs:
    difference = recall_cases(validation,**left)-recall_cases(validation,**right)
    assert difference.notna().equals(eligible)
    case_deltas.append(difference[eligible])
delta_sums = np.column_stack([delta.groupby(level='customerID').sum().reindex(clients,fill_value=0).to_numpy()
                              for delta in case_deltas])
def bootstrap_delta(draw, sums, case_counts):
    denominator = case_counts[draw].sum()
    return sums[draw].sum(axis=0)/denominator if denominator else np.full(sums.shape[1],np.nan)
rng = np.random.default_rng(bootstrap_seed)
draw_differences = np.full((bootstrap_repeats,len(bootstrap_specs)),np.nan)
for iteration in range(bootstrap_repeats):
    draw=rng.integers(0,len(clients),size=len(clients))
    draw_differences[iteration]=bootstrap_delta(draw,delta_sums,counts)
bootstrap_rows=[]
for column,(name,left,right) in enumerate(bootstrap_specs):
    values=draw_differences[:,column]; valid=np.isfinite(values)
    low,high = np.quantile(values[valid],[.025,.975]) if valid.any() else (np.nan,np.nan)
    bootstrap_rows.append({'comparison':name,'difference':float(case_deltas[column].mean()),
                           'ci_low':float(low),'ci_high':float(high),
                           'contains_zero':bool(low<=0<=high) if valid.any() else None,
                           'valid_draws':int(valid.sum()),'empty_draws':int((~valid).sum())})
bootstrap=pd.DataFrame(bootstrap_rows)
print(bootstrap.round(6).to_string(index=False))
''')
md('''
## 14. Итоговые проверки
Проверяем реальные top-5 после перемешивания строк с привязкой прогнозов к ключам и после
изменения будущих ответов при фиксированных прогнозах. Общие средние проверяем через правильные
месячные веса: учитываемые случаи для основных ranking-метрик, все случаи для Precision@5_all,
все пары для LogLoss. AUC так усреднять нельзя, он пересчитывается на общем срезе.
Кратность bootstrap проверяем и на примере, и независимым сбором случаев реальных выборок.
''')
code('''
def top_keys(top):
    return list(top.sort_values(case_keys+['_rank'])[['customerID','prediction_date','ISIN','_rank']]
                .itertuples(index=False,name=None))
shuffled=validation.sample(frac=1,random_state=123)
changed_labels=shuffled.assign(label=1-shuffled['label'])
for name,spec in {**method_specs,'popularity':diagnostic_specs['popularity']}.items():
    other=spec.copy()
    if 'probabilities' in other:
        keyed=pd.Series(other['probabilities'],index=pd.MultiIndex.from_frame(
            validation[['customerID','prediction_date','ISIN']]))
        keys=pd.MultiIndex.from_frame(shuffled[['customerID','prediction_date','ISIN']])
        other['probabilities']=pd.Series(keyed.reindex(keys).to_numpy(),index=shuffled.index)
    original_top=top_five(validation,**spec)
    assert original_top.groupby(case_keys).size().eq(5).all()
    assert not original_top.duplicated(['customerID','prediction_date','ISIN']).any()
    assert top_keys(original_top)==top_keys(top_five(shuffled,**other))
    assert top_keys(original_top)==top_keys(top_five(changed_labels,**other))
for name in method_specs:
    overall=comparison.set_index('method').loc[name]
    monthly=monthly_comparison[monthly_comparison['method']==name]
    for metric in ['Recall@5','Precision@5','NDCG@5']:
        assert np.isclose(overall[metric],np.average(monthly[metric],weights=monthly['eligible_user_months']))
    assert np.isclose(overall['Precision@5_all'],np.average(monthly['Precision@5_all'],weights=monthly['total_user_months']))
    if pd.notna(overall['LogLoss']):
        assert np.isclose(overall['LogLoss'],np.average(monthly['LogLoss'],weights=monthly['candidate_pairs']))
for name in diagnostic_specs:
    whole=purchase_diagnostics[(purchase_diagnostics['period']=='validation')&(purchase_diagnostics['method']==name)]
    assert whole['positive_pairs'].sum()==int(validation['label'].sum())
    for group in ['new_3m','repeat_3m']:
        full=whole[whole['group']==group].iloc[0]
        months=purchase_diagnostics[(purchase_diagnostics['period']!='validation')
                                  &(purchase_diagnostics['method']==name)&(purchase_diagnostics['group']==group)]
        valid=months['eligible_user_months']>0
        if valid.any(): assert np.isclose(full['Recall@5'],np.average(months.loc[valid,'Recall@5'],weights=months.loc[valid,'eligible_user_months']))
        if name=='manual_selected':
            stored=next(row for row in baseline['purchase_diagnostics'] if row['period']=='validation'
                        and row['method_id']=='boost_selected' and row['group']==group)
            assert np.isclose(full['Recall@5'],stored['Recall@5'],rtol=0,atol=1e-12)
# A имеет два случая, B один: выбор A,A,B означает пять случаев.
assert np.allclose(bootstrap_delta(np.array([0,0,1]),np.array([[.6],[.8]]),np.array([2,1])),[.4])
assert np.isnan(bootstrap_delta(np.array([0]),np.array([[0.]]),np.array([0]))).all()
# Независимая проверка первых десяти реальных bootstrap-выборок.
check_rng=np.random.default_rng(bootstrap_seed)
client_values=[{client:delta[delta.index.get_level_values('customerID')==client].to_numpy()
                for client in clients} for delta in case_deltas]
for iteration in range(10):
    draw=check_rng.integers(0,len(clients),size=len(clients))
    for column,values_by_client in enumerate(client_values):
        values=np.concatenate([values_by_client[clients[index]] for index in draw])
        expected=values.mean() if len(values) else np.nan
        assert np.isclose(draw_differences[iteration,column],expected,equal_nan=True)
checks.update(five_recommendations_per_case=True, key_aligned_row_shuffle_invariance=True,
              fixed_predictions_future_label_invariance=True, monthly_overall_weight_consistency=True,
              group_denominator_consistency=True, bootstrap_multiplicity_fixture=True,
              bootstrap_multiplicity_real_draws=True, bootstrap_empty_draw_handling=True)
print('Все проверки пройдены:', ', '.join(checks))
''')
md('''
## 15. Выводы и бюджет
Сравнение 100/7 отделяет добавление и отбор признаков от изменения настроек.
Сравнение выбранных моделей включает и признаки, и настройки. Recall — средняя доля найденных
купленных активов среди учитываемых случаев, не доля правильных рекомендаций.
Precision показывает попадания в пятёрке, NDCG — их позиции, LogLoss — качество вероятностей.

Бюджет учитывает модель отбора, контроль со всеми признаками, четыре настройки и возможное
повторное обучение ручного контроля отдельно. Перестановки требуют дополнительных прогнозов.
Число незадокументированных более ранних экспериментов неизвестно, общий бюджет не равен ручному.
Сравнение с фиксированным каталогом не доказывает качество для остальных активов или прибыльность.
''')
code('''
scores=comparison.set_index('method')
automatic=scores.loc['automatic_selected']; manual=scores.loc['manual_selected']
control_delta=scores.loc['selected_100_7','Recall@5']-scores.loc['manual_5_100_7','Recall@5']
conclusions=[
    f"Создано {len(auto_names)} кандидатов; добавлено {len(selected_new)} к пяти ручным, всего {len(selected_features)} признаков.",
    f"Выбраны настройки {best_key[0]}/{best_key[1]}; прежний отбор совпал: {selection_reproduced}, настройки совпали: {settings_reproduced}.",
    f"Добавление всех 17 при 100/7 меняет Recall на {scores.loc['all_17_100_7','Recall@5']-scores.loc['manual_5_100_7','Recall@5']:+.4f}.",
    f"Отбор при 100/7 меняет Recall относительно всех 17 на {scores.loc['selected_100_7','Recall@5']-scores.loc['all_17_100_7','Recall@5']:+.4f}.",
    f"Отобранный набор против ручных пяти при 100/7: разница Recall {control_delta:+.4f}.",
    f"Выбранные модели: ручной Recall {manual['Recall@5']:.4f}, автоматический {automatic['Recall@5']:.4f}; разница {automatic['Recall@5']-manual['Recall@5']:+.4f}.",
    f"Для выбранных моделей ΔLogLoss={automatic['LogLoss']-manual['LogLoss']:+.5f} (меньше лучше); ΔPrecision={automatic['Precision@5']-manual['Precision@5']:+.4f}; ΔNDCG={automatic['NDCG@5']-manual['NDCG@5']:+.4f}.",
    f"Покрытие каталога: {coverage_rows[0]['coverage']:.2%}, {denominator_rows[0]['eligible_user_months']} учитываемых случаев, {denominator_rows[0]['eligible_unique_clients']} клиентов.",
    "Метрики характеризуют разные стороны прогноза; автоматическую модель нельзя назвать лучшей по всем метрикам.",
    "Сохраняем ручную основу; это автоматическое расширение и эвристический отбор в заданном пространстве, не полный отказ от ручного проектирования."
]
if not selection_reproduced or not settings_reproduced:
    conclusions.append('Результат отличается от прежнего: требуется сверить версии и исходные артефакты; значения не подгоняются.')
for group in ['new_3m','repeat_3m']:
    subset=purchase_diagnostics[(purchase_diagnostics['period']=='validation')&(purchase_diagnostics['group']==group)].set_index('method')
    a=subset.loc['automatic_selected','Recall@5']; m=subset.loc['manual_selected','Recall@5']
    conclusions.append(f"Группа {group}: ручной Recall {m:.4f}, автоматический {a:.4f}, наблюдаемая разница {a-m:+.4f}; отдельный интервал для группы не рассчитан.")
for row in bootstrap_rows:
    note=('интервал включает ноль, устойчивое преимущество не подтверждено' if row['contains_zero']
          else 'интервал не включает ноль, но остаётся диагностическим на использованном validation')
    conclusions.append(f"{row['comparison']}: Δ={row['difference']:+.4f}, 95% CI [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]; {note}.")
fit_budget=6+manual_control_fits
conclusions.append(f"В этом запуске {fit_budget} обучений: 1 отбор + 1 контроль всех признаков + 4 настройки + {manual_control_fits} ручной контроль; перестановки дают {1+len(feature_names)*permutation_repeats} вызова прогнозов дополнительно.")
conclusions.append('Test не оценивался. Исходные ограничения FAR-Trans, реконструкций и полноты ноября сохраняются.')
print('\\n'.join(conclusions))
''')
md('''
## 16. Сохранение
Новые артефакты имеют суффикс `reviewed`, исходные файлы не перезаписываются.
Отдельный ручной контроль сохраняется только при его дополнительном обучении.
JSON содержит определения, все результаты, проверки, бюджет и хеши; пропуски записываются как `null`.
Для воспроизведения запусти **Restart → Run All** из папки проекта или `outputs`, с указанными версиями.
''')
code('''
def clean_json(value):
    if isinstance(value,dict): return {key:clean_json(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [clean_json(item) for item in value]
    if isinstance(value,(float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer): return int(value)
    if isinstance(value,np.bool_): return bool(value)
    return value
permutation_limits=[
    'Correlated manual features and neighboring windows may substitute for one another.',
    'Low importance does not prove absence of useful information.',
    'Positive mean over three shuffles is not a statistical significance test.',
    'Row-wise shuffling breaks within-user-month and within-item-month aggregate constancy.',
    'Selection by LogLoss does not guarantee better Recall or NDCG.'
]
output_dir=project_dir/'outputs'
record={
    'status':'automatic_extension_reviewed_validation_frozen',
    'approach':'automatic expansion of manual baseline and feature selection in a specified search space',
    'candidate_plan':feature_plan,'manual_features':manual_features,'selected_new_features':selected_new,
    'features':selected_features,
    'generation':'Buy counts for user/item/user-item over 7/14/30/60 days strictly before cutoff; calendar manual features retained',
    'feature_selection':{
        'method':'permutation_importance','scoring':'neg_log_loss','repeats':permutation_repeats,'seed':seed,
        'max_new_features':max_new_features,'positive_mean_required':True,
        'tie_rule':'importance descending, feature name ascending',
        'inner_train':['2020-04-01','2021-06-01'],'inner_check':['2021-07-01','2021-12-01'],
        'importance':importance_table.to_dict('records'),'limits':permutation_limits,
        'previous_selection_reproduced':selection_reproduced
    },
    'parameters':{'max_iter':best_key[0],'max_leaf_nodes':best_key[1],
                  'learning_rate':float(automatic_model.learning_rate),'early_stopping':False,'random_state':seed},
    'previous_settings_reproduced':settings_reproduced,
    'same_model_two_roles':best_key==(100,7),
    'comparison':comparison.to_dict('records'),'monthly_results':monthly_comparison.to_dict('records'),
    'denominators':denominator_rows,'tuning':tuning.to_dict('records'),
    'catalog_coverage':{'definition':baseline['catalog_coverage']['definition'],'results':coverage_rows},
    'purchase_diagnostics':purchase_diagnostics.to_dict('records'),
    'repeat_definition':baseline['repeat_definition'],'new_definition':baseline['new_definition'],
    'group_evaluation':'original top-5; recall within eligible group cases; never rerank on labels/groups; do not average group recalls to recover overall recall',
    'bootstrap':{'seed':bootstrap_seed,'repeats':bootstrap_repeats,'unit':'client','pool_size':len(clients),
                 'draw_size':len(clients),'replacement':True,'paired_draws_shared_for_all_comparisons':True,
                 'estimand':'macro Recall difference across eligible user-months with client multiplicity',
                 'interval':'percentiles 2.5% and 97.5%',
                 'empty_draws':'null difference, excluded from percentiles, counted explicitly',
                 'diagnostic_only':True,'validation_used_for_selection_and_development':True,'results':bootstrap_rows},
    'experiment_budget':{
        'boosting_fits_this_run':fit_budget,'selection_model_fits':1,'all_features_control_fits':1,
        'tuning_fits':4,'manual_control_fits':manual_control_fits,'manual_control_source':manual_control_source,
        'candidate_new_features':len(auto_names),'candidate_total_features':len(feature_names),
        'permutation_prediction_calls':1+len(feature_names)*permutation_repeats,
        'inner_quality_prediction_calls':1,'bootstrap_draws':bootstrap_repeats,
        'feature_generation_and_selection':'12 generated candidates, one selector fit, 52 permutation-score predictions',
        'model_tuning':'four selected-set configurations; all-17 and manual control excluded from winner selection',
        'manual_notebook_documented_fits':baseline['experiment_budget']['boosting_fits_in_this_notebook'],
        'previous_automatic_notebook_documented_fits':previous_auto['experiment_budget']['boosting_fits_per_full_run'],
        'earlier_undocumented_experiments':None,'equal_total_cost_with_manual_baseline':False,
        'counting_unit':'fits per full execution; verification reruns repeat the same configurations'
    },
    'source_artifact_sha256':source_artifact_hashes,'source_artifact_paths':{name:str(path) for name,path in artifact_paths.items()},
    'checks':checks,'conclusions':conclusions,'test_evaluated':False,'test_table_built':False,
    'versions':{'python':platform.python_version(),'pandas':pd.__version__,'numpy':np.__version__,
                'scikit_learn':sklearn.__version__,'joblib':joblib.__version__}
}
for key in ['train_months','validation_months','test_months','client_ids','asset_ids','cohort_random_state',
            'cohort_selection_window','metric_definitions','tie_break','observation','limits',
            'transactions_sha256','history_protocol','label_definition','selection_rule']:
    record[key]=baseline[key]
record['limits']=record['limits']+permutation_limits+[
    'Manual five features are retained; generation/search space and selection cap are human specified.',
    'Group differences are observed diagnostics without separate uncertainty intervals.'
]
if manual_control_fits:
    control_metadata={key:baseline[key] for key in
                      ['transactions_sha256','client_ids','asset_ids','train_months','validation_months']}
    control_metadata.update(metrics=evaluate(validation,manual_control_probabilities),
                            reference_results_sha256=source_artifact_hashes['manual_results'])
    joblib.dump({'model':manual_control_model,'features':manual_features,'metadata':control_metadata},manual_control_path)
record['manual_control_artifact']={'path':str(manual_control_path),
                                  'sha256':hashlib.sha256(manual_control_path.read_bytes()).hexdigest(),
                                  'source':manual_control_source}
record=clean_json(record)
joblib.dump({'model':automatic_model,'features':selected_features,'metadata':record},
            output_dir/'automatic_features_reviewed_model.joblib')
(output_dir/'automatic_features_reviewed_results.json').write_text(
    json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print('Сохранены automatic_features_reviewed_model.joblib и automatic_features_reviewed_results.json.')
print('Ручной контроль:',manual_control_path.name,'| Test не оценивался.')
''')
md('''
## Что мы получили
В этом ноутбуке выполнены генерация 12 кандидатов, автоматический отбор в заданных рамках,
обучение и сравнение моделей, месячная и групповая диагностика, bootstrap и проверки.
Фактические численные выводы находятся в разделе 15, метаданные и модели сохранены в разделе 16.
Ручные пять признаков сохраняются. Это автоматическое расширение ручного baseline,
а не полностью автономное проектирование. Test остаётся закрытым для будущей независимой проверки.
''')
notebook={'cells':cells,'metadata':source['metadata'],'nbformat':4,'nbformat_minor':5}
notebook['metadata']['reviewed_execution']={'mode':'sequential fresh Python process','test_evaluated':False}
path=root/'outputs/02_automatic_features_reviewed.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1))
print('Created',path,'cells',len(cells),'code cells',sum(c['cell_type']=='code' for c in cells))
