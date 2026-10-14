from pathlib import Path
import json
steps=[]
def step(title,explanation,code):
    steps.append({'title':title,'explanation':explanation,'code':code.strip()+'\n'})
step('1. Подготовить внутреннюю проверку для отбора',
'''Весь train уже собран. Его раннюю часть используем для обучения вспомогательной модели, а последние шесть месяцев — для оценки признаков. Основной validation на этом этапе не используем. Файл automatic_feature_metrics.py содержит те же определения метрик и порядок равенств, что ручной baseline; выносим их в общий файл, чтобы не переписывать большой блок.''',r'''
import os
import sys
import hashlib
import platform
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, str(project_dir / "outputs"))
from automatic_feature_metrics import prepare_ranking, evaluate, paired_bootstrap

seed = 42
max_new_features = 5
permutation_repeats = 3
inner_train = train[train["prediction_date"] < "2021-07-01"].copy()
inner_check = train[train["prediction_date"] >= "2021-07-01"].copy()
assert inner_train["prediction_date"].max() + pd.DateOffset(months=1) <= inner_check["prediction_date"].min()
assert inner_check["prediction_date"].max() + pd.DateOffset(months=1) <= validation["prediction_date"].min()
validation = prepare_ranking(validation)
print("Внутреннее обучение:", len(inner_train), "| Отбор:", len(inner_check))
''')
step('2. Автоматически отобрать новые признаки',
'''Обучаем вспомогательный бустинг на всех 17 признаках. Затем на inner_check перемешиваем значения одного столбца и смотрим, насколько вырос LogLoss. Если ошибка выросла, модель использовала этот столбец. Перемешивание повторяем три раза. Среди 12 новых кандидатов автоматически сохраняем до пяти с положительной средней важностью; пять ручных остаются. Порог, число повторов и лимит заданы до просмотра результатов. Важность относится к конкретной модели, а не доказывает универсальную полезность. Сходные окна могут заменять друг друга.''',r'''
selector_model = HistGradientBoostingClassifier(
    max_iter=100, max_leaf_nodes=7,
    early_stopping=False, random_state=seed
)
with threadpool_limits(limits=4):
    selector_model.fit(inner_train[feature_names], inner_train["label"])
    inner_probabilities = selector_model.predict_proba(inner_check[feature_names])[:, 1]
    importance = permutation_importance(
        selector_model, inner_check[feature_names], inner_check["label"],
        scoring="neg_log_loss", n_repeats=permutation_repeats,
        random_state=seed, n_jobs=1
    )

print("Внутренний LogLoss модели:", log_loss(inner_check["label"], inner_probabilities))
print("Внутренний LogLoss постоянной вероятности:", log_loss(
    inner_check["label"], np.full(len(inner_check), inner_train["label"].mean())))
importance_table = pd.DataFrame({
    "feature": feature_names,
    "importance": importance.importances_mean,
    "shuffle_std": importance.importances_std
})
new_importance = importance_table[importance_table["feature"].isin(auto_names)].sort_values(
    ["importance", "feature"], ascending=[False, True]
)
selected_new = new_importance[new_importance["importance"] > 0].head(max_new_features)["feature"].tolist()
selected_features = manual_features + selected_new
print(new_importance.round(6).to_string(index=False))
print("Выбраны новые:", selected_new)
print("Всего оставлено:", len(selected_features))
''')
step('3. Обучить контроль и выбрать настройки',
'''Контроль с 17 признаками при 100/7 показывает пользу добавления всех кандидатов. Для отобранного набора проверяем прежние четыре настройки: 100/7, 100/15, 200/7, 200/15. Каждая модель учится на полном train и остаётся фиксированной на validation. Выбор: выше Recall@5, затем ниже LogLoss, затем меньше итераций и листьев. Контроль с 17 признаками не участвует в выборе победителя: он отдельная диагностика добавления и отбора.''',r'''
def fit_version(columns, trees=100, leaves=7):
    model = HistGradientBoostingClassifier(
        max_iter=trees, max_leaf_nodes=leaves,
        early_stopping=False, random_state=seed
    )
    with threadpool_limits(limits=4):
        model.fit(train[columns], train["label"])
        probabilities = model.predict_proba(validation[columns])[:, 1]
    return model, probabilities

all_features_model, all_features_probabilities = fit_version(feature_names)
settings = [(100, 7), (100, 15), (200, 7), (200, 15)]
models, predictions, tuning_rows = {}, {}, []
for trees, leaves in settings:
    model, probabilities = fit_version(selected_features, trees, leaves)
    models[(trees, leaves)] = model
    predictions[(trees, leaves)] = probabilities
    tuning_rows.append({"max_iter": trees, "max_leaf_nodes": leaves,
                        **evaluate(validation, probabilities)})

tuning = pd.DataFrame(tuning_rows).sort_values(
    ["Recall@5", "LogLoss", "max_iter", "max_leaf_nodes"],
    ascending=[False, True, True, True]
).reset_index(drop=True)
best_key = (int(tuning.iloc[0]["max_iter"]), int(tuning.iloc[0]["max_leaf_nodes"]))
automatic_model = models[best_key]
automatic_probabilities = predictions[best_key]
print(tuning.round(5).to_string(index=False))
print("Выбрано:", best_key)
''')
step('4. Сравнить с ручным baseline и простым правилом',
'''Загружаем свою ранее сохранённую модель, не обучая её заново. Сравнение разделяет добавление кандидатов, их отбор и настройку модели. Простейшее сильное правило ранжирует по покупкам пары за три месяца, затем по популярности. Все методы оцениваются по тем же парам, ответам и равенствам. Основные ranking-метрики включают только клиент-месяцы с покупкой внутри каталога; Precision@5_all — все случаи. Правило не выдаёт вероятности, поэтому LogLoss и AUC для него пустые.''',r'''
manual_saved = joblib.load(project_dir / "outputs/manual_baseline_reviewed_model.joblib")
with threadpool_limits(limits=4):
    manual_probabilities = manual_saved["model"].predict_proba(
        validation[manual_saved["features"]])[:, 1]
manual_metrics = evaluate(validation, manual_probabilities)
old_selected = next(row for row in baseline["comparison"] if row["method_id"] == "boost_selected")
for metric in ["Recall@5", "LogLoss", "ROC-AUC"]:
    assert np.isclose(manual_metrics[metric], old_selected[metric], rtol=0, atol=1e-12)

old_control = next(row for row in baseline["comparison"] if row["method_id"] == "boost_5_control")
rule_criteria = ["user_item_buy_count_3m", "item_buy_count_1m"]
method_specs = {
    "manual_selected": {"probabilities": manual_probabilities},
    "all_17_100_7": {"probabilities": all_features_probabilities},
    "selected_100_7": {"probabilities": predictions[(100, 7)]},
    "automatic_selected": {"probabilities": automatic_probabilities},
    "pair_then_popularity": {"criteria": rule_criteria}
}
metric_names = ["Recall@5", "Precision@5", "NDCG@5", "Precision@5_all", "LogLoss", "ROC-AUC"]
comparison_rows = [{"method": "manual_5_100_7", **{name: old_control[name] for name in metric_names}}]
comparison_rows += [{"method": name, **evaluate(validation, **spec)} for name, spec in method_specs.items()]
comparison = pd.DataFrame(comparison_rows)
print(comparison[["method"] + metric_names].round(5).to_string(index=False, na_rep="—"))
monthly_rows = []
for date, month in validation.groupby("prediction_date"):
    for name, spec in method_specs.items():
        month_spec = ({"criteria": spec["criteria"]} if "criteria" in spec else
                      {"probabilities": pd.Series(spec["probabilities"], index=validation.index).reindex(month.index)})
        monthly_rows.append({"month": str(date.date()), "method": name, **evaluate(month, **month_spec)})
monthly_comparison = pd.DataFrame(monthly_rows)
print("Для основных метрик:", manual_metrics["eligible_user_months"],
      "случаев и", manual_metrics["eligible_unique_clients"], "клиентов")
''')
step('5. Проверить устойчивость разницы',
'''Повторяем парный bootstrap по всем 500 клиентам: 5000 выборок с возвращением, все месяцы клиента и кратность выбора сохраняются. Сравниваем выбранную автоматическую модель с ручной моделью и простым правилом. Это диагностика, не независимая проверка: настройки выбирали на том же validation. Интервал, содержащий ноль, не подтверждает устойчивое преимущество.''',r'''
bootstrap_seed = 20261013
bootstrap_repeats = 5000
bootstrap = paired_bootstrap(
    validation, automatic_probabilities,
    references={
        "manual_selected": {"probabilities": manual_probabilities},
        "pair_then_popularity": {"criteria": rule_criteria}
    },
    client_ids=baseline["client_ids"], repeats=bootstrap_repeats, seed=bootstrap_seed
)
print(bootstrap.round(5).to_string(index=False))
''')
step('6. Сформулировать результат без подгонки',
'''Числа берём из таблиц. Программа может проиграть ручной модели: это честный результат эксперимента. Генерация и отбор при этом работают. Не меняем окна, лимит или критерий отбора задним числом, чтобы получить красивый результат. Сравнение всех кандидатов и отобранного набора при одинаковых настройках показывает, полезен ли именно отбор.''',r'''
scores = comparison.set_index("method")
delta = scores.loc["automatic_selected", "Recall@5"] - scores.loc["manual_selected", "Recall@5"]
conclusions = [
    f"Создано {len(auto_names)} кандидатов, автоматически добавлено {len(selected_new)} к пяти ручным признакам.",
    f"Выбраны настройки {best_key[0]}/{best_key[1]}.",
    f"Recall@5 ручной модели: {scores.loc['manual_selected', 'Recall@5']:.4f}; автоматической: {scores.loc['automatic_selected', 'Recall@5']:.4f}; разница {delta:+.4f}.",
    f"Добавление всех кандидатов при 100/7 меняет Recall на {scores.loc['all_17_100_7','Recall@5']-scores.loc['manual_5_100_7','Recall@5']:+.4f}.",
    f"Отбор при 100/7 меняет Recall относительно всех 17 признаков на {scores.loc['selected_100_7','Recall@5']-scores.loc['all_17_100_7','Recall@5']:+.4f}.",
    "Это ограниченный генератор покупочных счётчиков. Человек задаёт типы агрегатов, окна и правила отбора; лучшего возможного набора не гарантируем.",
    "Сходные признаки могут заменять друг друга. Отбор по LogLoss не гарантирует улучшения Recall.",
    "Validation использован для выбора настроек. Bootstrap диагностический. Test не оценён."
]
for row in bootstrap.to_dict("records"):
    note = ("интервал включает ноль; устойчивое преимущество не подтверждено"
            if row["contains_zero"] else "интервал не включает ноль; всё равно нужна независимая итоговая проверка")
    conclusions.append(f"Разница с {row['reference']}: [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]; {note}.")
print("\n".join(conclusions))
''')
step('7. Сохранить метод и результаты',
'''Сохраняем модель, кандидатов, выбранные признаки, все оценки и правила. Ручные артефакты не меняются. Этот запуск содержит 6 новых обучений: вспомогательный отбор, контроль всех 17 признаков и 4 настройки выбранного набора. Но это не равная общая стоимость с ручным подходом: перестановки требуют дополнительных прогнозов. Пять ручных признаков мы сохраняем, поэтому это автоматическое расширение ручной базы, а не полностью автономный поиск. Test остаётся отдельным будущим этапом.''',r'''
def clean_json(value):
    if isinstance(value, dict): return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value

record = {
    "status": "automatic_extension_validation_frozen",
    "candidate_plan": feature_plan, "manual_features": manual_features,
    "selected_new_features": selected_new, "features": selected_features,
    "generation": "Buy counts for user/item/user-item; preceding 7/14/30/60 days; strictly before cutoff",
    "selection": {"method": "permutation_importance", "scoring": "neg_log_loss",
        "inner_train_months": ["2020-04-01", "2021-06-01"],
        "inner_check_months": ["2021-07-01", "2021-12-01"],
        "max_new_features": max_new_features, "positive_mean_required": True,
        "tie_rule": "importance descending, feature name ascending",
        "repeats": permutation_repeats, "seed": seed,
        "importance": importance_table.to_dict("records")},
    "parameters": {"max_iter": best_key[0], "max_leaf_nodes": best_key[1],
        "learning_rate": float(automatic_model.learning_rate), "early_stopping": False, "random_state": seed},
    "selection_rule": baseline["selection_rule"],
    "comparison": comparison.to_dict("records"), "monthly_results": monthly_comparison.to_dict("records"),
    "tuning": tuning.to_dict("records"), "conclusions": conclusions,
    "bootstrap": {"seed": bootstrap_seed, "repeats": bootstrap_repeats,
        "unit": "client", "pool_size": 500, "draw_size": 500, "replacement": True,
        "paired_draws": True, "macro_over_eligible_user_months": True,
        "empty_draws_excluded_and_counted": True, "diagnostic_only": True,
        "results": bootstrap.to_dict("records")},
    "budget": {"boosting_fits_per_full_run": 6, "candidate_new_features": len(auto_names),
        "selection_model_fits": 1, "all_features_control_fits": 1, "tuning_model_fits": 4,
        "permutation_prediction_calls": 1 + len(feature_names)*permutation_repeats,
        "inner_quality_prediction_calls": 1,
        "feature_generation_selection_budget": "12 candidates, 1 fit, permutation diagnostics; elapsed cost not equated with manual approach",
        "model_tuning_budget": "4 selected-set settings; all-17 control excluded from winner selection",
        "manual_boosting_fits_in_reviewed_notebook": 6, "earlier_experiments": None},
    "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                 "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
    "metrics_helper_sha256": hashlib.sha256((project_dir/"outputs/automatic_feature_metrics.py").read_bytes()).hexdigest(),
    "test_evaluated": False
}
for key in ["train_months", "validation_months", "test_months", "client_ids", "asset_ids",
            "cohort_random_state", "cohort_selection_window", "metric_definitions", "tie_break",
            "observation", "limits", "catalog_coverage", "transactions_sha256", "history_protocol", "label_definition"]:
    record[key] = baseline[key]
assert hashlib.sha256((project_dir/"work/data/FAR-Trans/transactions.csv").read_bytes()).hexdigest() == record["transactions_sha256"]
record = clean_json(record)
output_dir = project_dir / "outputs"
joblib.dump({"model": automatic_model, "features": selected_features, "metadata": record},
            output_dir / "automatic_features_model.joblib")
(output_dir / "automatic_features_results.json").write_text(
    json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
print("Сохранены automatic_features_model.joblib и automatic_features_results.json")
''')
root=Path.cwd()
(root/'work/automatic_finish_steps.json').write_text(json.dumps(steps,ensure_ascii=False,indent=2))
text='# Завершение автоматической генерации и отбора признаков\n\nДобавляй блоки после построения train и validation. Каждый блок — новая кодовая ячейка; выполняй сверху вниз. Исходный ноутбук этим документом не изменяется.\n\n'
for step in steps:
    text += f"## {step['title']}\n\n{step['explanation']}\n\n```python\n{step['code']}```\n\n"
text += 'Метод перестановки описан в документации: https://scikit-learn.org/stable/modules/permutation_importance.html\n'
(root/'outputs/automatic_features_finish_steps.md').write_text(text)
print('Saved guide with',len(steps),'steps')
