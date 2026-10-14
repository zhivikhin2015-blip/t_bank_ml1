import contextlib
import io
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits


def recall_at_5(data, predictions):
    keys = ["customerID", "prediction_date"]
    ranked = data[["customerID", "prediction_date", "ISIN", "label"]].copy()
    ranked["score"] = predictions
    ranked["tie_break"] = np.random.default_rng(42).random(len(ranked))
    ranked = ranked.sort_values(
        ["customerID", "prediction_date", "score", "tie_break"],
        ascending=[True, True, False, True]
    )
    top5 = ranked.groupby(keys).head(5)
    total_bought = data.groupby(keys)["label"].sum()
    found_bought = top5.groupby(keys)["label"].sum()
    has_purchases = total_bought > 0
    return float((found_bought[has_purchases] / total_bought[has_purchases]).mean())


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()):
        scope = runpy.run_path('work/popularity_feature_check.py', run_name='__main__')
    train = scope['train_v2']
    validation = scope['validation_v2']
    predictions_v2 = scope['predictions_v2']
    features = ['user_buy_count_1m', 'user_item_buy_count_1m']
    model = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, early_stopping=False, random_state=42
    )
    with threadpool_limits(limits=4):
        model.fit(train[features], train.label)
        model_predictions = model.predict_proba(validation[features])[:, 1]
    rates = train.groupby(train.user_item_buy_count_1m > 0).label.mean()
    repeat_predictions = (validation.user_item_buy_count_1m > 0).map(rates).fillna(train.label.mean()).to_numpy()
    names = ['constant', 'repeat_rule', 'two_features', 'three_features']
    scores = [
        np.full(len(validation), train.label.mean()),
        repeat_predictions, model_predictions, predictions_v2
    ]
    results = {name: recall_at_5(validation, p) for name, p in zip(names, scores)}
    totals = validation.groupby(['customerID', 'prediction_date']).label.sum()
    results['evaluated_user_months'] = int((totals > 0).sum())
    results['all_user_months'] = len(totals)
    results['candidate_count'] = 50
    results['k'] = 5
    results['averaging'] = 'macro over user-months with at least one in-catalog buy'
    results['tie_break'] = 'seeded random tie-break, seed 42'
    results['test_evaluated'] = False

    # Hand-computed two-client example: A has recall 2/3, B has recall 1.
    fixture = pd.DataFrame({
        'customerID': ['A'] * 6 + ['B'] * 6,
        'prediction_date': ['2022-01-01'] * 12,
        'ISIN': list('abcdef') * 2,
        'label': [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    })
    assert abs(recall_at_5(fixture, list(range(6, 0, -1)) * 2) - 5/6) < 1e-12
    fixture['label'] = fixture['label'].where(fixture['customerID'] == 'A', 0)
    assert abs(recall_at_5(fixture, list(range(6, 0, -1)) * 2) - 2/3) < 1e-12

    Path('work/ranking_result.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))
