import json
import runpy
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits


def make_month_dataset_v2(prediction_date):
    result = make_month_dataset(prediction_date)
    cutoff = pd.Timestamp(prediction_date)
    start = cutoff - pd.DateOffset(months=1)

    # Popularity comes from all customers, not just the selected 500.
    past_all_buys = transactions[
        (transactions["transactionType"] == "Buy")
        & (transactions["timestamp"] >= start)
        & (transactions["timestamp"] < cutoff)
    ]
    item_counts = past_all_buys.groupby("ISIN").size().reset_index(
        name="item_buy_count_1m"
    )
    result = result.merge(
        item_counts, on="ISIN", how="left", validate="many_to_one"
    )
    result["item_buy_count_1m"] = result["item_buy_count_1m"].fillna(0).astype(int)
    return result


if __name__ == "__main__":
    transactions = pd.read_csv(
        'work/data/FAR-Trans/transactions.csv',
        dtype={'customerID': 'string', 'ISIN': 'string'},
        parse_dates=['timestamp'],
    )
    selection_buys = transactions[
        (transactions.transactionType == 'Buy')
        & (transactions.timestamp >= '2019-10-01')
        & (transactions.timestamp < '2020-04-01')
    ]
    large_client_ids = selection_buys.customerID.drop_duplicates().sample(
        n=500, random_state=42
    ).tolist()
    large_asset_ids = selection_buys.ISIN.value_counts().head(50).index.tolist()
    large_buys = transactions[
        (transactions.transactionType == 'Buy')
        & transactions.customerID.isin(large_client_ids)
    ].copy()
    scope = runpy.run_path('work/month_dataset_check.py', init_globals={
        'transactions': transactions,
        'large_buys': large_buys,
        'large_client_ids': large_client_ids,
        'large_asset_ids': large_asset_ids,
    })
    make_month_dataset = scope['make_month_dataset']

    april_v2 = make_month_dataset_v2('2020-04-01')
    april_original = make_month_dataset('2020-04-01')
    assert len(april_v2) == 25000
    assert april_v2[april_original.columns].equals(april_original)
    assert april_v2.groupby('ISIN').item_buy_count_1m.nunique().eq(1).all()
    for row in april_v2.iloc[[0, 1, 499]].itertuples():
        direct = transactions[
            (transactions.transactionType == 'Buy')
            & (transactions.timestamp >= '2020-03-01')
            & (transactions.timestamp < '2020-04-01')
            & (transactions.ISIN == row.ISIN)
        ]
        assert row.item_buy_count_1m == len(direct)
    print(april_v2[['ISIN', 'user_item_buy_count_1m', 'item_buy_count_1m']].head().to_string(index=False))

    # Perturb future rows and confirm features are unchanged.
    original_transactions = transactions
    transactions = pd.concat([
        transactions,
        transactions[transactions.timestamp >= '2020-04-01'],
    ], ignore_index=True)
    perturbed = make_month_dataset_v2('2020-04-01')
    assert perturbed['item_buy_count_1m'].equals(april_v2['item_buy_count_1m'])
    transactions = original_transactions

    development = pd.concat([
        make_month_dataset_v2(date)
        for date in pd.date_range('2020-04-01', '2022-06-01', freq='MS')
    ], ignore_index=True)
    train_v2 = development[development.prediction_date < '2022-01-01']
    validation_v2 = development[development.prediction_date >= '2022-01-01']
    features_v2 = [
        'user_buy_count_1m', 'user_item_buy_count_1m', 'item_buy_count_1m'
    ]
    model_v2 = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, early_stopping=False, random_state=42
    )
    with threadpool_limits(limits=4):
        model_v2.fit(train_v2[features_v2], train_v2.label)
        predictions_v2 = model_v2.predict_proba(validation_v2[features_v2])[:, 1]
    score_v2 = log_loss(validation_v2.label, predictions_v2)
    previous = json.loads(Path('work/first_boosting_result.json').read_text())
    results = {
        'features': features_v2,
        'validation_logloss': score_v2,
        'previous_validation_logloss': previous['model_validation_logloss'],
        'train_rows': len(train_v2),
        'validation_rows': len(validation_v2),
        'test_evaluated': False,
        'future_perturbation_check_passed': True,
    }
    Path('work/popularity_feature_result.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results, indent=2))
