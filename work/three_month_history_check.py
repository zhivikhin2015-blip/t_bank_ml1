import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits


def make_month_dataset_v3(prediction_date):
    result = make_month_dataset_v2(prediction_date)
    cutoff = pd.Timestamp(prediction_date)
    start = cutoff - pd.DateOffset(months=3)

    past_3m = large_buys[
        (large_buys["timestamp"] >= start)
        & (large_buys["timestamp"] < cutoff)
    ]
    user_counts_3m = past_3m.groupby("customerID").size().reset_index(
        name="user_buy_count_3m"
    )
    pair_counts_3m = past_3m.groupby(["customerID", "ISIN"]).size().reset_index(
        name="user_item_buy_count_3m"
    )
    result = result.merge(
        user_counts_3m, on="customerID", how="left", validate="many_to_one"
    )
    result = result.merge(
        pair_counts_3m, on=["customerID", "ISIN"], how="left", validate="one_to_one"
    )
    new_columns = ["user_buy_count_3m", "user_item_buy_count_3m"]
    result[new_columns] = result[new_columns].fillna(0).astype(int)
    return result


if __name__ == '__main__':
    transactions = pd.read_csv(
        'work/data/FAR-Trans/transactions.csv',
        dtype={'customerID': 'string', 'ISIN': 'string'}, parse_dates=['timestamp']
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
    base = runpy.run_path('work/month_dataset_check.py', init_globals={
        'transactions': transactions, 'large_buys': large_buys,
        'large_client_ids': large_client_ids, 'large_asset_ids': large_asset_ids
    })
    popularity = runpy.run_path('work/popularity_feature_check.py', init_globals={
        'transactions': transactions, 'make_month_dataset': base['make_month_dataset']
    })
    make_month_dataset_v2 = popularity['make_month_dataset_v2']
    recall_at_5 = runpy.run_path('work/ranking_check.py')['recall_at_5']

    april_v3 = make_month_dataset_v3('2020-04-01')
    april_v2 = make_month_dataset_v2('2020-04-01')
    assert april_v3[april_v2.columns].equals(april_v2)
    assert len(april_v3) == 25000
    assert not april_v3.isna().any().any()
    assert (april_v3.user_buy_count_3m >= april_v3.user_buy_count_1m).all()
    assert (april_v3.user_item_buy_count_3m >= april_v3.user_item_buy_count_1m).all()
    for row in april_v3.iloc[[0, 1, 499]].itertuples():
        direct = large_buys[
            (large_buys.customerID == row.customerID)
            & (large_buys.timestamp >= '2020-01-01')
            & (large_buys.timestamp < '2020-04-01')
        ]
        assert row.user_buy_count_3m == len(direct)
        assert row.user_item_buy_count_3m == len(direct[direct.ISIN == row.ISIN])
    print(april_v3[[
        'ISIN', 'user_buy_count_1m', 'user_buy_count_3m',
        'user_item_buy_count_1m', 'user_item_buy_count_3m'
    ]].head().to_string(index=False), flush=True)

    development_v3 = pd.concat([
        make_month_dataset_v3(date)
        for date in pd.date_range('2020-04-01','2022-06-01',freq='MS')
    ], ignore_index=True)
    train_v3 = development_v3[development_v3.prediction_date < '2022-01-01'].copy()
    validation_v3 = development_v3[development_v3.prediction_date >= '2022-01-01'].copy()
    features_v3 = [
        'user_buy_count_1m', 'user_item_buy_count_1m', 'item_buy_count_1m',
        'user_buy_count_3m', 'user_item_buy_count_3m'
    ]
    model_v3 = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=7, early_stopping=False, random_state=42
    )
    with threadpool_limits(limits=4):
        model_v3.fit(train_v3[features_v3], train_v3.label)
        predictions_v3 = model_v3.predict_proba(validation_v3[features_v3])[:, 1]
    results = {
        'feature_columns': features_v3,
        'train_rows': len(train_v3),
        'validation_rows': len(validation_v3),
        'validation_logloss': log_loss(validation_v3.label, predictions_v3),
        'validation_recall_at_5': recall_at_5(validation_v3, predictions_v3),
        'test_evaluated': False
    }
    Path('work/three_month_history_result.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results, indent=2))
