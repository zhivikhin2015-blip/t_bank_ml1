import pandas as pd


def make_month_dataset(prediction_date):
    cutoff = pd.Timestamp(prediction_date)
    start = cutoff - pd.DateOffset(months=1)
    end = cutoff + pd.DateOffset(months=1)

    if not cutoff.is_month_start or cutoff != cutoff.normalize():
        raise ValueError("Дата прогноза должна быть первым числом месяца")
    if end > transactions["timestamp"].max().normalize() + pd.Timedelta(days=1):
        raise ValueError("В данных ещё нет полного месяца для ответа")

    past = large_buys[
        (large_buys["timestamp"] >= start)
        & (large_buys["timestamp"] < cutoff)
    ]
    future = large_buys[
        (large_buys["timestamp"] >= cutoff)
        & (large_buys["timestamp"] < end)
    ]

    user_counts = past.groupby("customerID").size().reset_index(
        name="user_buy_count_1m"
    )
    pair_counts = past.groupby(["customerID", "ISIN"]).size().reset_index(
        name="user_item_buy_count_1m"
    )

    result = pd.DataFrame({"customerID": large_client_ids}).merge(
        pd.DataFrame({"ISIN": large_asset_ids}), how="cross"
    )
    result = result.merge(
        user_counts, on="customerID", how="left", validate="many_to_one"
    )
    result = result.merge(
        pair_counts, on=["customerID", "ISIN"], how="left",
        validate="one_to_one"
    )

    labels = future[["customerID", "ISIN"]].drop_duplicates().copy()
    labels["label"] = 1
    result = result.merge(
        labels, on=["customerID", "ISIN"], how="left", validate="one_to_one"
    )

    columns = ["user_buy_count_1m", "user_item_buy_count_1m", "label"]
    result[columns] = result[columns].fillna(0).astype(int)
    result["prediction_date"] = cutoff
    return result


if __name__ == "__main__":
    transactions = pd.read_csv(
        "work/data/FAR-Trans/transactions.csv",
        dtype={"customerID": "string", "ISIN": "string"},
        parse_dates=["timestamp"],
    )
    selection_buys = transactions[
        (transactions["transactionType"] == "Buy")
        & (transactions["timestamp"] >= "2019-10-01")
        & (transactions["timestamp"] < "2020-04-01")
    ]
    large_client_ids = selection_buys["customerID"].drop_duplicates().sample(
        n=500, random_state=42
    ).tolist()
    large_asset_ids = selection_buys["ISIN"].value_counts().head(50).index.tolist()
    large_buys = transactions[
        (transactions["transactionType"] == "Buy")
        & transactions["customerID"].isin(large_client_ids)
    ].copy()

    # Test the entire 32-month range without retaining all frames in memory.
    for cutoff in pd.date_range("2020-04-01", "2022-11-01", freq="MS"):
        result = make_month_dataset(cutoff)
        assert len(result) == 25000
        assert not result.duplicated(["customerID", "ISIN"]).any()
        assert not result.isna().any().any()
        assert result["label"].isin([0, 1]).all()
        assert (result["user_item_buy_count_1m"] <= result["user_buy_count_1m"]).all()
        assert result.groupby("customerID")["user_buy_count_1m"].nunique().eq(1).all()

        # Compare selected pairs with direct filtering rather than grouping/joins.
        for row in result.iloc[[0, 1, 51, 1000, 24999]].itertuples():
            history = large_buys[
                (large_buys["customerID"] == row.customerID)
                & (large_buys["timestamp"] >= cutoff - pd.DateOffset(months=1))
                & (large_buys["timestamp"] < cutoff)
            ]
            truth = large_buys[
                (large_buys["customerID"] == row.customerID)
                & (large_buys["ISIN"] == row.ISIN)
                & (large_buys["timestamp"] >= cutoff)
                & (large_buys["timestamp"] < cutoff + pd.DateOffset(months=1))
            ]
            assert row.user_buy_count_1m == len(history)
            assert row.user_item_buy_count_1m == len(history[history["ISIN"] == row.ISIN])
            assert row.label == int(len(truth) > 0)

        if cutoff <= pd.Timestamp("2020-06-01"):
            print(cutoff.date(), len(result), "positive pairs", result["label"].sum())
        if cutoff == pd.Timestamp("2020-04-01"):
            assert result["label"].sum() == 323

    for invalid in ["2020-04-02", "2022-12-01", "2020-04-01 12:00"]:
        try:
            make_month_dataset(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted invalid date: {invalid}")
    print("Checked 32 months, direct-filter comparisons and invalid-date guards")
