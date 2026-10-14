"""Независимая проверка реальных предсказаний и признаков, без обучения."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .features import Builder, Feature, load_data, select_cohort, sha256
from .metrics import Ranking


def metric_oracle(frame, scores, secondary=None):
    data = frame.copy()
    data["score"] = scores
    data["secondary"] = np.zeros(len(data)) if secondary is None else secondary
    tie = []
    for row in data.itertuples():
        payload = json.dumps([42, row.customerID, str(row.prediction_date.date()), row.ISIN],
                             ensure_ascii=False, separators=(",", ":")).encode()
        tie.append(int.from_bytes(hashlib.blake2b(payload, digest_size=8, person=b"FAR-tie").digest(), "big"))
    data["tie"] = tie
    recalls, ndcgs, precisions, all_precisions = [], [], [], []
    for _, group in data.groupby(["customerID", "prediction_date"], sort=True):
        ranked = group.sort_values(["score", "secondary", "tie", "ISIN"], ascending=[False, False, True, True])
        positive = set(group.loc[group.label == 1, "ISIN"])
        top = ranked.head(5).ISIN.tolist()
        hits = len(positive.intersection(top))
        all_precisions.append(hits / 5)
        if positive:
            recalls.append(hits / len(positive))
            precisions.append(hits / 5)
            dcg = sum(1 / np.log2(j + 2) for j, item in enumerate(top) if item in positive)
            ideal = sum(1 / np.log2(j + 2) for j in range(min(len(positive), 5)))
            ndcgs.append(dcg / ideal)
    return {"Recall@5": float(np.mean(recalls)), "NDCG@5": float(np.mean(ndcgs)),
            "Precision@5": float(np.mean(precisions)), "Precision@5_all": float(np.mean(all_precisions))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    from . import test_pipeline
    suite = unittest.defaultTestLoader.loadTestsFromModule(test_pipeline)
    test = unittest.TextTestRunner(verbosity=1).run(suite)
    assert test.wasSuccessful()
    protocol = json.loads((args.out / "protocol.json").read_text())
    evaluation = json.loads((args.out / "evaluation.json").read_text())
    for file, expected in protocol["source_hashes"].items():
        assert sha256(args.data_dir / file) == expected
    for file, expected in protocol["source_code_hashes"].items():
        assert sha256(Path(__file__).parent / file) == expected
    for file, expected in protocol.get("manual_source_code_hashes", {}).items():
        original = args.out / "provenance" / file
        candidate = original if original.exists() else Path(__file__).parent / file
        assert sha256(candidate) == expected
    data, metadata, conflicts = load_data(args.data_dir)
    users, items = select_cohort(data)
    builder = Builder(data, metadata, users, items)
    dates = list(pd.date_range("2022-07-01", "2022-10-01", freq="MS"))
    frame, _ = builder.labels(dates)
    saved = pd.read_csv(args.out / "predictions.csv.gz", dtype={"customerID": str, "ISIN": str},
                        float_precision="round_trip", parse_dates=["prediction_date"])
    pd.testing.assert_frame_equal(frame[["customerID", "ISIN", "prediction_date", "label"]],
        saved[["customerID", "ISIN", "prediction_date", "label"]], check_dtype=False)
    oracle_errors, predict_errors = {}, {}
    names = list(evaluation["metrics"])
    with threadpool_limits(limits=4):
        for name in names:
            secondary = saved.pair_popularity_secondary.to_numpy() if name == "pair_popularity" else None
            oracle = metric_oracle(frame, saved[name].to_numpy(), secondary)
            errors = {k: abs(v - evaluation["metrics"][name][k]) for k, v in oracle.items()}
            assert max(errors.values()) < 1e-12
            oracle_errors[name] = errors
            if name not in ("pair_popularity", "popularity"):
                y = frame.label.to_numpy()
                probabilities = saved[name].to_numpy()
                clipped = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
                independent_loss = float(-(y * np.log(clipped) + (1-y) * np.log1p(-clipped)).mean())
                n_positive = int(y.sum())
                ranks = pd.Series(probabilities).rank(method="average").to_numpy()
                independent_auc = float((ranks[y == 1].sum() - n_positive*(n_positive+1)/2) /
                                        (n_positive*(len(y)-n_positive)))
                oracle_errors[name]["LogLoss"] = abs(independent_loss - evaluation["metrics"][name]["LogLoss"])
                oracle_errors[name]["ROC-AUC"] = abs(independent_auc - evaluation["metrics"][name]["ROC-AUC"])
                assert max(oracle_errors[name].values()) < 1e-12
                artifact = joblib.load(args.out / f"{name}.joblib")
                x = builder.matrix(dates, artifact["features"])
                prediction = artifact["model"].predict_proba(x)[:, 1]
                err = float(np.max(np.abs(prediction - saved[name].to_numpy())))
                assert err == 0
                # Порядок строк не меняет prediction после обратной перестановки.
                ix = np.random.default_rng(17).permutation(len(x))
                back = np.empty(len(x))
                back[ix] = artifact["model"].predict_proba(x[ix])[:, 1]
                assert np.array_equal(prediction, back)
                predict_errors[name] = err
    # На реальных данных независимый построчный расчёт пяти видов признаков.
    cutoff = pd.Timestamp("2021-04-01")
    past, pairs = builder.context(cutoff)
    check_specs = [Feature("user", "count", 30), Feature("item", "unique", 90),
                   Feature("pair", "share", 90), Feature("pair", "recency", 180),
                   Feature("subcategory", "count", 90)]
    max_feature_error = 0.0
    for ix in np.random.default_rng(18).choice(len(pairs), size=40, replace=False):
        row = pairs.iloc[ix]
        for spec in check_specs:
            recent = past[past.timestamp >= cutoff - pd.Timedelta(days=spec.days)]
            if spec.relation == "user":
                match = recent[recent.customerID == row.customerID]
            elif spec.relation == "item":
                match = recent[recent.ISIN == row.ISIN]
            elif spec.relation == "pair":
                match = recent[(recent.customerID == row.customerID) & (recent.ISIN == row.ISIN)]
            else:
                match = recent[(recent.customerID == row.customerID) &
                               (recent.assetSubCategory == row.assetSubCategory)]
            if spec.operation == "count":
                expected = len(match)
            elif spec.operation == "unique":
                expected = match.customerID.nunique()
            elif spec.operation == "recency":
                expected = (cutoff - match.timestamp.max()).total_seconds() / 86400 if len(match) else 181
            else:
                denominator = len(recent[recent.customerID == row.customerID])
                expected = len(match) / denominator if denominator else 0
            error = abs(float(builder.column(cutoff, spec)[ix]) - expected)
            max_feature_error = max(error, max_feature_error)
            assert error < 1e-6
    # Добавление будущих событий и версий метаданных не меняет прошлые признаки.
    without_future = Builder(data[data.timestamp < cutoff], metadata[metadata.timestamp < cutoff], users, items)
    for spec in check_specs:
        np.testing.assert_array_equal(builder.column(cutoff, spec), without_future.column(cutoff, spec))
    # Проверить первые десять bootstrap-выборок явным повторением клиентских случаев.
    case_a = Ranking(frame).cases(saved.manual_plus_automatic.to_numpy())
    case_b = Ranking(frame).cases(saved.manual.to_numpy())
    delta = case_a.recall.to_numpy() - case_b.recall.to_numpy()
    rng = np.random.default_rng(20261013)
    for draw in rng.integers(0, len(users), size=(10, len(users))):
        repeated = np.concatenate([delta[case_a.customerID == users[j]] for j in draw])
        multiplicity = np.bincount(draw, minlength=len(users))
        weights = case_a.customerID.map(dict(zip(users, multiplicity))).to_numpy()
        good = np.isfinite(delta)
        fast = np.sum(delta[good] * weights[good]) / weights[good].sum()
        assert abs(fast - np.nanmean(repeated)) < 1e-12
    repeated_trials = 0
    archived = args.out / "provenance/interrupted_automatic_trials.json"
    if archived.exists():
        current = json.loads((args.out / "automatic_trials.json").read_text())
        key = lambda r: json.dumps({"features": r["features"], "params": r["params"]}, sort_keys=True)
        lookup = {key(r): r for r in current if "aggregate" in r}
        for old in json.loads(archived.read_text()):
            again = lookup[key(old)]
            for k, v in old["aggregate"].items():
                assert abs(v - again["aggregate"][k]) < 1e-12
            for a, b in zip(old["fold_metrics"], again["fold_metrics"]):
                for metric, value in a.items():
                    if value is not None:
                        assert abs(value - b[metric]) < 1e-12
            repeated_trials += 1
    result = {"unit_tests": test.testsRun, "metric_oracle_errors": oracle_errors,
              "prediction_replay_max_errors": predict_errors, "row_order_invariance": True,
              "real_feature_oracle_checks": 200, "max_feature_oracle_error": max_feature_error,
              "real_future_invariance": True, "bootstrap_draws_independently_checked": 10,
              "source_hashes_verified": True, "core_code_hashes_verified": True,
              "manual_training_source_verified": True,
              "repeated_cv_trials_metrics_verified": repeated_trials,
              "frozen_models_unchanged": all(sha256(args.out / f"{n}.joblib") == h
                  for n, h in evaluation["frozen_model_hashes"].items())}
    assert result["frozen_models_unchanged"]
    (args.out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
