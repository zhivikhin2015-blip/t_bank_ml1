"""Запуск: python -m research.feature_automation.experiment --data-dir ... --out ..."""
import argparse
from datetime import datetime, timezone
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import platform
import resource
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from .features import (Builder, Feature, OLD, RELATIONS, manual_features, generated_features,
                       load_data, select_cohort, sha256)
from .metrics import Ranking, paired_bootstrap


def unique(specs):
    return list(dict.fromkeys(specs))


REFERENCE = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=7,
                 min_samples_leaf=20, l2_regularization=0.0)
FOLDS = [("2021-04-01", "2021-07-01"), ("2021-07-01", "2021-10-01"),
         ("2021-10-01", "2022-01-01")]


def grid():
    keys = list(REFERENCE)
    candidates = [dict(zip(keys, v)) for v in itertools.product(
        (0.05, 0.1, 0.2), (100, 200, 300), (7, 15, 31), (20, 60), (0.0, 1.0))]
    rng = np.random.default_rng(42)
    picked = [REFERENCE, REFERENCE | {"max_iter": 200}]
    for i in rng.permutation(len(candidates)):
        if candidates[i] not in picked:
            picked.append(candidates[i])
        if len(picked) == 16:
            break
    return picked


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Experiment:
    def __init__(self, builder, out):
        self.builder, self.out = builder, out
        self.dates = list(pd.date_range("2020-04-01", "2022-06-01", freq="MS"))
        self.frame, self.coverage = builder.labels(self.dates)
        self.ranks = []
        self.splits = []
        for start, end in FOLDS:
            train = (self.frame.prediction_date < start).to_numpy()
            check = ((self.frame.prediction_date >= start) & (self.frame.prediction_date < end)).to_numpy()
            self.splits.append((train, check))
            self.ranks.append(Ranking(self.frame.loc[check]))
        self.cache, self.log = {}, []
        self.timing = {"fit_seconds": 0.0, "predict_seconds": 0.0,
                       "feature_seconds": 0.0, "fits": 0}

    def cv(self, specs, params, stage):
        specs = unique(specs)
        key = (tuple(specs), tuple(sorted(params.items())))
        if key in self.cache:
            record = self.cache[key]
            self.log.append({"stage": stage, "cache_hit": True, "source_trial": record["trial"]})
            return record
        begin = time.perf_counter()
        x = self.builder.matrix(self.dates, specs)
        feature_time = time.perf_counter() - begin
        self.timing["feature_seconds"] += feature_time
        results = []
        fit_time, predict_time = 0.0, 0.0
        for (train, check), rank in zip(self.splits, self.ranks):
            model = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=42)
            t = time.perf_counter()
            model.fit(x[train], self.frame.label.to_numpy()[train])
            fit_time += time.perf_counter() - t
            t = time.perf_counter()
            p = model.predict_proba(x[check])[:, 1]
            predict_time += time.perf_counter() - t
            results.append(rank.evaluate(p))
        self.timing["fits"] += 3
        self.timing["fit_seconds"] += fit_time
        self.timing["predict_seconds"] += predict_time
        eligible = sum(r["eligible_cases"] for r in results)
        rows = sum(r["rows"] for r in results)
        aggregate = {k: sum(r[k] * r["eligible_cases"] for r in results) / eligible
                     for k in ("Recall@5", "NDCG@5", "Precision@5")}
        aggregate["LogLoss"] = sum(r["LogLoss"] * r["rows"] for r in results) / rows
        record = {"trial": len(self.log), "stage": stage, "features": [s.name for s in specs],
                  "params": params, "fold_metrics": results, "aggregate": aggregate,
                  "feature_seconds": feature_time, "fit_seconds": fit_time,
                  "predict_seconds": predict_time}
        self.log.append(record)
        self.cache[key] = record
        with (self.out / "trials.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{stage}: trial={record['trial']} features={len(specs)} Recall={aggregate['Recall@5']:.5f}", flush=True)
        return record

    def tune(self, specs, name):
        records = [self.cv(specs, params, f"tune/{name}") for params in grid()]
        best = min(records, key=lambda r: (-r["aggregate"]["Recall@5"], r["aggregate"]["LogLoss"],
                                          r["params"]["max_iter"], r["params"]["max_leaf_nodes"]))
        return best

    def select(self, families, base, name):
        selected, remaining, steps = [], list(families), []
        current = unique(base)
        score = self.cv(current, REFERENCE, f"select/{name}/base")["aggregate"]["Recall@5"] if current else -1
        while remaining:
            candidates = []
            for family in remaining:
                specs = unique(current + families[family])
                r = self.cv(specs, REFERENCE, f"select/{name}/{family}")
                candidates.append((r["aggregate"]["Recall@5"], family, specs, r))
            value, family, specs, r = max(candidates, key=lambda v: (v[0], -len(v[2]), v[1]))
            if current and value <= score + 1e-6:
                break
            steps.append({"family": family, "previous_recall": score if score >= 0 else None,
                          "recall": value, "trial": r["trial"]})
            current, score = specs, value
            selected.append(family)
            remaining.remove(family)
        return current, selected, steps

    def refine(self, specs, families):
        specs = list(specs)
        log = []
        score = self.cv(specs, REFERENCE, "refine/base")["aggregate"]["Recall@5"]
        for family in families:
            windows = sorted({s.days for s in specs if s.relation == family and
                              s.operation not in ("recency", "trend")})
            dropped = []
            for days in windows:
                retained = [s for s in specs if not (s.relation == family and s.days == days and
                                                    s.operation not in ("recency", "trend"))]
                r = self.cv(retained, REFERENCE, f"refine/drop/{family}/{days}")
                dropped.append((score - r["aggregate"]["Recall@5"], days))
            if not dropped:
                continue
            _, chosen = max(dropped)
            step = max(1, chosen // 2)
            proposals = sorted({max(1, chosen - step), min(180, chosen + step)} - set(windows))
            # Если соседи уже представлены, проверяем промежуточное целое окно.
            if not proposals:
                lower = [d for d in windows if d < chosen]
                higher = [d for d in windows if d > chosen]
                midpoints = []
                if lower:
                    midpoints.append((max(lower) + chosen) // 2)
                if higher:
                    midpoints.append((min(higher) + chosen) // 2)
                proposals = sorted(set(midpoints) - set(windows))
            attempts = []
            for day in proposals:
                revised = unique([replace(s, days=day) if s.relation == family and s.days == chosen and
                                  s.operation not in ("recency", "trend") else s for s in specs])
                r = self.cv(revised, REFERENCE, f"refine/replace/{family}/{chosen}/{day}")
                attempts.append((r["aggregate"]["Recall@5"], day, revised, r["trial"]))
            accepted = None
            if attempts:
                value, day, revised, trial = max(attempts, key=lambda v: (v[0], -v[1]))
                if value > score + 1e-6:
                    accepted = {"old_days": chosen, "new_days": day, "trial": trial,
                                "previous_recall": score, "recall": value}
                    specs, score = revised, value
            log.append({"family": family, "drop_importance": dropped, "proposed_days": proposals,
                        "attempts": [{"days": d, "recall": v, "trial": t} for v, d, _, t in attempts],
                        "accepted": accepted})
        return specs, log

    def freeze(self, specs, params, name):
        t = time.perf_counter()
        x = self.builder.matrix(self.dates, specs)
        self.timing["feature_seconds"] += time.perf_counter() - t
        train = (self.frame.prediction_date < "2022-01-01").to_numpy()
        valid = ~train
        model = HistGradientBoostingClassifier(**params, early_stopping=False, random_state=42)
        t = time.perf_counter()
        model.fit(x[train], self.frame.label.to_numpy()[train])
        self.timing["fit_seconds"] += time.perf_counter() - t
        self.timing["fits"] += 1
        p = model.predict_proba(x[valid])[:, 1]
        joblib.dump({"model": model, "features": specs, "params": params}, self.out / f"{name}.joblib")
        return model, Ranking(self.frame.loc[valid]).evaluate(p), p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=["manual", "automatic", "evaluate"], required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, metadata, conflicts = load_data(args.data_dir)
    users, items = select_cohort(data)
    builder = Builder(data, metadata, users, items)
    with threadpool_limits(limits=4):
        if args.phase == "manual":
            if (args.out / "protocol.json").exists():
                raise FileExistsError("Use a fresh output directory; completed inputs are immutable")
            protocol = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": 42,
                "cohort": {"users": users, "items": items, "selection": ["2019-10-01", "2020-04-01"]},
                "folds": FOLDS, "train": ["2020-04-01", "2021-12-01"],
                "validation": ["2022-01-01", "2022-06-01"],
                "evaluation": ["2022-07-01", "2022-10-01"], "november_excluded": True,
                "evaluation_status": "exploratory: prior audit disclosed outcomes for this period",
                "primary_metric": "Recall@5, mean over user-months with in-catalog buys",
                "primary_comparison": "manual_plus_automatic minus manual",
                "other_comparisons": ["automatic_selected minus manual", "manual minus pair_popularity"],
                "selection": "forward relation families, positive inner-CV gain > 1e-6; then window refinement",
                "refinement": "leave-one-window-out; +/- floor(window/2) within 1..180; if empty, test midpoint to nearest initial windows",
                "hyperparameters": grid(), "manual_features": [s.description() for s in manual_features()],
                "generated_features": [s.description() for s in generated_features()],
                "training_policy": "final weights trained only through December 2021; no refit after validation",
                "tie_break": "BLAKE2b64 FAR-tie compact UTF-8 [42,user,date,item], digest asc then ISIN asc",
                "source_hashes": {f: sha256(args.data_dir / f) for f in
                                  ("transactions.csv", "asset_information.csv")},
                "source_code_hashes": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                "metadata_conflicts": conflicts,
                "limits": ["No exposure logs; zero is no recorded Buy", "Retrospectively reconstructed FAR-Trans",
                           "Fixed 500 existing clients and 50 historical popular assets", "No investment return claim"]}
            write_json(args.out / "protocol.json", protocol)
            exp = Experiment(builder, args.out)
            old = exp.tune(OLD, "old_5")
            manual = manual_features()
            best = exp.tune(manual, "manual")
            ablations = []
            groups = {"recency": lambda s: s.operation == "recency", "trend": lambda s: s.operation == "trend",
                      "category_relations": lambda s: s.relation in ("category", "subcategory", "sector"),
                      "unique": lambda s: s.operation == "unique", "share": lambda s: s.operation == "share"}
            for name, predicate in groups.items():
                retained = [s for s in manual if not predicate(s)]
                r = exp.cv(retained, best["params"], f"ablation/{name}")
                ablations.append({"removed": name, "recall": r["aggregate"]["Recall@5"],
                                  "delta_vs_full": r["aggregate"]["Recall@5"] - best["aggregate"]["Recall@5"]})
            controls = {"old_5": exp.cv(OLD, REFERENCE, "control/old_5"),
                        "manual": exp.cv(manual, REFERENCE, "control/manual")}
            frozen = {}
            for name, specs, params in [("old_5", OLD, old["params"]), ("manual", manual, best["params"])]:
                _, metric, _ = exp.freeze(specs, params, name)
                frozen[name] = {"params": params, "validation": metric, "model_sha256": sha256(args.out / f"{name}.joblib")}
            rank = Ranking(exp.frame.loc[exp.frame.prediction_date >= "2022-01-01"])
            dates = list(pd.date_range("2022-01-01", "2022-06-01", freq="MS"))
            pair = builder.matrix(dates, [Feature("pair", "count", months=3)]).ravel()
            pop = builder.matrix(dates, [Feature("item", "count", months=1)]).ravel()
            write_json(args.out / "manual_stage.json", {"best": {"old_5": old, "manual": best},
                       "controls": controls, "ablations": ablations, "frozen": frozen,
                       "rule_validation": rank.evaluate(pair, False, pop), "timing": exp.timing})
        elif args.phase == "automatic":
            if (args.out / "automatic_stage.json").exists():
                raise FileExistsError("Automatic stage already completed")
            if not (args.out / "manual_stage.json").exists():
                raise FileNotFoundError("Complete and freeze the manual baseline first")
            manual_artifact_hash = sha256(args.out / "manual.joblib")
            exp = Experiment(builder, args.out)
            all_specs = generated_features()
            families = {r: [s for s in all_specs if s.relation == r] for r in RELATIONS}
            selected, selected_families, steps = exp.select(families, [], "automatic")
            selected, refinement = exp.refine(selected, selected_families)
            combined, added, combined_steps = exp.select(
                {r: [s for s in selected if s.relation == r] for r in selected_families},
                manual_features(), "combined")
            variants = {"automatic_all": all_specs, "automatic_selected": selected,
                        "manual_plus_automatic": combined}
            best, frozen, controls = {}, {}, {}
            for name, specs in variants.items():
                controls[name] = exp.cv(specs, REFERENCE, f"control/{name}")
                best[name] = exp.tune(specs, name)
                _, metric, _ = exp.freeze(specs, best[name]["params"], name)
                frozen[name] = {"params": best[name]["params"], "validation": metric,
                                "features": [s.description() for s in specs],
                                "model_sha256": sha256(args.out / f"{name}.joblib")}
            assert sha256(args.out / "manual.joblib") == manual_artifact_hash
            write_json(args.out / "automatic_stage.json", {"selected_families": selected_families,
                "selection_steps": steps, "window_refinement": refinement, "combined_added_families": added,
                "combined_steps": combined_steps, "best": best, "frozen": frozen, "controls": controls,
                "manual_model_unchanged": True, "timing": exp.timing})
        else:
            if (args.out / "evaluation.json").exists():
                raise FileExistsError("Evaluation already completed")
            protocol = json.loads((args.out / "protocol.json").read_text())
            for f, expected in protocol["source_hashes"].items():
                assert sha256(args.data_dir / f) == expected
            dates = list(pd.date_range("2022-07-01", "2022-10-01", freq="MS"))
            frame, coverage = builder.labels(dates)
            rank = Ranking(frame)
            # Ни один fit не допускается в фазе финальной оценки.
            def forbid_fit(*a, **k):
                raise RuntimeError("Final evaluation must not train")
            HistGradientBoostingClassifier.fit = forbid_fit
            predictions, metrics, cases, timings = {}, {}, {}, {}
            names = ["old_5", "manual", "automatic_all", "automatic_selected", "manual_plus_automatic"]
            frozen_before = {n: sha256(args.out / f"{n}.joblib") for n in names}
            for name in names:
                artifact = joblib.load(args.out / f"{name}.joblib")
                t = time.perf_counter()
                x = builder.matrix(dates, artifact["features"])
                feature_time = time.perf_counter() - t
                t = time.perf_counter()
                p = artifact["model"].predict_proba(x)[:, 1]
                predict_time = time.perf_counter() - t
                predictions[name] = p
                timings[name] = {"feature_seconds": feature_time, "predict_seconds": predict_time,
                                 "features": len(artifact["features"]), "feature_cache": "shared in listed order"}
                metrics[name] = rank.evaluate(p)
                cases[name] = rank.cases(p)
            pair = builder.matrix(dates, [Feature("pair", "count", months=3)]).ravel()
            pop = builder.matrix(dates, [Feature("item", "count", months=1)]).ravel()
            for name, score, secondary in [("pair_popularity", pair, pop), ("popularity", pop, None)]:
                metrics[name] = rank.evaluate(score, False, secondary)
                cases[name] = rank.cases(score, secondary)
                predictions[name] = score
            monthly = {n: [{"month": str(d.date()),
                           "Recall@5": float(c[c.prediction_date == d].recall.mean()),
                           "NDCG@5": float(c[c.prediction_date == d].ndcg.mean()),
                           "eligible_cases": int((c[c.prediction_date == d].positives > 0).sum())}
                          for d in dates] for n, c in cases.items()}
            comparisons = [("manual_plus_automatic", "manual"), ("automatic_selected", "manual"),
                           ("manual", "pair_popularity"), ("manual_plus_automatic", "pair_popularity")]
            bootstrap = {f"{a} - {b}": paired_bootstrap(cases[a], cases[b], users) for a, b in comparisons}
            group_metrics = {}
            for kind, mask in [("repeat_3m", pair > 0), ("new_3m", pair == 0)]:
                subset = frame.copy()
                subset["label"] = subset.label.to_numpy() * mask
                grouped_rank = Ranking(subset)
                group_metrics[kind] = {n: grouped_rank.evaluate(p, False, pop if n == "pair_popularity" else None)
                                       for n, p in predictions.items()}
            assert frozen_before == {n: sha256(args.out / f"{n}.joblib") for n in names}
            frame.assign(**predictions, pair_popularity_secondary=pop).to_csv(
                args.out / "predictions.csv.gz", index=False, float_format="%.17g")
            write_json(args.out / "evaluation.json", {"status": protocol["evaluation_status"],
                "metrics": metrics, "monthly": monthly, "groups": group_metrics, "coverage": coverage,
                "bootstrap": bootstrap, "timing": timings, "frozen_model_hashes": frozen_before,
                "evaluation_fits": 0, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "environment": {"python": platform.python_version(), "numpy": np.__version__,
                                "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 4}})
        if args.phase != "evaluate":
            write_json(args.out / f"{args.phase}_trials.json", exp.log)
    print(f"Completed {args.phase}", flush=True)


if __name__ == "__main__":
    main()
