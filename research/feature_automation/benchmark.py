"""Изолированный замер генерации/инференса каждой замороженной модели."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import joblib
import pandas as pd
from threadpoolctl import threadpool_limits
from .features import Builder, load_data, select_cohort


NAMES = ["old_5", "manual", "automatic_all", "automatic_selected", "manual_plus_automatic"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", choices=NAMES)
    args = parser.parse_args()
    if args.model:
        data, metadata, _ = load_data(args.data_dir)
        users, items = select_cohort(data)
        artifact = joblib.load(args.out / f"{args.model}.joblib")
        b = Builder(data, metadata, users, items)
        dates = list(pd.date_range("2020-04-01", "2022-06-01", freq="MS"))
        t = time.perf_counter()
        x = b.matrix(dates, artifact["features"])
        generation = time.perf_counter() - t
        shape = x.shape
        del x, b
        b = Builder(data, metadata, users, items)
        evaluation_dates = list(pd.date_range("2022-07-01", "2022-10-01", freq="MS"))
        t = time.perf_counter()
        x = b.matrix(evaluation_dates, artifact["features"])
        evaluation_generation = time.perf_counter() - t
        with threadpool_limits(limits=4):
            t = time.perf_counter()
            artifact["model"].predict_proba(x)
            predict = time.perf_counter() - t
        print(json.dumps({"model": args.model, "development_matrix_shape": shape,
            "development_generation_seconds": generation, "evaluation_generation_seconds": evaluation_generation,
            "evaluation_predict_seconds": predict, "evaluation_rows": len(x),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "design": "one fresh process/cache per model; OS cache and run order uncontrolled; four model threads"}))
    else:
        results = []
        for name in NAMES:
            command = [sys.executable, "-m", "research.feature_automation.benchmark", "--data-dir",
                       str(args.data_dir), "--out", str(args.out), "--model", name]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            results.append(json.loads(result.stdout))
        (args.out / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
