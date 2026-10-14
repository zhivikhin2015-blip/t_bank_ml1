"""Метрики и парный клиентский bootstrap, независимые от обучения."""
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


class Ranking:
    def __init__(self, frame):
        keys = ["customerID", "prediction_date"]
        if frame.duplicated(keys + ["ISIN"]).any():
            raise ValueError("Duplicate candidate")
        if not frame.label.isin([0, 1]).all():
            raise ValueError("Nonbinary target")
        self.frame = frame.reset_index(drop=True)
        self.y = self.frame.label.to_numpy()
        self.group = self.frame.groupby(keys, sort=True).ngroup().to_numpy()
        self.n = self.group.max() + 1
        self.total = np.bincount(self.group, weights=self.y, minlength=self.n)
        self.eligible = self.total > 0
        self.keys = self.frame.groupby(keys, sort=True).first().reset_index()[keys]
        payloads = zip(self.frame.customerID, self.frame.prediction_date, self.frame.ISIN)
        self.tie = np.array([int.from_bytes(hashlib.blake2b(
            json.dumps([42, str(u), str(pd.Timestamp(d).date()), str(i)],
                       ensure_ascii=False, separators=(",", ":")).encode(),
            digest_size=8, person=b"FAR-tie").digest(), "big") for u, d, i in payloads], dtype=np.uint64)
        self.item = self.frame.ISIN.astype(str).to_numpy()
        self.idcg = np.array([sum(1 / np.log2(j + 2) for j in range(min(int(r), 5)))
                              for r in self.total])

    def cases(self, scores, secondary=None):
        scores = np.asarray(scores)
        if scores.shape != self.y.shape or not np.isfinite(scores).all():
            raise ValueError("Bad scores")
        sec = np.zeros_like(scores) if secondary is None else np.asarray(secondary)
        order = np.lexsort((self.item, self.tie, -sec, -scores, self.group))
        g = self.group[order]
        starts = np.r_[0, np.flatnonzero(g[1:] != g[:-1]) + 1]
        rank = np.arange(len(g)) - np.repeat(starts, np.diff(np.r_[starts, len(g)]))
        top = rank < 5
        hits = np.bincount(g[top], weights=self.y[order[top]], minlength=self.n)
        dcg = np.bincount(g[top], weights=self.y[order[top]] / np.log2(rank[top] + 2), minlength=self.n)
        return self.keys.assign(positives=self.total, hits=hits,
            recall=np.divide(hits, self.total, out=np.full(self.n, np.nan), where=self.eligible),
            precision=hits / 5,
            ndcg=np.divide(dcg, self.idcg, out=np.full(self.n, np.nan), where=self.eligible))

    def evaluate(self, scores, probability=True, secondary=None):
        c = self.cases(scores, secondary)
        result = {"Recall@5": float(c.recall.mean()), "NDCG@5": float(c.ndcg.mean()),
                  "Precision@5": float(c.loc[self.eligible, "precision"].mean()),
                  "Precision@5_all": float(c.precision.mean()), "rows": len(self.y),
                  "positives": int(self.y.sum()), "eligible_cases": int(self.eligible.sum()),
                  "all_cases": int(self.n)}
        if probability:
            result["LogLoss"] = float(log_loss(self.y, scores, labels=[0, 1]))
            result["ROC-AUC"] = float(roc_auc_score(self.y, scores)) if len(np.unique(self.y)) == 2 else None
        else:
            result.update({"LogLoss": None, "ROC-AUC": None})
        return result


def paired_bootstrap(a, b, users, repeats=5000, seed=20261013):
    if not a[["customerID", "prediction_date"]].equals(b[["customerID", "prediction_date"]]):
        raise ValueError("Unpaired cases")
    if not np.array_equal(a.positives, b.positives):
        raise ValueError("Different targets")
    eligible = a.positives > 0
    per = pd.DataFrame({"customerID": a.customerID, "delta": a.recall - b.recall,
                        "n": eligible.astype(int)}).groupby("customerID").agg(delta=("delta", "sum"), n=("n", "sum"))
    per = per.reindex(users, fill_value=0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(users), size=(repeats, len(users)))
    numerators = per.delta.to_numpy()[draws].sum(axis=1)
    denominators = per.n.to_numpy()[draws].sum(axis=1)
    diffs = np.divide(numerators, denominators, out=np.full(repeats, np.nan), where=denominators > 0)
    return {"delta": float((a.recall - b.recall).mean()),
            "ci95": np.nanquantile(diffs, [0.025, 0.975]).tolist(), "repeats": repeats,
            "empty_draws": int(np.isnan(diffs).sum()), "seed": seed,
            "scope": "fixed fitted models and observed months; client sampling only"}
