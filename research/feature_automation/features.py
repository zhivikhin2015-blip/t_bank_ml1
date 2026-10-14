"""История строго до cutoff; ответы рассчитываются отдельной функцией."""
from dataclasses import dataclass, asdict
import hashlib
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Feature:
    relation: str
    operation: str
    days: int = 0
    months: int = 0

    @property
    def name(self):
        window = f"{self.months}m" if self.months else f"{self.days}d"
        return f"{self.relation}__{self.operation}__{window}"

    def description(self):
        return asdict(self) | {"name": self.name}


RELATIONS = {
    "user": ["customerID"], "item": ["ISIN"],
    "pair": ["customerID", "ISIN"],
    "category": ["customerID", "assetCategory"],
    "subcategory": ["customerID", "assetSubCategory"],
    "sector": ["customerID", "sector"],
}
OLD = [Feature("user", "count", months=1), Feature("pair", "count", months=1),
       Feature("item", "count", months=1), Feature("user", "count", months=3),
       Feature("pair", "count", months=3)]


def manual_features():
    specs = list(OLD)
    for days in (30, 90, 180):
        specs += [Feature(r, "count", days) for r in ("user", "item", "pair")]
        specs += [Feature(r, "unique", days) for r in ("user", "item")]
        specs += [Feature("pair", "share", days)]
    for r in RELATIONS:
        specs.append(Feature(r, "recency", 180))
        specs.append(Feature(r, "trend", 30))
    for r in ("category", "subcategory", "sector"):
        specs += [Feature(r, op, 90) for op in ("count", "share", "unique")]
    return specs


def generated_features(windows=(7, 14, 30, 60, 90, 180)):
    result = []
    for r in RELATIONS:
        for d in windows:
            result.append(Feature(r, "count", d))
            if r != "pair":
                result.append(Feature(r, "unique", d))
            if r not in ("user", "item"):
                result.append(Feature(r, "share", d))
        result += [Feature(r, "recency", 180), Feature(r, "trend", 30)]
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data(directory):
    t = pd.read_csv(directory / "transactions.csv", dtype={"customerID": str, "ISIN": str})
    t["timestamp"] = pd.to_datetime(t["timestamp"], errors="raise")
    t = t.loc[(t.transactionType == "Buy") & (t.timestamp >= "2019-10-01")].copy()
    a = pd.read_csv(directory / "asset_information.csv", dtype={"ISIN": str})
    a["timestamp"] = pd.to_datetime(a["timestamp"], errors="raise")
    fields = ["assetCategory", "assetSubCategory", "sector"]
    # Конфликтующие значения в одной версии нельзя разрешать порядком CSV.
    conflicts = []
    rows = []
    for (isin, ts), group in a.groupby(["ISIN", "timestamp"], sort=True):
        row = {"ISIN": isin, "timestamp": ts}
        for field in fields:
            values = group[field].dropna().unique()
            row[field] = values[0] if len(values) == 1 else np.nan
            if len(values) > 1:
                conflicts.append({"ISIN": isin, "timestamp": str(ts), "field": field,
                                  "values": sorted(map(str, values))})
        rows.append(row)
    return t, pd.DataFrame(rows), conflicts


def select_cohort(buys):
    past = buys[(buys.timestamp >= "2019-10-01") & (buys.timestamp < "2020-04-01")]
    users = past.customerID.drop_duplicates().sample(n=500, random_state=42).tolist()
    items = past.ISIN.value_counts().head(50).index.tolist()
    return users, items


class Builder:
    def __init__(self, buys, metadata, users, items):
        self.buys, self.metadata = buys, metadata
        self.users, self.items = list(users), list(items)
        self.pairs = pd.DataFrame({"customerID": users}).merge(
            pd.DataFrame({"ISIN": items}), how="cross")
        self.cache = {}
        self._contexts = {}

    def context(self, cutoff):
        cutoff = pd.Timestamp(cutoff)
        if cutoff not in self._contexts:
            a = self.metadata[self.metadata.timestamp < cutoff].sort_values(
                ["timestamp", "ISIN"]).drop_duplicates("ISIN", keep="last").drop(columns="timestamp")
            pairs = self.pairs.merge(a, on="ISIN", how="left", validate="many_to_one")
            past = self.buys[(self.buys.timestamp < cutoff) &
                             (self.buys.timestamp >= cutoff - pd.Timedelta(days=180))]
            past = past.merge(a, on="ISIN", how="left", validate="many_to_one")
            self._contexts[cutoff] = (past, pairs)
        return self._contexts[cutoff]

    def clear_contexts(self):
        self._contexts.clear()

    def column(self, cutoff, spec):
        cutoff = pd.Timestamp(cutoff)
        key = (cutoff, spec)
        if key in self.cache:
            return self.cache[key]
        past, pairs = self.context(cutoff)
        start = cutoff - (pd.DateOffset(months=spec.months) if spec.months else
                          pd.Timedelta(days=spec.days))
        events = past[past.timestamp >= start]
        keys = RELATIONS[spec.relation]
        valid = events.dropna(subset=keys)

        def mapped(aggregate):
            if len(keys) == 1:
                return pairs[keys[0]].map(aggregate).to_numpy(dtype=float)
            idx = pd.MultiIndex.from_frame(pairs[keys])
            return aggregate.reindex(idx).to_numpy(dtype=float)

        if spec.operation == "recency":
            durations = (cutoff - valid.groupby(keys).timestamp.max()).dt.total_seconds() / 86400
            values = mapped(durations)
            values = np.nan_to_num(values, nan=181.0)
        elif spec.operation == "trend":
            recent = self.column(cutoff, Feature(spec.relation, "count", spec.days))
            double = self.column(cutoff, Feature(spec.relation, "count", 2 * spec.days))
            values = np.log1p(recent) - np.log1p(double - recent)
        else:
            if spec.operation == "unique":
                distinct = "customerID" if spec.relation == "item" else "ISIN"
                counts = valid.groupby(keys)[distinct].nunique()
            else:
                counts = valid.groupby(keys).size()
            values = np.nan_to_num(mapped(counts), nan=0.0)
            if spec.operation == "share":
                denominator = self.column(cutoff, Feature("user", "count", spec.days, spec.months))
                values = np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0)
        values = values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite feature {spec}")
        self.cache[key] = values
        return values

    def matrix(self, dates, specs):
        return np.concatenate([np.column_stack([self.column(d, s) for s in specs]) for d in dates])

    def labels(self, dates):
        frames, coverage = [], []
        for cutoff in map(pd.Timestamp, dates):
            future = self.buys[(self.buys.timestamp >= cutoff) &
                               (self.buys.timestamp < cutoff + pd.DateOffset(months=1)) &
                               self.buys.customerID.isin(self.users)]
            positives = future[["customerID", "ISIN"]].drop_duplicates()
            f = self.pairs.merge(positives.assign(label=1), how="left", on=["customerID", "ISIN"])
            f["label"] = f.label.fillna(0).astype(np.int8)
            f["prediction_date"] = cutoff
            frames.append(f)
            coverage.append({"month": str(cutoff.date()), "in_catalog": int(f.label.sum()),
                             "all_assets": len(positives)})
        return pd.concat(frames, ignore_index=True), coverage
