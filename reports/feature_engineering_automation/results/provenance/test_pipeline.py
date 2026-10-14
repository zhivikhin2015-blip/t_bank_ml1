"""Независимые примеры для временных признаков и метрик."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from .features import Builder, Feature
from .metrics import Ranking, paired_bootstrap


class PipelineTests(unittest.TestCase):
    def builder(self):
        buys = pd.DataFrame({"customerID": ["u", "u", "u", "v", "u"],
            "ISIN": ["a", "a", "b", "a", "b"],
            "timestamp": pd.to_datetime(["2021-02-01", "2021-02-20", "2021-02-25", "2021-02-28", "2021-03-01"])})
        meta = pd.DataFrame({"ISIN": ["a", "b", "a"], "assetCategory": ["bond", "stock", "stock"],
            "assetSubCategory": ["x", "y", "y"], "sector": ["S", "T", "T"],
            "timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2021-03-01"])})
        return Builder(buys, meta, ["u", "v", "empty"], ["a", "b"])

    def test_temporal_counts_unique_share_recency(self):
        b = self.builder()
        d = "2021-03-01"
        np.testing.assert_array_equal(b.column(d, Feature("user", "count", 30)), [3, 3, 1, 1, 0, 0])
        np.testing.assert_array_equal(b.column(d, Feature("pair", "count", 30)), [2, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(b.column(d, Feature("item", "unique", 30)), [2, 1, 2, 1, 2, 1])
        np.testing.assert_allclose(b.column(d, Feature("pair", "share", 30)), [2/3, 1/3, 1, 0, 0, 0])
        np.testing.assert_array_equal(b.column(d, Feature("pair", "recency", 180)), [9, 4, 1, 181, 181, 181])
        # Будущее изменение категории на cutoff не присоединяется.
        np.testing.assert_array_equal(b.column(d, Feature("category", "count", 30)), [2, 1, 1, 0, 0, 0])

    def test_future_invariance_and_row_order(self):
        b = self.builder()
        other = Builder(b.buys[b.buys.timestamp < "2021-03-01"].sample(frac=1, random_state=4),
                        b.metadata[b.metadata.timestamp < "2021-03-01"].sample(frac=1, random_state=3),
                        b.users, b.items)
        for r in ("user", "item", "pair", "category"):
            for op in ("count", "recency", "trend"):
                s = Feature(r, op, 180 if op == "recency" else 30)
                np.testing.assert_array_equal(b.column("2021-03-01", s), other.column("2021-03-01", s))

    def test_labels_separate(self):
        b = self.builder()
        f, _ = b.labels(["2021-03-01"])
        self.assertEqual(f.label.tolist(), [0, 1, 0, 0, 0, 0])
        b.buys = b.buys.iloc[:0]
        b.cache.clear()
        b.clear_contexts()
        np.testing.assert_array_equal(b.column("2021-03-01", Feature("user", "count", 30)), np.zeros(6))

    def frame(self):
        return pd.DataFrame({"customerID": ["u"] * 6 + ["zero"] * 6,
            "prediction_date": pd.Timestamp("2021-01-01"), "ISIN": list("abcdef") * 2,
            "label": [1, 1, 0, 0, 0, 1] + [0] * 6})

    def test_independent_metric_oracle(self):
        f = self.frame()
        p = np.tile(np.arange(6, 0, -1) / 7, 2)
        result = Ranking(f).evaluate(p)
        self.assertAlmostEqual(result["Recall@5"], 2 / 3)
        self.assertAlmostEqual(result["Precision@5"], 2 / 5)
        self.assertAlmostEqual(result["Precision@5_all"], 1 / 5)
        dcg = 1 + 1 / np.log2(3)
        self.assertAlmostEqual(result["NDCG@5"], dcg / (dcg + 1 / np.log2(4)))
        expected = -(f.label.to_numpy() * np.log(p) + (1-f.label.to_numpy()) * np.log(1-p)).mean()
        self.assertAlmostEqual(result["LogLoss"], expected)

    def test_ties_independent_of_labels_and_order(self):
        f = self.frame()
        a = Ranking(f).cases(np.ones(12))
        shuffled = f.sample(frac=1, random_state=8)
        b = Ranking(shuffled).cases(np.ones(12))
        pd.testing.assert_frame_equal(a, b)
        changed = f.copy()
        changed.label = 1 - changed.label
        np.testing.assert_array_equal(Ranking(f).tie, Ranking(changed).tie)

    def test_one_class_auc_and_zero_buyers(self):
        f = self.frame()
        f.label = 0
        result = Ranking(f).evaluate(np.full(12, 0.2))
        self.assertIsNone(result["ROC-AUC"])
        self.assertEqual(result["eligible_cases"], 0)
        self.assertEqual(result["Precision@5_all"], 0)

    def test_bootstrap_multiplicity(self):
        a = pd.DataFrame({"customerID": ["a", "a", "b"],
                          "prediction_date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-01-01"]),
                          "positives": [1, 1, 1], "recall": [1., 0., 0.]})
        b = a.copy()
        b.recall = [0., 0., 1.]
        # Клиент a выбран дважды, клиент b один раз: (2*1 - 1)/(2*2 + 1)=0.2.
        class RNG:
            def integers(self, *args, **kwargs):
                return np.array([[0, 0, 1]])
        with patch("numpy.random.default_rng", return_value=RNG()):
            r = paired_bootstrap(a, b, ["a", "b", "unused"], repeats=1)
        np.testing.assert_allclose(r["ci95"], [0.2, 0.2])


if __name__ == "__main__":
    unittest.main()
