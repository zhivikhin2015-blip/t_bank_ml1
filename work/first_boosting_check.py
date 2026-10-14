import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits


scope = runpy.run_path('work/month_dataset_check.py', run_name='__main__')
make_month_dataset = scope['make_month_dataset']
# Build only train and validation; test outcomes are not evaluated here.
large_dataset = pd.concat([
    make_month_dataset(date)
    for date in pd.date_range('2020-04-01', '2022-06-01', freq='MS')
], ignore_index=True)
train = large_dataset[large_dataset['prediction_date'] < '2022-01-01'].copy()
validation = large_dataset[large_dataset['prediction_date'] >= '2022-01-01'].copy()

model_features = ['user_buy_count_1m', 'user_item_buy_count_1m']
model = HistGradientBoostingClassifier(
    max_iter=100,
    max_leaf_nodes=7,
    early_stopping=False,
    random_state=42,
)
with threadpool_limits(limits=4):
    model.fit(train[model_features], train['label'])
    model_predictions = model.predict_proba(validation[model_features])[:, 1]

baseline_probability = float(train['label'].mean())
baseline_score = log_loss(
    validation['label'], np.full(len(validation), baseline_probability)
)
model_score = log_loss(validation['label'], model_predictions)
manual_score = float(-(
    validation['label'].to_numpy() * np.log(model_predictions)
    + (1 - validation['label'].to_numpy()) * np.log(1 - model_predictions)
).mean())
assert abs(model_score - manual_score) < 1e-12
assert list(model.classes_) == [0, 1]
assert np.isfinite(model_predictions).all()
assert ((model_predictions > 0) & (model_predictions < 1)).all()
assert model.n_iter_ == 100

results = {
    'sklearn_version': sklearn.__version__,
    'numpy_version': np.__version__,
    'pandas_version': pd.__version__,
    'feature_columns': model_features,
    'train_rows': len(train),
    'validation_rows': len(validation),
    'baseline_probability': baseline_probability,
    'baseline_validation_logloss': baseline_score,
    'model_validation_logloss': model_score,
    'max_iter': 100,
    'max_leaf_nodes': 7,
    'early_stopping': False,
    'random_state': 42,
    'test_evaluated': False,
}
Path('work/first_boosting_result.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2) + '\n'
)
print(json.dumps(results, ensure_ascii=False, indent=2))
