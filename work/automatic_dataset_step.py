# Полная история для train/validation: test сразу исключаем
transactions = pd.read_csv(
    project_dir / 'work/data/FAR-Trans/transactions.csv',
    dtype={'customerID': 'string', 'ISIN': 'string'}
)
transactions['timestamp'] = pd.to_datetime(transactions['timestamp'])
test_start = pd.Timestamp('2022-07-01')
all_buys = transactions[
    (transactions['transactionType'] == 'Buy')
    & (transactions['timestamp'] >= '2020-01-01')
    & (transactions['timestamp'] < test_start)
].copy()
del transactions

manual_features = baseline['features']
auto_names = [feature['name'] for feature in feature_plan]
feature_names = manual_features + auto_names
pair_keys = ['customerID', 'ISIN']


def make_dataset_month(date):
    date = pd.Timestamp(date)
    end = date + pd.DateOffset(months=1)
    if not date.is_month_start or date != date.normalize():
        raise ValueError('Нужно первое число месяца')
    if date < pd.Timestamp('2020-04-01') or end > test_start:
        raise ValueError('Строим только месяцы train/validation')
    if end > pd.Timestamp(baseline['observation']['end_exclusive']):
        raise ValueError('Месяц выходит за документированный период')

    result = generate_features(all_buys, candidate_pairs, date, feature_plan)

    # Сохраняем пять ручных признаков с календарными окнами
    for months in [1, 3]:
        past = all_buys[
            (all_buys['timestamp'] >= date - pd.DateOffset(months=months))
            & (all_buys['timestamp'] < date)
        ]
        for group in (['user', 'user_item', 'item'] if months == 1 else ['user', 'user_item']):
            columns = group_columns[group]
            name = f'{group}_buy_count_{months}m'
            counts = past.groupby(columns).size().reset_index(name=name)
            result = result.merge(counts, on=columns, how='left', validate='many_to_one')
            result[name] = result[name].fillna(0).astype(int)

    # Ответ: была ли хотя бы одна покупка за целевой месяц
    future = all_buys[(all_buys['timestamp'] >= date) & (all_buys['timestamp'] < end)]
    labels = future[pair_keys].drop_duplicates().assign(label=1)
    result = result.merge(labels, on=pair_keys, how='left', validate='one_to_one')
    result['label'] = result['label'].fillna(0).astype(int)
    return result[pair_keys + ['prediction_date'] + feature_names + ['label']]


dates = pd.date_range('2020-04-01', '2022-06-01', freq='MS')
dataset = pd.concat([make_dataset_month(date) for date in dates], ignore_index=True)
train = dataset[dataset['prediction_date'] < '2022-01-01'].copy()
validation = dataset[dataset['prediction_date'] >= '2022-01-01'].copy()
assert not dataset.duplicated(pair_keys + ['prediction_date']).any()
assert not dataset[feature_names].isna().any().any()
assert dataset['label'].isin([0, 1]).all()
assert dataset.groupby(['customerID', 'prediction_date']).size().eq(50).all()
print('Train:', len(train), '| Validation:', len(validation))
print('Признаков:', len(feature_names), '= 5 ручных + 12 новых')
train.head()
