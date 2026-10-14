# Ручной feature engineering и автоматизация признаков для рекомендаций финансовых активов

## Аннотация

Исследуется месячный прогноз записанной покупки актива по истории взаимодействий. На FAR-Trans построен ручной HistGradientBoostingClassifier с 44 признаками, затем реализован генератор 102 кандидатов с отбором связей и локальным уточнением временных окон. Сравнение выполнено на одинаковых 500 клиентах и 50 активах.

На июле–октябре 2022 ручная модель получила 65,53% Recall@5, автоматический отбор — 64,32%, объединённый вариант — 64,39%. Основная разность объединённого и ручного вариантов: -1.14 п.п., 95% клиентский интервал [-2.92; +0.42] п.п. Преимущество автоматизации по основной метрике не подтверждено. Ручной набор улучшил наблюдаемый результат относительно пяти счётчиков.

**Статус оценки — исследовательский:** итоговый период уже был раскрыт при предшествующем аудите. Текущие настройки выбраны на внутренних блоках, но этот период не является новым независимым подтверждением.

## Содержание

1. [Постановка задачи](#1-постановка-задачи)
2. [Данные и построение таблицы](#2-данные-и-построение-таблицы)
3. [Временной протокол](#3-временной-протокол)
4. [Ручные признаки](#4-ручные-признаки)
5. [Бустинг и настройка](#5-бустинг-и-настройка)
6. [Автоматизация](#6-автоматизация)
7. [Вклад ручных признаков](#7-вклад-ручных-признаков)
8. [Метрики и расчёты](#8-метрики-и-расчёты)
9. [Результаты и неопределённость](#9-результаты-и-неопределённость)
10. [Затраты и проверка](#10-затраты-и-проверка)
11. [Рекомендации и ограничения](#11-рекомендации-и-ограничения)
12. [Воспроизведение и источники](#12-воспроизведение-и-источники)
13. [Полный каталог признаков](#13-каталог-признаков)

## 1. Постановка задачи

| Требование задания | Реализованный результат | Где описан |
|---|---|---|
| Ручной бустинг с содержательными признаками | 44 признака, временная настройка, контроль из пяти счётчиков | Разделы 4–5, 7 |
| История строго до момента прогноза | Временные фильтры событий и версий метаданных, проверка будущих событий | Разделы 2–3, 10 |
| Идея автоматизации | Грамматика 102 кандидатов, отбор семейств, локальный поиск окон | Раздел 6 |
| Экспериментальное сравнение | Пять ML-вариантов, два правила, метрики, срезы и интервалы | Разделы 8–10 |
| Читаемый отчёт | Объяснения, реальные примеры, полный каталог и воспроизведение | Настоящий документ |

Проверено выполнение процедуры автоматизации; превосходство над ручными признаками не установлено. Предложения SQL-агента, графов и трансформера рассматриваются в задании как возможные направления, а не как обязательные три реализации.

Признак — числовое описание прошлой истории, например количество покупок пары за 90 дней. Модель по признакам оценивает вероятность покупки. Ранжирование — сортировка активов по этой оценке; top-5 — первые пять позиций. Baseline — исходное решение для сравнения; validation — промежуточная проверка; итоговая evaluation — оценка уже зафиксированных моделей.

Задание содержит два последовательных этапа: разработать хорошие ручные признаки и настроить градиентный бустинг; предложить автоматизацию расчёта признаков и показать экспериментальные результаты её работы.

В момент $t$, первое число месяца, для клиента $u$ и актива $i$ оценивается $p_{u,i,t}=P(y_{u,i,t}=1\mid x_{u,i,t})$. Цель $y=1$ означает хотя бы одну записанную покупку пары в $[t,t+1\text{ месяц})$. Признаки $x$ вычисляются только по предыдущим событиям. По убыванию $p$ выбираются пять активов.

Здесь проверяются предсказание покупок и ранжирование. Доходность, риск и пригодность продукта клиенту не входят в целевую переменную. Нули — отсутствие записанной покупки, а не доказанные отрицательные предпочтения: логов показов нет.

Исследовательские вопросы: помогает ли расширенный ручной набор; какие группы в нём полезны; способен ли генератор получить конкурентные признаки; даёт ли добавление автоматически выбранных признаков прирост ручной базе. Основное формальное сравнение — объединённая модель минус ручная по Recall@5.

## 2. Данные и построение таблицы

Используется публичный [FAR-Trans, University of Glasgow](https://researchdata.gla.ac.uk/1658/). В расчётах нужны transactions.csv и asset_information.csv; цены, продажи, анкеты риска и размер капитала не подаются модели. История покупок ограничена событиями начиная с октября 2019.

По октябрю 2019–марту 2020 выбираются 500 случайных прежних покупателей с seed=42 и 50 самых покупаемых активов. Состав фиксируется для всех месяцев. На месяц создаётся $500\cdot50=25000$ кандидатных пар. Сделки внутри одного целевого месяца объединяются в бинарную метку; для исторических счётчиков сделки сохраняются.

| Часть | Месяцев | Пар | Положительных пар | Клиент-месяцев | Случаев для Recall |
|---|---|---|---|---|---|
| train | 21 | 525000 | 4307 | 10500 | 2085 |
| validation | 6 | 150000 | 1068 | 3000 | 481 |
| evaluation | 4 | 100000 | 508 | 2000 | 236 |

Для финальной оценки: $25000\cdot4=100000$ пар; $500\cdot4=2000$ клиент-месяцев; положительных пар 508, то есть 0,508% строк. Основные ranking-метрики используют 236 клиент-месяцев с покупками внутри каталога — 11,8% всех случаев. Эти случаи принадлежат 129 разным клиентам. Число кандидатных строк не равно числу независимых покупок; это существенно для интерпретации качества.

Вне каталога остаются покупки: покрытие финального периода $508/678=74,93$%. Оно измеряет долю фактических уникальных покупок клиент–актив–месяц, представленную каталогом. Наличие актива в прошлом не доказывает его текущую торговую доступность.

Метаданные присоединяются по последней версии строго до $t$. В трёх парах ISIN/дата обнаружены конфликтующие категории. Неоднозначное поле заменяется неизвестным. Исторические сделки группируются по классификации, известной на дату прогноза. Эти фильтры не доказывают истинную историческую доступность ретроспективно реконструированных исходных данных. Авторы описывают реконструкцию покупок и изменения дат в [§3.3.1 публикации FAR-Trans](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).

### Код подготовки данных и целевого ответа

Ниже приведён действующий код чтения данных и выбора выборки. `Buy` — покупка; `timestamp` — дата события; `customerID` и `ISIN` — ключи клиента и актива. Отбор клиентов и каталога использует только период до апреля 2020.

```python
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
```

Целевой ответ вычисляет метод `Builder.labels`: повторные покупки внутри месяца объединяются через `drop_duplicates`, а отсутствие покупки даёт ноль. `self.pairs` — все 500 × 50 кандидатных пар; его построение показано в полном классе `Builder` в разделе 4.

```python
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
```

## 3. Временной протокол

| Этап | Обучение | Проверяемые месяцы | Назначение |
|---|---|---|---|
| Внутренний блок 1 | Апрель 2020 — март 2021 | Апрель–июнь 2021 | Выбор признаков и настроек |
| Внутренний блок 2 | Апрель 2020 — июнь 2021 | Июль–сентябрь 2021 | То же |
| Внутренний блок 3 | Апрель 2020 — сентябрь 2021 | Октябрь–декабрь 2021 | То же |
| Диагностическая validation | Апрель 2020 — декабрь 2021 | Январь–июнь 2022 | Без выбора решений |
| Финальная оценка | Те же фиксированные веса | Июль–октябрь 2022 | Исследовательское сравнение |

Внутренний Recall объединяет все учитываемые клиент-месяцы трёх блоков; это не среднее трёх процентов с одинаковым весом блоков. Выбор уже использует внутренние данные, поэтому внутренний максимум может быть оптимистичен. Это общий риск отбора моделей, описанный [Cawley & Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html).

Веса финальных моделей обучаются до декабря 2021, после validation не переобучаются. История при переходе к следующему месяцу обновляется уже наблюдёнными событиями; это последовательный прогноз с фиксированной моделью. Ноябрь исключён: документированный сбор заканчивается 29 ноября, полнота календарного месяца не подтверждена.

Период и правила сохранены в [protocol.json](results/final/protocol.json). До текущей финальной оценки исправлена обработка границы поиска окон; исходный протокол и код сохранены в provenance. После получения текущих финальных метрик признаки, параметры и веса не менялись.

### Код временных разбиений

`FOLDS` задаёт начало и конец каждого внутреннего проверочного блока. `train` включает строки строго до начала блока, `check` — его месяцы. Конструктор класса `Experiment` сохраняет эти маски и объекты расчёта метрик; будущие месяцы не попадают в соответствующий `fit`.

```python
FOLDS = [("2021-04-01", "2021-07-01"), ("2021-07-01", "2021-10-01"),
         ("2021-10-01", "2022-01-01")]

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
```

## 4. Ручные признаки

Ручная база — 44 признака. Она включает активность клиента и спрос на актив, повторные покупки пары, разнообразие активов и покупателей, давность, доли покупок и динамику. Косвенные связи с категорией, подкатегорией и сектором дают возможность оценивать активы, которые клиент ещё не покупал.

| Связь | Ручных | Генерируемых | Отобранных автоматических |
|---|---|---|---|
| Клиент | 10 | 14 | 0 |
| Актив | 9 | 14 | 14 |
| Клиент–актив | 10 | 14 | 14 |
| Клиент–категория | 5 | 20 | 0 |
| Клиент–подкатегория | 5 | 20 | 0 |
| Клиент–сектор | 5 | 20 | 0 |

Используются точные 30/90/180 дней и исходные календарные 1/3 месяца. При отсутствии покупок count/unique/share равны нулю; recency равен 181. Динамика сравнивает два соседних 30-дневных окна через разность логарифмов. Все формулы, **44 конкретных имени**, выбранные автоматические признаки и шесть добавлений выделены в [приложении с каталогом признаков](#13-каталог-признаков).

### Код определения и расчёта признаков

Код ниже взят из `features.py`. Для его чтения нужны стандартные импорты:

```python
from dataclasses import dataclass, asdict, replace
import hashlib
import itertools
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
```

`Feature` описывает один признак: связь, операцию и окно. `RELATIONS` определяет ключи группировки. Например, связь `pair` группирует покупки одновременно по клиенту и активу. `OLD` — пять исходных счётчиков, с которыми сравнивается расширенная база.

```python
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
```

В полном классе `Builder` реализованы все операции. `context` ограничивает историю условием `timestamp < cutoff`; `column` дополнительно задаёт левую границу окна. `groupby(...).size()` считает покупки, `nunique()` — разные сущности, `timestamp.max()` — последнюю покупку, `np.divide` — долю. `np.log1p(x)` означает `ln(1 + x)`, поэтому ветка `trend` буквально реализует формулу из каталога. `matrix` объединяет значения в матрицу «строки × признаки».

```python
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
```

Для доли `pair__share__90d` числитель — `pair__count__90d`, знаменатель — `user__count__90d`. Условие `where=denominator > 0` предотвращает деление на ноль. Код 181 для давности обозначает отсутствие покупки в 180-дневной истории. Кэш сохраняет уже рассчитанные столбцы и не добавляет будущие события.

### Реальный пример расчёта признаков

Кейс A выбран детерминированно: первый по месяцу и идентификатору клиент с не менее чем тремя положительными активами внутри каталога. Качество прогноза не участвует в выборе примера. Момент прогноза — 2022-07-01, рассматриваемый актив — GRS003003035.

| Признак | Фактическое значение |
|---|---|
| `user__count__90d` | 15,000000 |
| `pair__count__90d` | 2,000000 |
| `pair__share__90d` | 0,133333 |
| `pair__recency__180d` | 9,000000 |
| `item__count__30d` | 354,000000 |
| `user__unique__90d` | 8,000000 |

Доля покупок пары за 90 дней равна 2 / 15 = 0,133333. Последняя покупка этой пары — 2022-06-22; давность = (2022-07-01 − эта дата) = 9 дней. Приведённые значения — реальные расчёты из источника; это описание входов модели, а не оценка причинной важности отдельного признака.

## 5. Бустинг и настройка

Используется HistGradientBoostingClassifier — ансамбль деревьев, в котором новые итерации корректируют предыдущие предсказания. Число листьев ограничивает сложность дерева, learning_rate задаёт вклад новой итерации, min_samples_leaf и L2 контролируют регуляризацию. Обучение минимизирует бинарный LogLoss; выбор гиперпараметров ориентирован на Recall@5. Это разные уровни: функция обучения вероятностей и критерий качества итогового top-5. Все модели проверяют один список 16 конфигураций пяти параметров, seed=42; случайный early stopping выключен.

Полный пул, из которого детерминированно выбраны конфигурации: learning_rate {0,05; 0,1; 0,2}, max_iter {100; 200; 300}, max_leaf_nodes {7; 15; 31}, min_samples_leaf {20; 60}, L2 {0; 1}. Исходные 100/7 и 200/7 с lr=0,1 включены явно. Это ограниченный подбор, а не доказательство оптимальности во всём пространстве. Список попыток — в protocol.json и trials.jsonl.

Правило выбора: максимальный внутренний Recall, затем меньший LogLoss, затем меньше деревьев и листьев. Выбранные параметры:

| Модель | Признаков | learning_rate | max_iter | max_leaf_nodes | min_samples_leaf | L2 |
|---|---|---|---|---|---|---|
| Пять счётчиков, новый подбор | 5 | 0.05 | 100 | 7 | 20 | 0.0 |
| Ручной бустинг | 44 | 0.05 | 100 | 7 | 20 | 0.0 |
| Все исходные автоматические | 102 | 0.05 | 100 | 7 | 60 | 0.0 |
| Автоматический отбор | 28 | 0.1 | 100 | 7 | 20 | 0.0 |
| Ручные + автоматические | 50 | 0.05 | 100 | 7 | 20 | 0.0 |

Идентификаторы, дата и ответ не подаются модели как признаки. Все методы используют одинаковые пары и определения метрик. При равных оценках порядок задан BLAKE2b-хешем ключа (seed, клиент, дата, актив); он не зависит от ответа и порядка строк.

### Код настройки и получения вероятностей

`grid` составляет один и тот же список 16 конфигураций. Метод `cv` обучает бустинг отдельно на каждом временном блоке, получает вероятность положительного класса через `predict_proba(... )[:, 1]` и усредняет Recall с весом числа учитываемых клиент-месяцев. Поэтому это среднее по случаям, а не по трём процентам. Код деревьев и оптимизации LogLoss используется из scikit-learn.

```python
REFERENCE = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=7,
                 min_samples_leaf=20, l2_regularization=0.0)

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
```

После выбора параметров `freeze` обучает финальную модель только до января 2022, рассчитывает диагностическую validation и сохраняет веса. `specs` — определения признаков; `params` — выбранные параметры. Эти методы относятся к классу `Experiment`, конструктор которого показан в разделе 3.

```python
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
```

## 6. Автоматизация

Идея — составлять признаки как «связь × операция × временное окно». Допустимы шесть связей: user, item, pair, category, subcategory, sector; операции count, unique, share, recency, trend. Начальные окна 7, 14, 30, 60, 90, 180 дней дают 102 столбца. Это собственный ограниченный генератор, использующий принцип агрегации по отношениям, родственный [Deep Feature Synthesis, Kanter & Veeramachaneni, 2015](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf). Полный DFS не реализован, теоретическая новизна этого принципа не заявляется.

1. При фиксированном бустинге 100/7, lr=0,1 проверяются семейства связей.
2. Выбирается лучшее первое семейство, затем последовательно добавляются другие, только если внутренний Recall растёт более чем на $10^{-6}$.
3. Для выбранных семейств исключается каждое окно; окно с наибольшим падением качества проверяется заменой соседними значениями. Если оба соседних варианта уже представлены или выходят за 1–180 дней, проверяется промежуточное окно между ближайшими исходными значениями.
4. По выбранному набору настраивается бустинг по тем же 16 конфигурациям.
5. Отдельно проверяется добавление выбранных семейств к зафиксированной ручной базе.

Человек определяет грамматику операций и связей; программа создаёт конкретные столбцы, выбирает семейства и проверяет окна. Отбор групп не равен индивидуальному отбору каждого столбца. Это локальный поиск, не исчерпывающий перебор всех целых окон или всех возможных комбинаций.

| Шаг | Внутренний Recall, % | Решение |
|---|---|---|
| Только pair | 53,42 | Первое выбранное семейство |
| Добавление item | 59,79 | Принято; итог 28 признаков |
| Замена pair 180 → 135 дней | 58,58 | Отклонено |
| Замена item 180 → 135 дней | 59,66 | Отклонено |
| Ручная база, фиксированные параметры | 59,03 | Контроль |
| Ручная база + item | 59,63 | Принято; итог 50 признаков |

Контроль «все автоматические» использует исходные 102 столбца. Уточнения не были приняты. Отдельное сравнение фиксированных и настроенных параметров позволяет видеть вклад представления и вклад настройки:

| Модель | Внутренний Recall при 100/7, lr=0,1, % | После подбора, % | Январь–июнь, % |
|---|---|---|---|
| Пять счётчиков, новый подбор | 54,56 | 54,62 | 62,17 |
| Ручной бустинг | 59,03 | 59,97 | 65,19 |
| Все исходные автоматические | 59,50 | 60,19 | 64,70 |
| Автоматический отбор | 59,79 | 59,79 | 62,25 |
| Ручные + автоматические | 59,63 | 60,32 | 64,66 |

### Код генератора, отбора и поиска окон

`generated_features` создаёт комбинации связей, операций и окон. Это генерация программой внутри заданной грамматики.

```python
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
```

В алгоритме используются два небольших вспомогательных определения: `unique` убирает одинаковые определения признаков, `replace` создаёт копию определения с новым окном.

```python
def unique(specs):
    return list(dict.fromkeys(specs))
```

Метод `select` выбирает лучшее семейство, затем принимает только добавления с приростом внутреннего Recall больше `1e-6`. При непустой `base` поиск добавляет признаки к уже заданному набору; при пустой — строит автоматическую модель с нуля.

```python
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
```

`refine` оценивает исключение каждого окна и пробует изменить самое полезное. `chosen` — исходное окно, `proposals` — новые значения. Запись `accepted` появляется только при улучшении внутренней метрики; отклонённые варианты также остаются в журнале.

```python
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
```

## 7. Вклад ручных признаков

В полной ручной модели фиксируются выбранные параметры, затем по одному исключаются типы признаков. Модель каждого варианта заново обучается только на соответствующих внутренних train-блоках.

| Исключённая группа | Внутренний Recall, % | Разность относительно полного набора, п.п. |
|---|---|---|
| recency | 59,49 | -0.48 |
| trend | 60,22 | +0.25 |
| category_relations | 59,75 | -0.21 |
| unique | 60,37 | +0.40 |
| share | 59,78 | -0.19 |

Исключение recency, долей или категориальных связей снижает внутренний Recall; исключение trend и unique немного повышает его. Следовательно, увеличение числа признаков само по себе не является улучшением. Эти результаты диагностические: группы пересекаются, взаимодействуют, а интервалы разности здесь не рассчитаны. Полная ручная база была зафиксирована протоколом; диагностика не превращена в дополнительный выбор по финальному периоду.

## 8. Метрики и расчёты

Пусть $R_{u,t}$ — множество фактически купленных внутри каталога активов, $L_{u,t}$ — пять рекомендованных активов, $h=|R\cap L|$. Основная популяция $E$ включает только клиент-месяцы с $|R|>0$.

$$
\mathrm{Recall@5}=\frac{1}{|E|}\sum_{(u,t)\in E}\frac{h_{u,t}}{|R_{u,t}|},\qquad
\mathrm{Precision@5}=\frac{1}{|E|}\sum_{(u,t)\in E}\frac{h_{u,t}}{5}.
$$

Precision_all использует все 2000 клиент-месяцев, включая месяцы без покупки. Основной Recall — **среднее долей по случаям**, а не отношение всех попаданий ко всем покупкам.

Для бинарной релевантности $y_j$ на позиции $j$:

$$
\mathrm{DCG@5}=\sum_{j=1}^5\frac{y_j}{\log_2(j+1)},\qquad
\mathrm{IDCG@5}=\sum_{j=1}^{\min(|R|,5)}\frac{1}{\log_2(j+1)},\qquad
\mathrm{NDCG@5}=\frac{\mathrm{DCG@5}}{\mathrm{IDCG@5}}.
$$

NDCG усредняется по $E$: она учитывает и наличие попаданий, и их порядок. Пары без положительных активов не входят в это среднее.

$$
\mathrm{LogLoss}=-\frac{1}{N}\sum_{n=1}^N\left[y_n\ln p_n+(1-y_n)\ln(1-p_n)\right].
$$

LogLoss считается по всем парам; меньшая ошибка лучше. Вероятности ограничиваются машинным epsilon для устойчивости. Глобальный ROC-AUC — вероятность, что случайная положительная пара выше отрицательной, плюс половина вероятности равенства. При одном классе AUC не определён. Глобальный AUC не заменяет ранжирование внутри клиента. Для правил LogLoss не определён без вероятностной калибровки. ROC-AUC для этих контролей не рассчитывался: они включены для сравнения top-5. Сам по себе AUC может вычисляться по ранговым оценкам без вероятностей.

### Код ранжирования и метрик

Ниже приведён полный класс `Ranking` из `metrics.py`. `group` обозначает клиент-месяц, `self.total` — число купленных в каталоге активов. В `cases` выбираются первые пять позиций, `hits` считает попадания, а деления реализуют Recall, Precision и NDCG. `self.idcg` — идеальный DCG. NumPy нумерует позиции с нуля, поэтому `log2(rank + 2)` соответствует `log2(j + 1)` в формуле с нумерацией с единицы.

```python
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
```

`evaluate` усредняет Recall/NDCG по ненулевым покупкам: остальные случаи имеют `NaN` и пропускаются в `mean`. Precision_all включает все случаи. LogLoss и глобальный AUC вызываются из scikit-learn.

Чтобы проверка не зависела только от этих вызовов, LogLoss и AUC также пересчитываются явно. Это действующий фрагмент независимой проверки: `frame` содержит ответы, `saved[name]` — сохранённые вероятности одной модели.

```python
y = frame.label.to_numpy()
probabilities = saved[name].to_numpy()
clipped = np.clip(probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps)
independent_loss = float(-(y * np.log(clipped) + (1-y) * np.log1p(-clipped)).mean())
n_positive = int(y.sum())
ranks = pd.Series(probabilities).rank(method="average").to_numpy()
independent_auc = float((ranks[y == 1].sum() - n_positive*(n_positive+1)/2) /
                        (n_positive*(len(y)-n_positive)))
```

`independent_loss` буквально повторяет формулу LogLoss. Для AUC используются средние ранги одинаковых вероятностей: сумма рангов положительных примеров за вычетом `n_positive*(n_positive+1)/2` делится на число положительно-отрицательных пар. Этот фрагмент выполняется для пяти ML-моделей, у которых оба класса присутствуют.

Отдельный расчёт ranking-метрик использует обычный цикл и пересечение множеств. Функция ниже из `verify.py` проверяет ту же формулу другим способом; правила равных оценок сохраняются.

```python
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
```

### Реальный top-5 и расчёт одного случая

Для кейса A модель даёт следующий выход. Здесь «рекомендация» означает выход экспериментального алгоритма ранжирования по покупке, а не проверенную инвестиционную пригодность.

| Позиция | ISIN | Оценка покупки | Факт покупки в месяце |
|---|---|---|---|
| 1 | GRS003003035 | 0,523439 | 1 |
| 2 | GRS014003032 | 0,519113 | 1 |
| 3 | GRS282183003 | 0,372730 | 0 |
| 4 | GRS015003007 | 0,181047 | 0 |
| 5 | GRS323003012 | 0,181047 | 0 |

В каталоге фактически куплены 3 актива, найдено 2. Поэтому Recall = 2/3 = 66,67%, Precision = 2/5 = 40,00%.

DCG = 1,630930, IDCG = 2,130930, NDCG = 1,630930/2,130930 = 0,765361. Положительные активы на первых двух позициях дают $1+1/\log_2 3$ в DCG; третий купленный актив в top-5 не попал.

Для положительной пары с $p=0,523439$ вклад в LogLoss = $-\ln p=0,647335$. Для отрицательной пары с $p=0,372730$ вклад = $-\ln(1-p)=0,466379$. Общий LogLoss — среднее по 100000 парам, а не по двум примерам.

### Расчёт агрегатов ручной модели

Получено 298 попаданий в $2000\cdot5=10000$ рекомендациях. Precision_all = $298/10000=2,98$%. Если считать долю найденных покупок суммарно, получится $298/508=58,66$% — это дополнительный micro-Recall, **не основная метрика**. Основной macro-Recall = среднее 236 индивидуальных долей = 65,53%. Эти величины различаются из-за разного числа покупок у клиентов.

Выбор примера, номера строк CSV, формулы, численные значения и хеши входов сохранены в [report_calculations.json](results/evidence/report_calculations.json). Документы собираются скриптом без обучения модели.

## 9. Результаты и неопределённость

### Общая оценка июля–октября

| Метод | Recall@5, % | NDCG@5 | Precision@5, % | Precision_all, % | LogLoss ↓ | ROC-AUC |
|---|---|---|---|---|---|---|
| Пять счётчиков, новый подбор | 59,18 | 0,5100 | 22,54 | 2,66 | 0,019692 | 0,9371 |
| Ручной бустинг | 65,53 | 0,5614 | 25,25 | 2,98 | 0,018243 | 0,9532 |
| Все исходные автоматические | 63,68 | 0,5589 | 24,75 | 2,92 | 0,018388 | 0,9536 |
| Автоматический отбор | 64,32 | 0,5635 | 24,75 | 2,92 | 0,019283 | 0,9259 |
| Ручные + автоматические | 64,39 | 0,5636 | 24,83 | 2,93 | 0,018321 | 0,9533 |
| Повторные покупки + популярность | 61,71 | 0,5303 | 23,56 | 2,78 | — | — |
| Популярность | 44,05 | 0,3217 | 16,61 | 1,96 | — | — |

Пять счётчиков настроены заново на внутренних блоках; это другая модель, чем исходная модель 200/7 из прежнего аудита. У сравниваемых новых моделей с 5 и 44 признаками параметры одинаковы. Наблюдаемый вклад расширения: $100(0.655308533-0.591758904)=6.3550$ п.п. Отдельный подтверждающий интервал этой разности не рассчитан.

### Парные клиентские интервалы

| Сравнение A − B | Разность Recall, п.п. | 95% интервал, п.п. |
|---|---|---|
| Ручные + автоматические − Ручной бустинг | -1.14 | [-2.92; +0.42] |
| Автоматический отбор − Ручной бустинг | -1.21 | [-3.84; +1.31] |
| Ручной бустинг − Повторные покупки + популярность | +3.82 | [+0.17; +7.51] |
| Ручные + автоматические − Повторные покупки + популярность | +2.69 | [-1.25; +6.62] |

Для каждой из 5000 выборок с seed=20261013 выбираются 500 клиентов с возвращением. Если клиент выбран несколько раз, все его месяцы повторяются с той же кратностью. Разность двух моделей считается на одних и тех же случаях:

$$
\Delta_b=\frac{\sum_u m_{u,b}\sum_{t:(u,t)\in E}(\mathrm{Recall}^A_{u,t}-\mathrm{Recall}^B_{u,t})}{\sum_u m_{u,b}\,n_u},\qquad
\mathrm{CI}_{95\%}=[Q_{0.025}(\Delta_b),Q_{0.975}(\Delta_b)].
$$

Здесь $m_{u,b}$ — кратность выбора, $n_u$ — число учитываемых месяцев клиента. Пустых выборок не было. Интервал описывает вариацию клиентов при фиксированных моделях и периоде, а не новые рыночные режимы, переобучение или неопределённость отбора.

### Код парного клиентского bootstrap

`a` и `b` — таблицы показателей двух моделей на одинаковых клиент-месяцах; `users` — все 500 клиентов. `draws` содержит 5000 выборок клиентов с возвращением. Индексация повторяет вклад клиента столько раз, сколько он выбран; сумма `n` задаёт соответствующий знаменатель. Затем берутся квантили 2,5% и 97,5%.

```python
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
```

Основной интервал включает ноль: прирост автоматизации по Recall@5 не подтверждён, эквивалентность также не установлена. Ручной вариант выше правила по наблюдаемой разности; его диагностический интервал положителен. С учётом раскрытого периода это не независимое подтверждение будущего эффекта. Остальные сравнения диагностические, коррекция множественных проверок не применялась.

### Помесячное качество

| Месяц | Ручной Recall, % | Автоотбор, % | Объединённый, % | Правило, % | Случаев |
|---|---|---|---|---|---|
| 2022-07-01 | 68,68 | 64,73 | 68,68 | 63,74 | 72 |
| 2022-08-01 | 75,08 | 74,16 | 70,83 | 75,49 | 49 |
| 2022-09-01 | 61,30 | 63,47 | 61,87 | 59,11 | 71 |
| 2022-10-01 | 56,57 | 54,04 | 54,29 | 47,22 | 44 |

Октябрь слабее предыдущих месяцев у всех методов. Нельзя приписывать это доказанному рыночному режиму: диагноз причины не проводился. Покрытие каталога меняется по месяцам:

| Месяц | Покупки внутри каталога | Все покупки активов | Покрытие, % |
|---|---|---|---|
| 2022-07-01 | 164 | 220 | 74,55 |
| 2022-08-01 | 95 | 117 | 81,20 |
| 2022-09-01 | 166 | 208 | 79,81 |
| 2022-10-01 | 83 | 133 | 62,41 |

### Новые и повторные покупки

| Срез | Положительных пар | Случаев | Ручной Recall, % | Автоотбор, % | Объединённый, % | Правило, % |
|---|---|---|---|---|---|---|
| repeat_3m | 280 | 134 | 90,16 | 88,99 | 87,78 | 89,72 |
| new_3m | 228 | 149 | 40,51 | 39,74 | 40,51 | 32,57 |

Срез repeat_3m означает наличие покупки пары за три предыдущих календарных месяца; new_3m — её отсутствие. Это не признак отсутствия взаимодействий за всю жизнь. В каждом срезе меняется только множество положительных ответов для оценки, исходный top-5 не перестраивается. Случаи с покупками обоих типов входят в оба среза, поэтому их нельзя усреднить для получения общего Recall. Групповые интервалы не рассчитаны.

Числа таблиц берутся из [evaluation.json](results/final/evaluation.json), [manual_stage.json](results/final/manual_stage.json), [automatic_stage.json](results/final/automatic_stage.json).

## 10. Затраты и проверка

| Вариант | Генерация 27 месяцев, с | Генерация 4 месяцев, с | predict 100000 пар, с | Пиковая память, MiB |
|---|---|---|---|---|
| Пять счётчиков, новый подбор | 0,61 | 0,07 | 0,029 | 362,5 |
| Ручной бустинг | 5,66 | 0,71 | 0,052 | 757,6 |
| Все исходные автоматические | 10,48 | 1,35 | 0,144 | 1192,3 |
| Автоматический отбор | 2,85 | 0,36 | 0,030 | 593,7 |
| Ручные + автоматические | 5,85 | 0,73 | 0,051 | 813,0 |

Каждый вариант измерялся в отдельном процессе с новым кэшем агрегатов и четырьмя потоками модели. Пиковая память относится ко всему процессу; кэш ОС и порядок запусков не контролировались. Это профилирование, не строгий статистический benchmark.

Ручной этап: 113 fit, 511,63 с измеренного обучения. Автоматический: 240 fit, 1263,99 с. Это включает внутренние сравнения, подбор и финальное обучение. Общий вычислительный лимит заданием не требовался; затраты измерены как дополнительная характеристика. Время ручного придумывания признаков не измерено, поэтому общая экономия разработки не доказана.

Первый технически неудачный запуск выполнил три fit до исправления JSON-сериализации. Первый автоматический поиск остановлен из-за пустого уточнения у границы 180 дней; сохранены 99 завершённых fit и не полностью записанная попытка внутри следующего CV (до трёх дополнительных fit). После добавления промежуточного окна поиск повторён до финальной оценки. Успешные этапы таблиц не включают эти дополнительные затраты; история — [execution_notes.json](results/final/execution_notes.json) и provenance.

Проверки: 8 тестов; независимый пересчёт ranking-метрик всех семи методов; LogLoss по формуле и AUC через ранги; точное воспроизведение вероятностей пяти моделей; 200 отдельных расчётов признаков; инвариантность проверенных признаков к будущим событиям; независимость ранжирования от ответов и порядка строк; явная проверка кратности клиентов в десяти bootstrap-выборках; совпадение CV-метрик 33 повторённых конфигураций. Финальная фаза запрещает fit, хеши моделей до/после совпадают. Максимальная ошибка независимых ranking-метрик — порядка $10^{-16}$, вероятностей — 0. Подтверждение: [verification.json](results/final/verification.json).

### Код проверки сохранённых вероятностей

Фрагмент из `verify.py` повторно строит признаки по исходным данным и сравнивает вероятности сохранённой модели с опубликованным CSV. `err == 0` проверяет точное совпадение. Следующая перестановка проверяет, что результат не меняется от порядка строк после возвращения к исходному порядку. Здесь выполняется только прогноз, обучение не вызывается.

```python
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
```

## 11. Рекомендации и ограничения

**Выбор текущего решения:** ручной бустинг с 44 признаками имеет лучший наблюдаемый Recall@5 и LogLoss из сравниваемых ML-вариантов. Его следует представлять как основной результат ручного этапа. Автоматический отбор — отдельный исследованный метод с 28 признаками и меньшими измеренными затратами генерации, но без доказанного превосходства или эквивалентности по Recall.

Отобранный и объединённый варианты немного выше по NDCG; интервал этой разности не рассчитан. Поэтому нельзя заявлять подтверждённое преимущество по этой дополнительной метрике. Методическая ценность автоматизации — воспроизводимые генерация, отбор и журнал экспериментов, включая неудачные варианты.

Следующие эксперименты ещё не выполнены. Их приоритет определяется ограничениями текущего результата:

| Приоритет | Основание | Следующий эксперимент | Критерий проверки |
|---|---|---|---|
| Первый | Итоговый период ранее раскрывался | Новый закрытый период или независимый датасет | Основное сравнение и метрику фиксировать заранее; одна финальная оценка |
| Второй | Ноль смешивает неизвестную категорию с отсутствием покупок | Индикаторы доступности категорий и сектора | Проверить на внутренних временных блоках |
| Второй | Исключение trend и unique немного повышало внутренний Recall | Отбор отдельных групп и окон с повторной настройкой | Разделить выбор и оценку; проверить устойчивость по блокам |
| Второй | Семейства выбираются при одних параметрах, а итоговый бустинг настраивается отдельно | Совместный или чередующийся поиск признаков и параметров | Вложенные временные проверки, одинаковые условия сравнения |
| Третий | Каталог покрывает 74,93% покупок | Расширенный каталог и проверка доступности активов | Одновременно оценивать покрытие, ранжирование и затраты |
| Третий | Исследованы прежние покупатели и популярные активы | Отдельные сценарии новых клиентов и активов | Явно определить отсутствие истории; new_3m не считать полным cold-start |
| Третий | Замеры времени единичные | Повторное профилирование в одинаковом окружении | Свежие процессы, несколько повторов, контроль порядка и потоков |

Для ускорения расчёта целесообразно вычислять только признаки выбранной модели и переиспользовать агрегаты по дате, связи и окну. Кэш уже реализован. Изменения реализации нужно проверять на совпадение признаков и предсказаний; расширение метода поиска является новым экспериментом.

Ограничения: раскрытый финальный период, ретроспективное происхождение части событий, 500 прежних покупателей и 50 исторически популярных активов, отсутствие показов, смешение неизвестных категорий с нулевой историей, ограниченный подбор и локальный поиск. Предсказание поведения не устанавливает финансовую полезность, доходность или пригодность продукта клиенту. Абсолютное отсутствие утечки в исходнике не доказано; проверена корректность временных фильтров реализации.

## 12. Воспроизведение и источники

Все формулы, реализации, примеры и выводы для оценки содержатся в настоящем документе. Дополнительные выполненные notebooks: [ручной этап](notebooks/01_manual_baseline.ipynb), [автоматизация](notebooks/02_feature_automation.ipynb), [сравнение](notebooks/03_experiments.ipynb). Обучение выполнено фазами CLI; notebooks читают журналы и пересчитывают результаты. Полное повторное обучение запускается отдельно либо через RUN_TRAINING с новой папкой результатов.

Окружение исходного эксперимента: Python 3.15.0, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1. Остальные версии перечислены в [environment/requirements-lock.txt](environment/requirements-lock.txt). Публичный архив сохранён в data/raw; источник, лицензия и контрольные суммы — в [data/manifest.json](data/manifest.json). Подготовка проверяет архив и оба используемых CSV по SHA-256.

### Установка и подготовка данных

Репозиторий приватный: для clone необходим предоставленный доступ. Команды выполняются из его корня.

```bash
git clone --branch main https://github.com/zhivikhin2015-blip/t_bank_ml1.git
cd t_bank_ml1
python3.15 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements-lock.txt
python -m pip install -e . --no-deps
python -m research.feature_automation.prepare_data --out data/processed
```

Папка назначения данных должна быть новой. Никакие файлы из прежних черновиков для запуска не требуются.

### Проверка готового результата без обучения

```bash
python -m unittest discover -s tests -v
mkdir -p .tmp
cp -R results/final .tmp/verification
python -m research.feature_automation.verify \
  --data-dir data/processed --out .tmp/verification
```

Проверка записывает verification.json только в копию результатов. Она пересчитывает метрики, предсказания, примеры признаков и проверяет контрольные суммы. Сохранённые модели следует загружать в совместимом окружении.

### Полное повторное обучение

Для нового запуска задаётся новая папка результатов. Веса опубликованных моделей сохраняются.

```bash
python -m research.feature_automation.experiment \
  --data-dir data/processed --out results/runs/reproduction --phase manual
python -m research.feature_automation.experiment \
  --data-dir data/processed --out results/runs/reproduction --phase automatic
python -m research.feature_automation.experiment \
  --data-dir data/processed --out results/runs/reproduction --phase evaluate
python -m research.feature_automation.verify \
  --data-dir data/processed --out results/runs/reproduction
python -m research.feature_automation.benchmark \
  --data-dir data/processed --out results/runs/reproduction
```

manual фиксирует протокол и ручную модель; automatic выполняет поиск после ручного этапа; evaluate использует сохранённые веса и запрещает fit. Успешные исходные этапы включают 113 и 240 обучений; дополнительные технические запуски отражены в execution_notes.json. Время, память и отметки времени могут отличаться. Повторная оценка тех же месяцев не превращает их в новый независимый тест.

### Повторная сборка отчёта

```bash
python scripts/build_report.py \
  --data-dir data/processed --destination .tmp/report
```

Сборщик использует фиксированные results/final, проверяет входные данные, пересчитывает реальные примеры и формирует настоящий документ, CSV-каталог и JSON расчётов. Он не обучает модели и не выбирает параметры.

### Карта доказательств

| Артефакт | Назначение |
|---|---|
| results/final/protocol.json | Разбиения, критерии выбора, пространство поиска и хеши |
| results/final/manual_stage.json, automatic_stage.json | Настройка, отбор, абляции и validation |
| results/final/trials.jsonl, *_trials.json | Фактически проверенные варианты и кэшированные обращения |
| results/final/evaluation.json | Итоговые метрики, месяцы, срезы, интервалы |
| results/final/predictions.csv.gz | Все 100000 кандидатных пар, ответы и оценки |
| results/final/*.joblib | Пять зафиксированных моделей и определения признаков |
| results/final/verification.json | Независимые проверки результата |
| results/final/benchmark.json | Профилирование в отдельных процессах |
| results/final/provenance/ | Исходный код и протокол до технического исправления поиска |
| results/history/prior_evaluation.json | Документированная прежняя оценка того же периода |
| results/evidence/report_calculations.json | Реальные примеры, знаменатели, номера строк и хеши |
| results/evidence/feature_catalog.csv | Все 107 уникальных определений и принадлежность моделям |

### Структура проекта

Исходники отделены от сырых данных, подготовленных таблиц и результатов. В src сохранено имя модуля research.feature_automation, необходимое для загрузки зафиксированных моделей. Новые результаты и подготовленные данные исключены из Git.

```text
REPORT.md                  единый документ для оценки
README.md                  краткая точка входа
src/research/feature_automation/  признаки, обучение, метрики, проверки
tests/                     обнаружение тестов исходного эксперимента
notebooks/                 три выполненных этапа исследования
data/raw/                  неизменяемый публичный архив и его описание
data/manifest.json         происхождение, лицензия, контрольные суммы
results/final/             фиксированные модели и результаты
results/history/           свидетельство прежней оценки периода
results/evidence/          каталог и проверяемые расчёты отчёта
scripts/build_report.py    сборка документа без обучения
environment/               точные версии окружения
pyproject.toml             установка пакета и команды CLI
```

### Первичные источники

1. [FAR-Trans, University of Glasgow, DOI 10.5525/gla.researchdata.1658](https://researchdata.gla.ac.uk/1658/): источник данных и лицензия CC BY 4.0.
2. [Sanz-Cruzado, Droukas, McCreadie, 2024](https://arxiv.org/html/2407.08692v1): происхождение и обработка FAR-Trans. Наш месячный протокол с повторными покупками отличается от протокола статьи; численные результаты напрямую не сравниваются.
3. [Kanter & Veeramachaneni, 2015, Deep Feature Synthesis](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf): генерация признаков по отношениям.
4. [Cawley & Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html): оптимизм при выборе модели.
5. [HistGradientBoostingClassifier, официальная документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html): используемый класс бустинга.

## 13. Каталог признаков

Признаки вычисляются из покупок строго до даты прогноза. Идентификаторы клиента и актива используются для агрегаций и соединений, но не подаются бустингу отдельными столбцами.

### 13.1. Сколько признаков используется

| Связь | Ручных | Генерируемых | Отобранных автоматических |
|---|---|---|---|
| Клиент | 10 | 14 | 0 |
| Актив | 9 | 14 | 14 |
| Клиент–актив | 10 | 14 | 14 |
| Клиент–категория | 5 | 20 | 0 |
| Клиент–подкатегория | 5 | 20 | 0 |
| Клиент–сектор | 5 | 20 | 0 |

Ручная база содержит 44 столбца, генератор — 102, отобранная автоматическая модель — 28, объединённая — 50. Между ручной и генерируемой базами совпадают 39 столбцов: это **107 уникальных определений**, а не 146 независимых признаков. Отбор выполняется по семействам связей, поэтому выбранное семейство сохраняет несколько коррелирующих окон.

### 13.2. Формулы и обработка отсутствующей истории

Пусть $H_r(t,w)$ — покупки по связи $r$ в полуоткрытом окне $[t-w,t)$. Для категориальных связей используется классификация, известная на $t$.

| Операция | Формула / правило | Гипотеза |
|---|---|---|
| count | $C_r(t,w)=\operatorname{card}(H_r(t,w))$ | Активность и повторное потребление |
| unique | Число разных активов; для item — разных покупателей | Разнообразие интересов или широта спроса |
| share | $C_r(t,w)/C_{user}(t,w)$; при нулевом знаменателе 0 | Относительное предпочтение |
| recency | Дни после последней покупки в 180-дневной истории; 181, если её нет | Недавнее поведение информативно |
| trend | $\ln(1+C_r(t,30))-\ln(1+C_r(t,60)-C_r(t,30))$ | Изменение двух соседних 30-дневных окон |

`1m` и `3m` — календарные месяцы; `30d` — ровно 30 дней. Они могут давать разные значения. Для recency код `180d` обозначает максимальную глубину истории, для trend `30d` — длину каждого из двух сравниваемых окон.

Покупки клиента включают активы вне каталога; популярность актива — покупки всех клиентов, а не только выбранных 500. Повторные сделки считаются отдельно в count, но дают один положительный ответ за месяц.

Временная версия метаданных выбирается строго до $t$. При конфликте значений одной версии поле становится неизвестным. Неизвестные категории не объединяются в общую категорию; их агрегаты равны нулю. Ноль поэтому частично смешивает отсутствие истории и отсутствие информации.

### 13.3. Все 44 ручных признака

| № | Имя в коде | Смысл |
|---|---|---|
| 1 | `user__count__1m` | Число покупок; клиент; 1 календ. мес. |
| 2 | `pair__count__1m` | Число покупок; клиент–актив; 1 календ. мес. |
| 3 | `item__count__1m` | Число покупок; актив; 1 календ. мес. |
| 4 | `user__count__3m` | Число покупок; клиент; 3 календ. мес. |
| 5 | `pair__count__3m` | Число покупок; клиент–актив; 3 календ. мес. |
| 6 | `user__count__30d` | Число покупок; клиент; 30 дней |
| 7 | `item__count__30d` | Число покупок; актив; 30 дней |
| 8 | `pair__count__30d` | Число покупок; клиент–актив; 30 дней |
| 9 | `user__unique__30d` | Число разных активов; клиент; 30 дней |
| 10 | `item__unique__30d` | Число разных покупателей; актив; 30 дней |
| 11 | `pair__share__30d` | Доля покупок; клиент–актив; 30 дней |
| 12 | `user__count__90d` | Число покупок; клиент; 90 дней |
| 13 | `item__count__90d` | Число покупок; актив; 90 дней |
| 14 | `pair__count__90d` | Число покупок; клиент–актив; 90 дней |
| 15 | `user__unique__90d` | Число разных активов; клиент; 90 дней |
| 16 | `item__unique__90d` | Число разных покупателей; актив; 90 дней |
| 17 | `pair__share__90d` | Доля покупок; клиент–актив; 90 дней |
| 18 | `user__count__180d` | Число покупок; клиент; 180 дней |
| 19 | `item__count__180d` | Число покупок; актив; 180 дней |
| 20 | `pair__count__180d` | Число покупок; клиент–актив; 180 дней |
| 21 | `user__unique__180d` | Число разных активов; клиент; 180 дней |
| 22 | `item__unique__180d` | Число разных покупателей; актив; 180 дней |
| 23 | `pair__share__180d` | Доля покупок; клиент–актив; 180 дней |
| 24 | `user__recency__180d` | Давность покупки; клиент; до 180 дней |
| 25 | `user__trend__30d` | Изменение активности; клиент; последние 30 / предыдущие 30 дней |
| 26 | `item__recency__180d` | Давность покупки; актив; до 180 дней |
| 27 | `item__trend__30d` | Изменение активности; актив; последние 30 / предыдущие 30 дней |
| 28 | `pair__recency__180d` | Давность покупки; клиент–актив; до 180 дней |
| 29 | `pair__trend__30d` | Изменение активности; клиент–актив; последние 30 / предыдущие 30 дней |
| 30 | `category__recency__180d` | Давность покупки; клиент–категория; до 180 дней |
| 31 | `category__trend__30d` | Изменение активности; клиент–категория; последние 30 / предыдущие 30 дней |
| 32 | `subcategory__recency__180d` | Давность покупки; клиент–подкатегория; до 180 дней |
| 33 | `subcategory__trend__30d` | Изменение активности; клиент–подкатегория; последние 30 / предыдущие 30 дней |
| 34 | `sector__recency__180d` | Давность покупки; клиент–сектор; до 180 дней |
| 35 | `sector__trend__30d` | Изменение активности; клиент–сектор; последние 30 / предыдущие 30 дней |
| 36 | `category__count__90d` | Число покупок; клиент–категория; 90 дней |
| 37 | `category__share__90d` | Доля покупок; клиент–категория; 90 дней |
| 38 | `category__unique__90d` | Число разных активов; клиент–категория; 90 дней |
| 39 | `subcategory__count__90d` | Число покупок; клиент–подкатегория; 90 дней |
| 40 | `subcategory__share__90d` | Доля покупок; клиент–подкатегория; 90 дней |
| 41 | `subcategory__unique__90d` | Число разных активов; клиент–подкатегория; 90 дней |
| 42 | `sector__count__90d` | Число покупок; клиент–сектор; 90 дней |
| 43 | `sector__share__90d` | Доля покупок; клиент–сектор; 90 дней |
| 44 | `sector__unique__90d` | Число разных активов; клиент–сектор; 90 дней |

### 13.4. Грамматика 102 автоматических кандидатов

Для user и item: count и unique в шести окнах — $2\times6+2=14$ признаков с учётом recency и trend. Для pair: count и share — также 14. Для category, subcategory, sector: count, unique и share — $3\times6+2=20$ для каждой связи. Всего $14+14+14+20+20+20=102$.

Начальные окна: 7, 14, 30, 60, 90, 180 дней. Уточнение проверило замену 180 на 135 дней в выбранных семействах; обе замены отклонены. Начальные 102 столбца и уточнения различаются: контроль без отбора использует только исходный набор.

### 13.5. Какие 28 признаков выбрал алгоритм

| Имя | Смысл | Есть в ручной базе |
|---|---|---|
| `item__count__7d` | Число покупок; актив; 7 дней | нет |
| `item__unique__7d` | Число разных покупателей; актив; 7 дней | нет |
| `item__count__14d` | Число покупок; актив; 14 дней | нет |
| `item__unique__14d` | Число разных покупателей; актив; 14 дней | нет |
| `item__count__30d` | Число покупок; актив; 30 дней | да |
| `item__unique__30d` | Число разных покупателей; актив; 30 дней | да |
| `item__count__60d` | Число покупок; актив; 60 дней | нет |
| `item__unique__60d` | Число разных покупателей; актив; 60 дней | нет |
| `item__count__90d` | Число покупок; актив; 90 дней | да |
| `item__unique__90d` | Число разных покупателей; актив; 90 дней | да |
| `item__count__180d` | Число покупок; актив; 180 дней | да |
| `item__unique__180d` | Число разных покупателей; актив; 180 дней | да |
| `item__recency__180d` | Давность покупки; актив; до 180 дней | да |
| `item__trend__30d` | Изменение активности; актив; последние 30 / предыдущие 30 дней | да |
| `pair__count__7d` | Число покупок; клиент–актив; 7 дней | нет |
| `pair__share__7d` | Доля покупок; клиент–актив; 7 дней | нет |
| `pair__count__14d` | Число покупок; клиент–актив; 14 дней | нет |
| `pair__share__14d` | Доля покупок; клиент–актив; 14 дней | нет |
| `pair__count__30d` | Число покупок; клиент–актив; 30 дней | да |
| `pair__share__30d` | Доля покупок; клиент–актив; 30 дней | да |
| `pair__count__60d` | Число покупок; клиент–актив; 60 дней | нет |
| `pair__share__60d` | Доля покупок; клиент–актив; 60 дней | нет |
| `pair__count__90d` | Число покупок; клиент–актив; 90 дней | да |
| `pair__share__90d` | Доля покупок; клиент–актив; 90 дней | да |
| `pair__count__180d` | Число покупок; клиент–актив; 180 дней | да |
| `pair__share__180d` | Доля покупок; клиент–актив; 180 дней | да |
| `pair__recency__180d` | Давность покупки; клиент–актив; до 180 дней | да |
| `pair__trend__30d` | Изменение активности; клиент–актив; последние 30 / предыдущие 30 дней | да |

### 13.6. Что добавлено к ручному бустингу

| Новый признак | Смысл |
|---|---|
| `item__count__7d` | Число покупок; актив; 7 дней |
| `item__unique__7d` | Число разных покупателей; актив; 7 дней |
| `item__count__14d` | Число покупок; актив; 14 дней |
| `item__unique__14d` | Число разных покупателей; актив; 14 дней |
| `item__count__60d` | Число покупок; актив; 60 дней |
| `item__unique__60d` | Число разных покупателей; актив; 60 дней |

Добавлены шесть признаков по связи item; новые признаки пары не улучшили внутренний критерий объединённого варианта. Полный набор объединённой модели — 44 ручных плюс эти шесть.

Машиночитаемый каталог: [feature_catalog.csv](results/evidence/feature_catalog.csv). Определения: [features.py](src/research/feature_automation/features.py). Алгоритм отбора: [experiment.py](src/research/feature_automation/experiment.py). Результаты исключения групп: [полный отчёт](#7-вклад-ручных-признаков).
