"""Собрать документы для сдачи из зафиксированных результатов; fit не используется."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.feature_automation.features import Builder, manual_features, generated_features, Feature, load_data, sha256
from research.feature_automation.metrics import Ranking

LABELS = {"old_5": "Пять счётчиков, новый подбор", "manual": "Ручной бустинг",
          "automatic_all": "Все исходные автоматические", "automatic_selected": "Автоматический отбор",
          "manual_plus_automatic": "Ручные + автоматические", "pair_popularity": "Повторные покупки + популярность",
          "popularity": "Популярность"}
REL = {"user": "Клиент", "item": "Актив", "pair": "Клиент–актив", "category": "Клиент–категория",
       "subcategory": "Клиент–подкатегория", "sector": "Клиент–сектор"}
OP = {"count": "Число покупок", "unique": "Число разных покупателей" , "share": "Доля покупок",
      "recency": "Давность покупки", "trend": "Изменение активности"}
R = "reports/feature_engineering_automation/results"


def pct(x, digits=2):
    return f"{100*x:.{digits}f}".replace(".", ",")


def num(x, digits=4):
    return f"{x:.{digits}f}".replace(".", ",")


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def window(s):
    if s.operation == "recency":
        return "до 180 дней"
    if s.operation == "trend":
        return "последние 30 / предыдущие 30 дней"
    return f"{s.months} календ. мес." if s.months else f"{s.days} дней"


def description(s):
    operation = "Число разных активов" if s.operation == "unique" and s.relation != "item" else OP[s.operation]
    return f"{operation}; {REL[s.relation].lower()}; {window(s)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    out = args.destination.resolve()
    (out / "docs/evidence").mkdir(parents=True, exist_ok=True)
    result_dir = ROOT / R
    read = lambda name: json.loads((result_dir / name).read_text())
    protocol, manual, auto, evaluation, verified, bench = [read(n) for n in
        ("protocol.json", "manual_stage.json", "automatic_stage.json", "evaluation.json", "verification.json", "benchmark.json")]
    for f, expected in protocol["source_hashes"].items():
        assert sha256(args.data_dir / f) == expected
    manual_specs = manual_features()
    generated = generated_features()
    assert [s.description() for s in manual_specs] == protocol["manual_features"]
    assert [s.description() for s in generated] == protocol["generated_features"]
    chosen_names = {s["name"] for s in auto["frozen"]["automatic_selected"]["features"]}
    combined_names = {s["name"] for s in auto["frozen"]["manual_plus_automatic"]["features"]}
    manual_names = {s.name for s in manual_specs}
    generated_names = {s.name for s in generated}
    union = list(dict.fromkeys(manual_specs + generated))
    assert len(manual_names)==44 and len(generated_names)==102 and len(chosen_names)==28 and len(combined_names)==50
    with (out / "docs/features.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name","relation","operation","days","months","description_ru",
            "manual","generated","selected_automatic","combined"], lineterminator=chr(10))
        writer.writeheader()
        for s in union:
            writer.writerow(s.description() | {"description_ru":description(s), "manual":int(s.name in manual_names),
                "generated":int(s.name in generated_names), "selected_automatic":int(s.name in chosen_names),
                "combined":int(s.name in combined_names)})
    data, metadata, conflicts = load_data(args.data_dir)
    users, items = protocol["cohort"]["users"], protocol["cohort"]["items"]
    builder = Builder(data, metadata, users, items)
    periods = {"train":pd.date_range("2020-04-01","2021-12-01",freq="MS"),
               "validation":pd.date_range("2022-01-01","2022-06-01",freq="MS"),
               "evaluation":pd.date_range("2022-07-01","2022-10-01",freq="MS")}
    populations = {}
    for name, dates in periods.items():
        frame, coverage = builder.labels(dates)
        counts = frame.groupby(["customerID","prediction_date"]).label.sum()
        populations[name] = {"months":len(dates),"rows":len(frame),"positives":int(frame.label.sum()),
            "all_cases":len(counts),"eligible_cases":int((counts>0).sum()),
            "in_catalog":sum(c["in_catalog"] for c in coverage),"all_assets":sum(c["all_assets"] for c in coverage)}
    saved = pd.read_csv(result_dir / "predictions.csv.gz", dtype={"customerID":str,"ISIN":str},
                        parse_dates=["prediction_date"], float_precision="round_trip")
    rank = Ranking(saved)
    cases = {}
    for name, metrics in evaluation["metrics"].items():
        secondary = saved.pair_popularity_secondary.to_numpy() if name=="pair_popularity" else None
        case = rank.cases(saved[name].to_numpy(), secondary)
        cases[name] = case
        assert abs(float(case.recall.mean()) - metrics["Recall@5"]) < 1e-12
        assert abs(float(case.ndcg.mean()) - metrics["NDCG@5"]) < 1e-12
    counts = saved.groupby(["prediction_date","customerID"],sort=True).label.sum()
    example_date, example_user = counts[counts>=3].index[0]
    example = saved[(saved.prediction_date==example_date)&(saved.customerID==example_user)].copy()
    example["source_row"] = example.index
    example["tie"] = Ranking(example).tie
    example = example.sort_values(["manual","tie","ISIN"],ascending=[False,True,True])
    top = example.head(5)
    positives = int(example.label.sum())
    hits = int(top.label.sum())
    dcg = sum(int(row.label)/math.log2(j+2) for j,row in enumerate(top.itertuples()))
    idcg = sum(1/math.log2(j+2) for j in range(min(positives,5)))
    positive_example = example[example.label==1].iloc[0]
    negative_example = example[example.label==0].iloc[0]
    loss_positive = -math.log(float(positive_example.manual))
    loss_negative = -math.log1p(-float(negative_example.manual))
    best_item = top.iloc[0].ISIN
    row_index = builder.pairs.index[(builder.pairs.customerID==example_user)&(builder.pairs.ISIN==best_item)][0]
    sample_specs = [Feature("user","count",90),Feature("pair","count",90),Feature("pair","share",90),
                   Feature("pair","recency",180),Feature("item","count",30),Feature("user","unique",90)]
    sample_values = {s.name:float(builder.column(example_date,s)[row_index]) for s in sample_specs}
    macro = evaluation["metrics"]["manual"]["Recall@5"]
    total_hits = int(cases["manual"].hits.sum())
    total_positives = int(saved.label.sum())
    primary = evaluation["bootstrap"]["manual_plus_automatic - manual"]
    calculate = {"source_sha256":{"predictions.csv.gz":sha256(result_dir/"predictions.csv.gz"),
            "evaluation.json":sha256(result_dir/"evaluation.json"),"protocol.json":sha256(result_dir/"protocol.json")},
        "population":populations,"feature_counts":{"manual":44,"generated":102,"selected":28,"combined":50,
            "distinct_manual_and_generated":len(union),"overlap":len(manual_names & generated_names)},
        "example_rule":"first (month, customerID) with at least three in-catalog positive assets; scores do not select the case",
        "example":{"month":str(example_date.date()),"source_row_indices":example.source_row.astype(int).tolist(),
            "top5":[{"rank":j+1,"source_row":int(row.source_row),"ISIN":row.ISIN,"p":float(row.manual),"label":int(row.label)}
                    for j,row in enumerate(top.itertuples())],"positives":positives,"hits":hits,"recall":hits/positives,
            "precision":hits/5,"dcg":dcg,"idcg":idcg,"ndcg":dcg/idcg,"sample_features":sample_values,
            "logloss_positive":{"source_row":int(positive_example.source_row),"p":float(positive_example.manual),"loss":loss_positive},
            "logloss_negative":{"source_row":int(negative_example.source_row),"p":float(negative_example.manual),"loss":loss_negative}},
        "manual_global":{"hits":total_hits,"positive_assets":total_positives,"macro_recall":macro,
            "micro_recall":total_hits/total_positives,"recommendations":len(cases['manual'])*5,
            "precision_all":total_hits/(len(cases['manual'])*5)},"evaluation_fits":0}
    (out / "docs/evidence/submission_calculations.json").write_text(json.dumps(calculate,ensure_ascii=False,indent=2)+"\n")
    counts_table = md_table(["Связь","Ручных","Генерируемых","Отобранных автоматических"],
        [[REL[r],sum(s.relation==r for s in manual_specs),sum(s.relation==r for s in generated),
          sum(s.relation==r and s.name in chosen_names for s in generated)] for r in REL])
    manual_table = md_table(["№","Имя в коде","Смысл"],[[j+1,f"`{s.name}`",description(s)] for j,s in enumerate(manual_specs)])
    chosen_table = md_table(["Имя","Смысл","Есть в ручной базе"],
        [[f"`{s.name}`",description(s),"да" if s.name in manual_names else "нет"] for s in generated if s.name in chosen_names])
    new_combined = [s for s in generated if s.name in combined_names - manual_names]
    feature_doc = rf'''# Каталог признаков

Признаки вычисляются из покупок строго до даты прогноза. Идентификаторы клиента и актива используются для агрегаций и соединений, но не подаются бустингу отдельными столбцами.

## 1. Сколько признаков используется

{counts_table}

Ручная база содержит 44 столбца, генератор — 102, отобранная автоматическая модель — 28, объединённая — 50. Между ручной и генерируемой базами совпадают 39 столбцов: это **107 уникальных определений**, а не 146 независимых признаков. Отбор выполняется по семействам связей, поэтому выбранное семейство сохраняет несколько коррелирующих окон.

## 2. Формулы и обработка отсутствующей истории

Пусть $H_r(t,w)$ — покупки по связи $r$ в полуоткрытом окне $[t-w,t)$. Для категориальных связей используется классификация, известная на $t$.

| Операция | Формула / правило | Гипотеза |
|---|---|---|
| count | $C_r(t,w)=\operatorname{{card}}(H_r(t,w))$ | Активность и повторное потребление |
| unique | Число разных активов; для item — разных покупателей | Разнообразие интересов или широта спроса |
| share | $C_r(t,w)/C_{{user}}(t,w)$; при нулевом знаменателе 0 | Относительное предпочтение |
| recency | Дни после последней покупки в 180-дневной истории; 181, если её нет | Недавнее поведение информативно |
| trend | $\ln(1+C_r(t,30))-\ln(1+C_r(t,60)-C_r(t,30))$ | Изменение двух соседних 30-дневных окон |

`1m` и `3m` — календарные месяцы; `30d` — ровно 30 дней. Они могут давать разные значения. Для recency код `180d` обозначает максимальную глубину истории, для trend `30d` — длину каждого из двух сравниваемых окон.

Покупки клиента включают активы вне каталога; популярность актива — покупки всех клиентов, а не только выбранных 500. Повторные сделки считаются отдельно в count, но дают один положительный ответ за месяц.

Временная версия метаданных выбирается строго до $t$. При конфликте значений одной версии поле становится неизвестным. Неизвестные категории не объединяются в общую категорию; их агрегаты равны нулю. Ноль поэтому частично смешивает отсутствие истории и отсутствие информации.

## 3. Все 44 ручных признака

{manual_table}

## 4. Грамматика 102 автоматических кандидатов

Для user и item: count и unique в шести окнах — $2\times6+2=14$ признаков с учётом recency и trend. Для pair: count и share — также 14. Для category, subcategory, sector: count, unique и share — $3\times6+2=20$ для каждой связи. Всего $14+14+14+20+20+20=102$.

Начальные окна: 7, 14, 30, 60, 90, 180 дней. Уточнение проверило замену 180 на 135 дней в выбранных семействах; обе замены отклонены. Начальные 102 столбца и уточнения различаются: контроль без отбора использует только исходный набор.

## 5. Какие 28 признаков выбрал алгоритм

{chosen_table}

## 6. Что добавлено к ручному бустингу

{md_table(['Новый признак','Смысл'],[[f'`{s.name}`',description(s)] for s in new_combined])}

Добавлены шесть признаков по связи item; новые признаки пары не улучшили внутренний критерий объединённого варианта. Полный набор объединённой модели — 44 ручных плюс эти шесть.

Машиночитаемый каталог: [features.csv](features.csv). Определения: [features.py](../research/feature_automation/features.py). Алгоритм отбора: [experiment.py](../research/feature_automation/experiment.py). Результаты исключения групп: [полный отчёт](../REPORT.md#7-вклад-ручных-признаков).
'''
    (out / "docs/FEATURES.md").write_text(feature_doc)
    feature_example = md_table(["Признак","Фактическое значение"],[[f"`{s.name}`",num(sample_values[s.name],6)] for s in sample_specs])
    top_table = md_table(["Позиция","ISIN","Оценка покупки","Факт покупки в месяце"],
        [[j+1,row.ISIN,num(float(row.manual),6),int(row.label)] for j,row in enumerate(top.itertuples())])
    results_table = md_table(["Метод","Recall@5, %","NDCG@5","Precision@5, %","Precision_all, %","LogLoss ↓","ROC-AUC"],
        [[LABELS[n],pct(m['Recall@5']),num(m['NDCG@5']),pct(m['Precision@5']),pct(m['Precision@5_all']),
          num(m['LogLoss'],6) if m['LogLoss'] is not None else '—',num(m['ROC-AUC']) if m['ROC-AUC'] is not None else '—']
         for n,m in evaluation['metrics'].items()])
    params = {n:v['params'] for n,v in manual['frozen'].items()} | {n:v['params'] for n,v in auto['frozen'].items()}
    param_table = md_table(['Модель','Признаков','learning_rate','max_iter','max_leaf_nodes','min_samples_leaf','L2'],
        [[LABELS[n],{'old_5':5,'manual':44,'automatic_all':102,'automatic_selected':28,'manual_plus_automatic':50}[n],
          p['learning_rate'],p['max_iter'],p['max_leaf_nodes'],p['min_samples_leaf'],p['l2_regularization']]
         for n,p in params.items()])
    cv_table = md_table(['Модель','Внутренний Recall при 100/7, lr=0,1, %','После подбора, %','Январь–июнь, %'],
        [[LABELS[n],pct(stage['controls'][n]['aggregate']['Recall@5']),pct(stage['best'][n]['aggregate']['Recall@5']),
          pct(stage['frozen'][n]['validation']['Recall@5'])] for stage in [manual,auto] for n in stage['frozen']])
    population_table = md_table(['Часть','Месяцев','Пар','Положительных пар','Клиент-месяцев','Случаев для Recall'],
        [[n,v['months'],v['rows'],v['positives'],v['all_cases'],v['eligible_cases']] for n,v in populations.items()])
    month_table = md_table(['Месяц','Ручной Recall, %','Автоотбор, %','Объединённый, %','Правило, %','Случаев'],
        [[evaluation['monthly']['manual'][j]['month']] + [pct(evaluation['monthly'][n][j]['Recall@5']) for n in
          ['manual','automatic_selected','manual_plus_automatic','pair_popularity']] + [evaluation['monthly']['manual'][j]['eligible_cases']]
         for j in range(4)])
    group_table = md_table(['Срез','Положительных пар','Случаев','Ручной Recall, %','Автоотбор, %','Объединённый, %','Правило, %'],
        [[kind,v['manual']['positives'],v['manual']['eligible_cases']] + [pct(v[n]['Recall@5']) for n in
         ['manual','automatic_selected','manual_plus_automatic','pair_popularity']] for kind,v in evaluation['groups'].items()])
    bootstrap_table = md_table(['Сравнение A − B','Разность Recall, п.п.','95% интервал, п.п.'],
        [[f"{LABELS[n.split(' - ')[0]]} − {LABELS[n.split(' - ')[1]]}",f"{100*v['delta']:+.2f}",
          f"[{100*v['ci95'][0]:+.2f}; {100*v['ci95'][1]:+.2f}]"] for n,v in evaluation['bootstrap'].items()])
    ablation_table = md_table(['Исключённая группа','Внутренний Recall, %','Разность относительно полного набора, п.п.'],
        [[r['removed'],pct(r['recall']),f"{100*r['delta_vs_full']:+.2f}"] for r in manual['ablations']])
    source_links = f"[evaluation.json]({R}/evaluation.json), [manual_stage.json]({R}/manual_stage.json), [automatic_stage.json]({R}/automatic_stage.json)"
    report = rf'''# Ручной feature engineering и автоматизация признаков для рекомендаций финансовых активов

## Аннотация

Исследуется месячный прогноз записанной покупки актива по истории взаимодействий. На FAR-Trans построен ручной HistGradientBoostingClassifier с 44 признаками, затем реализован генератор 102 кандидатов с отбором связей и локальным уточнением временных окон. Сравнение выполнено на одинаковых 500 клиентах и 50 активах.

На июле–октябре 2022 ручная модель получила {pct(macro)}% Recall@5, автоматический отбор — {pct(evaluation['metrics']['automatic_selected']['Recall@5'])}%, объединённый вариант — {pct(evaluation['metrics']['manual_plus_automatic']['Recall@5'])}%. Основная разность объединённого и ручного вариантов: {100*primary['delta']:+.2f} п.п., 95% клиентский интервал [{100*primary['ci95'][0]:+.2f}; {100*primary['ci95'][1]:+.2f}] п.п. Преимущество автоматизации по основной метрике не подтверждено. Ручной набор улучшил наблюдаемый результат относительно пяти счётчиков.

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

## 1. Постановка задачи

Задание содержит два последовательных этапа: разработать хорошие ручные признаки и настроить градиентный бустинг; предложить автоматизацию расчёта признаков и показать экспериментальные результаты её работы.

В момент $t$, первое число месяца, для клиента $u$ и актива $i$ оценивается $p_{{u,i,t}}=P(y_{{u,i,t}}=1\mid x_{{u,i,t}})$. Цель $y=1$ означает хотя бы одну записанную покупку пары в $[t,t+1\text{{ месяц}})$. Признаки $x$ вычисляются только по предыдущим событиям. По убыванию $p$ выбираются пять активов.

Здесь проверяются предсказание покупок и ранжирование. Доходность, риск и пригодность продукта клиенту не входят в целевую переменную. Нули — отсутствие записанной покупки, а не доказанные отрицательные предпочтения: логов показов нет.

Исследовательские вопросы: помогает ли расширенный ручной набор; какие группы в нём полезны; способен ли генератор получить конкурентные признаки; даёт ли добавление автоматически выбранных признаков прирост ручной базе. Основное формальное сравнение — объединённая модель минус ручная по Recall@5.

## 2. Данные и построение таблицы

Используется публичный [FAR-Trans, University of Glasgow](https://researchdata.gla.ac.uk/1658/). В расчётах нужны transactions.csv и asset_information.csv; цены, продажи, анкеты риска и размер капитала не подаются модели. История покупок ограничена событиями начиная с октября 2019.

По октябрю 2019–марту 2020 выбираются 500 случайных прежних покупателей с seed=42 и 50 самых покупаемых активов. Состав фиксируется для всех месяцев. На месяц создаётся $500\cdot50=25000$ кандидатных пар. Сделки внутри одного целевого месяца объединяются в бинарную метку; для исторических счётчиков сделки сохраняются.

{population_table}

Для финальной оценки: $25000\cdot4=100000$ пар; $500\cdot4=2000$ клиент-месяцев; положительных пар {total_positives}, то есть {pct(total_positives/len(saved),3)}% строк. Основные ranking-метрики используют 236 клиент-месяцев с покупками внутри каталога — 11,8% всех случаев. Это существенно для интерпретации качества.

Вне каталога остаются покупки: покрытие финального периода $508/678={pct(508/678)}$%. Оно измеряет долю фактических уникальных покупок клиент–актив–месяц, представленную каталогом. Наличие актива в прошлом не доказывает его текущую торговую доступность.

Метаданные присоединяются по последней версии строго до $t$. В трёх парах ISIN/дата обнаружены конфликтующие категории. Неоднозначное поле заменяется неизвестным. Исторические сделки группируются по классификации, известной на дату прогноза. Эти фильтры не доказывают истинную историческую доступность ретроспективно реконструированных исходных данных. Авторы описывают реконструкцию покупок и изменения дат в [§3.3.1 публикации FAR-Trans](https://arxiv.org/html/2407.08692v1#S3.SS3.SSS1).

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

Период и правила сохранены в [protocol.json]({R}/protocol.json). До текущей финальной оценки исправлена обработка границы поиска окон; исходный протокол и код сохранены в provenance. После получения текущих финальных метрик признаки, параметры и веса не менялись.

## 4. Ручные признаки

Ручная база — 44 признака. Она включает активность клиента и спрос на актив, повторные покупки пары, разнообразие активов и покупателей, давность, доли покупок и динамику. Косвенные связи с категорией, подкатегорией и сектором дают возможность оценивать активы, которые клиент ещё не покупал.

{counts_table}

Используются точные 30/90/180 дней и исходные календарные 1/3 месяца. При отсутствии покупок count/unique/share равны нулю; recency равен 181. Динамика сравнивает два соседних 30-дневных окна через разность логарифмов. Все формулы, **44 конкретных имени**, выбранные автоматические признаки и шесть добавлений выделены в [каталоге FEATURES](docs/FEATURES.md).

### Реальный пример расчёта признаков

Кейс A выбран детерминированно: первый по месяцу и идентификатору клиент с не менее чем тремя положительными активами внутри каталога. Качество прогноза не участвует в выборе примера. Момент прогноза — {example_date.date()}, рассматриваемый актив — {best_item}.

{feature_example}

Доля покупок пары за 90 дней равна {num(sample_values['pair__count__90d'],0)} / {num(sample_values['user__count__90d'],0)} = {num(sample_values['pair__share__90d'],6)}. Последняя покупка этой пары — {(example_date-pd.Timedelta(days=sample_values["pair__recency__180d"])).date()}; давность = ({example_date.date()} − эта дата) = {num(sample_values["pair__recency__180d"],0)} дней. Приведённые значения — реальные расчёты из источника; это описание входов модели, а не оценка причинной важности отдельного признака.

## 5. Бустинг и настройка

Используется HistGradientBoostingClassifier — ансамбль деревьев, в котором новые итерации корректируют предыдущие предсказания. Число листьев ограничивает сложность дерева, learning_rate задаёт вклад новой итерации, min_samples_leaf и L2 контролируют регуляризацию. Обучение минимизирует бинарный LogLoss; выбор гиперпараметров ориентирован на Recall@5. Это разные уровни: функция обучения вероятностей и критерий качества итогового top-5. Все модели проверяют один список 16 конфигураций пяти параметров, seed=42; случайный early stopping выключен.

Полный пул, из которого детерминированно выбраны конфигурации: learning_rate {{0,05; 0,1; 0,2}}, max_iter {{100; 200; 300}}, max_leaf_nodes {{7; 15; 31}}, min_samples_leaf {{20; 60}}, L2 {{0; 1}}. Исходные 100/7 и 200/7 с lr=0,1 включены явно. Это ограниченный подбор, а не доказательство оптимальности во всём пространстве. Список попыток — в protocol.json и trials.jsonl.

Правило выбора: максимальный внутренний Recall, затем меньший LogLoss, затем меньше деревьев и листьев. Выбранные параметры:

{param_table}

Идентификаторы, дата и ответ не подаются модели как признаки. Все методы используют одинаковые пары и определения метрик. При равных оценках порядок задан BLAKE2b-хешем ключа (seed, клиент, дата, актив); он не зависит от ответа и порядка строк.

## 6. Автоматизация

Идея — составлять признаки как «связь × операция × временное окно». Допустимы шесть связей: user, item, pair, category, subcategory, sector; операции count, unique, share, recency, trend. Начальные окна 7, 14, 30, 60, 90, 180 дней дают 102 столбца. Это собственный ограниченный генератор, использующий принцип агрегации по отношениям, родственный [Deep Feature Synthesis, Kanter & Veeramachaneni, 2015](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf). Полный DFS не реализован, теоретическая новизна этого принципа не заявляется.

1. При фиксированном бустинге 100/7, lr=0,1 проверяются семейства связей.
2. Выбирается лучшее первое семейство, затем последовательно добавляются другие, только если внутренний Recall растёт более чем на $10^{{-6}}$.
3. Для выбранных семейств исключается каждое окно; окно с наибольшим падением качества проверяется заменой соседними значениями. Если оба соседних варианта уже представлены или выходят за 1–180 дней, проверяется промежуточное окно между ближайшими исходными значениями.
4. По выбранному набору настраивается бустинг по тем же 16 конфигурациям.
5. Отдельно проверяется добавление выбранных семейств к зафиксированной ручной базе.

Человек определяет грамматику операций и связей; программа создаёт конкретные столбцы, выбирает семейства и проверяет окна. Отбор групп не равен индивидуальному отбору каждого столбца. Это локальный поиск, не исчерпывающий перебор всех целых окон или всех возможных комбинаций.

| Шаг | Внутренний Recall, % | Решение |
|---|---|---|
| Только pair | {pct(auto['selection_steps'][0]['recall'])} | Первое выбранное семейство |
| Добавление item | {pct(auto['selection_steps'][1]['recall'])} | Принято; итог 28 признаков |
| Замена pair 180 → 135 дней | 58,58 | Отклонено |
| Замена item 180 → 135 дней | 59,66 | Отклонено |
| Ручная база, фиксированные параметры | {pct(auto['combined_steps'][0]['previous_recall'])} | Контроль |
| Ручная база + item | {pct(auto['combined_steps'][0]['recall'])} | Принято; итог 50 признаков |

Контроль «все автоматические» использует исходные 102 столбца. Уточнения не были приняты. Отдельное сравнение фиксированных и настроенных параметров позволяет видеть вклад представления и вклад настройки:

{cv_table}

## 7. Вклад ручных признаков

В полной ручной модели фиксируются выбранные параметры, затем по одному исключаются типы признаков. Модель каждого варианта заново обучается только на соответствующих внутренних train-блоках.

{ablation_table}

Исключение recency, долей или категориальных связей снижает внутренний Recall; исключение trend и unique немного повышает его. Следовательно, увеличение числа признаков само по себе не является улучшением. Эти результаты диагностические: группы пересекаются, взаимодействуют, а интервалы разности здесь не рассчитаны. Полная ручная база была зафиксирована протоколом; диагностика не превращена в дополнительный выбор по финальному периоду.

## 8. Метрики и расчёты

Пусть $R_{{u,t}}$ — множество фактически купленных внутри каталога активов, $L_{{u,t}}$ — пять рекомендованных активов, $h=|R\cap L|$. Основная популяция $E$ включает только клиент-месяцы с $|R|>0$.

$$
\mathrm{{Recall@5}}=\frac{{1}}{{|E|}}\sum_{{(u,t)\in E}}\frac{{h_{{u,t}}}}{{|R_{{u,t}}|}},\qquad
\mathrm{{Precision@5}}=\frac{{1}}{{|E|}}\sum_{{(u,t)\in E}}\frac{{h_{{u,t}}}}{{5}}.
$$

Precision_all использует все 2000 клиент-месяцев, включая месяцы без покупки. Основной Recall — **среднее долей по случаям**, а не отношение всех попаданий ко всем покупкам.

Для бинарной релевантности $y_j$ на позиции $j$:

$$
\mathrm{{DCG@5}}=\sum_{{j=1}}^5\frac{{y_j}}{{\log_2(j+1)}},\qquad
\mathrm{{IDCG@5}}=\sum_{{j=1}}^{{\min(|R|,5)}}\frac{{1}}{{\log_2(j+1)}},\qquad
\mathrm{{NDCG@5}}=\frac{{\mathrm{{DCG@5}}}}{{\mathrm{{IDCG@5}}}}.
$$

NDCG усредняется по $E$: она учитывает и наличие попаданий, и их порядок. Пары без положительных активов не входят в это среднее.

$$
\mathrm{{LogLoss}}=-\frac{{1}}{{N}}\sum_{{n=1}}^N\left[y_n\ln p_n+(1-y_n)\ln(1-p_n)\right].
$$

LogLoss считается по всем парам; меньшая ошибка лучше. Вероятности ограничиваются машинным epsilon для устойчивости. Глобальный ROC-AUC — вероятность, что случайная положительная пара выше отрицательной, плюс половина вероятности равенства. При одном классе AUC не определён. Глобальный AUC не заменяет ранжирование внутри клиента. Для правил LogLoss не определён без вероятностной калибровки. ROC-AUC для этих контролей не рассчитывался: они включены для сравнения top-5. Сам по себе AUC может вычисляться по ранговым оценкам без вероятностей.

### Реальный top-5 и расчёт одного случая

Для кейса A модель даёт следующий выход. Здесь «рекомендация» означает выход экспериментального алгоритма ранжирования по покупке, а не проверенную инвестиционную пригодность.

{top_table}

В каталоге фактически куплены {positives} актива, найдено {hits}. Поэтому Recall = {hits}/{positives} = {pct(hits/positives)}%, Precision = {hits}/5 = {pct(hits/5)}%.

DCG = {num(dcg,6)}, IDCG = {num(idcg,6)}, NDCG = {num(dcg,6)}/{num(idcg,6)} = {num(dcg/idcg,6)}. Положительные активы на первых двух позициях дают $1+1/\log_2 3$ в DCG; третий купленный актив в top-5 не попал.

Для положительной пары с $p={num(float(positive_example.manual),6)}$ вклад в LogLoss = $-\ln p={num(loss_positive,6)}$. Для отрицательной пары с $p={num(float(negative_example.manual),6)}$ вклад = $-\ln(1-p)={num(loss_negative,6)}$. Общий LogLoss — среднее по 100000 парам, а не по двум примерам.

### Расчёт агрегатов ручной модели

Получено {total_hits} попаданий в $2000\cdot5=10000$ рекомендациях. Precision_all = ${total_hits}/10000={pct(total_hits/10000)}$%. Если считать долю найденных покупок суммарно, получится ${total_hits}/{total_positives}={pct(total_hits/total_positives)}$% — это дополнительный micro-Recall, **не основная метрика**. Основной macro-Recall = среднее 236 индивидуальных долей = {pct(macro)}%. Эти величины различаются из-за разного числа покупок у клиентов.

Выбор примера, номера строк CSV, формулы, численные значения и хеши входов сохранены в [submission_calculations.json](docs/evidence/submission_calculations.json). Документы собираются скриптом без обучения модели.

## 9. Результаты и неопределённость

### Общая оценка июля–октября

{results_table}

Пять счётчиков настроены заново на внутренних блоках; это другая модель, чем исходная модель 200/7 из прежнего аудита. У сравниваемых новых моделей с 5 и 44 признаками параметры одинаковы. Наблюдаемый вклад расширения: $100({macro:.9f}-{evaluation['metrics']['old_5']['Recall@5']:.9f})={100*(macro-evaluation['metrics']['old_5']['Recall@5']):.4f}$ п.п. Отдельный подтверждающий интервал этой разности не рассчитан.

### Парные клиентские интервалы

{bootstrap_table}

Для каждой из 5000 выборок с seed=20261013 выбираются 500 клиентов с возвращением. Если клиент выбран несколько раз, все его месяцы повторяются с той же кратностью. Разность двух моделей считается на одних и тех же случаях:

$$
\Delta_b=\frac{{\sum_u m_{{u,b}}\sum_{{t:(u,t)\in E}}(\mathrm{{Recall}}^A_{{u,t}}-\mathrm{{Recall}}^B_{{u,t}})}}{{\sum_u m_{{u,b}}\,n_u}},\qquad
\mathrm{{CI}}_{{95\%}}=[Q_{{0.025}}(\Delta_b),Q_{{0.975}}(\Delta_b)].
$$

Здесь $m_{{u,b}}$ — кратность выбора, $n_u$ — число учитываемых месяцев клиента. Пустых выборок не было. Интервал описывает вариацию клиентов при фиксированных моделях и периоде, а не новые рыночные режимы, переобучение или неопределённость отбора.

Основной интервал включает ноль: прирост автоматизации по Recall@5 не подтверждён, эквивалентность также не установлена. Ручной вариант выше правила по наблюдаемой разности; его диагностический интервал положителен. С учётом раскрытого периода это не независимое подтверждение будущего эффекта. Остальные сравнения диагностические, коррекция множественных проверок не применялась.

### Помесячное качество

{month_table}

Октябрь слабее предыдущих месяцев у всех методов. Нельзя приписывать это доказанному рыночному режиму: диагноз причины не проводился. Покрытие каталога меняется по месяцам:

{md_table(['Месяц','Покупки внутри каталога','Все покупки активов','Покрытие, %'],[[r['month'],r['in_catalog'],r['all_assets'],pct(r['in_catalog']/r['all_assets'])] for r in evaluation['coverage']])}

### Новые и повторные покупки

{group_table}

Срез repeat_3m означает наличие покупки пары за три предыдущих календарных месяца; new_3m — её отсутствие. Это не признак отсутствия взаимодействий за всю жизнь. В каждом срезе меняется только множество положительных ответов для оценки, исходный top-5 не перестраивается. Случаи с покупками обоих типов входят в оба среза, поэтому их нельзя усреднить для получения общего Recall. Групповые интервалы не рассчитаны.

Числа таблиц берутся из {source_links}.

## 10. Затраты и проверка

{md_table(['Вариант','Генерация 27 месяцев, с','Генерация 4 месяцев, с','predict 100000 пар, с','Пиковая память, MiB'],[[LABELS[r['model']],num(r['development_generation_seconds'],2),num(r['evaluation_generation_seconds'],2),num(r['evaluation_predict_seconds'],3),num(r['peak_rss_bytes']/1024**2,1)] for r in bench])}

Каждый вариант измерялся в отдельном процессе с новым кэшем агрегатов и четырьмя потоками модели. Пиковая память относится ко всему процессу; кэш ОС и порядок запусков не контролировались. Это профилирование, не строгий статистический benchmark.

Ручной этап: {manual['timing']['fits']} fit, {num(manual['timing']['fit_seconds'],2)} с измеренного обучения. Автоматический: {auto['timing']['fits']} fit, {num(auto['timing']['fit_seconds'],2)} с. Это включает внутренние сравнения, подбор и финальное обучение. Общий вычислительный лимит заданием не требовался; затраты измерены как дополнительная характеристика. Время ручного придумывания признаков не измерено, поэтому общая экономия разработки не доказана.

Первый технически неудачный запуск выполнил три fit до исправления JSON-сериализации. Первый автоматический поиск остановлен из-за пустого уточнения у границы 180 дней; сохранены 99 завершённых fit и не полностью записанная попытка внутри следующего CV (до трёх дополнительных fit). После добавления промежуточного окна поиск повторён до финальной оценки. Успешные этапы таблиц не включают эти дополнительные затраты; история — [execution_notes.json]({R}/execution_notes.json) и provenance.

Проверки: {verified['unit_tests']} тестов; независимый пересчёт ranking-метрик всех семи методов; LogLoss по формуле и AUC через ранги; точное воспроизведение вероятностей пяти моделей; 200 отдельных расчётов признаков; инвариантность проверенных признаков к будущим событиям; независимость ранжирования от ответов и порядка строк; явная проверка кратности клиентов в десяти bootstrap-выборках; совпадение CV-метрик 33 повторённых конфигураций. Финальная фаза запрещает fit, хеши моделей до/после совпадают. Максимальная ошибка независимых ranking-метрик — порядка $10^{{-16}}$, вероятностей — 0. Подтверждение: [verification.json]({R}/verification.json).

## 11. Рекомендации и ограничения

**Выбор текущего решения:** ручной бустинг с 44 признаками имеет лучший наблюдаемый Recall@5 и LogLoss из сравниваемых ML-вариантов. Его следует представлять как основной результат ручного этапа. Автоматический отбор — отдельный исследованный метод с 28 признаками и меньшими измеренными затратами генерации, но без доказанного превосходства или эквивалентности по Recall.

Отобранный и объединённый варианты немного выше по NDCG; интервал этой разности не рассчитан. Поэтому нельзя заявлять подтверждённое преимущество по этой дополнительной метрике. Методическая ценность автоматизации — воспроизводимые генерация, отбор и журнал экспериментов, включая неудачные варианты.

**Следующие действия** выделены в [RECOMMENDATIONS](docs/RECOMMENDATIONS.md): отдельная маркировка неизвестных категорий; удаление избыточных групп и окон; согласование отбора с настройкой модели; проверка на новом закрытом периоде; анализ расширенного каталога. Это рекомендации, не уже выполненные улучшения.

Ограничения: раскрытый финальный период, ретроспективное происхождение части событий, 500 прежних покупателей и 50 исторически популярных активов, отсутствие показов, смешение неизвестных категорий с нулевой историей, ограниченный подбор и локальный поиск. Предсказание поведения не устанавливает финансовую полезность, доходность или пригодность продукта клиенту. Абсолютное отсутствие утечки в исходнике не доказано; проверена корректность временных фильтров реализации.

## 12. Воспроизведение и источники

Для чтения достаточно этого отчёта и выполненных notebooks: [ручной этап](reports/feature_engineering_automation/01_manual_baseline.ipynb), [автоматизация](reports/feature_engineering_automation/02_feature_automation.ipynb), [сравнение](reports/feature_engineering_automation/03_experiments.ipynb). Обучение выполнено фазами CLI; notebooks читают журналы и пересчитывают результаты. Полное повторное обучение запускается отдельно либо через RUN_TRAINING с новой папкой результатов.

Окружение: Python 3.15.0, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1; остальные точные версии — environment_freeze.txt. Публичный архив включён в снимки входов; подготовка данных проверяет SHA-256. [Инструкция воспроизведения](docs/REPRODUCIBILITY.md) описывает чтение, проверку фиксированных моделей и полный эксперимент.

1. [FAR-Trans, University of Glasgow, DOI 10.5525/gla.researchdata.1658](https://researchdata.gla.ac.uk/1658/): источник данных и лицензия CC BY 4.0.
2. [Sanz-Cruzado, Droukas, McCreadie, 2024](https://arxiv.org/html/2407.08692v1): происхождение и обработка FAR-Trans. Наш месячный протокол с повторными покупками отличается от протокола статьи; численные результаты напрямую не сравниваются.
3. [Kanter & Veeramachaneni, 2015, Deep Feature Synthesis](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf): генерация признаков по отношениям.
4. [Cawley & Talbot, 2010](https://jmlr.org/papers/v11/cawley10a.html): оптимизм при выборе модели.
5. [HistGradientBoostingClassifier, официальная документация](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html): используемый класс бустинга.
'''
    (out / "REPORT.md").write_text(report)
    print(json.dumps({"documents":["REPORT.md","docs/FEATURES.md","docs/features.csv","docs/evidence/submission_calculations.json"],
        "manual_recall":macro,"manual_hits":total_hits,"example_recall":hits/positives,"example_ndcg":dcg/idcg,
        "distinct_features":len(union)},ensure_ascii=False))


if __name__ == "__main__":
    main()
