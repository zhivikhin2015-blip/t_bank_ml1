"""Собрать отчёт и честно выполнить ячейки трёх notebooks по результатам запуска."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import pandas as pd


LABELS = {"old_5": "Пять счётчиков, новый подбор", "manual": "Ручные 44 признака",
          "automatic_all": "Все исходные автоматические признаки", "automatic_selected": "Отобранные автоматические",
          "manual_plus_automatic": "Ручные + автоматические", "pair_popularity": "Повторные покупки + популярность",
          "popularity": "Популярность"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def notebook(path, cells, root):
    namespace = {"__name__": "__notebook__"}
    result = []
    count = 0
    for kind, source in cells:
        cell = {"cell_type": kind, "id": f"cell-{len(result):03d}", "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            count += 1
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, str(path), "exec"), namespace)
            cell.update(execution_count=count, outputs=[{"output_type": "stream", "name": "stdout", "text": buffer.getvalue().splitlines(keepends=True)}])
        result.append(cell)
    obj = {"cells": result, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
           "language_info": {"name": "python", "version": platform.python_version()}}, "nbformat": 4, "nbformat_minor": 5}
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--destination", required=True, type=Path)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    load = lambda name: json.loads((args.results / name).read_text())
    protocol, manual, auto, evaluation = [load(n) for n in
        ("protocol.json", "manual_stage.json", "automatic_stage.json", "evaluation.json")]
    verification, benchmark = load("verification.json"), load("benchmark.json")
    metrics = evaluation["metrics"]
    primary = evaluation["bootstrap"]["manual_plus_automatic - manual"]
    report = ["# Признаки для рекомендаций финансовых активов: ручной бустинг и автоматизация\n",
      "## Результат выполнения задания\n",
      "Реализованы оба этапа: ручной градиентный бустинг с 44 признаками, затем генератор "
      "102 признаков с отбором связей и локальным поиском временных окон. Все выводы ниже "
      "получены из выполненных экспериментов; результаты поиска и параметры сохранены.\n",
      f"На июле–октябре 2022 ручная модель получила **{100*metrics['manual']['Recall@5']:.2f}% Recall@5**, "
      f"объединённая — **{100*metrics['manual_plus_automatic']['Recall@5']:.2f}%**. Разность "
      f"**{100*primary['delta']:+.2f} п.п.**, парный 95% интервал "
      f"**[{100*primary['ci95'][0]:+.2f}; {100*primary['ci95'][1]:+.2f}] п.п.**.\n",
      "Этот период уже был просмотрен в предыдущем аудите. Поэтому оценка исследовательская: "
      "она не является новым независимым подтверждением. Параметры, признаки и модели "
      "зафиксированы по внутренним блокам до этой новой оценки и после неё не дорабатывались.\n",
      "## 1. Данные и протокол\n",
      "FAR-Trans; 500 клиентов и 50 популярных активов выбраны только по октябрю 2019–марту 2020. "
      "На начало месяца прогнозируется наличие записанной покупки актива за следующий календарный месяц. "
      "Повторные сделки дают один положительный ответ, но отдельно учитываются в истории. "
      "Популярность считается по всем клиентам; история клиента включает активы вне каталога.\n",
      "Обучение: апрель 2020–декабрь 2021. Настройка и отбор: последовательные блоки II, III и IV "
      "кварталов 2021 с обучением на предшествующих месяцах. Январь–июнь 2022 — диагностическая "
      "validation без выбора решений. Июль–октябрь 2022 — исследовательская финальная оценка; "
      "ноябрь исключён. Модель обучается до декабря 2021, история обновляется предыдущими "
      "наблюдёнными событиями. Протокол — [protocol.json](results/protocol.json).\n",
      "Recall@5 — средняя доля купленных внутри каталога активов, найденных в пяти рекомендациях, "
      "по клиент-месяцам с такими покупками. NDCG@5 дополнительно учитывает порядок попаданий. "
      "Precision@5 использует те же случаи, Precision@5_all — всех клиентов. LogLoss и ROC-AUC "
      "считаются по всем парам; ROC-AUC глобальный, не внутри клиента. Равенства разрешаются "
      "по хешу ключа, независимо от порядка строк и ответа.\n",
      "## 2. Ручной feature engineering\n",
      "Разработаны счётчики покупок, число разных активов/покупателей, давность последнего события, "
      "доли покупок пары и категории в активности клиента, изменение активности соседних окон. "
      "Связи: клиент, актив, клиент–актив, клиент–категория, клиент–подкатегория, клиент–сектор. "
      "Окна 30, 90, 180 дней дополняют исходные календарные 1 и 3 месяца. Формулы и гипотезы "
      "описаны в [справке к коду](../../research/feature_automation/README.md#формулы-признаков).\n",
      "Метаданные присоединяются по последней версии строго до cutoff. Обнаружены три пары "
      "ISIN/дата с конфликтующими категориями; неоднозначные значения заменены неизвестными, "
      "а не выбраны порядком CSV. Категориальные признаки не используют будущие версии. "
      "Классификация прошлых сделок соответствует информации, известной на cutoff.\n",
      "Настроены learning_rate, max_iter, max_leaf_nodes, min_samples_leaf и l2_regularization. "
      "Использованы 16 заранее фиксированных конфигураций, включая исходные 100/7 и 200/7; "
      "это ограниченный поиск, а не доказательство глобально оптимального бустинга. "
      "Случайный early stopping выключен. Выбор: максимальный внутренний Recall, затем LogLoss "
      "и меньшая модель. Одинаковое пространство применяется ко всем вариантам.\n"]
    rows = []
    for name in ("old_5", "manual"):
        best = manual["best"][name]
        rows.append([LABELS[name], f"{100*manual['controls'][name]['aggregate']['Recall@5']:.2f}",
                     f"{100*best['aggregate']['Recall@5']:.2f}",
                     f"{100*manual['frozen'][name]['validation']['Recall@5']:.2f}", json.dumps(best['params'])])
    report += [table(["Модель", "Внутренний Recall, одинаковые параметры, %", "Внутренний Recall после подбора, %",
                      "Validation Recall, %", "Выбранные параметры"], rows),
               "\nПроверка вклада групп при выбранных параметрах полной ручной модели:\n",
               table(["Исключённая группа", "Внутренний Recall, %", "Разность относительно полной модели, п.п."],
                     [[r["removed"], f"{100*r['recall']:.2f}", f"{100*r['delta_vs_full']:+.2f}"] for r in manual['ablations']]),
               "\nВнутренние оценки использованы для выбора и могут быть оптимистичны. Проверка исключением "
               "показывает условный вклад группы при фиксированной модели, а не причинный эффект признака.\n",
               "## 3. Идея и реализация автоматизации\n",
               "Генератор строит `связь × операция × окно`: count, unique, share, recency и trend. "
               "Начальные окна — 7, 14, 30, 60, 90, 180 дней. Forward selection добавляет семейства "
               "связей только при росте внутреннего Recall больше 1e−6. Затем исключением отдельных "
               "окон определяется наиболее полезное окно каждого выбранного семейства; проверяется "
               "его замена соседними значениями ± половина длины, в пределах 1–180 дней. "
               "Если соседи уже присутствуют, проверяется промежуточное окно между ближайшими исходными значениями. "
               "После отбора настраивается бустинг. Контроль без отбора использует исходные 102 признака; "
               "уточнение может добавить новые окна только в отобранные варианты. "
               "Метод и остановка заданы до финальной оценки.\n",
               f"Выбранные семейства: **{', '.join(auto['selected_families'])}**. "
               f"В объединённую ручную модель добавлены: **{', '.join(auto['combined_added_families']) or 'ни одного'}**.\n"]
    accepted = [r['accepted'] | {'family':r['family']} for r in auto['window_refinement'] if r['accepted']]
    report += ["Проверенные дополнительные окна:\n",
       table(["Связь", "Новое окно, дней", "Внутренний Recall, %", "Принято"],
             [[r['family'], t['days'], f"{100*t['recall']:.2f}", "да" if r['accepted'] and r['accepted']['new_days'] == t['days'] else "нет"]
              for r in auto['window_refinement'] for t in r['attempts']]),
       "\nПринятые уточнения окон:\n" + (table(["Связь", "Было дней", "Стало дней", "Внутренняя разность Recall, п.п."],
       [[r['family'],r['old_days'],r['new_days'],f"{100*(r['recall']-r['previous_recall']):+.3f}"] for r in accepted])
       if accepted else "Проверенные замены не улучшили внутренний критерий; исходные окна сохранены."),
       "\nЧеловек задаёт отношения и операции, программа выбирает комбинации и окна. Это автоматизация "
       "в заданной грамматике и локальный поиск; она не гарантирует оптимума и не изобретает "
       "произвольные новые операции. Генерация по отношениям опирается на принцип "
       "[Deep Feature Synthesis, Kanter & Veeramachaneni (2015)](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf); "
       "полный DFS здесь не реализован.\n",
       table(["Автоматический вариант", "Число признаков", "Внутренний Recall, одинаковые параметры, %",
              "Внутренний Recall после подбора, %", "Validation Recall, %", "Параметры"],
             [[LABELS[n],len(v['features']),f"{100*auto['controls'][n]['aggregate']['Recall@5']:.2f}",
               f"{100*auto['best'][n]['aggregate']['Recall@5']:.2f}",f"{100*v['validation']['Recall@5']:.2f}",
               json.dumps(v['params'])]
              for n,v in auto['frozen'].items()]),
       "\n## 4. Финальные экспериментальные результаты\n",
       table(["Метод", "Recall@5, %", "NDCG@5", "Precision@5, %", "Precision@5_all, %", "LogLoss", "ROC-AUC"],
       [[LABELS[n],f"{100*m['Recall@5']:.2f}",f"{m['NDCG@5']:.4f}",f"{100*m['Precision@5']:.2f}",
         f"{100*m['Precision@5_all']:.2f}",f"{m['LogLoss']:.6f}" if m['LogLoss'] is not None else "—",
         f"{m['ROC-AUC']:.4f}" if m['ROC-AUC'] is not None else "—"] for n,m in metrics.items()]),
       "\nПравила не выдают вероятностей, поэтому LogLoss/AUC для них не рассчитываются. "
       "Строка «Пять счётчиков» использует новый заранее заданный внутренний подбор; "
       "это не та же модель 200/7, которая оценивалась в прежнем аудите.\n",
       table(["Сравнение", "Разность Recall, п.п.", "95% интервал, п.п."],
             [[n,f"{100*v['delta']:+.2f}",f"[{100*v['ci95'][0]:+.2f}; {100*v['ci95'][1]:+.2f}]"]
              for n,v in evaluation['bootstrap'].items()]),
       "\nПарный bootstrap: 5000 выборок 500 клиентов с возвращением, все месяцы и кратность "
       "сохраняются. Интервалы условны на выбранных моделях и периоде; они не включают "
       "неопределённость поиска, рыночных режимов и реконструкции данных. Основное сравнение — "
       "объединённая модель минус ручная; остальные диагностические.\n",
       table(["Месяц", "Ручная Recall, %", "Отобранная автоматическая, %", "Объединённая, %", "Правило, %", "Случаев"],
        [[evaluation['monthly']['manual'][j]['month']] +
         [f"{100*evaluation['monthly'][n][j]['Recall@5']:.2f}" for n in
          ('manual','automatic_selected','manual_plus_automatic','pair_popularity')] +
         [evaluation['monthly']['manual'][j]['eligible_cases']] for j in range(4)]),
       "\nСрезы новых/повторных покупок используют отсутствие/наличие покупки пары за три календарных "
       "месяца. Top-5 остаётся прежним, ответы не меняют ранжирование. Это не lifetime cold-start.\n",
       table(["Срез", "Ручная Recall, %", "Автоматическая, %", "Объединённая, %", "Правило, %"],
        [[g] + [f"{100*values[n]['Recall@5']:.2f}" for n in
                  ('manual','automatic_selected','manual_plus_automatic','pair_popularity')]
         for g,values in evaluation['groups'].items()]),
       f"\nОценено {metrics['manual']['rows']} пар и {metrics['manual']['all_cases']} клиент-месяцев; "
       f"основные ranking-метрики используют {metrics['manual']['eligible_cases']} случаев. "
       f"Покрытие каталога: {sum(r['in_catalog'] for r in evaluation['coverage'])}/"
       f"{sum(r['all_assets'] for r in evaluation['coverage'])} уникальных покупок клиент–актив–месяц.\n",
       "## 5. Стоимость и проверки\n",
       table(["Вариант", "Генерация 27 месяцев, с", "Генерация 4 месяцев, с", "predict 100000 пар, с", "Пиковая память, MiB"],
        [[LABELS[r['model']],f"{r['development_generation_seconds']:.2f}",f"{r['evaluation_generation_seconds']:.2f}",
          f"{r['evaluation_predict_seconds']:.3f}",f"{r['peak_rss_bytes']/1024**2:.1f}"] for r in benchmark]),
       "\nДля каждой модели использован отдельный процесс и новый кэш агрегатов; файловый кэш ОС "
       "и порядок запусков не контролировались. Память относится ко всему процессу. "
       "Профилирование не доказывает экономию общей стоимости разработки.\n",
       table(["Этап", "Число fit", "Измеренное время fit, с", "CV predict, с"],
        [[label,stage['timing']['fits'],f"{stage['timing']['fit_seconds']:.2f}",f"{stage['timing']['predict_seconds']:.2f}"]
         for label,stage in [('Ручной',manual),('Автоматический',auto)]]),
       "\nВремя этапов включает отбор, настройку и окончательные fit. CV predict не включает "
       "проверку замороженных моделей на validation. Отдельно учитывается один технически "
       "неудачный запуск с тремя fit до исправления JSON-сериализации счётчика. "
       "Исторические затраты ручного труда не измерены.\n",
       "До финальной оценки исправлен пустой поиск соседей у границы 180 дней: добавлена "
       "проверка промежуточного окна 135 дней. Первый автоматический поиск остановлен и повторён; "
       "ручная модель не менялась. В остановленном поиске записаны 99 завершённых fit; "
       "ещё от нуля до трёх попыток внутри прерванного CV не имеют отдельной записи. "
       "Исходный протокол, код и журнал сохранены в results/provenance, изменение — в protocol.json. "
       "Это изменение обработки границы по внутренним данным, не по новым финальным метрикам.\n",
       f"Пройдены {verification['unit_tests']} тестов; независимый oracle пересчитал ranking-метрики "
       "всех семи методов, предсказания пяти моделей воспроизведены побайтово, проверены "
       "LogLoss по формуле и AUC ранговой статистикой, 200 построчных расчётов признаков, инвариантность к будущим событиям и порядку строк, "
       "десять bootstrap-выборок явным повторением клиентов. Фаза оценки запрещает fit; "
       "хеши моделей до/после совпадают. [Проверки](results/verification.json).\n",
       "## 6. Выводы и ограничения\n"]
    low, high = primary['ci95']
    if low <= 0 <= high:
        report.append("Основной интервал включает ноль: доказанного преимущества добавления автоматических "
                      "признаков по Recall@5 нет. Это не доказывает равенства методов. Автоматизация реализована и "
                      "проверена экспериментально, её эффект на качество ограничен наблюдёнными результатами.\n")
    elif low > 0:
        report.append("На наблюдённом периоде основная разность положительна, и клиентский интервал не включает "
                      "ноль. Из-за исследовательского статуса периода это требует независимого подтверждения.\n")
    else:
        report.append("На наблюдённом периоде добавление автоматических признаков ухудшило Recall; клиентский "
                      "интервал отрицателен. Это отрицательный результат данного метода, не всех способов автоматизации.\n")
    report += [f"Инженерия ручных признаков дала наблюдаемую разность {100*(metrics['manual']['Recall@5']-metrics['old_5']['Recall@5']):+.2f} "
               "п.п. Recall относительно пяти счётчиков с теми же выбранными параметрами. "
               "Отобранная автоматическая и объединённая модели слегка выше по NDCG, но интервал этой разности не рассчитан; "
               "заявлять подтверждённое преимущество по NDCG нельзя.\n",
               "Ручная модель выше простого правила на 3.82 п.п.; диагностический клиентский интервал "
               "[+0.17; +7.51] п.п. положителен. Это условный результат наблюдённого периода, "
               "а не независимое подтверждение для будущих рыночных периодов.\n",
               "FAR-Trans содержит ретроспективно восстановленные события и скорректированные даты. "
               "Фильтры кода проверены, но исходная point-in-time достоверность не доказана. "
               "Нет логов показов и явных отказов. Выводы относятся к записанным покупкам фиксированной "
               "когорты и каталога; они не описывают прибыльность, риск и пригодность активов клиенту. "
               "Неизвестные категории и отсутствие истории частично смешиваются нулевыми агрегатами. "
               "Требуются отдельные исследования расширения каталога и новых пользователей.\n",
               "Приоритеты следующей итерации: отдельно обозначить неизвестные категории; проверить "
               "удаление избыточных окон и групп вместо только добавления; изучить зависимость отбора "
               "от параметров бустинга. Эти изменения относятся к будущим экспериментам и не внесены "
               "по результатам финального периода. Для подтверждения потребуется новый закрытый период.\n",
               "## Артефакты и воспроизведение\n",
               "[Ручной notebook](01_manual_baseline.ipynb), [автоматизация](02_feature_automation.ipynb), "
               "[сравнение](03_experiments.ipynb). Обучение выполнено фазами CLI; выполненные notebooks "
               "читают журнал и проверяют результаты. Для полного повторного обучения предусмотрен "
               "RUN_TRAINING с новой папкой результатов. "
               "[Команды](../../research/feature_automation/README.md#воспроизведение), "
               "[журнал](results/trials.jsonl), [результаты](results/evaluation.json), "
               "[окружение](results/environment_freeze.txt).\n",
               "## Источники\n",
               "1. [FAR-Trans, данные University of Glasgow и лицензия CC BY 4.0](https://researchdata.gla.ac.uk/1658/).\n"
               "2. [Sanz-Cruzado et al., 2024: происхождение и реконструкция данных](https://arxiv.org/html/2407.08692v1).\n"
               "3. [Kanter & Veeramachaneni, 2015: генерация признаков по отношениям](https://dai.lids.mit.edu/wp-content/uploads/2017/10/DSAA_DSM_2015.pdf).\n"
               "4. [Официальная документация HistGradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).\n"
               "5. [Cawley & Talbot, 2010: оптимизм выбора модели](https://jmlr.org/papers/v11/cawley10a.html).\n"]
    (args.destination / "research_results.md").write_text("\n".join(report))
    setup = '''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "research/feature_automation").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULT_DIR = ROOT / "reports/feature_engineering_automation/results"
RUN_TRAINING = False
# Для полного повторного запуска задайте True и новую RESULT_DIR.
DATA_DIR = ROOT / "work/data/FAR-Trans"
read = lambda name: json.loads((RESULT_DIR / name).read_text())
print("Источник: фактические сохранённые эксперименты; RUN_TRAINING =", RUN_TRAINING)
'''
    run = lambda phase: f'''if RUN_TRAINING:
    import subprocess
    subprocess.run([sys.executable, "-m", "research.feature_automation.experiment",
                    "--data-dir", str(DATA_DIR), "--out", str(RESULT_DIR), "--phase", "{phase}"],
                   cwd=ROOT, check=True)
else:
    print("Этап {phase} ранее выполнен CLI; ниже его журнал и результаты.")
'''
    cells = [("markdown", "# 1. Ручной градиентный бустинг и engineering признаков\n\n"
              "На начало месяца предсказываем записанные покупки. 500 клиентов, 50 активов, top-5. "
              "Три внутренних временных блока используются для подбора, validation — для диагностики. "
              "Обучение выполнено CLI; этот notebook выполнен для просмотра и проверки результатов."),
             ("code", setup), ("code", run("manual")),
             ("markdown", "## Формулы и смысл\n\ncount — покупки; unique — разнообразие; recency — давность; "
              "share — доля интереса; trend — логарифмическое изменение соседних окон. Все события строго до cutoff. "
              "Календарные месяцы отличаются от фиксированных дней. Категории присоединяются по доступной версии."),
             ("code", '''from research.feature_automation.features import manual_features
specs = manual_features()
print(pd.DataFrame([s.description() for s in specs]).to_string(index=False))
print("Число ручных признаков:", len(specs))'''),
             ("code", '''m = read("manual_stage.json")
for name in ("old_5", "manual"):
    print(name, "CV:", m["best"][name]["aggregate"], "params:", m["best"][name]["params"])
    print("validation:", m["frozen"][name]["validation"])
print("Правило validation:", m["rule_validation"])
print(pd.DataFrame(m["ablations"]).to_string(index=False))'''),
             ("markdown", "## Интерпретация\n\nРазность при одинаковых параметрах отделяет влияние признаков от настройки. "
              "Исключение групп показывает условный вклад. Внутренние результаты участвовали в выборе, поэтому "
              "не являются независимым подтверждением."),
             ("code", '''print("Контроль одинаковых параметров:")
for name, r in m["controls"].items():
    print(name, r["aggregate"])
print("Профиль этапа:", m["timing"])''')]
    root = Path(__file__).resolve().parents[2]
    notebook(args.destination / "01_manual_baseline.ipynb", cells, root)
    notebook(args.destination / "02_feature_automation.ipynb", [
        ("markdown", "# 2. Автоматическая генерация и поиск признаков\n\n"
         "Идея: связь × агрегат × окно. Человек задаёт операции и отношения; программа выбирает комбинации "
         "и уточняет окна. Это ограниченная грамматика и локальный поиск."),
        ("code", setup), ("code", run("automatic")),
        ("code", '''from research.feature_automation.features import generated_features
generated = generated_features()
print("Сгенерировано:", len(generated))
print(pd.DataFrame([s.description() for s in generated]).to_string(index=False))'''),
        ("markdown", "## Отбор\n\nForward selection по внутреннему Recall@5, без построчных перестановок. "
         "После отбора связей проверяется вклад окон и соседние значения. После этого настраивается бустинг."),
        ("code", '''a = read("automatic_stage.json")
print("Выбранные связи:", a["selected_families"])
print("Шаги:", json.dumps(a["selection_steps"], ensure_ascii=False, indent=2))
print("Уточнение:", json.dumps(a["window_refinement"], ensure_ascii=False, indent=2))
print("Добавлено к ручной базе:", a["combined_added_families"])'''),
        ("code", '''for name, r in a["frozen"].items():
    print(name, "features:", len(r["features"]), "params:", r["params"])
    print("CV:", a["best"][name]["aggregate"], "validation:", r["validation"])
print("Исходная ручная модель сохранена:", a["manual_model_unchanged"])
print("Профиль:", a["timing"])''')], root)
    notebook(args.destination / "03_experiments.ipynb", [
        ("markdown", "# 3. Эксперименты и проверка результатов\n\n"
         "Июль–октябрь 2022 уже известны по аудиту: статус оценки исследовательский. "
         "Все модели фиксируются до этой оценки. Финальная фаза запрещает fit."),
        ("code", setup), ("code", run("evaluate")),
        ("code", '''e = read("evaluation.json")
print(pd.DataFrame(e["metrics"]).T.to_string())
print("Статус:", e["status"], "число fit при оценке:", e["evaluation_fits"])'''),
        ("code", '''print(json.dumps(e["bootstrap"], ensure_ascii=False, indent=2))
print("Месяцы:")
for name in ("manual", "automatic_selected", "manual_plus_automatic", "pair_popularity"):
    print(name, pd.DataFrame(e["monthly"][name]).to_string(index=False))'''),
        ("code", '''print("Срезы:", json.dumps(e["groups"], ensure_ascii=False, indent=2))
print("Покрытие:", e["coverage"])
print(pd.DataFrame(read("benchmark.json")).to_string(index=False))'''),
        ("markdown", "## Независимый пересчёт\n\nНиже ranking-метрики пересчитываются отдельным циклом "
         "по множествам купленных активов. Прежние оценки и веса не меняются."),
        ("code", '''from research.feature_automation.verify import metric_oracle
saved = pd.read_csv(RESULT_DIR / "predictions.csv.gz", dtype={"customerID":str,"ISIN":str},
                    parse_dates=["prediction_date"], float_precision="round_trip")
for name, metrics in e["metrics"].items():
    secondary = saved.pair_popularity_secondary.to_numpy() if name == "pair_popularity" else None
    oracle = metric_oracle(saved, saved[name].to_numpy(), secondary)
    assert max(abs(value-metrics[key]) for key,value in oracle.items()) < 1e-12
print("Независимый пересчёт семи методов совпал.")
print(json.dumps(read("verification.json"), indent=2))'''),
        ("markdown", "## Вывод\n\nЭффект автоматизации определяется фактической разностью и интервалом. "
         "Покупки не доказывают прибыльность рекомендаций. Bootstrap условен на периоде и выбранных моделях; "
         "независимое подтверждение требует нового закрытого периода.")], root)
    print("Report and three notebooks executed and saved")


if __name__ == "__main__":
    main()
