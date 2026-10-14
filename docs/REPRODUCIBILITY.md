# Как прочитать и воспроизвести исследование

## 1. Чтение без вычислений

Откройте [REPORT.md](../REPORT.md), [каталог признаков](FEATURES.md) и [рекомендации](RECOMMENDATIONS.md). Три notebooks уже выполнены и содержат результаты. Модели, вероятности и журналы находятся в `reports/feature_engineering_automation/results/`.

Ни локальные рабочие файлы автора, ни папки outputs/work из старой работы для воспроизведения опубликованного комплекта не требуются. Входы восстанавливаются из включённого публичного архива.

## 2. Окружение и данные

Команды выполняются из корня checkout ветки `codex/feature-engineering-automation`. Проверенное окружение: Python **3.15.0**, scikit-learn **1.9.1**, NumPy **2.5.4**, pandas **3.0.6**. Загрузка сохранённых моделей требует совместимого окружения; результаты иной версии следует обозначать отдельно.

```bash
git clone --branch codex/feature-engineering-automation https://github.com/zhivikhin2015-blip/t_bank_ml1.git
cd t_bank_ml1
python3.15 -m venv .venv
.venv/bin/python -m pip install -r reports/feature_engineering_automation/results/environment_freeze.txt
.venv/bin/python -m research.feature_automation.prepare_data --out /tmp/finance-submission-data
```

Репозиторий приватный: команда clone использует доступ уже авторизованной учётной записи. `prepare_data` извлекает только нужные файлы и проверяет архив и transactions.csv по SHA-256. Укажите новый путь, если папка уже существует.

## 3. Проверка готовых результатов без обучения

```bash
.venv/bin/python -m unittest research.feature_automation.test_pipeline -v
.venv/bin/python -m research.feature_automation.verify \
  --data-dir /tmp/finance-submission-data \
  --out reports/feature_engineering_automation/results
```

Проверяются входы и исходники, метрики, вероятности, порядок строк, примеры временных признаков, bootstrap и неизменность моделей. Проверка записывает verification.json; если нужно сохранить опубликованный файл побайтово, предварительно скопируйте всю папку results в новую временную папку и передайте её в `--out`.

## 4. Полный новый запуск

Папка `/tmp/finance-submission-run` должна быть новой. Этапы выполняются последовательно:

```bash
.venv/bin/python -m research.feature_automation.experiment \
  --data-dir /tmp/finance-submission-data --out /tmp/finance-submission-run --phase manual
.venv/bin/python -m research.feature_automation.experiment \
  --data-dir /tmp/finance-submission-data --out /tmp/finance-submission-run --phase automatic
.venv/bin/python -m research.feature_automation.experiment \
  --data-dir /tmp/finance-submission-data --out /tmp/finance-submission-run --phase evaluate
.venv/bin/python -m research.feature_automation.verify \
  --data-dir /tmp/finance-submission-data --out /tmp/finance-submission-run
.venv/bin/python -m research.feature_automation.benchmark \
  --data-dir /tmp/finance-submission-data --out /tmp/finance-submission-run
```

manual создаёт протокол, настраивает и фиксирует ручной baseline; automatic выполняется после него; evaluate запрещает fit и использует сохранённые веса. Полный запуск включает сотни обучений. В текущем исследовании успешные этапы выполнили 113 и 240 fit; дополнительные технические запуски учтены отдельно в execution_notes.json.

Для полного повторного обучения через notebooks задайте RUN_TRAINING=True, DATA_DIR с подготовленными данными и новую RESULT_DIR последовательно в трёх notebooks. По умолчанию они читают опубликованные результаты. Значения метрик проверяются с допуском 1e−12; время, память, отметки времени и технические хеши нового запуска могут различаться. Нельзя подбирать новые параметры по финальным месяцам ради совпадения или улучшения результата.

## 5. Пересборка отчёта и расчётов

```bash
.venv/bin/python scripts/build_submission.py \
  --data-dir /tmp/finance-submission-data \
  --destination /tmp/finance-submission-documents
```

Скрипт читает **опубликованный фиксированный** results из checkout, сверяет хеши данных, формирует REPORT.md, каталог признаков и JSON реальных расчётов. Он не содержит fit и не меняет модель. Для оформления результатов нового научного эксперимента требуется отдельный комплект с собственным протоколом.

## 6. Основные артефакты

| Файл | Что подтверждает |
|---|---|
| protocol.json | Даты, определения признаков, список настроек, критерий выбора, поправку протокола |
| manual_stage.json / automatic_stage.json | Внутренние сравнения, параметры и validation |
| trials.jsonl / *_trials.json | Фактически проверенные конфигурации и использование кэша |
| evaluation.json | Финальные метрики, месяцы, срезы, интервалы и нулевое число fit при оценке |
| predictions.csv.gz | 100000 пар, ответы, оценки всех методов и дополнительный ключ правила |
| *.joblib | Пять зафиксированных моделей, определения признаков и параметры |
| verification.json | Независимые проверки результатов |
| benchmark.json | Измеренные затраты в отдельных процессах |
| docs/evidence/submission_calculations.json | Выбор реального примера, номера строк, формулы и агрегаты |

Все они доступны в [папке результатов](../reports/feature_engineering_automation/results/), кроме последнего — в [доказательствах отчёта](evidence/submission_calculations.json).
