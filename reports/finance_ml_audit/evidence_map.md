# Карта свидетельств

Номера ячеек — индексы с нуля в JSON ноутбука. Снимки исходного кода сохранены без изменений.

| Код | Источник | Ячейка | Что проверяется |
|---|---|---:|---|
| M2 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 2 | Выбор когорты и каталога до первого прогноза; закрытие test |
| M4 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 4 | Левые и правые границы истории, бинарные ответы |
| M8 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 8 | Детерминированные равенства и метрики |
| M11 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 11 | Сетка из четырёх настроек; выбор по validation |
| M19 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 19 | Проверки метрик, порядка строк и кратности bootstrap |
| M23 | [manual_baseline_reviewed.ipynb](source_inputs/outputs/manual_baseline_reviewed.ipynb) | 23 | Метаданные, бюджет и исторический source hash |
| A2 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 2 | 3 группы × 4 окна, count-агрегаты |
| A6 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 6 | Проверка исходных артефактов; генератор |
| A8 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 8 | Сборка train/validation и границы |
| A12 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 12 | Отбор neg_log_loss, три shuffle, максимум пять новых признаков |
| A18 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 18 | Все признаки и сетка настроек выбранного набора |
| A20 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 20 | Ручной контроль, сравнение и совпадение двух ролей модели |
| A26 | [02_automatic_features_reviewed.ipynb](source_inputs/outputs/02_automatic_features_reviewed.ipynb) | 26 | Парный bootstrap на validation |
| F4 | [03_final_comparison.ipynb](source_inputs/outputs/03_final_comparison.ipynb) | 4 | Шлюз периода и замораживание протокола |
| F6 | [03_final_comparison.ipynb](source_inputs/outputs/03_final_comparison.ipynb) | 6 | Генерация признаков до cutoff |
| F12 | [03_final_comparison.ipynb](source_inputs/outputs/03_final_comparison.ipynb) | 12 | Окно данных, исключение ноября |
| F14 | [03_final_comparison.ipynb](source_inputs/outputs/03_final_comparison.ipynb) | 14 | Только predict_proba замороженных моделей |
| F18 | [03_final_comparison.ipynb](source_inputs/outputs/03_final_comparison.ipynb) | 18 | Клиентский bootstrap и проверки исходных хешей |

Историческое построение test: `outputs/manual_baseline.ipynb`, ячейка 6 (сохранённый вывод: 125000 строк), и `outputs/01_far_trans_learning.ipynb`, ячейка 43 (125000 строк). Эти оригиналы не изменялись; свидетельства сохранены отдельно в `historical_test_access.json`.
