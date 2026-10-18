# STATUS.md — где работа сейчас

## Где мы
Данные для Next Item Prediction готовы: `results/vocab.parquet` (топ-500 000 треков),
`results/sequences.parquet` (9 238 пользователей, train/test). `.venv/bin/python checks/check.py`
даёт код 0; результат совпадает с независимой реализацией на pandas.
Статья PLUM сжата в `memory/L1.md`–`L3.md`, 5 тезисов L2 сверены по PDF.

## Дальше
Обучить трансформер next item prediction на `results/sequences.parquet` в двух вариантах —
случайная инициализация и инициализация весами Qwen — и сравнить Recall@10 на test.
Перед этим заменить цель в разделах 0–1 `AGENTS.md` и написать для неё проверку.

## Действует
- `data/` и `checks/` только для чтения (`chmod -R a-w`).
- Пакеты — только в `.venv`.
- Словарь — топ-500 000 по числу прослушиваний; тест — `timestamp >= max(timestamp) - 86400`.
