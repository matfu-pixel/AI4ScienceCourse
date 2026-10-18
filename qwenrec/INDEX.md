# INDEX.md — где что лежит

- `AGENTS.md` — правила работы агента в проекте.
- `STATUS.md` — где работа сейчас; читать в начале сессии.
- `diary.md` — журнал проекта; только дописывается.
- `data/listens.parquet` — Yambda 50m: 46 467 212 прослушиваний, `uid, timestamp, item_id, is_organic, played_ratio_pct, track_length_seconds`; только чтение.
- `data/embeddings.parquet` — контентные эмбеддинги всех 7,7 млн треков Yambda (`item_id, embed, normalized_embed`, 128 чисел); 13,8 ГБ, только чтение.
- `data/embeddings_listens.parquet` — те же эмбеддинги только для треков из `data/listens.parquet` (819 432 строки); пишет `src/filter_embeddings.py::main()`.
- `src/prepare_data.py` — словарь и последовательности, `main()`; пишет `results/vocab.parquet` и `results/sequences.parquet`.
- `src/filter_embeddings.py` — фильтрация эмбеддингов по трекам из прослушиваний, `main()`.
- `results/vocab.parquet` — словарь: `item_id, token, n_listens`, топ-500 000 треков.
- `results/sequences.parquet` — `uid, train_tokens, test_tokens` по пользователям.
- `checks/check.py` — проверка `results/` пересчётом из `data/listens.parquet`; только чтение.
- `memory/L2.md` — тезисы статьи PLUM со сверкой по PDF; читать вместо PDF.
- `memory/L1.md` — реферат PLUM по частям со страницами; `memory/L3.md` — главный результат одной строкой.
- `memory/notes.md` — заметки агента о `src/prepare_data.py` (как строится отображение `item_id` → токен).
- `paper/paper.pdf` — статья PLUM, arXiv:2510.07784v1; `paper/paper.txt` — её текст по страницам.
- `paper/tg_posts.md` — черновики постов о проекте с предварительными результатами.
- `cascade.sh` — каскадное сжатие статьи агентом: `bash cascade.sh paper/paper.pdf`.
