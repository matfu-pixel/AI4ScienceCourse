"""Проверка подготовки данных: .venv/bin/python checks/check.py [папка_результатов]

Код 0 — результат принят, 1 — нет. Эталон пересчитывается из data/listens.parquet
независимо от кода агента; готовых ответов в проверке нет. Папка результатов
в аргументе — только для проверки самой проверки (тогда src/ не требуется).
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CUSTOM_RESULTS = len(sys.argv) > 1
RESULTS = Path(sys.argv[1]) if CUSTOM_RESULTS else ROOT / "results"
LISTENS = ROOT / "data" / "listens.parquet"
VOCAB_SIZE = 500_000
TEST_WINDOW = 86_400  # последние сутки по timestamp — тест
errors = []


def fail(msg):
    print(msg)
    sys.exit(1)


# 1. Файлы есть и имеют нужные столбцы.
for name in ("vocab.parquet", "sequences.parquet"):
    if not (RESULTS / name).is_file():
        fail(f"нет файла {RESULTS.name}/{name}")
if not CUSTOM_RESULTS and not (ROOT / "src" / "prepare_data.py").is_file():
    errors.append("нет скрипта src/prepare_data.py")
vocab = pq.read_table(RESULTS / "vocab.parquet")
seqs = pq.read_table(RESULTS / "sequences.parquet")
for table, name, need in ((vocab, "vocab", {"item_id", "token", "n_listens"}),
                          (seqs, "sequences", {"uid", "train_tokens", "test_tokens"})):
    if not need <= set(table.column_names):
        fail(f"в {name}.parquet нужны столбцы {sorted(need)}, есть {table.column_names}")

# 2. Эталон из исходных данных.
listens = pq.read_table(LISTENS, columns=["uid", "timestamp", "item_id"])
uid = listens["uid"].to_numpy()
ts = listens["timestamp"].to_numpy()
item = listens["item_id"].to_numpy()

items, counts = np.unique(item, return_counts=True)
order = np.lexsort((items, -counts))[:VOCAB_SIZE]   # по убыванию числа прослушиваний, при равенстве — по item_id
ref_items, ref_counts = items[order], counts[order]

# 3. Словарь совпадает с эталоном целиком.
v = vocab.sort_by("token")
v_token = v["token"].to_numpy()
if len(v) != VOCAB_SIZE:
    errors.append(f"в словаре {len(v)} строк, ожидалось {VOCAB_SIZE}")
elif not np.array_equal(v_token, np.arange(VOCAB_SIZE)):
    errors.append("токены словаря — не 0..N-1 без пропусков и повторов")
elif not np.array_equal(v["item_id"].to_numpy(), ref_items):
    errors.append("состав или порядок треков в словаре не совпадает с топ-500000 по числу прослушиваний")
elif not np.array_equal(v["n_listens"].to_numpy(), ref_counts):
    errors.append("n_listens в словаре не совпадает с пересчётом")

# 4. Последовательности: пользователи, объёмы train/test, диапазон токенов.
by_item = np.argsort(ref_items)
pos = np.searchsorted(ref_items, item, sorter=by_item)
pos[pos == len(ref_items)] = 0
in_vocab = ref_items[by_item[pos]] == item
token_of = by_item[pos]                               # токен = позиция в эталонном словаре
t_split = ts.max() - TEST_WINDOW
is_test = ts >= t_split

s_uid = seqs["uid"].to_numpy()
all_uids = np.unique(uid[in_vocab])
for col in ("train_tokens", "test_tokens"):
    flat = pc.list_flatten(seqs[col])
    if len(flat) and (pc.min(flat).as_py() < 0 or pc.max(flat).as_py() >= VOCAB_SIZE):
        errors.append(f"в {col} есть токены вне 0..{VOCAB_SIZE - 1}")
if len(np.unique(s_uid)) != len(s_uid):
    errors.append("uid в sequences.parquet повторяются")
elif not np.array_equal(np.sort(s_uid), all_uids):
    errors.append(f"пользователей {len(s_uid)}, ожидалось {len(all_uids)} (все, у кого есть прослушивания из словаря)")
else:
    # 5. Все последовательности целиком: порядок по timestamp, при равенстве — как в listens.parquet.
    s = seqs.take(pa.array(np.argsort(s_uid)))
    sel = np.flatnonzero(in_vocab)
    sel = sel[np.lexsort((ts[sel], uid[sel]))]       # lexsort устойчив: равные timestamp — в порядке файла
    user_idx = np.searchsorted(all_uids, uid[sel])
    for col, mask in (("train_tokens", ~is_test[sel]), ("test_tokens", is_test[sel])):
        exp_len = np.bincount(user_idx[mask], minlength=len(all_uids))
        got_len = pc.list_value_length(s[col]).fill_null(0).to_numpy()
        if not np.array_equal(got_len, exp_len):
            k = np.flatnonzero(got_len != exp_len)[0]
            errors.append(f"длины {col} не совпадают с пересчётом, первое расхождение у uid {all_uids[k]} "
                          f"(порог теста: max(timestamp) - {TEST_WINDOW})")
            continue
        got = pc.list_flatten(s[col]).to_numpy(zero_copy_only=False)
        exp = token_of[sel[mask]]
        if not np.array_equal(got, exp):
            k = np.flatnonzero(got != exp)[0]
            u = all_uids[np.searchsorted(np.cumsum(exp_len), k, side="right")]
            errors.append(f"{col}: состав или порядок токенов не совпадает с пересчётом, первое расхождение у uid {u}")

print("\n".join(errors) if errors else "принято")
sys.exit(1 if errors else 0)
