"""Prepare vocab and sequences for Next Item Prediction from listens.parquet."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

VOCAB_SIZE = 500_000
TEST_WINDOW = 86_400  # last day by timestamp → test

def main():
    # 1. Read listens
    listens = pq.read_table("data/listens.parquet", columns=["uid", "timestamp", "item_id"])
    uid = listens["uid"].to_numpy()
    ts = listens["timestamp"].to_numpy()
    item = listens["item_id"].to_numpy()

    # 2. Build vocab: top VOCAB_SIZE items by listen count
    #    Sort by (-count, item_id ascending) — matches np.lexsort((items, -counts))
    items, counts = np.unique(item, return_counts=True)
    order = np.lexsort((items, -counts))[:VOCAB_SIZE]
    ref_items = items[order]
    ref_counts = counts[order]

    # 3. Write vocab.parquet
    vocab_table = pa.table({
        "item_id": ref_items,
        "token": np.arange(VOCAB_SIZE, dtype=np.uint32),
        "n_listens": ref_counts,
    })
    pq.write_table(vocab_table, "results/vocab.parquet")
    print(f"vocab.parquet: {len(ref_items)} items written")

    # 4. Map item_id → token (0..VOCAB_SIZE-1) or -1 if not in vocab
    #    ref_items is sorted by (-count, item_id). Need to search by item_id.
    #    Exact match of check.py logic.
    by_item = np.argsort(ref_items)  # indices to sort ref_items by item_id
    pos = np.searchsorted(ref_items, item, sorter=by_item)
    pos[pos == len(ref_items)] = 0  # dummy for out-of-range
    in_vocab = ref_items[by_item[pos]] == item
    token_of = np.full_like(pos, -1, dtype=np.int64)
    token_of[in_vocab] = by_item[pos[in_vocab]]

    # 5. Split: test = timestamp >= max_ts - TEST_WINDOW
    max_ts = ts.max()
    t_split = max_ts - TEST_WINDOW
    is_test = ts >= t_split

    # 6. Filter to users who have at least one listen in vocab
    sel = np.flatnonzero(in_vocab)
    sel_uid = uid[sel]
    sel_ts = ts[sel]
    sel_token = token_of[sel]
    sel_is_test = is_test[sel]

    # Sort by uid first, then by timestamp (stable — preserves file order for ties)
    sort_idx = np.lexsort((sel_ts, sel_uid))
    sel_uid = sel_uid[sort_idx]
    sel_ts = sel_ts[sort_idx]
    sel_token = sel_token[sort_idx]
    sel_is_test = sel_is_test[sort_idx]

    # 7. Group by uid and collect train/test tokens
    #    Find uid boundaries
    uid_changed = np.concatenate([[True], sel_uid[1:] != sel_uid[:-1]])
    uid_groups = np.flatnonzero(uid_changed)
    uid_groups = np.append(uid_groups, len(sel_uid))

    all_uids = sel_uid[uid_changed]
    train_lists = []
    test_lists = []

    for i in range(len(uid_groups) - 1):
        start = uid_groups[i]
        end = uid_groups[i + 1]
        group_token = sel_token[start:end]
        group_is_test = sel_is_test[start:end]
        train_lists.append(group_token[~group_is_test])
        test_lists.append(group_token[group_is_test])

    # 8. Write sequences.parquet
    seq_table = pa.table({
        "uid": all_uids,
        "train_tokens": [t.tolist() for t in train_lists],
        "test_tokens": [t.tolist() for t in test_lists],
    })
    pq.write_table(seq_table, "results/sequences.parquet")
    print(f"sequences.parquet: {len(all_uids)} users written")
    print(f"  train tokens total: {sum(len(t) for t in train_lists)}")
    print(f"  test tokens total: {sum(len(t) for t in test_lists)}")


if __name__ == "__main__":
    main()