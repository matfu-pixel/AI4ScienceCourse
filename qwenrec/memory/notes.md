# QwenRec — Notes for Future Sessions

## Data Pipeline (prepare_data.py)
- Input: `data/listens.parquet` — 46.4M rows, columns: uid (uint32), timestamp (uint32), item_id (uint32)
- Vocab: top 500k items by listen count, sorted by (-count, item_id ascending). Token = position 0..499999.
- Token mapping: `searchsorted(ref_items, item, sorter=by_item)` where `by_item = np.argsort(ref_items)`.
  Must check `ref_items[by_item[pos]] == item` for exact match (searchsorted returns insertion point, not just match).
- Train/test split: `timestamp >= max(timestamp) - 86400` → test. All others → train.
- Users with no vocab listens are excluded.

## Key Gotchas
1. `np.searchsorted(a, v, sorter=s)` searches in `a[s]` (sorted version). Returns insertion point, not just match.
   Must verify `a[s[pos]] == v` for exact membership.
2. `np.lexsort((ts, uid))` sorts by `uid` first (last key), then by `ts` within each uid.
3. `np.lexsort((ts[sel], uid[sel]))` in check.py: sorts selected rows by uid, then timestamp.

## Verification
- Run: `.venv/bin/python checks/check.py`
- Returns "принято" + exit code 0 on success.