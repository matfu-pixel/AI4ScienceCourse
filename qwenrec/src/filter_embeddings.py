"""Filter content embeddings to only item_ids present in listens.parquet.

Output: data/embeddings_listens.parquet (derived file, kept separate from raw data).
"""
import pandas as pd

EMB_PATH = "data/embeddings.parquet"
LIS_PATH = "data/listens.parquet"
OUT_PATH = "data/embeddings_listens.parquet"


def main() -> None:
    """Filter content embeddings to only item_ids present in listens.parquet.

    Данные: data/embeddings.parquet, data/listens.parquet
    Проверка: (нет в checks/check.py)
    Используется в: data/embeddings_listens.parquet
    """
    emb = pd.read_parquet(EMB_PATH)
    lis = pd.read_parquet(LIS_PATH, columns=["item_id"])

    # Unique item_ids present in listens.
    listen_ids = set(lis["item_id"].unique())

    before = len(emb)
    mask = emb["item_id"].isin(listen_ids)
    filtered = emb[mask].reset_index(drop=True)
    after = len(filtered)

    filtered.to_parquet(OUT_PATH, index=False)

    print(f"embeddings rows: {before} -> {after}")
    print(f"unique item_ids in listens: {len(listen_ids)}")
    print(f"dropped rows: {before - after}")
    print(f"wrote: {OUT_PATH}")


if __name__ == "__main__":
    main()