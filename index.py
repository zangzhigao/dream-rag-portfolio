# -*- coding: utf-8 -*-
"""Step 2 —— 用 embed.py 生成的向量构建 FAISS 索引。

单独运行（需先跑过 embed.py）：
    python index.py
产物：
    data/faiss.index
"""
from pathlib import Path

import numpy as np
import faiss

from embed import EMB_PATH

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "data" / "faiss.index"


def main() -> None:
    if not EMB_PATH.exists():
        raise SystemExit("找不到 embeddings.npy，请先运行：python embed.py")

    emb = np.load(EMB_PATH).astype("float32")
    dim = emb.shape[1]

    # 向量已归一化 -> 用内积索引(IndexFlatIP)即等价于余弦相似度检索。
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, str(INDEX_PATH))
    print(f"[index] 已为 {index.ntotal} 条向量(维度 {dim})建立索引 -> {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
