# -*- coding: utf-8 -*-
"""Step 3 —— 加载索引与模型，提供语义检索函数 search()。

单独运行（需先跑过 embed.py 和 index.py）：
    python search.py 梦见自己在水里游泳
"""
import sys
import json
# faiss 是重依赖 → 移入 _load() 惰性导入，使云端 BM25-only 模式无需安装 faiss 即可 import。

from embed import get_model, META_PATH, QUERY_INSTRUCTION
from index import INDEX_PATH

_index = None
_meta = None


def _load():
    """惰性加载索引与元数据，只读一次。"""
    global _index, _meta
    if _index is None:
        import faiss   # 惰性导入：仅本地 FAISS 检索时需要
        if not INDEX_PATH.exists() or not META_PATH.exists():
            raise SystemExit("索引或元数据缺失，请先运行：python embed.py 然后 python index.py")
        _index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
    return _index, _meta


def search(query: str, top_k: int = 5) -> list[dict]:
    """返回与 query 最相关的 top_k 条记录，按相似度降序。

    每条是记录字典（topic/content/tags/...）外加一个 'similarity_score' 字段：
    向量已归一化 + 索引用内积(IndexFlatIP)，所以该分数就是 **余弦相似度**，
    范围约 -1~1（实际多在 0.3~0.7），分数越高越相关。
    """
    index, meta = _load()
    model = get_model()

    q = model.encode(
        [QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
    ).astype("float32")

    scores, idxs = index.search(q, top_k)
    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i != -1:
            rec = dict(meta[i])                       # 复制，避免污染缓存的 meta
            rec["similarity_score"] = float(score)    # 余弦相似度，越高越相关
            results.append(rec)
    return results


def main() -> None:
    query = " ".join(sys.argv[1:]) or "梦见自己在水里游泳"
    print(f"查询：{query}\n")
    for rank, rec in enumerate(search(query, top_k=5), 1):
        print(f"#{rank}  similarity={rec['similarity_score']:.3f}  [{rec['topic']}]  ({rec['id']})")
        print(f"     {rec['content'][:80]}...")
        print(f"     标签：{'、'.join(rec['tags'])}\n")


if __name__ == "__main__":
    main()
