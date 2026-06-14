# -*- coding: utf-8 -*-
"""FAISS 语义检索的薄封装。

不改动现有 search.py 的任何逻辑——仅复用其 search()，
对外暴露与 bm25_topk 对称的 faiss_topk 接口。
"""
from search import search   # 复用现有 FAISS 检索（原逻辑不变）


def faiss_topk(query: str, top_k: int = 5) -> list[dict]:
    """返回 FAISS 语义检索的前 top_k 条；每条带 'similarity_score'（余弦相似度）。"""
    return search(query, top_k=top_k)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "梦见蛇追我"
    print(f"FAISS 查询：{q}\n")
    for rank, r in enumerate(faiss_topk(q, top_k=5), 1):
        print(f"#{rank}  sim={r['similarity_score']:.3f}  [{r['topic']}]  ({r['id']})")
