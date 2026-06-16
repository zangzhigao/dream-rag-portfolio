# -*- coding: utf-8 -*-
"""FAISS 语义检索的薄封装（带云端安全降级）。

复用 search.py 的 search()，对外暴露与 bm25_topk 对称的 faiss_topk 接口。
当语义模型不可用（如 Streamlit Cloud 无法下载 BGE）或索引缺失时，
faiss_topk 不抛异常、返回空列表，由上层 hybrid_topk 降级为 BM25-only。
"""


def faiss_available() -> bool:
    """FAISS 语义检索此刻是否可用：模型可加载 且 索引+元数据已就绪。

    单一事实来源：UI 横幅与 run_pipeline 的门控都据此判断是否处于降级模式。
    文件存在性每次实时检查（本地首次构建索引后即恢复），模型探测结果由
    embed.semantic_available() 缓存，不重复尝试慢速下载。
    """
    try:
        from embed import semantic_available, META_PATH
        from index import INDEX_PATH
    except Exception:
        return False
    if not semantic_available():
        return False
    return INDEX_PATH.exists() and META_PATH.exists()


def faiss_topk(query: str, top_k: int = 5) -> list[dict]:
    """返回 FAISS 语义检索的前 top_k 条；每条带 'similarity_score'（余弦相似度）。

    语义不可用时返回 []（不抛异常），让上层走 BM25-only 降级。
    """
    if not faiss_available():
        return []
    try:
        from search import search   # 延迟导入：search 顶部 import faiss，降级时避免硬依赖
        return search(query, top_k=top_k)
    except Exception as e:
        print(f"[faiss] 语义检索运行失败，本次降级为 BM25-only：{e}")
        return []


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "梦见蛇追我"
    print(f"FAISS 查询：{q}\n")
    for rank, r in enumerate(faiss_topk(q, top_k=5), 1):
        print(f"#{rank}  sim={r['similarity_score']:.3f}  [{r['topic']}]  ({r['id']})")
