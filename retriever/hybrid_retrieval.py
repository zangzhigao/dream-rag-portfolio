# -*- coding: utf-8 -*-
"""混合检索：融合 FAISS(语义) 与 BM25(词法) 两路 top5。

⚠️ 量纲问题：FAISS 余弦分(0~1) 与 BM25 分(无界，常 0~20) 完全不可比，
直接 0.7*faiss + 0.3*bm25 会被 BM25 的大数值主导、权重失去意义。
因此先把两路分数各自 min-max 归一化到 [0,1]，再按权重融合：

    hybrid_score = 0.7 * faiss_norm + 0.3 * bm25_norm

返回结果仍保留两路的**原始分数**（faiss_score / bm25_score）便于核对，
hybrid_score 则基于归一化分数计算。
"""
from .faiss_retriever import faiss_topk
from .bm25_retriever import bm25_topk

FAISS_WEIGHT = 0.7
BM25_WEIGHT = 0.3


def _minmax(results: list[dict], key: str) -> dict:
    """把一组结果的 key 分数 min-max 归一化到 [0,1]，返回 {id: norm}。"""
    if not results:
        return {}
    vals = [r[key] for r in results]
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return {r["id"]: ((r[key] - lo) / span if span > 0 else 1.0) for r in results}


def hybrid_topk(query: str, top_k: int = 5,
                faiss_weight: float = FAISS_WEIGHT,
                bm25_weight: float = BM25_WEIGHT) -> list[dict]:
    """同时跑 FAISS 与 BM25 各取 top5，归一化后加权融合，返回 top_k。

    每条字段：id, topic, faiss_score(原始), bm25_score(原始), hybrid_score。
    某条只被一路召回时，另一路的原始分记为 0.0、归一化分按缺省 0 计入。
    """
    faiss_res = faiss_topk(query, top_k=5)
    bm25_res = bm25_topk(query, top_k=5)

    faiss_raw = {r["id"]: r["similarity_score"] for r in faiss_res}
    bm25_raw = {r["id"]: r["bm25_score"] for r in bm25_res}
    topic = {r["id"]: r["topic"] for r in (faiss_res + bm25_res)}

    fnorm = _minmax(faiss_res, "similarity_score")
    bnorm = _minmax(bm25_res, "bm25_score")

    merged = []
    for rid in topic:                       # 两路结果的并集
        hs = faiss_weight * fnorm.get(rid, 0.0) + bm25_weight * bnorm.get(rid, 0.0)
        merged.append({
            "id": rid,
            "topic": topic[rid],
            "faiss_score": round(faiss_raw.get(rid, 0.0), 4),
            "bm25_score": round(bm25_raw.get(rid, 0.0), 4),
            "hybrid_score": round(hs, 4),
        })

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return merged[:top_k]


# ---- CLI：python -m retriever.hybrid_retrieval <query> ----
def _print_table(rows):
    import unicodedata
    def w(s):
        return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in str(s))
    def pad(s, width, right=False):
        gap = width - w(s)
        return (" " * gap + str(s)) if right else (str(s) + " " * gap)
    headers = ["id", "topic", "faiss_score", "bm25_score", "hybrid_score"]
    aligns_r = [False, False, True, True, True]
    cols = list(zip(*([headers] + [[r["id"], r["topic"],
                f"{r['faiss_score']:.4f}", f"{r['bm25_score']:.4f}",
                f"{r['hybrid_score']:.4f}"] for r in rows])))
    widths = [max(w(c) for c in col) for col in cols]
    def line(row):
        return "  ".join(pad(row[i], widths[i], aligns_r[i]) for i in range(len(headers)))
    print(line(headers))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for r in rows:
        print(line([r["id"], r["topic"], f"{r['faiss_score']:.4f}",
                    f"{r['bm25_score']:.4f}", f"{r['hybrid_score']:.4f}"]))


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "梦见在水里挣扎快淹死了"
    print(f"\n混合检索  query: {q}   权重 faiss={FAISS_WEIGHT} / bm25={BM25_WEIGHT}（归一化后融合）\n")
    _print_table(hybrid_topk(q, top_k=5))
