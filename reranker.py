# -*- coding: utf-8 -*-
"""STEP 2 —— Cross-Encoder 重排（rerank）。

流程：hybrid_topk 给出粗排候选 → 本模块用 cross-encoder 对 (query, doc) 逐对精排 →
按 final_score 取 top5。

- Cross-encoder 与双塔(embedding)不同：它把 query 和 doc 一起喂进模型、直接打相关分，
  更准但更慢，因此只用于对少量候选做精排。
- 若 cross-encoder 模型不可用（未下载 / 无网络）→ 自动降级为 mock 打分，保证流程可跑。

分数说明：
- 真模型输出 logits（无界）、mock 输出字面覆盖率，二者都先 min-max 归一化到 [0,1]，
  再与同为 [0,1] 的 hybrid_score 融合：
      final_score = 0.5 * hybrid_score + 0.5 * rerank_score
"""
import os
import json
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "data" / "dream_kb.json"

RERANK_MODEL = "BAAI/bge-reranker-base"               # 中文 cross-encoder
LOCAL_RERANK_DIR = BASE_DIR / "models" / "bge-reranker-base"

HYBRID_WEIGHT = 0.5
RERANK_WEIGHT = 0.5

_id2rec = None
_cross_encoder = None
_ce_tried = False


def _kb() -> dict:
    global _id2rec
    if _id2rec is None:
        _id2rec = {}
        with open(KB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    _id2rec[r["id"]] = r
    return _id2rec


def _doc_text(rid: str, fallback_topic: str = "") -> str:
    rec = _kb().get(rid, {})
    return f"{rec.get('topic', fallback_topic)}。{rec.get('content', '')}"


def _get_cross_encoder():
    """尝试加载 cross-encoder；任何失败都返回 None（触发 mock）。"""
    global _cross_encoder, _ce_tried
    if not _ce_tried:
        _ce_tried = True
        try:
            from sentence_transformers import CrossEncoder
            path = str(LOCAL_RERANK_DIR) if LOCAL_RERANK_DIR.exists() else RERANK_MODEL
            _cross_encoder = CrossEncoder(path)
        except Exception:
            _cross_encoder = None      # 模型缺失 / 无网络 → mock
    return _cross_encoder


def is_mock() -> bool:
    """当前是否处于 mock 模式（无可用 cross-encoder）。"""
    return _get_cross_encoder() is None


def _minmax(vals: list[float]) -> list[float]:
    """把一组分数 min-max 归一化到 [0,1]；全相等时统一记为 1.0。"""
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    span = hi - lo
    return [((v - lo) / span if span > 0 else 1.0) for v in vals]


def _mock_score(query: str, doc_text: str) -> float:
    """无模型时的确定性兜底：query 字符在文档中的覆盖率（[0,1]，越高越相关）。"""
    q = set(query)
    if not q:
        return 0.0
    d = set(doc_text)
    return len(q & d) / len(q)


def rerank(query: str, hybrid_results: list[dict], top_k: int = 5) -> list[dict]:
    """对 hybrid_topk 结果做 cross-encoder 精排，返回 reranked_top5。

    输出每条字段：id, topic, hybrid_score, rerank_score, final_score。
    """
    if not hybrid_results:
        return []

    pairs = [(query, _doc_text(h["id"], h.get("topic", ""))) for h in hybrid_results]

    ce = _get_cross_encoder()
    if ce is not None:
        raw = list(ce.predict(pairs))                       # 真模型：logits
    else:
        raw = [_mock_score(q, d) for (q, d) in pairs]        # mock：字面覆盖率

    rr_norm = _minmax([float(x) for x in raw])               # 归一化到 [0,1]

    out = []
    for h, r in zip(hybrid_results, rr_norm):
        final = HYBRID_WEIGHT * h["hybrid_score"] + RERANK_WEIGHT * r
        out.append({
            "id": h["id"],
            "topic": h["topic"],
            "hybrid_score": round(h["hybrid_score"], 4),
            "rerank_score": round(r, 4),
            "final_score": round(final, 4),
        })

    out.sort(key=lambda x: x["final_score"], reverse=True)
    return out[:top_k]


# ---- CLI：python reranker.py <query> ----
def _print_table(rows):
    import unicodedata
    def w(s):
        return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in str(s))
    def pad(s, width, right=False):
        gap = width - w(s)
        return (" " * gap + str(s)) if right else (str(s) + " " * gap)
    headers = ["id", "topic", "hybrid_score", "rerank_score", "final_score"]
    right = [False, False, True, True, True]
    table = [headers] + [[r["id"], r["topic"], f"{r['hybrid_score']:.4f}",
                          f"{r['rerank_score']:.4f}", f"{r['final_score']:.4f}"] for r in rows]
    widths = [max(w(row[i]) for row in table) for i in range(len(headers))]
    def line(row):
        return "  ".join(pad(row[i], widths[i], right[i]) for i in range(len(headers)))
    print(line(headers))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in table[1:]:
        print(line(row))


if __name__ == "__main__":
    import sys
    from retriever import hybrid_topk

    q = " ".join(sys.argv[1:]) or "梦见在水里挣扎快淹死了"
    candidates = hybrid_topk(q, top_k=10)        # 粗排候选（最多 10 条）
    reranked = rerank(q, candidates, top_k=5)

    mode = "mock（无 cross-encoder 模型，降级打分）" if is_mock() else "cross-encoder（真模型）"
    print(f"\nrerank query: {q}   模式: {mode}")
    print(f"final_score = {HYBRID_WEIGHT}*hybrid + {RERANK_WEIGHT}*rerank（rerank 已归一化）\n")
    _print_table(reranked)
