# -*- coding: utf-8 -*-
"""梦境 / 周易 / 风水 RAG —— 统一入口。

单一入口函数 run_pipeline(query) 串起完整流水线：
    1. 检索 (FAISS + BM25)
    2. 混合融合 (hybrid fusion)
    3. 重排 (rerank)
    4. 置信度计算 (confidence)
    5. 坏样本分类 (badcase classify → data/badcases_v2.jsonl)
    6. 生成最终答案 (LLM)

统一输出标准 JSON（仅这 5 个字段）：
    { answer, sources, confidence, error_type, retrieval_breakdown }
（retrieval_score 等内部信号保存在 retrieval_breakdown.top_faiss 中，不再单列）

CLI：
    python app.py "梦见水很多是什么意思"
    python app.py                 # 无参数进入交互模式
"""
import sys
import json
import contextlib
from pathlib import Path

import embed
import index
from retriever import hybrid_topk, faiss_topk, bm25_topk, faiss_available  # 融合 + 原始两路 + 能力探测
from reranker import rerank                 # 重排
from prompt import build_rag_prompt
from llm import call_deepseek
from parser import parse_json_response, FALLBACK_REASON
from evaluator import badcase_v2                           # v2 自动分类 + 落盘
from evaluator.stats import format_stats

VERSION = "v1.0-stable"     # 冻结基线版本标记（详见 docs/CHANGELOG.md）

BASE_DIR = Path(__file__).resolve().parent
EMB_PATH = BASE_DIR / "data" / "embeddings.npy"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"
KB_PATH = BASE_DIR / "data" / "dream_kb.json"

TOP_K = 5            # 最终送入生成 / 输出的条数
CANDIDATE_K = 10     # 粗排候选数（hybrid → rerank）
THRESHOLD = 0.35     # 最高 FAISS 余弦分低于此 → 门控拒答
LEX_OVERLAP_THRESHOLD = 0.30  # BM25-only 降级：top1 实义词命中率低于此 → 门控拒答
                              # （经离题/对题样本标定：对题≥0.33，离题≤0.25，0.30 干净分隔）

_id2rec = None


def _kb() -> dict:
    """惰性加载 id -> 记录(含 content/source) 查表。"""
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


def ensure_ready() -> None:
    """缺向量就生成，缺索引就构建；语义模型不可用时跳过（云端 BM25-only 降级）。"""
    if not embed.semantic_available():
        print(">> 语义模型不可用，进入 BM25-only 降级模式（云端演示），跳过向量/索引构建。")
        return
    if not EMB_PATH.exists():
        print(">> 未发现向量，开始生成 ...")
        embed.main()
    if not INDEX_PATH.exists():
        print(">> 未发现索引，开始构建 ...")
        index.main()


def cloud_mode() -> bool:
    """是否处于云端降级模式（语义检索不可用，仅 BM25）。供 UI 顶部横幅判断。

    消费端（streamlit / pages）经此函数从统一入口获知模式，无需直接 import retriever，
    符合单一管线架构（arch_check 守卫）。
    """
    return not faiss_available()


def _clean_sources(model_sources, hits: list[dict]) -> list[dict]:
    """用真实检索命中校验模型 sources：丢弃编造 id，重建 topic/source，附融合分。"""
    by_id = {h["id"]: h for h in hits}
    kb = _kb()
    cleaned, seen = [], set()
    for s in model_sources or []:
        cid = s.get("id") if isinstance(s, dict) else None
        if cid in by_id and cid not in seen:
            seen.add(cid)
            h, rec = by_id[cid], kb.get(cid, {})
            cleaned.append({
                "id": cid,
                "topic": rec.get("topic", h["topic"]),
                "source": rec.get("source", ""),
                "score": h["hybrid_score"],
            })
    return cleaned


def _compute_confidence(hits: list[dict], semantic: bool,
                        top_faiss: float, bm25_hit: bool,
                        gate_score: float, gate_threshold: float):
    """基于检索信号的客观置信度。返回 (retrieval_score, coverage_factor, final)。

    semantic=True ：用 FAISS 余弦分判定覆盖度（本地完整模式）。
    semantic=False：BM25-only 降级模式——无语义交叉验证，实义词命中率达标则覆盖度封顶 0.6。
    """
    if not hits:
        return 0.0, 0.3, 0.0
    top1_hybrid = hits[0]["hybrid_score"]
    if semantic:
        if top_faiss < gate_threshold:
            coverage = 0.3
        elif top_faiss > 0.4 and bm25_hit:
            coverage = 1.0
        else:
            coverage = 0.7
    else:
        coverage = 0.6 if gate_score >= gate_threshold else 0.3
    return round(top1_hybrid, 3), coverage, round(top1_hybrid * coverage, 3)


def run_pipeline(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD,
                 debug_mode: bool = False) -> dict:
    """RAG 统一入口：检索 → 融合 → 重排 → 置信度 → 分类 → 生成。

    debug_mode=True 时在标准输出外追加 debug 字段（faiss_topk / bm25_topk /
    hybrid_scores / rerank_scores / threshold_decision_trace），仅供开发调试，
    默认 False，不影响 UI 与标准 JSON。
    """
    # ---- 1 & 2. 检索 (FAISS + BM25) + 混合融合 ----
    hits = hybrid_topk(query, top_k=CANDIDATE_K)

    # 语义路是否真正参与：faiss_score 全为 0 → 云端 BM25-only 降级模式
    semantic_on = any(h.get("faiss_score", 0) > 0 for h in hits)

    # ---- 3. 重排 ----
    reranked = rerank(query, hits, top_k=top_k)

    # ---- 4. 门控信号 + 置信度 ----
    top_faiss = round(max((h["faiss_score"] for h in hits), default=0.0), 3)
    top_bm25 = round(max((h["bm25_score"] for h in hits), default=0.0), 3)
    bm25_hit = top_bm25 > 0

    # 门控信号按模式选择：
    #   语义模式 → FAISS 余弦分 vs THRESHOLD；
    #   BM25-only 降级 → top1 实义词命中率 vs LEX_OVERLAP_THRESHOLD
    #   （纯靠单字虚词命中的离题问题命中率≈0，会被门控拦下，而原始 BM25 分无法区分）。
    if semantic_on:
        gate_score, gate_threshold = top_faiss, threshold
    else:
        gate_score = round(hits[0].get("lex_overlap", 0.0), 3) if hits else 0.0
        gate_threshold = LEX_OVERLAP_THRESHOLD

    rs, coverage, confidence = _compute_confidence(
        hits, semantic_on, top_faiss, bm25_hit, gate_score, gate_threshold)

    by_id = {h["id"]: h for h in hits}
    candidates = [{
        "id": r["id"], "topic": r["topic"],
        "faiss_score": by_id.get(r["id"], {}).get("faiss_score", 0.0),
        "bm25_score": by_id.get(r["id"], {}).get("bm25_score", 0.0),
        "hybrid_score": r["hybrid_score"],
        "rerank_score": r["rerank_score"],
        "final_score": r["final_score"],
    } for r in reranked]
    retrieval_breakdown = {
        "mode": "faiss+bm25" if semantic_on else "bm25_only",
        "top_faiss": top_faiss,
        "top_bm25": top_bm25,
        "bm25_hit": bm25_hit,
        "coverage_factor": coverage,
        "candidates": candidates,
    }

    # ---- 调试信息（仅 debug_mode 计算与返回，不影响标准输出 / UI）----
    debug = None
    if debug_mode:
        faiss_raw = faiss_topk(query, top_k=CANDIDATE_K)
        bm25_raw = bm25_topk(query, top_k=CANDIDATE_K)
        gate_passed = bool(hits) and gate_score >= gate_threshold
        debug = {
            "faiss_topk": [{"id": r["id"], "topic": r["topic"],
                            "similarity_score": round(r["similarity_score"], 4)} for r in faiss_raw],
            "bm25_topk": [{"id": r["id"], "topic": r["topic"],
                           "bm25_score": round(r["bm25_score"], 4)} for r in bm25_raw],
            "hybrid_scores": [{"id": h["id"], "topic": h["topic"],
                               "hybrid_score": round(h["hybrid_score"], 4)} for h in hits],
            "rerank_scores": [{"id": r["id"], "topic": r["topic"],
                               "rerank_score": r["rerank_score"], "final_score": r["final_score"]}
                              for r in reranked],
            "threshold_decision_trace": {
                "mode": "faiss+bm25" if semantic_on else "bm25_only",
                "gate_score": gate_score,
                "gate_threshold": gate_threshold,
                "top_faiss": top_faiss,
                "top_bm25": top_bm25,
                "gate_passed": gate_passed,
                "decision": ("gate_score ≥ threshold → 进入生成阶段" if gate_passed
                             else "gate_score < threshold → 触发 unknown_gate_trigger"),
            },
        }

    def _finalize(full: dict) -> dict:
        """---- 5. 坏样本分类 + 落盘 v2，然后裁剪为标准 5 字段输出 ----

        full 内部携带 unknown / reason / retrieval_score 供分类与日志使用，
        但对外只暴露标准 JSON：answer, sources, confidence, error_type, retrieval_breakdown。
        debug_mode 时再附加 debug 字段。
        """
        et = badcase_v2.classify_error(full, gate_threshold)
        if et:
            badcase_v2.log_badcase_v2(query, full, et)
        out = {
            "answer": full.get("answer", ""),
            "sources": full.get("sources", []),
            "confidence": full.get("confidence", 0.0),
            "error_type": et or "",
            "retrieval_breakdown": full.get("retrieval_breakdown", {}),
        }
        if debug is not None:
            out["debug"] = debug
        return out

    # 门控：检索不达标 → unknown（不调用 LLM）
    if not hits or gate_score < gate_threshold:
        gate_name = "FAISS相似度" if semantic_on else "BM25词法分"
        return _finalize({
            "answer": "知识库中没有找到足够相关的信息，暂时无法判断。",
            "sources": [],
            "confidence": confidence,
            "unknown": True,
            "error_type": "",
            "retrieval_breakdown": retrieval_breakdown,
            "reason": f"最高{gate_name} {gate_score} 低于阈值 {gate_threshold}。",
            "retrieval_score": gate_score,
        })

    # ---- 6. 生成最终答案（用 rerank 后的候选作为上下文）----
    kb = _kb()
    contexts = []
    for r in reranked:
        rec = kb.get(r["id"])
        if rec:
            contexts.append({"id": r["id"], "topic": rec["topic"],
                             "source": rec.get("source", ""), "content": rec["content"]})

    try:
        raw = call_deepseek(build_rag_prompt(query, contexts))
    except Exception as e:
        return _finalize({
            "answer": "模型服务暂时不可用，请稍后重试。",
            "sources": [],
            "confidence": 0.0,
            "unknown": True,
            "error_type": "",
            "retrieval_breakdown": retrieval_breakdown,
            "reason": f"LLM调用失败：{e}",
            "retrieval_score": gate_score,
        })

    parsed = parse_json_response(raw)
    unknown = bool(parsed.get("unknown", False)) or (parsed.get("reason") == FALLBACK_REASON)
    return _finalize({
        "answer": parsed.get("answer", ""),
        "sources": _clean_sources(parsed.get("sources"), hits),
        "confidence": confidence,
        "unknown": unknown,
        "error_type": "",
        "retrieval_breakdown": retrieval_breakdown,
        "reason": parsed.get("reason", ""),
        "retrieval_score": gate_score,
    })


# 向后兼容：Streamlit (streamlit_app.py) 仍调用 rag_answer
rag_answer = run_pipeline


def main() -> None:
    ensure_ready()

    # CLI 一次性模式：python app.py "查询" [--debug]
    if len(sys.argv) > 1:
        debug = "--debug" in sys.argv[1:]
        query = " ".join(a for a in sys.argv[1:] if a != "--debug").strip()
        with contextlib.redirect_stdout(sys.stderr):   # 让模型加载等杂讯走 stderr，stdout 纯 JSON
            result = run_pipeline(query, debug_mode=debug)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 交互模式
    print(f"\n=== RAG 统一入口 · {VERSION} ===  输入查询，/stats 查看统计，q 退出\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"q", "quit", "exit"}:
            break
        if query.lower() == "/stats":
            print(format_stats())
            print("-" * 60 + "\n")
            continue
        print(json.dumps(run_pipeline(query), ensure_ascii=False, indent=2))
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
