# -*- coding: utf-8 -*-
"""梦境解析 RAG · V2 —— 检索 + 生成 + 结构化输出 + 检索分/阈值机制。

运行前设置 Key：
    $env:DEEPSEEK_API_KEY = "你的key"
运行：
    python app.py
"""
import json
from pathlib import Path

import embed
import index
from search import search
from prompt import build_rag_prompt
from llm import call_deepseek
from parser import parse_json_response, FALLBACK_REASON
from badcase import log_badcase
from stats import format_stats

BASE_DIR = Path(__file__).resolve().parent
EMB_PATH = BASE_DIR / "data" / "embeddings.npy"
INDEX_PATH = BASE_DIR / "data" / "faiss.index"

# ---- V2 可调参数 ----
TOP_K = 5          # 每次检索返回多少条候选
THRESHOLD = 0.35   # 最高检索分低于此值 → 视为“无相关知识”，unknown=true


def ensure_ready() -> None:
    """缺向量就生成，缺索引就构建。"""
    if not EMB_PATH.exists():
        print(">> 未发现向量，开始生成 ...")
        embed.main()
    if not INDEX_PATH.exists():
        print(">> 未发现索引，开始构建 ...")
        index.main()


def _clean_sources(model_sources, contexts: list[dict]) -> list[dict]:
    """用真实检索结果校验模型 sources：丢弃编造的 id，重建 topic/source，并附上该来源的检索分。"""
    by_id = {c["id"]: c for c in contexts}
    cleaned, seen = [], set()
    for s in model_sources or []:
        cid = s.get("id") if isinstance(s, dict) else None
        if cid in by_id and cid not in seen:
            seen.add(cid)
            rec = by_id[cid]
            cleaned.append({
                "id": cid,
                "topic": rec["topic"],
                "source": rec["source"],
                "score": round(rec["similarity_score"], 3),   # 每条来源的检索分
            })
    return cleaned


def rag_answer(query: str, top_k: int = TOP_K, threshold: float = THRESHOLD) -> dict:
    # 1. 检索 top_k 候选（每条带 similarity_score）
    results = search(query, top_k=top_k)
    top_score = round(results[0]["similarity_score"], 3) if results else 0.0

    # 2. 阈值门：最高检索分不达标 → unknown（不浪费一次 LLM 调用）
    if top_score < threshold:
        result = {
            "answer": "知识库中没有找到足够相关的信息，暂时无法判断。",
            "sources": [], "confidence": 0.0, "unknown": True,
            "reason": f"最高检索分 {top_score} 低于阈值 {threshold}。",
            "retrieval_score": top_score,
        }
        log_badcase(query, result, "low_retrieval_score")
        return result

    # 3. 只保留达标的候选喂给模型
    contexts = [r for r in results if r["similarity_score"] >= threshold]

    # 4. 构建 prompt + 调用 LLM
    prompt = build_rag_prompt(query, contexts)
    try:
        raw_answer = call_deepseek(prompt)
    except Exception as e:
        result = {
            "answer": "模型服务暂时不可用，请稍后重试。",
            "sources": [], "confidence": 0.0, "unknown": True,
            "reason": f"LLM调用失败：{e}",
            "retrieval_score": top_score,
        }
        log_badcase(query, result, "llm_api_error")
        return result

    # 5. 解析 + 注入检索分
    result = parse_json_response(raw_answer)
    result["retrieval_score"] = top_score          # 最高检索分

    # 5.1 JSON 解析失败 → 记 json_parse_error，直接返回兜底
    if result.get("reason") == FALLBACK_REASON:
        log_badcase(query, result, "json_parse_error")
        return result

    # 6. 校验来源 + bad case 记录
    result["sources"] = _clean_sources(result.get("sources"), contexts)
    if result.get("unknown") is True:
        log_badcase(query, result, "unknown_answer")
    else:
        if not result.get("sources"):
            log_badcase(query, result, "missing_citation")
        conf = result.get("confidence")
        if isinstance(conf, (int, float)) and conf < 0.5:
            log_badcase(query, result, "low_confidence")

    return result


def main() -> None:
    ensure_ready()
    print("\n=== 梦境解析助手（RAG · V3）===")
    print(f"top_k={TOP_K}  threshold={THRESHOLD}")
    print("描述你的梦境，回车获取解读；输入 /stats 查看统计，q 退出\n")
    while True:
        try:
            query = input("梦境> ").strip()
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
        result = rag_answer(query)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
