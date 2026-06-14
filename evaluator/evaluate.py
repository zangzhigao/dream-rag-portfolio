# -*- coding: utf-8 -*-
"""STEP 6 —— Evaluation mode（离线批量评测）。

从 data/test_queries.json 读取测试集，对每条 query 跑 run_pipeline，
收集 answer / confidence / error_type / retrieval_score，汇总统计后写入
evaluation_report.json。

用法：
    python -m evaluator.evaluate     # 在项目根目录运行

说明：评测期间临时关闭 v2 bad case 落盘，避免测试 query 污染线上 badcases_v2.jsonl。
"""
import sys
import json
import contextlib
from collections import Counter
from pathlib import Path

from app import run_pipeline, ensure_ready
from evaluator import badcase_v2

BASE_DIR = Path(__file__).resolve().parent.parent      # 项目根（evaluator/ 的上一级）
TEST_PATH = BASE_DIR / "data" / "test_queries.json"
REPORT_PATH = BASE_DIR / "evaluation_report.json"


def load_queries() -> list[str]:
    """支持 ["q1","q2"] 或 [{"query":"q1"}, ...] 两种格式。"""
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = []
    for item in data:
        if isinstance(item, str):
            queries.append(item)
        elif isinstance(item, dict) and item.get("query"):
            queries.append(item["query"])
    return queries


def evaluate() -> dict:
    queries = load_queries()

    # 评测期间关闭 v2 落盘（仍保留分类逻辑，只是不写文件）
    _orig_log = badcase_v2.log_badcase_v2
    badcase_v2.log_badcase_v2 = lambda *a, **k: None

    results = []
    try:
        for q in queries:
            with contextlib.redirect_stdout(sys.stderr):   # 模型加载杂讯走 stderr
                r = run_pipeline(q)
            results.append({
                "query": q,
                "answer": r.get("answer", ""),
                "confidence": r.get("confidence", 0.0),
                "error_type": r.get("error_type", ""),
                # retrieval_score 现统一从标准输出的 retrieval_breakdown 取
                "retrieval_score": (r.get("retrieval_breakdown") or {}).get("top_faiss", 0.0),
            })
    finally:
        badcase_v2.log_badcase_v2 = _orig_log

    # 标准输出不再单列 unknown，从 error_type 推导"未给出有效答案"的情形
    UNKNOWN_ERRORS = {"unknown_gate_trigger", "retrieval_fail"}
    n = len(results) or 1
    summary = {
        "avg_confidence": round(sum(x["confidence"] for x in results) / n, 3),
        "unknown_rate": round(sum(x["error_type"] in UNKNOWN_ERRORS for x in results) / n, 3),
        "retrieval_fail_rate": round(sum(x["error_type"] == "retrieval_fail" for x in results) / n, 3),
        "hallucination_risk_rate": round(sum(x["error_type"] == "hallucination_risk" for x in results) / n, 3),
        "error_type_distribution": dict(Counter(x["error_type"] or "ok" for x in results)),
    }

    report = {"total_queries": len(results), "summary": summary, "results": results}
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    ensure_ready()
    report = evaluate()
    s = report["summary"]
    print(f"\n=== Evaluation Report（{report['total_queries']} 条）===")
    print(f"avg_confidence          : {s['avg_confidence']}")
    print(f"unknown_rate            : {s['unknown_rate']}")
    print(f"retrieval_fail_rate     : {s['retrieval_fail_rate']}")
    print(f"hallucination_risk_rate : {s['hallucination_risk_rate']}")
    print(f"error_type 分布          : {s['error_type_distribution']}")
    print(f"\n明细已写入: {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
