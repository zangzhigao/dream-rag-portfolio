# -*- coding: utf-8 -*-
"""V3 —— Bad case 统计（Evaluation / Monitoring 的最小实现）。

读取 data/badcases.jsonl，汇总各类问题计数，并列出最近若干条。
单独运行：
    python stats.py
也被 app.py 的 /stats 命令调用。
"""
import json
from collections import Counter
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "data" / "badcases.jsonl"


def load_badcases() -> list[dict]:
    """读取 JSONL 日志；文件不存在或某行损坏都安全跳过。"""
    if not LOG_PATH.exists():
        return []
    records = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _confidence(record: dict):
    c = (record.get("result") or {}).get("confidence")
    return c if isinstance(c, (int, float)) else None


def _is_answered(record: dict) -> bool:
    return not (record.get("result") or {}).get("unknown", False)


def compute_stats(records: list[dict]) -> dict:
    by_tag = Counter(r.get("tag", "未知类型") for r in records)

    # 低置信度：模型给了答案(unknown=false)但 confidence < 0.5（排除错误/无答案的兜底 0.0）
    low_conf = sum(
        1 for r in records
        if _is_answered(r) and (_confidence(r) is not None) and _confidence(r) < 0.5
    )

    return {
        "total": len(records),
        "unknown_answer": by_tag.get("unknown_answer", 0),
        "missing_citation": by_tag.get("missing_citation", 0),
        "llm_api_error": by_tag.get("llm_api_error", 0),
        "json_parse_error": by_tag.get("json_parse_error", 0),
        "low_confidence": low_conf,
        "by_tag": dict(by_tag),
    }


def format_stats(records: list[dict] | None = None, recent: int = 10) -> str:
    if records is None:
        records = load_badcases()
    s = compute_stats(records)

    lines = ["=== Bad Case 统计 ==="]
    lines.append(f"总数               : {s['total']}")
    lines.append(f"unknown_answer     : {s['unknown_answer']}")
    lines.append(f"missing_citation   : {s['missing_citation']}")
    lines.append(f"llm_api_error      : {s['llm_api_error']}")
    lines.append(f"json_parse_error   : {s['json_parse_error']}")
    lines.append(f"低置信度(答了但<0.5): {s['low_confidence']}")

    # 其它类型（如 low_retrieval_score / low_confidence 标签等）一并展示
    named = {"unknown_answer", "missing_citation", "llm_api_error", "json_parse_error"}
    others = {k: v for k, v in s["by_tag"].items() if k not in named}
    if others:
        lines.append("其它类型           : " + "，".join(f"{k}={v}" for k, v in others.items()))

    lines.append(f"\n最近 {recent} 条：")
    if not records:
        lines.append("  （暂无 bad case）")
    else:
        for r in records[-recent:][::-1]:
            lines.append(f"  [{r.get('time', '')}] {str(r.get('tag', '')):18s} {r.get('query', '')}")
    return "\n".join(lines)


def main() -> None:
    print(format_stats())


if __name__ == "__main__":
    main()
