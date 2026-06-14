# -*- coding: utf-8 -*-
"""STEP 4 —— Bad case 系统 v2：自动分类 + 结构化字段，写入 data/badcases_v2.jsonl。

error_type（按优先级级联，每条只归一类）：
  · unknown_gate_trigger : unknown 且 retrieval_score < threshold —— 检索分不达标，门控直接拒答
  · retrieval_fail       : unknown 但 retrieval_score >= threshold —— 过了检索门槛却没产出有据答案
                           （LLM 调用失败 / JSON 解析失败等管线故障）
  · hallucination_risk   : 已作答(unknown=false) 但 sources 为空 —— 答案无引用支撑，编造风险
  · low_confidence       : 已作答且有来源，但 confidence < 0.5 —— 置信不足
  · None                 : 正常答案（有来源、置信达标），不记录

保存字段：query, retrieval_score, confidence, final_answer, error_type, timestamp
"""
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "badcases_v2.jsonl"
CONF_THRESHOLD = 0.5


def classify_error(result: dict, threshold: float) -> str | None:
    """自动判定 error_type；正常答案返回 None。"""
    unknown = bool(result.get("unknown"))
    rs = result.get("retrieval_score", 0.0)
    conf = result.get("confidence", 0.0)
    sources = result.get("sources") or []

    if unknown:
        if isinstance(rs, (int, float)) and rs < threshold:
            return "unknown_gate_trigger"
        return "retrieval_fail"
    if not sources:
        return "hallucination_risk"
    if isinstance(conf, (int, float)) and conf < CONF_THRESHOLD:
        return "low_confidence"
    return None


def log_badcase_v2(query: str, result: dict, error_type: str) -> None:
    """写入一条 v2 bad case（6 字段）。写日志失败不影响主流程。"""
    record = {
        "query": query,
        "retrieval_score": result.get("retrieval_score", 0.0),
        "confidence": result.get("confidence", 0.0),
        "final_answer": result.get("answer", ""),
        "error_type": error_type,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[badcase_v2] 写日志失败（已忽略）：{e}")


def autolog(query: str, result: dict, threshold: float) -> str | None:
    """自动分类 + 落盘（非 bad case 不写）。返回判定的 error_type（或 None）。"""
    error_type = classify_error(result, threshold)
    if error_type:
        log_badcase_v2(query, result, error_type)
    return error_type
