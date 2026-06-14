# -*- coding: utf-8 -*-
"""容错解析模型返回的 JSON，并保证输出符合 schema。

注意：'parser' 是个很常见的名字，若以后和其他库冲突，可改名为 json_parser.py。
"""
import json
import re

# 解析失败兜底结果的 reason —— app.py 用它判断“是否发生了 JSON 解析错误”
FALLBACK_REASON = "模型输出不是合法JSON。"

_FALLBACK = {
    "answer": "系统暂时无法生成结构化结果，请稍后重试。",
    "sources": [],
    "confidence": 0.0,
    "unknown": True,
    "reason": FALLBACK_REASON,
}


def _ensure_schema(d: dict) -> dict:
    """补齐缺失字段：合法 JSON ≠ 合法 schema。"""
    return {
        "answer": d.get("answer", ""),
        "sources": d.get("sources", []),
        "confidence": d.get("confidence", 0.0),
        "unknown": d.get("unknown", False),
        "reason": d.get("reason", ""),
    }


def parse_json_response(raw_text) -> dict:
    """把模型输出解析成 schema 字典；任何异常都回退到 _FALLBACK，永不崩溃。"""
    if not isinstance(raw_text, str) or not raw_text.strip():   # 防 None/非字符串/空
        return dict(_FALLBACK)

    text = raw_text.strip()
    candidates = [text]
    m = re.search(r"\{[\s\S]*\}", text)                          # 兜底：抠出 {...}
    if m:
        candidates.append(m.group())

    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):                               # 挡掉 JSON 是数组/数字的情况
            return _ensure_schema(obj)

    return dict(_FALLBACK)
