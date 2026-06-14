# -*- coding: utf-8 -*-
"""把异常 / 低质量回答记录成日志，便于事后复盘改进。

每行一条 JSON（JSONL），追加写入 data/badcases.jsonl。
"""
import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "data" / "badcases.jsonl"


def log_badcase(query: str, result: dict, tag: str) -> None:
    """追加一条 bad case 记录。tag 标明类型，如 no_retrieval_result / llm_api_error 等。"""
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "query": query,
        "result": result,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
