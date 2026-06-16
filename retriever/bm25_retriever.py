# -*- coding: utf-8 -*-
"""BM25 词法检索（独立模块，与 FAISS 语义检索互不影响）。

- FAISS：基于向量的"语义"相似（懂同义、近义）。
- BM25 ：基于词频的"字面"匹配（命中关键词强，速度快，零模型依赖）。

中文需先分词，这里用 jieba；打分用 rank_bm25 的 BM25Okapi。
语料与 FAISS 相同（data/dream_kb.json 的全部 264 条），保证两路检索覆盖同一知识库。
"""
import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

# 项目根：retriever/ 的上一级
BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "dream_kb.json"

_bm25 = None
_records = None


def _load_records() -> list[dict]:
    recs = []
    with open(KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _doc_text(rec: dict) -> str:
    """用于 BM25 匹配的文本：主题 + 关键词/标签 + 正文。
    与向量侧不同——词法匹配下正文越全，关键词命中率越高。"""
    kws = rec.get("keywords") or rec.get("tags") or []
    return f"{rec.get('topic', '')} {' '.join(kws)} {rec.get('content', '')}"


def _tokenize(text: str) -> list[str]:
    """jieba 分词，去掉空白 token。"""
    return [t for t in jieba.lcut(text) if t.strip()]


def query_overlap(query: str, rec: dict) -> float:
    """查询"实义词"在文档(主题+关键词+正文)中的命中率，范围 [0,1]。

    BM25-only 降级模式下的"语义近似"门控信号：只统计多字词（len≥2，近似实义词），
    过滤掉"的/去/我/票"等单字虚词——离题问题往往仅靠虚词凑出 BM25 分，
    实义词命中率却接近 0，借此把它们与真正对题的问题区分开。
    查询无多字词时（如纯单字/数字），退回用全部 token，避免误杀。
    """
    q = {t for t in _tokenize(query) if len(t) >= 2}
    if not q:
        q = set(_tokenize(query))
    if not q:
        return 0.0
    d = set(_tokenize(_doc_text(rec)))
    return len(q & d) / len(q)


def _build():
    """惰性构建 BM25 索引，只建一次。"""
    global _bm25, _records
    if _bm25 is None:
        _records = _load_records()
        corpus = [_tokenize(_doc_text(r)) for r in _records]
        _bm25 = BM25Okapi(corpus)
    return _bm25, _records


def bm25_topk(query: str, top_k: int = 5) -> list[dict]:
    """返回 BM25 词法检索的前 top_k 条。

    每条是记录字典外加 'bm25_score'（BM25 相关度，越大越相关；
    无字面重合时可能为 0，与 FAISS 的余弦分不可直接比较）。
    """
    bm25, records = _build()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in order:
        rec = dict(records[i])
        rec["bm25_score"] = float(scores[i])
        results.append(rec)
    return results


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "梦见蛇追我"
    print(f"BM25 查询：{q}\n")
    for rank, r in enumerate(bm25_topk(q, top_k=5), 1):
        print(f"#{rank}  bm25={r['bm25_score']:.3f}  [{r['topic']}]  ({r['id']})")
