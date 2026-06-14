# -*- coding: utf-8 -*-
"""📊 System Showcase —— 一页看懂系统能力（v1.1）。

只调用 app.run_pipeline（不碰检索/重排内部），符合单一管线架构。
内容：架构图 / pipeline 流程 / 4 核心指标 / 10 条 demo 快速切换 / 一键批量。
"""
import sys
import json
import contextlib
from pathlib import Path

import streamlit as st

from evaluator import badcase_v2
from app import run_pipeline, ensure_ready, VERSION

st.set_page_config(page_title="📊 System Showcase", page_icon="📊", layout="wide")

if "ready" not in st.session_state:
    ensure_ready()
    st.session_state["ready"] = True

BASE = Path(__file__).resolve().parent.parent
DEMO_PATH = BASE / "data" / "demo_queries.json"
UNKNOWN_ERRORS = {"unknown_gate_trigger", "retrieval_fail"}

st.title("📊 System Showcase")
st.caption(f"基于 RAG 的文化知识增强问答系统　·　{VERSION}")

# ---- 1. 系统架构（文字版）----
st.subheader("① 系统架构")
st.code(
    """
                 ┌─────────── app.run_pipeline(query) ── 唯一入口 ───────────┐
   用户查询  ──▶  │  检索(FAISS 语义 + BM25 词法) → Hybrid 融合 → Rerank 重排   │ ──▶ 标准 JSON
                 │  → 置信度计算 → 错误分类 → LLM 生成                          │
                 └──────────────────────────────────────────────────────────┘
   三入口共用 ：   CLI(app.py)     网页(streamlit)     批量评测(evaluator/evaluate.py)
   实现层     ：   retriever/(faiss·bm25·hybrid)   reranker.py   evaluator/badcase_v2.py
   守卫       ：   arch_check.py（锁定单一管线，违规即拦截）
""",
    language="text",
)

# ---- 2. Pipeline 流程（run_pipeline）----
st.subheader("② Pipeline 流程（run_pipeline）")
st.code(
    """
query
 │ 1. retrieval      FAISS top-k  ＋  BM25 top-k
 │ 2. hybrid fusion  归一化后  0.7*faiss + 0.3*bm25
 │ 3. rerank         cross-encoder（缺模型→mock）  0.5*hybrid + 0.5*rerank
 │ 4. confidence     top1_hybrid × coverage_factor(1.0 / 0.7 / 0.3)
 │ 5. classify       unknown_gate / retrieval_fail / hallucination / low_conf
 │ 6. generate       LLM 基于检索内容生成（接地、可拒答）
 ▼
{ answer, sources, confidence, error_type, retrieval_breakdown }
""",
    language="text",
)

# ---- 3. 4 个核心指标卡片 ----
st.subheader("③ 核心指标（运行 Demo Batch 后生成）")
m = st.session_state.get("showcase_metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("avg confidence", m["avg_confidence"] if m else "—")
c2.metric("unknown rate", m["unknown_rate"] if m else "—")
c3.metric("retrieval_fail rate", m["retrieval_fail_rate"] if m else "—")
c4.metric("hallucination risk", m["hallucination_risk_rate"] if m else "—")

# ---- 加载 demo 集 ----
try:
    cases = json.loads(DEMO_PATH.read_text(encoding="utf-8"))
except Exception as e:
    st.error(f"读取 demo_queries.json 失败：{e}")
    cases = []

# ---- 5. 一键运行 Demo Batch ----
st.subheader("④ Demo")
if st.button("▶ 运行 Demo Batch（10 条）", type="primary") and cases:
    _orig = badcase_v2.log_badcase_v2
    badcase_v2.log_badcase_v2 = lambda *a, **k: None     # 展示期不落盘
    rows = []
    try:
        prog = st.progress(0.0, "运行中 ...")
        for i, c in enumerate(cases, 1):
            with contextlib.redirect_stdout(sys.stderr):
                r = run_pipeline(c["query"])
            rows.append({
                "category": c["category"],
                "query": c["query"],
                "error_type": r["error_type"] or "ok",
                "confidence": r["confidence"],
                "retrieval_score": r["retrieval_breakdown"]["top_faiss"],
            })
            prog.progress(i / len(cases), f"{i}/{len(cases)}")
        prog.empty()
    finally:
        badcase_v2.log_badcase_v2 = _orig

    n = len(rows) or 1
    st.session_state["showcase_metrics"] = {
        "avg_confidence": round(sum(x["confidence"] for x in rows) / n, 3),
        "unknown_rate": round(sum(x["error_type"] in UNKNOWN_ERRORS for x in rows) / n, 3),
        "retrieval_fail_rate": round(sum(x["error_type"] == "retrieval_fail" for x in rows) / n, 3),
        "hallucination_risk_rate": round(sum(x["error_type"] == "hallucination_risk" for x in rows) / n, 3),
    }
    st.session_state["showcase_rows"] = rows
    st.rerun()

if st.session_state.get("showcase_rows"):
    st.dataframe(st.session_state["showcase_rows"], use_container_width=True, hide_index=True)

# ---- 4. Demo case 快速切换（10 条）----
st.markdown("**快速切换 Demo（点按钮单条运行）**")
cols = st.columns(2)
for i, c in enumerate(cases):
    label = f"{c['category']}｜{c['query'][:14]}…"
    if cols[i % 2].button(label, key=f"demo_{c['id']}", use_container_width=True):
        with contextlib.redirect_stdout(sys.stderr):
            st.session_state["single_result"] = (c, run_pipeline(c["query"]))

sr = st.session_state.get("single_result")
if sr:
    c, r = sr
    st.divider()
    st.markdown(f"**[{c['category']}]** {c['query']}")
    st.write(r.get("answer", ""))
    st.caption(
        f"confidence = {r['confidence']}　·　error_type = {r['error_type'] or '正常'}"
        f"　·　top_faiss = {r['retrieval_breakdown']['top_faiss']}"
    )
