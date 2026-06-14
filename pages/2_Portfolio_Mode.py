# -*- coding: utf-8 -*-
"""🚀 Portfolio Mode —— 极简产品发布页（v1.1）。

一页呈现：System Overview · 4 指标 · 可切换 Demo · Sources 表。
每个模块带一句"给面试官"的解释标签。只调用 run_pipeline（单一管线）。
默认自动加载 1 个最佳 demo。
"""
import sys
import json
import contextlib
from pathlib import Path

import streamlit as st

import badcase_v2
from app import run_pipeline, ensure_ready, VERSION

st.set_page_config(page_title="🚀 Portfolio Mode", page_icon="🚀", layout="wide")

if "ready" not in st.session_state:
    ensure_ready()
    st.session_state["ready"] = True

BASE = Path(__file__).resolve().parent.parent
DEMO_PATH = BASE / "data" / "demo_queries.json"
REPORT_PATH = BASE / "evaluation_report.json"
BEST_DEMO_ID = "demo_06"                              # 高置信案例（掉牙）作默认最佳 demo
UNKNOWN_ERRORS = {"unknown_gate_trigger", "retrieval_fail"}

cases = json.loads(DEMO_PATH.read_text(encoding="utf-8"))


def _run(query):
    with contextlib.redirect_stdout(sys.stderr):       # 模型加载杂讯走 stderr
        return run_pipeline(query)


# ---- 默认加载最佳 demo（仅首次进入）----
if "pm_result" not in st.session_state:
    best = next((c for c in cases if c["id"] == BEST_DEMO_ID), cases[0])
    st.session_state["pm_case"] = best
    st.session_state["pm_result"] = _run(best["query"])

st.title("🚀 Portfolio Mode")
st.caption(f"基于 RAG 的文化知识增强问答系统　·　{VERSION}")

# ===== 1. System Overview =====
with st.container(border=True):
    st.markdown("#### 🧭 System Overview")
    st.markdown("### 模糊的一句话 → **有出处、有把握、敢说不知道** 的回答")
    st.markdown("`Hybrid 检索 (FAISS+BM25)` → `Rerank` → `Confidence` → `Unknown Gate` → `LLM 生成`")
    st.caption("👉 给面试官：单一管线 `run_pipeline`，CLI / 网页 / 评测三端共用，架构守卫锁定，不可绕过。")

st.write("")

# ===== 2. 4 Metrics Cards =====
st.markdown("#### 📊 Quality Metrics")
st.caption("👉 给面试官：质量不靠感觉——批量评测得到的客观指标。")
metrics = st.session_state.get("pm_metrics")
if metrics is None and REPORT_PATH.exists():
    try:
        metrics = json.loads(REPORT_PATH.read_text(encoding="utf-8"))["summary"]
    except Exception:
        metrics = None
m1, m2, m3, m4 = st.columns(4)
m1.metric("Avg Confidence", metrics["avg_confidence"] if metrics else "—")
m2.metric("Unknown Rate", metrics["unknown_rate"] if metrics else "—")
m3.metric("Retrieval-Fail Rate", metrics["retrieval_fail_rate"] if metrics else "—")
m4.metric("Hallucination Risk", metrics["hallucination_risk_rate"] if metrics else "—")
if st.button("▶ 运行基准评测（10 条）"):
    _orig = badcase_v2.log_badcase_v2
    badcase_v2.log_badcase_v2 = lambda *a, **k: None
    rows = []
    try:
        with st.spinner("评测中 ..."):
            rows = [_run(c["query"]) for c in cases]
    finally:
        badcase_v2.log_badcase_v2 = _orig
    n = len(rows) or 1
    st.session_state["pm_metrics"] = {
        "avg_confidence": round(sum(x["confidence"] for x in rows) / n, 3),
        "unknown_rate": round(sum(x["error_type"] in UNKNOWN_ERRORS for x in rows) / n, 3),
        "retrieval_fail_rate": round(sum(x["error_type"] == "retrieval_fail" for x in rows) / n, 3),
        "hallucination_risk_rate": round(sum(x["error_type"] == "hallucination_risk" for x in rows) / n, 3),
    }
    st.rerun()

st.write("")

# ===== 3. Live Demo（可切换）=====
st.markdown("#### 🎯 Live Demo")
st.caption("👉 给面试官：切换不同类型的真实问题，看系统如何应对——包括对跑题问题直接拒答。")
labels = [f"{c['category']}｜{c['query'][:18]}" for c in cases]
default_idx = next((i for i, c in enumerate(cases)
                    if c["id"] == st.session_state["pm_case"]["id"]), 0)
sel = st.selectbox("选择 demo", range(len(cases)), index=default_idx,
                   format_func=lambda i: labels[i])
if cases[sel]["id"] != st.session_state["pm_case"]["id"]:     # 仅切换时重跑
    st.session_state["pm_case"] = cases[sel]
    st.session_state["pm_result"] = _run(cases[sel]["query"])

res = st.session_state["pm_result"]
case = st.session_state["pm_case"]
with st.container(border=True):
    st.markdown(f"**Q：{case['query']}**　`{case['category']}`")
    st.write(res.get("answer", ""))
    d1, d2, d3 = st.columns(3)
    d1.metric("Confidence", res.get("confidence", 0.0))
    d2.metric("error_type", res.get("error_type") or "正常")
    d3.metric("Retrieval Score", res.get("retrieval_breakdown", {}).get("top_faiss", 0.0))

st.write("")

# ===== 4. Sources Table =====
st.markdown("#### 📚 Sources")
st.caption("👉 给面试官：每条答案都可溯源——标注来源与 FAISS / BM25 / 融合各路得分。")
cands = res.get("retrieval_breakdown", {}).get("candidates", [])
if cands:
    st.dataframe(
        [{"id": c["id"], "topic": c["topic"],
          "faiss": round(c["faiss_score"], 3), "bm25": round(c["bm25_score"], 3),
          "hybrid": round(c["hybrid_score"], 3), "final": round(c["final_score"], 3)}
         for c in cands],
        use_container_width=True, hide_index=True,
    )
else:
    st.caption("（本次无检索候选）")
