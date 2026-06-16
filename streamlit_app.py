# -*- coding: utf-8 -*-
"""Streamlit 界面 —— 统一入口 run_pipeline + 批量评测（带配色与状态提示）。

页面元素：
  1. 查询输入框
  2. answer
  3. sources / 检索明细表格（FAISS蓝 / BM25绿 / rerank橙 三色列）
  4. confidence 条形图 + 分档配色徽标（<0.3红 / 0.3~0.7黄 / >0.7绿）
  5. error_type badge
  6. "Run Evaluation Mode" 按钮

启动：
    $env:DEEPSEEK_API_KEY = "你的key"
    streamlit run streamlit_app.py
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from app import run_pipeline, ensure_ready, cloud_mode
from evaluator import evaluate

st.set_page_config(page_title="RAG 统一问答与评测", page_icon="🔮")

# ---- Demo Snapshot 导出 ----
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "demo_cases"
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf", r"C:\Windows\Fonts\simsun.ttc",
]


def _font(size):
    for fp in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap(text, n):
    out = []
    for para in (text or "").split("\n"):
        if not para:
            out.append("")
            continue
        while len(para) > n:
            out.append(para[:n])
            para = para[n:]
        out.append(para)
    return out


def _render_png(snap, path):
    """把一条 demo 渲染成快照卡片 PNG（非浏览器截图，用 Pillow 绘制）。"""
    W, pad, lh = 900, 30, 38
    f_title, f_h, f_b = _font(30), _font(22), _font(18)
    rows = [
        ("Demo Snapshot", f_title, (25, 25, 25)),
        (f"时间：{snap['timestamp']}", f_b, (130, 130, 130)),
        ("", f_b, (0, 0, 0)),
        ("Query", f_h, (70, 70, 70)),
    ]
    rows += [(ln, f_b, (0, 0, 0)) for ln in _wrap(snap["query"], 42)]
    conf = float(snap["confidence"])
    cc = (231, 76, 60) if conf < 0.3 else ((243, 156, 18) if conf <= 0.7 else (39, 174, 96))
    rows += [
        ("", f_b, (0, 0, 0)),
        (f"Confidence: {conf:.3f}    error_type: {snap['error_type'] or '正常'}", f_h, cc),
        ("", f_b, (0, 0, 0)),
        ("Answer", f_h, (70, 70, 70)),
    ]
    rows += [(ln, f_b, (0, 0, 0)) for ln in _wrap(snap["answer"], 42)]
    rows += [("", f_b, (0, 0, 0)), ("Sources", f_h, (70, 70, 70))]
    for s in snap["sources"]:
        rows.append((f"· {s.get('id', '')}  {s.get('topic', '')}  (score={round(float(s.get('score', 0)), 3)})",
                     f_b, (40, 40, 40)))

    H = pad * 2 + len(rows) * lh
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=(106, 90, 205))   # 顶部紫色条
    y = pad
    for text, font, color in rows:
        d.text((pad, y), text, font=font, fill=color)
        y += lh
    img.save(path)


def export_snapshot(snap):
    """保存 JSON + PNG 到 outputs/demo_cases/，返回两个路径。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"demo_{stamp}.json"
    png_path = OUTPUT_DIR / f"demo_{stamp}.png"
    json_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    _render_png(snap, png_path)
    return json_path, png_path

if "ready" not in st.session_state:
    ensure_ready()
    st.session_state["ready"] = True

# 配色
COL_FAISS = "#dbeafe"   # 蓝
COL_BM25 = "#dcfce7"    # 绿
COL_RERANK = "#ffedd5"  # 橙
ET_COLORS = {
    "": "#27ae60",
    "low_confidence": "#f39c12",
    "unknown_gate_trigger": "#7f8c8d",
    "retrieval_fail": "#e74c3c",
    "hallucination_risk": "#c0392b",
}


def _badge(text, color):
    return (f"<span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:8px;font-weight:600'>{text}</span>")


def _conf_color(c):
    return "#e74c3c" if c < 0.3 else ("#f39c12" if c <= 0.7 else "#27ae60")


st.title("🔮 RAG 统一问答系统")
st.caption("混合检索 (FAISS + BM25) → 重排 → 生成 ｜ 含置信度与错误分类 ｜ v1.0-stable")

# ---- 云端降级模式横幅（语义模型不可用时）----
if cloud_mode():
    st.warning(
        "☁️ 当前为云端演示模式：使用 BM25 关键词检索，"
        "完整本地版支持 FAISS + BM25 + Rerank。"
    )

# ---- 顶部：系统说明卡片 ----
with st.container(border=True):
    st.markdown(
        """
##### 📖 本系统：基于 RAG 的文化知识增强问答系统

**能力包括**
- 🔵 语义检索（FAISS）
- 🟢 关键词检索（BM25）
- 🟣 混合检索（Hybrid）
- 🟠 重排序（Rerank）
- 📊 置信度评估（Confidence Calibration）
- ⚠️ 异常识别（Error Classification）

**目标**：将模糊的自然语言问题 → 可解释的知识检索结果
"""
    )

# ---- 1. 查询输入框 ----
query = st.text_input("输入查询", placeholder="例如：梦见水很多是什么意思")

if st.button("开始解析", type="primary"):
    if query.strip():
        # ---- 4(spinner). 状态提示：语义检索 → rerank ----
        with st.status("处理中 ...", expanded=True) as status:
            st.write("🔍 正在进行语义检索...")
            st.write("🔀 正在 rerank...")
            st.session_state["result"] = run_pipeline(query.strip())
            st.session_state["query"] = query.strip()
            status.update(label="✅ 完成", state="complete", expanded=False)
    else:
        st.warning("请先输入查询。")

result = st.session_state.get("result")
if result:
    st.divider()

    # ---- 5. error_type badge ----
    et = result.get("error_type", "")
    et_label = et if et else "正常"
    st.markdown("error_type：" + _badge(et_label, ET_COLORS.get(et, "#7f8c8d")),
                unsafe_allow_html=True)

    # ---- 2. answer ----
    st.subheader("Answer")
    st.write(result.get("answer", ""))

    # ---- 4. confidence：分档配色徽标 + 条形图 ----
    st.subheader("Confidence")
    conf = float(result.get("confidence", 0.0))
    rs = float((result.get("retrieval_breakdown") or {}).get("top_faiss", 0.0))
    st.markdown(
        _badge(f"confidence = {conf:.3f}", _conf_color(conf))
        + "　<span style='color:#888'>（&lt;0.3 红 / 0.3~0.7 黄 / &gt;0.7 绿）</span>",
        unsafe_allow_html=True,
    )
    st.bar_chart(pd.DataFrame({"分数": [conf, rs]}, index=["confidence", "retrieval_score"]))

    # ---- 3. sources / 检索明细：三色列 ----
    st.subheader("Sources（检索明细）")
    st.caption("🟦 FAISS（语义）　🟩 BM25（词法）　🟧 rerank（重排）")
    cands = (result.get("retrieval_breakdown") or {}).get("candidates", [])
    if cands:
        df = pd.DataFrame(cands)[
            ["id", "topic", "faiss_score", "bm25_score", "hybrid_score", "rerank_score", "final_score"]
        ]
        styler = (
            df.style
            .set_properties(subset=["faiss_score"], **{"background-color": COL_FAISS})
            .set_properties(subset=["bm25_score"], **{"background-color": COL_BM25})
            .set_properties(subset=["rerank_score"], **{"background-color": COL_RERANK})
            .format({c: "{:.3f}" for c in
                     ["faiss_score", "bm25_score", "hybrid_score", "rerank_score", "final_score"]})
        )
        st.dataframe(styler, use_container_width=True, hide_index=True)
    else:
        st.caption("（本次无检索候选）")

    # ---- 导出 Demo Snapshot（JSON + PNG → outputs/demo_cases/）----
    st.subheader("导出")
    if st.button("📸 Export Demo Snapshot"):
        snap = {
            "query": st.session_state.get("query", ""),
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "error_type": result.get("error_type", ""),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        json_path, png_path = export_snapshot(snap)
        st.success(f"已导出到 outputs/demo_cases/：{json_path.name} ＋ {png_path.name}")
        st.image(str(png_path), caption="快照卡片预览")
        cda, cdb = st.columns(2)
        cda.download_button("下载 JSON", json_path.read_bytes(),
                            file_name=json_path.name, mime="application/json")
        cdb.download_button("下载 PNG", png_path.read_bytes(),
                            file_name=png_path.name, mime="image/png")

st.divider()

# ---- 6. Run Evaluation Mode ----
st.subheader("批量评测")
st.caption("对 data/test_queries.json 全量跑一遍流水线，汇总指标并写入 evaluation_report.json")

if st.button("Run Evaluation Mode"):
    with st.spinner("批量评测中（逐条跑 run_pipeline）..."):
        report = evaluate.evaluate()
    s = report["summary"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("avg confidence", s["avg_confidence"])
    c2.metric("unknown rate", s["unknown_rate"])
    c3.metric("retrieval_fail rate", s["retrieval_fail_rate"])
    c4.metric("hallucination risk", s["hallucination_risk_rate"])
    st.caption(f"error_type 分布：{s['error_type_distribution']}　|　共 {report['total_queries']} 条")
    st.dataframe([
        {"query": r["query"], "error_type": r["error_type"],
         "confidence": r["confidence"], "retrieval_score": r["retrieval_score"]}
        for r in report["results"]
    ], use_container_width=True, hide_index=True)
    st.success("评测完成，明细已写入 evaluation_report.json")
