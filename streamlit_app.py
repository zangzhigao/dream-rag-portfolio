# -*- coding: utf-8 -*-
"""V4 —— Streamlit 可视化界面（复用 app.rag_answer，与命令行同一套逻辑）。

启动：
    pip install streamlit
    $env:DEEPSEEK_API_KEY = "你的key"
    streamlit run streamlit_app.py
"""
import streamlit as st

from app import rag_answer, ensure_ready
from badcase import log_badcase

st.set_page_config(page_title="基于 RAG 的梦境知识增强问答系统", page_icon="🌙")

# 首次启动确保向量/索引就绪（每个会话只跑一次）
if "ready" not in st.session_state:
    ensure_ready()
    st.session_state["ready"] = True

st.title("🌙 基于 RAG 的梦境知识增强问答系统")
st.caption("输入梦境 → 语义检索知识库 → 大模型基于检索内容生成解读")

query = st.text_input(
    "请输入你的梦境问题",
    placeholder="例如：我梦见很多水，而且很害怕，是什么意思？",
)

if st.button("开始解析", type="primary"):
    if not query.strip():
        st.warning("请先输入梦境问题。")
    else:
        with st.spinner("检索 + 生成中 ..."):
            st.session_state["result"] = rag_answer(query.strip())
        st.session_state["query"] = query.strip()
        st.session_state["feedback_done"] = False

result = st.session_state.get("result")
if result:
    st.divider()

    # Unknown 状态
    if result.get("unknown"):
        st.warning("⚠️ Unknown：知识库中没有足够相关的信息，或模型无法作答")
    else:
        st.success("✅ 已基于知识库作答")

    # Answer
    st.subheader("Answer · 解读")
    st.write(result.get("answer", ""))

    # Confidence / Retrieval Score
    c1, c2 = st.columns(2)
    c1.metric("Confidence（模型自评）", result.get("confidence", 0.0))
    c2.metric("Retrieval Score（最高检索分）", result.get("retrieval_score", 0.0))

    # Reason
    st.subheader("Reason · 判断依据")
    st.write(result.get("reason", ""))

    # Sources（id / topic / source / score）
    st.subheader("Sources · 引用来源")
    sources = result.get("sources", [])
    if sources:
        st.table(sources)
    else:
        st.caption("（本次无引用来源）")

    # 用户反馈
    st.subheader("这条解析对你有帮助吗？")
    fb1, fb2 = st.columns(2)
    if fb1.button("👍 有用"):
        st.success("感谢你的反馈！")
    if fb2.button("👎 无用"):
        if not st.session_state.get("feedback_done"):
            log_badcase(st.session_state.get("query", ""), result, "user_negative_feedback")
            st.session_state["feedback_done"] = True
        st.info("已记录你的反馈（user_negative_feedback），我们会据此改进。")

# 项目说明
st.divider()
st.markdown(
    """
### 项目说明
- 本项目基于 **RAG（检索增强生成）**
- 使用 **Embedding + FAISS** 进行语义检索
- 使用 **LLM（DeepSeek）** 生成解读
- 支持 **Citation（引用来源）/ Confidence（置信度）/ Unknown（不确定）/ Bad case（坏样本记录）**
"""
)
