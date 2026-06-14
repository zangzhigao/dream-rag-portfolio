# -*- coding: utf-8 -*-
"""RAG 的 prompt 构造：把检索到的记录组织成给大模型的提示词。

与检索、调用解耦——只负责"怎么把上下文 + 问题拼成提示词"。
"""


def build_rag_prompt(query: str, contexts: list[dict]) -> str:
    """让模型基于知识库输出结构化 JSON（answer/sources/confidence/unknown/reason）。"""
    context_text = ""
    for i, item in enumerate(contexts, start=1):
        context_text += f"""
[来源{i}]
id: {item.get("id")}
topic: {item.get("topic")}
source: {item.get("source")}
content: {item.get("content")}
"""

    return f"""你是一个基于知识库的传统文化/梦境解析助手。

你必须严格遵守以下规则：
1. 只能基于【知识库内容】回答。
2. 如果知识库内容不足以回答，必须把 unknown 设为 true。
3. 不允许编造知识库以外的内容。
4. 必须输出合法 JSON。
5. 不要输出 Markdown。
6. 不要输出解释性废话。
7. sources 必须来自提供的知识库来源，不能自己编造。

【知识库内容】
{context_text}

【用户问题】
{query}

请严格输出以下 JSON 格式：

{{
  "answer": "",
  "sources": [
    {{
      "id": "",
      "topic": "",
      "source": ""
    }}
  ],
  "confidence": 0.0,
  "unknown": false,
  "reason": ""
}}
"""
