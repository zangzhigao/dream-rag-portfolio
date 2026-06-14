# -*- coding: utf-8 -*-
"""RAG 的生成(G)步：调用 DeepSeek。

本函数遇错直接抛异常（key 缺失 / 网络 / HTTP 错误），由调用方 try/except 处理——
与 app.py 里 rag_answer 的错误处理 + bad case 记录配套。

API Key 从环境变量读取，绝不要写死在代码里：
    PowerShell:  $env:DEEPSEEK_API_KEY = "你的key"
"""
import os

import requests

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
SYSTEM_PROMPT = "你是一个严谨的RAG问答系统，只能基于提供的知识库内容回答，并严格输出合法JSON。"


def call_deepseek(prompt: str, temperature: float = 0.2, json_mode: bool = True) -> str:
    """调用 DeepSeek 并返回回答文本。失败时抛异常（不静默吞掉）。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")          # 调用时读，而非导入时
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 1000,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}   # 保证返回合法 JSON

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
