# -*- coding: utf-8 -*-
"""Step 1 —— 读取知识库，生成句向量并保存到磁盘。

单独运行：
    python embed.py
产物：
    data/embeddings.npy   每条数据的向量（已归一化）
    data/meta.json        与向量行一一对应的元数据
"""
import os

# 国内访问 huggingface.co 常常超时，提前指向镜像（可被外部环境变量覆盖）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- 配置（被 index.py / search.py 复用） ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "dream_kb.json"
EMB_PATH = BASE_DIR / "data" / "embeddings.npy"
META_PATH = BASE_DIR / "data" / "meta.json"

# bge-small-zh 体积小(~95MB)、中文检索效果好；查询侧需加下面的指令前缀。
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 优先使用项目内已下载好的本地模型（见 download_model.py），离线可用、不依赖网络。
LOCAL_MODEL_DIR = BASE_DIR / "models" / "bge-small-zh-v1.5"

_model = None


def get_model() -> SentenceTransformer:
    """惰性加载模型：首次调用时才载入。

    若本地已下载模型（models/ 目录）则直接离线加载；否则回退到在线下载。
    """
    global _model
    if _model is None:
        if LOCAL_MODEL_DIR.exists():
            print(f"[embed] 从本地加载模型 {LOCAL_MODEL_DIR.name} ...")
            _model = SentenceTransformer(str(LOCAL_MODEL_DIR))
        else:
            print(f"[embed] 本地无模型，尝试在线加载 {MODEL_NAME} ...")
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def load_kb() -> list[dict]:
    """读取 JSONL 格式的知识库，每行一条记录。"""
    records = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_text(rec: dict) -> str:
    """构造“用于检索匹配”的文本：主题 + 标签。

    刻意只用主题与标签这类高信息密度的短文本，而非整段正文——
    长正文会把向量稀释成模糊的“大致主题”，反而降低检索区分度
    （实测“从高楼坠落”会被正文里谈及高度的“飞机失事”抢走）。
    完整正文仍保存在 meta.json 中用于展示，不影响返回内容。
    """
    tags = "、".join(rec.get("tags", []))
    return f"{rec['topic']}。关键词：{tags}"


def main() -> None:
    records = load_kb()
    print(f"[embed] 读取到 {len(records)} 条数据")

    texts = [build_text(r) for r in records]
    model = get_model()
    emb = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,   # 归一化后，向量内积 == 余弦相似度
        show_progress_bar=True,
    ).astype("float32")

    np.save(EMB_PATH, emb)

    meta = [
        {
            "id": r["id"],
            "topic": r["topic"],
            "content": r["content"],
            "tags": r.get("tags", []),
            "source": r.get("source", ""),
        }
        for r in records
    ]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[embed] 向量已保存 {emb.shape} -> {EMB_PATH.name}")
    print(f"[embed] 元数据已保存 -> {META_PATH.name}")


if __name__ == "__main__":
    main()
