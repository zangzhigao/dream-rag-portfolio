# -*- coding: utf-8 -*-
"""把嵌入模型下载到项目内的 models/ 目录（离线可用）。

国内直连 huggingface.co 常失败，这里用 ModelScope（魔搭）下载，稳定快速。
    pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
    python download_model.py
"""
from pathlib import Path

from modelscope import snapshot_download

from embed import MODEL_NAME, LOCAL_MODEL_DIR


def main() -> None:
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
        print(f"[download] 模型已存在 -> {LOCAL_MODEL_DIR}")
        return
    print(f"[download] 从 ModelScope 下载 {MODEL_NAME} ...")
    path = snapshot_download(MODEL_NAME, local_dir=str(LOCAL_MODEL_DIR))
    print(f"[download] 完成 -> {path}")


if __name__ == "__main__":
    main()
