# -*- coding: utf-8 -*-
"""架构守卫 —— 锁定 Single Pipeline Architecture（单一数据流入口）。

规则：
  1. run_pipeline 只能在 app.py 定义（唯一实现入口）。
  2. 消费端（streamlit_app.py / evaluate.py）必须只经 run_pipeline 取数据，
     禁止直接导入检索/重排/置信度的内部模块或函数（即禁止绕过管线）。
  3. 禁止重复实现 retrieval / rerank / confidence（消费端不得引入底层检索库）。

用法（可接入 CI / pre-commit）：
    python arch_check.py        # 通过 → 退出码 0；违规 → 打印违规项并退出码 1

说明：retriever/、reranker.py 是管线的"唯一实现"，retriever_compare.py 是诊断工具，
均不属于"数据流入口"，不在本守卫约束范围内。
"""
import ast
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent

OWNER = "app.py"                                  # 唯一可编排管线的文件
CONSUMERS = [                                      # 必须只经 run_pipeline 的入口
    "streamlit_app.py",
    "evaluate.py",
    "pages/1_System_Showcase.py",
]

# 消费端禁止直接导入的"管线内部"模块 + 底层检索/重排库
FORBIDDEN_MODULES = {
    "retriever", "reranker", "search",
    "retriever.bm25_retriever", "retriever.faiss_retriever", "retriever.hybrid_retrieval",
    "faiss", "rank_bm25", "sentence_transformers", "jieba",
}
# 消费端禁止直接导入的管线内部函数
FORBIDDEN_NAMES = {"hybrid_topk", "faiss_topk", "bm25_topk", "rerank", "_compute_confidence"}


def parse_file(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules, names = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                modules.add(a.name)
                modules.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                modules.add(n.module)
                modules.add(n.module.split(".")[0])
            for a in n.names:
                names.add(a.name)
    return tree, modules, names


def defines_func(tree, func: str) -> bool:
    return any(isinstance(n, ast.FunctionDef) and n.name == func for n in ast.walk(tree))


def main() -> int:
    violations = []

    # 规则 1：OWNER 必须定义 run_pipeline
    owner_tree, _, _ = parse_file(BASE / OWNER)
    if not defines_func(owner_tree, "run_pipeline"):
        violations.append(f"{OWNER}：未定义 run_pipeline（管线唯一实现入口缺失）")

    # 规则 2 & 3：消费端检查
    for fname in CONSUMERS:
        tree, modules, names = parse_file(BASE / fname)
        bad_m = modules & FORBIDDEN_MODULES
        bad_n = names & FORBIDDEN_NAMES
        if bad_m:
            violations.append(f"{fname}：绕过管线——直接导入内部模块/库 {sorted(bad_m)}")
        if bad_n:
            violations.append(f"{fname}：绕过管线——直接导入内部函数 {sorted(bad_n)}")
        if "run_pipeline" not in names:
            violations.append(f"{fname}：未从 app 导入 run_pipeline（必须经统一入口）")
        if defines_func(tree, "run_pipeline"):
            violations.append(f"{fname}：重复定义了 run_pipeline")

    if violations:
        print("❌ 架构校验失败（Single Pipeline Architecture 被破坏）：")
        for v in violations:
            print("   -", v)
        return 1

    print("✅ 架构校验通过：")
    print(f"   · run_pipeline 唯一定义于 {OWNER}")
    print(f"   · 消费端 {CONSUMERS} 均只经 run_pipeline 取数据")
    print("   · retrieval / rerank / confidence 无重复实现，未被绕过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
