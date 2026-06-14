# -*- coding: utf-8 -*-
"""检索对比脚本：同一 query 下，并排对比 FAISS（语义）与 BM25（词法）的 top5。

用法（在 rag_project/ 目录下）：
    python retriever_compare.py 梦见蛇追我
    python retriever_compare.py            # 不带参数则交互输入

只读不改：仅复用 retriever 包，不修改任何现有代码。
"""
import sys
import unicodedata

# ---- 导入检索器；BM25 若不可用则降级为 mock ----
from retriever import faiss_topk

try:
    from retriever import bm25_topk
    BM25_MODE = "real"
except Exception:                                   # BM25 缺失 -> mock 占位
    BM25_MODE = "mock"

    def bm25_topk(query, top_k=5):
        return [{"id": f"mock_{i+1}", "topic": "（BM25未实现·mock）", "bm25_score": 0.0}
                for i in range(top_k)]


# ---- 中文对齐的表格打印（按显示宽度，全角字符算 2） ----
def _w(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("F", "W") else 1 for c in str(s))


def _pad(s, width, align):
    s = str(s)
    gap = width - _w(s)
    if gap <= 0:
        return s
    if align == "r":
        return " " * gap + s
    if align == "c":
        left = gap // 2
        return " " * left + s + " " * (gap - left)
    return s + " " * gap


def print_table(headers, rows, aligns):
    cols = len(headers)
    widths = [_w(headers[i]) for i in range(cols)]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], _w(row[i]))

    def rule(left, mid, right):
        return left + mid.join("─" * (widths[i] + 2) for i in range(cols)) + right

    def fmt(row, row_aligns):
        return "│" + "│".join(" " + _pad(row[i], widths[i], row_aligns[i]) + " "
                              for i in range(cols)) + "│"

    print(rule("┌", "┬", "┐"))
    print(fmt(headers, ["c"] * cols))
    print(rule("├", "┼", "┤"))
    for row in rows:
        print(fmt(row, aligns))
    print(rule("└", "┴", "┘"))


def _cell(results, i, score_key):
    if i < len(results):
        r = results[i]
        return r.get("id", "-"), r.get("topic", "-"), f"{r.get(score_key, 0.0):.3f}"
    return "-", "-", "-"


def main():
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        try:
            query = input("请输入查询: ").strip()
        except EOFError:
            query = ""
    if not query:
        query = "梦见蛇追我"

    faiss_res = faiss_topk(query, top_k=5)
    bm25_res = bm25_topk(query, top_k=5)

    headers = ["Rank", "FAISS id", "FAISS topic", "sim", "BM25 id", "BM25 topic", "bm25"]
    aligns = ["c", "l", "l", "r", "l", "l", "r"]
    rows = []
    for i in range(5):
        fid, ftopic, fsc = _cell(faiss_res, i, "similarity_score")
        bid, btopic, bsc = _cell(bm25_res, i, "bm25_score")
        rows.append([str(i + 1), fid, ftopic, fsc, bid, btopic, bsc])

    print(f"\n查询: {query}    |  FAISS=语义检索  BM25={BM25_MODE}（词法检索）\n")
    print_table(headers, rows, aligns)
    print("\n注：sim 为余弦相似度(0~1)，bm25 为 BM25 相关度(无界)，两者不可直接比大小。")


if __name__ == "__main__":
    main()
