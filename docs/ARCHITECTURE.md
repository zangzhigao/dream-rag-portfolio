# 架构约定 · Single Pipeline Architecture（单一数据流入口）

本项目锁定为**单一管线架构**：所有"查询 → 结果"的数据流，必须且只能经过
唯一入口 **`app.run_pipeline(query)`**。

## 数据流（唯一路径）

```
            ┌──────────────── app.py ────────────────┐
            │  run_pipeline(query)   ← 唯一实现/编排    │
            │   hybrid_topk(FAISS+BM25) → rerank →     │
            │   _compute_confidence → classify → 生成   │
            │   返回标准 JSON（5 字段）                  │
            └───────────────┬─────────────────────────┘
        ┌───────────────────┼───────────────────────┐
  python app.py "q"   streamlit_app.py     evaluator/evaluate.py
   (CLI)               (网页)                  (批量评测)
        └──── 三个入口只调 run_pipeline，绝不自行检索 ────┘

实现层（被 run_pipeline 调用，唯一实现处，不得复制）：
  retriever/  (faiss / bm25 / hybrid)   reranker.py   app._compute_confidence
诊断工具（非数据流入口，可直接访问底层，仅供开发对比）：
  retriever_compare.py
```

## 锁定规则

1. **禁止绕过**：任何用户侧数据流入口都必须经 `run_pipeline`，不得直接调用
   `hybrid_topk / faiss_topk / bm25_topk / rerank / _compute_confidence`。
2. **统一入口**：`app.py`（CLI）、`streamlit_app.py`、`evaluator/evaluate.py` 一律调用 `run_pipeline`。
3. **禁止重复实现**：retrieval / rerank / confidence 各自只有一处实现，消费端不得引入
   底层库（`faiss / rank_bm25 / sentence_transformers / jieba`）自行实现。

## 标准输出契约（仅 5 字段）

```json
{ "answer": "...", "sources": [...], "confidence": 0.x,
  "error_type": "", "retrieval_breakdown": { ... } }
```
检索分等内部信号统一收纳于 `retrieval_breakdown`（如 `top_faiss`）。
`debug_mode=True` 时额外附 `debug` 字段，默认关闭、不影响 UI。

## 强制执行

静态守卫 `arch_check.py` 会校验上述规则，违规即非零退出：

```bash
python arch_check.py        # 通过 → 0；违规 → 列出违规项并返回 1
```

建议接入 **pre-commit / CI**，使违反单一管线架构的改动无法提交 / 合并。
