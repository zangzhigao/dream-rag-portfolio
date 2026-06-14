# Changelog

## v1.0-stable — 基线冻结（2026-06-14）

将当前系统冻结为 **v1.0 稳定基线**。

### 能力快照
- 检索：语义 FAISS + 词法 BM25 + 混合 Hybrid（归一化后加权融合）
- 重排：cross-encoder rerank（缺模型时自动 mock 降级）
- 置信度：客观计算 `retrieval_score(top1 hybrid) × coverage_factor`
- 异常分类：v2 自动归类（unknown_gate_trigger / retrieval_fail / hallucination_risk / low_confidence）+ 落盘
- 评测：`evaluate.py` 批量跑测试集，产出 `evaluation_report.json`
- 统一入口：`app.run_pipeline(query)` —— CLI / Streamlit / evaluate 三端共用
- 标准输出契约（5 字段）：`answer, sources, confidence, error_type, retrieval_breakdown`
- 调试：`run_pipeline(..., debug_mode=True)` 附 `debug` 字段，默认关闭
- 架构守卫：`arch_check.py` 锁定单一管线（违规非零退出）
- UI：系统说明卡片、三色检索表、置信度配色、error_type badge、Export Demo Snapshot

### 冻结策略（自 v1.0-stable 起生效）
1. **不再新增 retrieval / UI 功能。**
2. **仅允许 bug fix。**
3. **`run_pipeline` 管线与标准输出契约锁定**，结构性变更须经评审。
4. 任何改动须先通过 `python arch_check.py`（单一管线守卫）。
5. 新功能一律在 **v1.1+ 分支**进行，不污染 v1.0 基线。

> 版本标记：`VERSION` 文件 / `app.VERSION` / git tag `v1.0-stable`。
