# Changelog

## v1.1-dev — 仓库结构整理（GitHub 上传准备）

为开源 / 作品集上传做的工程化整理，**不改动 `run_pipeline` 管线与标准输出契约**（v1.0 基线行为不变）。

- 评测 / 监控相关模块归入 `evaluator/` 包：`evaluate.py`、`stats.py`、`badcase.py`、`badcase_v2.py`。
- 文档归入 `docs/`：`ARCHITECTURE.md`、`CHANGELOG.md`。
- 新增 `outputs/.gitkeep`，保留导出目录结构（产物本身仍被 `.gitignore` 忽略）。
- 相应更新导入路径（`from evaluator import ...`）、`arch_check.py` 消费端清单与 `README` 命令（`python -m evaluator.evaluate`）。
- `arch_check.py` 在 Windows GBK 控制台下输出 UTF-8，避免打印 `✅` 时崩溃（修复，不影响退出码）。

### 云端安全降级（Streamlit Cloud）
- 线上无法下载 BGE 句向量模型时**不再红屏崩溃**：`embed.try_get_model()` / `semantic_available()`
  捕获加载失败并缓存，`faiss_topk` 改为返回空、`hybrid_topk` 自动降级为 **BM25-only** 检索。
- 降级模式下的拒答门控改用 **实义词命中率**（多字词 overlap，阈值 0.30）近似语义相关——
  纯靠单字虚词凑出 BM25 分的离题问题命中率≈0，仍能被门控拦下（保留"敢说不知道"）。
- 页面顶部显示横幅：「☁️ 当前为云端演示模式：使用 BM25 关键词检索，完整本地版支持 FAISS + BM25 + Rerank」。
- sources / confidence / unknown / badcase / evaluation 展示全部保留；本地有模型时仍走完整 FAISS+BGE 路径。
- 可设环境变量 `RAG_FORCE_BM25=1` 显式强制 BM25-only（云端推荐：跳过注定失败的下载尝试）。

## v1.0-stable — 基线冻结（2026-06-14）

将当前系统冻结为 **v1.0 稳定基线**。

### 能力快照
- 检索：语义 FAISS + 词法 BM25 + 混合 Hybrid（归一化后加权融合）
- 重排：cross-encoder rerank（缺模型时自动 mock 降级）
- 置信度：客观计算 `retrieval_score(top1 hybrid) × coverage_factor`
- 异常分类：v2 自动归类（unknown_gate_trigger / retrieval_fail / hallucination_risk / low_confidence）+ 落盘
- 评测：`evaluator/evaluate.py` 批量跑测试集，产出 `evaluation_report.json`
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
