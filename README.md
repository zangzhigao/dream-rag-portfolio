# 🔮 文化知识 RAG 问答系统

> 把"模糊的一句话"变成 **有出处、有把握、敢说不知道** 的回答。

`Python` · `Streamlit` · `FAISS + BM25` · `Cross-Encoder Rerank` · `DeepSeek` · `v1.0-stable`

---

## 1. What I Built
一个面向**中国传统文化（梦境 / 周易 / 风水）**的检索增强（RAG）问答系统：用户随口问一句白话，系统先从知识库检索、再让大模型**基于检索内容**作答，并附上**引用来源、客观置信度，以及"知识库没有就老实拒答"的能力**。

---

## 2. Problem —— 传统 RAG 的四个痛点
- 🔍 **只用一种检索**：纯语义或纯关键词，用户换个说法就召回不到。
- 🤥 **不懂装懂**：知识库里没有，也硬编一个答案（幻觉）。
- 🤷 **不会拒答、不报置信**：永远给答案，却不告诉你"它有多确定"。
- 🕳️ **黑箱**：出错难定位，质量无法量化，迭代靠感觉。

---

## 3. My Solution —— 四个针对性机制（重点）

### 🔀 Hybrid Retrieval｜混合检索
**FAISS（语义，懂近义换词）+ BM25（关键词，命中精确术语）** 双路召回、归一化加权融合。
> "梦见往下坠"（语义）和"坠落"（关键词）都能找到——单路各有盲区，融合互补。

### 📊 Confidence System｜置信度系统
置信度**不取自大模型自评**（它常自吹），而是从检索信号客观计算：
`confidence = top1融合分 × 覆盖度系数(1.0 / 0.7 / 0.3)`。检索越强、双路越一致，分越高。

### 🚦 Unknown Gate｜拒答门控
最高检索分低于阈值时**直接拒答**（"未找到相关信息"），**不调用大模型、不产生幻觉**。
> 问它"红烧肉怎么做"，它会老实说不知道，而不是硬扯。

### 🧾 Badcase Logging｜坏样本系统
每次不理想结果**自动归类**（拒答门控 / 检索失败 / 幻觉风险 / 低置信）并落盘，形成
"发现问题 → 量化 → 改进"的数据闭环。

---

## 4. Architecture —— 单一管线
所有入口（命令行 / 网页 / 批量评测）共用**唯一函数** `run_pipeline(query)`：

```
query
  → 检索 (FAISS 语义 + BM25 词法)
  → Hybrid 融合
  → Rerank 重排
  → 置信度计算
  → 错误分类
  → LLM 生成
  → { answer, sources, confidence, error_type, retrieval_breakdown }
```

> 架构由 `arch_check.py` **守卫锁定**：任何"绕过管线 / 重复实现检索"的改动都会被自动拦截，保证团队协作下数据流不跑偏。

---

## 5. Key Features —— 界面一览
> 截图建议放在 `outputs/demo_cases/`，下方为各界面说明。

- **🟦🟢🟧 三色来源表**：每条来源标出 FAISS（蓝）/ BM25（绿）/ Rerank（橙）各自得分，检索过程一目了然。
  _（截图位：成功问答 + 来源表）_
- **置信度配色**：`<0.3 红 / 0.3~0.7 黄 / >0.7 绿`，一眼看出可信度。
  _（截图位：confidence 条形图 + 徽标）_
- **error_type 徽标**：正常 / 拒答 / 检索失败 / 幻觉风险 / 低置信，彩色标签直观呈现。
  _（截图位：⚠️ Unknown 拒答案例）_
- **📊 System Showcase 页**：架构图 + 流程图 + 4 指标卡 + 10 条 demo 一键切换/批量。
  _（截图位：Showcase 全景）_
- **📸 Export Demo Snapshot**：一键把当前问答导出为 JSON + 卡片 PNG，方便做案例集。

---

## 6. Evaluation System —— 用数字说话
内置批量评测（`evaluator/evaluate.py` / Showcase 一键），对测试集汇总 **4 个核心指标**：

| 指标 | 含义 |
|---|---|
| **avg confidence** | 平均置信度 |
| **unknown rate** | 拒答 / 无有效答案占比 |
| **retrieval_fail rate** | 检索未能支撑答案占比 |
| **hallucination risk** | 答了却无引用的占比 |

> 改了阈值、换了模型、补了知识——**跑一遍就知道好坏**，而不是凭感觉。

---

## 7. Demo Flow —— 如何跑
```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python download_model.py                 # 下载本地中文向量模型（离线可用）
$env:DEEPSEEK_API_KEY = "你的key"         # 生成用（不设也能跑，仅检索/拒答）

streamlit run streamlit_app.py           # ① 网页：问答 + 📊 Showcase
python app.py "梦见水很多是什么意思"        # ② 命令行：输出标准 JSON
python -m evaluator.evaluate             # ③ 批量评测 → evaluation_report.json
```

---

## 8. Why It Matters —— 业务价值
- ✅ **可信**：每条答案有出处、可溯源——不是"裸 LLM"的随口一说。
- 🛡️ **安全**：不知道就拒答，杜绝一本正经地胡说，降低业务风险。
- 📈 **可量化**：质量有指标、改进有依据，符合工程化迭代。
- 🔧 **可维护**：单一管线 + 架构守卫，多人协作也不跑偏。
- 🔁 **可迁移**：换一份知识库即可用于**法律 / 医疗 / 客服 FAQ** 等任意领域——这套"检索 + 融合 + 置信 + 拒答 + 评测"的框架是通用的。

---

> 技术细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)（架构契约）与 [`docs/CHANGELOG.md`](docs/CHANGELOG.md)（版本与冻结策略）。
