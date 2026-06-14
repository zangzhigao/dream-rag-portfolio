# 梦境解析 RAG · V4

一个中文梦境解析的检索增强（RAG）小系统：输入一段梦境描述，先用
`sentence-transformers` + `FAISS` 在知识库里检索最相关的条目，再交给
DeepSeek 大模型生成**结构化 JSON** 解读（带来源引用、检索分、置信度、unknown 标记），
并对异常 / 低质量结果做记录与统计。

## 功能（V3）

- 梦境自然语言输入，命令行连续问答
- FAISS 语义检索（中文模型 `bge-small-zh-v1.5`，本地离线运行）
- **检索分机制**：每条结果带 `similarity_score`（余弦相似度，越高越相关）
- **top_k + threshold**：取前 top_k 候选，最高分低于 threshold 则 `unknown=true`
- DeepSeek 生成，**JSON 模式**保证结构化输出
- 输出含 `answer / sources / confidence / unknown / reason / retrieval_score`
- `sources` 引用经代码校验（丢弃编造来源），并附每条来源的检索分
- **Bad case 记录与统计**：异常 / 低质量结果写入 `data/badcases.jsonl`，
  命令行 `/stats` 一键查看统计（V3 新增）
- **Streamlit 网页界面**：可视化问答 + 来源展示 + 用户反馈（V4 新增）

## 目录结构与职责

```
rag_project/
├── data/
│   ├── dream_kb.json     # 知识库（JSONL，每行一条，54 条）
│   ├── embeddings.npy    # 生成物：句向量
│   ├── meta.json         # 生成物：与向量对应的元数据
│   ├── faiss.index       # 生成物：FAISS 索引
│   └── badcases.jsonl    # 运行时：bad case 日志
├── models/bge-small-zh-v1.5/   # 本地嵌入模型（离线）
├── embed.py     # 知识库 -> 向量（+ 模型加载、共享配置）
├── index.py     # 向量 -> FAISS 索引（IndexFlatIP）
├── search.py    # 查询 -> top_k 记录（带 score）
├── prompt.py    # 记录 + 问题 -> 结构化 JSON 提示词
├── llm.py       # 提示词 -> DeepSeek 回答（env 读 key，JSON 模式）
├── parser.py    # 模型文本 -> schema 字典（容错，永不崩）
├── badcase.py   # 写 bad case 日志
├── stats.py     # 读 badcases.jsonl 做统计（/stats、Evaluation/Monitoring）
├── app.py       # 主流程 rag_answer + 命令行循环（含 /stats 命令）
├── streamlit_app.py    # V4 网页界面（复用 rag_answer）
└── download_model.py   # 从 ModelScope 下载嵌入模型到 models/
```

## 安装

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

嵌入模型已放在 `models/`，离线可用。若缺失，运行 `python download_model.py`
从 ModelScope（国内稳定）下载。

## 运行

```powershell
# 1. 设置 DeepSeek Key（只在当前终端有效；不要写进代码）
$env:DEEPSEEK_API_KEY = "你的key"

# 2. 启动（首次会自动生成向量和索引）
python app.py
```

然后按提示输入梦境，回车即可得到 JSON 结果；输入 `/stats` 查看 bad case 统计，`q` 退出。

也可单独查看统计：

```powershell
python stats.py
```

未设置 Key 时程序不会崩溃：检索照常进行，但生成步骤会返回
`LLM调用失败` 的兜底结果并记入 bad case。

## 网页界面（V4）

命令行之外，提供一个 Streamlit 可视化页面：

```powershell
pip install streamlit
$env:DEEPSEEK_API_KEY = "你的key"
streamlit run streamlit_app.py
```

浏览器会自动打开（默认 http://localhost:8501）。页面包含：

- 输入框 + “开始解析”按钮
- 解读结果 `answer`、`confidence`、`unknown` 状态、`reason`
- `sources` 引用来源表（含 `id / topic / source / score`）
- 用户反馈按钮“有用 / 无用”；点击“无用”会把这条记入
  `data/badcases.jsonl`（`badcase_type = user_negative_feedback`），用于改进
- 页面底部的项目说明

网页与命令行共用同一套 `rag_answer`，逻辑完全一致。

### 单步调试（可选）

```powershell
python embed.py                  # 1. 生成向量
python index.py                  # 2. 构建索引
python search.py 梦见很多水       # 3. 只看检索结果
```

## 输出格式

```json
{
  "answer": "……解读文本……",
  "sources": [
    { "id": "dream_001", "topic": "梦境-水", "source": "梦境解析知识库", "score": 0.63 }
  ],
  "confidence": 0.6,
  "unknown": false,
  "reason": "……为什么这样判断……",
  "retrieval_score": 0.63
}
```

- `retrieval_score`：本次检索的**最高相似度**，衡量“知识库里有多贴近的内容”。
- `sources[].score`：每条来源各自的检索分。
- `unknown=true`：最高检索分低于 `threshold`，或模型判断无法作答。
- `sources`：只会出现确实被检索到的条目（编造的 id 会被剔除）。
- `confidence`：模型自评的相关程度，**仅供参考，非校准概率**（与 `retrieval_score` 不同：后者是客观算出来的）。

## V2 机制：top_k / threshold / retrieval_score 的产品意义

这三个旋钮一起决定了“答得准、答得稳、敢说不知道”，都在 `app.py` 顶部配置：

```python
TOP_K = 5          # 检索返回多少条候选
THRESHOLD = 0.35   # 最高检索分低于此 → unknown
```

- **top_k（召回宽度）**：一次取回前 k 条候选。k 太小可能漏掉相关条目；k 太大会
  把不相关的内容也喂给模型、增加噪声与成本。5 是知识库当前规模下的折中。
- **threshold（敢不敢答的红线）**：最高检索分低于它，系统直接返回 `unknown=true`，
  **不调用大模型**——省钱，也避免“硬答”出幻觉。这是产品上“知之为知之”的体现。
  调高 → 更保守（宁缺毋滥）；调低 → 更敢答（可能答些边缘问题）。默认 0.35 偏宽松。
- **retrieval_score（这次有多靠谱）**：本次最高相似度，写进每个结果。它是**客观**
  信号，可用于前端展示置信提示、做 A/B、或离线分析哪些问题召回不佳。
  与模型自评的 `confidence` 互补：一个看“检索到的内容有多贴近”，一个看“模型自认为多确定”。

> 经验范围：`bge-small-zh-v1.5` 的相似度多落在 0.45~0.7（相关）与 0.25~0.35（不相关）
> 之间。0.35 是一道偏宽松的红线，主要拦截完全跑题的问题；想更严格可上调到 0.45。

## V3 机制：bad case / Evaluation / Monitoring 的产品意义

RAG 系统上线后，真正的难点不是"能不能答"，而是"答错了怎么发现、怎么改进"。
V3 用最轻量的方式（一个 JSONL 文件）建立这条反馈闭环。

- **Bad case（坏样本）**：每次出现"不理想结果"就落一条日志到 `data/badcases.jsonl`。
  目前记录这几类：
  - `low_retrieval_score`：检索分太低，没敢答
  - `llm_api_error`：模型调用失败
  - `json_parse_error`：模型没返回合法 JSON
  - `unknown_answer`：模型自己说不知道
  - `missing_citation`：答了却没引用任何来源
  - `low_confidence`：答了但 confidence < 0.5
  bad case 是改进的"原材料"——知道系统在哪些问题上翻车，才知道该补哪些知识、调哪个阈值。

- **Evaluation（评估）**：`stats.py` 把这些日志聚合成可量化的指标（各类数量、低置信度比例、
  最近样本）。有了数字，"系统好不好"就不再靠感觉，而是可以对比：改了阈值/换了模型/补了知识后，
  bad case 是变多还是变少。这就是离线评估的雏形。

- **Monitoring（监控）**：`/stats` 让你随时查看运行状况，像看仪表盘一样。线上系统据此设报警
  （如 `llm_api_error` 突增 = 接口出问题；`low_retrieval_score` 偏高 = 知识库覆盖不足）。

> 现在用 JSONL 文件足够；当数据量大、需要多人查询或实时看板时，再迁移到数据库 / 日志平台。
> V3 先把"记录 → 统计 → 改进"这条链路跑通。

查看统计：命令行内 `/stats`，或单独 `python stats.py`。

## 关键设计说明

- **检索匹配只用「主题 + 标签」嵌入**，完整正文存在 `meta.json` 用于展示——
  长正文会稀释向量、降低区分度。
- **归一化向量 + `IndexFlatIP`** = 余弦相似度检索。
- **同一模型贯穿建库与查询**：换模型必须重跑 `embed.py` + `index.py`。
- 想换知识库：替换 `data/dream_kb.json`，删除 `embeddings.npy / meta.json / faiss.index` 后重跑。

## 已知边界

- `confidence` 来自模型自评，不可当作真实概率。
- 口语化、跨域的问题（如含动漫专有名词）检索分数偏低，可能命中较泛的主题
  或返回 `unknown`。
- 嵌入模型 `bge-small-zh-v1.5` 的相似度通常落在 0.45~0.7，阈值据此设定。
