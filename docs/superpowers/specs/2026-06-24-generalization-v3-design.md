# Generalization V3 Design

日期：2026-06-24

## 目标

在不读取 gold answer、不按样本 ID 或表格 ID 分支的前提下，允许 WTQ、TabFact、
CRT 使用不同的数据输入和答案协议，修复 blind36 暴露的通用失败模式，并在新的
冻结盲测上与 MACT 配对比较。

下一轮工程验收标准：

- 每个数据集 20 条未见样本，共 60 条；排除所有历史样本 ID 和表格 ID。
- myAgent 总准确率不低于同配置 MACT。
- 至少两个数据集的 myAgent 准确率不低于 MACT。
- myAgent 平均真实 API token 不超过 MACT 的 40%。
- myAgent 执行失败率不超过 2%。

该 60 条只作为第三轮工程盲测。专利和论文最终结果仍需每数据集至少 100 条的
预注册冻结测试。

## 防过拟合边界

允许：

- 根据 `source_dataset` 选择公开的数据集协议。
- 根据问题文本、表头、列类型、列值摘要和公开元数据推断答案结构。
- 处理重复表头、空值、`TBA`、破损转义等通用数据质量问题。
- 针对 WTQ denotation、TabFact 二分类和 CRT 混合答案执行不同输出契约。

禁止：

- 运行时读取 gold answer、canonical answer 或评测结果。
- 按样本 ID、表格 ID、具体实体名或 blind36 题目全文分支。
- 解封新盲测后修改代码并继续把同一批数据称为测试集。
- 将 benchmark18、blind36 或后续开发集结果写成最终泛化成绩。

## 方案选择

### 方案 A：继续扩写统一 prompt

改动小，但规则之间会互相干扰，且开发集选择偏差无法控制。不采用。

### 方案 B：题型专用求解器

可为日期、比分、排名等题型手写求解器，短期准确率可能较高，但规则数量会快速
增长，难以证明通用性。不作为主方案。

### 方案 C：数据集配置层与统一推理内核

每个数据集只定义输入清洗、答案协议和提示约束；Router、Compressor、Planner、
Calculator、Critic 和 Grounding Validator 保持统一。采用该方案。

## 架构

### 1. DatasetProfile

新增 `code/dataset_profiles.py`。`DatasetProfile` 只接收数据集名称和问题文本，输出：

- 数据集提示约束。
- 默认答案模式和动态候选标签。
- 多答案/双答案的结构要求。
- 平均值精度策略。
- 可忽略的缺失值标记。

WTQ 配置强调 denotation 列表、问题中的复数实体和历史时间问题；TabFact 配置将
包含数字、比较或聚合的陈述强制升级为代码推理；CRT 配置解析显式标签，并将
`A or B` 比较建模为动态候选标签，而不是 Yes/No。

### 2. AnswerContract V3

扩展 `AnswerContract`：

- `kind` 支持 `scalar`、`list`、`label`、`tuple`。
- `allowed_labels` 可由问题动态提供。
- `arity` 描述固定答案数量。
- `decimal_places` 允许 `None`、固定小数位或整数精度。
- 契约说明包含数据集 profile 的输出要求。

归一化层必须保留结构化列表/tuple，禁止把多答案提前拼成自然语言。

### 3. 全表列画像

扩展 `_build_table_schema()`，为每列提供：

- pandas dtype 和语义类型（numeric/text/mixed）。
- 非空数、唯一值数。
- 从完整 DataFrame 抽取的代表性唯一值，避免只看前 8 行。
- 缺失值/TBA 等标记数量。

Planner 和 Critic 同时看到列画像。这样可避免对数值列调用 `.str`、只根据前 8 行
把 non-finisher 等价为 Fell，以及忽略后部类别。

### 4. 执行恢复

Calculator 安全白名单加入只读的 `re`、`isinstance` 等常用能力。Pipeline 在代码
执行失败时优先把真实 `exec_error` 反馈给 Planner，不能被“答案为空”的契约错误
覆盖。Planner 重规划时必须处理列类型和缺失值。

### 5. 数据集提示约束

- WTQ：复数 `which/what` 实体返回列表；处理并列最大值；数值计算忽略 TBA/空值；
  历史相对时间若缺少参考年份，标记为时效歧义而不是使用当前年份猜测。
- TabFact：陈述中出现数字、比例、比较或计数时走 Planner/Calculator；简单实体
  存在/等值判断保留分类器路径。
- CRT：优先解析题目明确给出的候选标签；`home or away` 返回 `home/away`；双计数
  问题返回固定二元结构；精度根据问题单位和源列类型决定。

### 6. 评测协议

开发阶段仅使用 benchmark18 和 blind36。修复完成并通过单元测试后：

1. 从 full 数据集中排除所有历史 ID 和表格 ID。
2. 使用加密随机种子抽取 20 条/数据集并冻结 SHA-256。
3. 先运行 myAgent 三组，再运行 MACT 三组。
4. 六组全部完成前不计算正确率，不修改代码。
5. 统一报告准确率、真实 token、执行失败、Wilson 区间和配对 McNemar 检验。

## 测试策略

- DatasetProfile：WTQ 列表、TabFact 数字陈述、CRT 动态选项与 tuple 契约。
- Schema profile：混合列、后部类别、TBA 统计。
- Calculator：`re`、`isinstance`、辅助函数和禁止模块。
- Pipeline：执行错误优先反馈、契约重规划、分类与代码路径分流。
- Evaluator：WTQ 多答案、CRT tuple 和既有三数据集规则不回归。

## 风险

- 60 条盲测仍然较小，不能证明统计显著优势。
- 数据集 profile 可能提升基准适配但降低跨数据集零样本能力，因此必须通过消融
  单独报告 profile 的收益。
- 历史相对时间题可能没有可恢复的参考年份，应在正式结果中单独标记数据时效问题。
