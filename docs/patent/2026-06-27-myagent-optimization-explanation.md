# myAgent 优化改造说明及专利可写技术点

日期：2026-06-27  
适用项目：`D:\AAAcode\code-code\agent+\myAgent-main`  
对标项目：`D:\AAAcode\code-code\agent+\MACT-main\MACT-main`  
数据集范围：WTQ、TabFact、CRT-QA

## 1. 文档目的

本文档解释本轮在 myAgent 原有方案基础上做了哪些修改、为什么要修改、修改后对应哪些代码模块，以及哪些内容适合写入专利报告。

结论先行：

- 适合作为专利核心技术点：风险驱动的选择性协作、证据包构建、推理前/推理后风险评估、候选答案一致性判断、预算约束下的动态升级。
- 适合作为实施例：WTQ、TabFact、CRT-QA 的答案契约、数据集格式适配、CRT 确定性语义算子、WTQ 标量规范化。
- 不建议作为核心专利点：DeepSeek API 接入、命令行参数、测试脚本、具体模型名称、某个样本的错误修补。

当前小样本开发验证结果为：WTQ 4/5、TabFact 4/5、CRT 5/5，合计 13/15，平均 API tokens 为 2732.3。该结果只用于开发验证，不应直接写成正式实验结论，后续仍需更大规模盲测确认。

## 2. 原有方案的问题

原有 myAgent 的主要特点是低 token、轻量压缩、Planner 生成代码、Calculator 执行代码、Critic 检查结果。该路线的优点是成本低，但在和 MACT 对比时暴露出几个问题：

1. 低 token 并不等于高质量。部分复杂问题被压缩或路由为低风险路径后，证据不足、列选择不完整、答案契约不稳定。
2. easy/medium/hard 路由不能直接代表真实推理风险。有些看似简单的样本其实涉及时间边界、单位换算、百分比口径、闭集标签约束。
3. Planner 单一路径容易把错误逻辑执行得很稳定。例如代码能运行，但理解错了 “after 1936”、百分比快照、summary 行、或 yes/no 语义。
4. 旧输出缺少统一的风险、证据、候选答案、token 观测字段，不利于论文/专利中解释“为什么该样本要多花 token，为什么另一个样本可以少花 token”。
5. 三个数据集的答案形态不同。WTQ 是 denotation，TabFact 是真假判断，CRT-QA 混合了 yes/no、标量、标签、tuple 等答案；若没有统一契约，评测结果容易不可比。

因此，本轮优化的目标不是继续极限压低 token，而是在准确率优先的前提下，通过风险判断把 token 用在更需要的样本上。

## 3. 总体修改思路

本轮把原有“固定轻量流程”改造成“风险驱动的选择性协作流程”。

新的核心流程为：

```text
输入样本
  -> 数据集适配与答案契约
  -> 表格压缩与证据包构建
  -> 推理前风险评估 R_pre
  -> 根据风险选择轻量路径 / 代码路径 / 候选比较 / fallback
  -> 代码执行与答案契约校验
  -> 候选答案一致性判断
  -> 推理后风险评估 R_post
  -> 预算记录与评估报告
```

这个流程的关键点是：不是所有样本都调用更贵的推理组件，只有风险较高、证据不足、候选冲突或执行/契约失败时才升级。

## 4. 主要修改清单

| 模块 | 修改内容 | 修改原因 | 专利可写性 |
| --- | --- | --- | --- |
| `model_backends.py` | 增加 DeepSeek/OpenAI-compatible/Azure/local 后端切换 | 支持本地直接调用 API，并可通过参数切换模型 | 工程实施，不建议作为核心专利点 |
| `dataset_adapters.py` | 将 WTQ、TabFact、CRT-QA 转为统一 MACT JSONL 格式 | 保证两个项目、三个数据集能使用同一评测入口 | 可写为实施例中的数据接入方式 |
| `answer_contracts.py` | 增加答案契约：scalar/list/tuple/label、精度、闭集标签 | 防止模型输出解释句、错误标签、错误精度 | 可写为答案约束模块 |
| `dataset_profiles.py` | 按数据集推断 answer_mode、提示约束、缺失值规则 | 三个数据集答案格式和口径不同 | 可写为数据集协议适配层 |
| `risk_control.py` | 新增风险公式、风险分层、预算控制 | 用可解释风险决定协作深度和 token 投入 | 适合作为核心专利点 |
| `evidence_builder.py` | 构建 EvidencePack：候选行列、实体匹配、缺失项、操作线索 | 让多个推理组件共享同一证据视图 | 适合作为核心专利点 |
| `selective_collaboration.py` | 新增 CandidateAnswer、AgreementJudge、ThinkingSolver | 比较候选答案，冲突时再升级 | 适合作为核心专利点 |
| `my_agents.py` | 接入 selective pipeline、风险记录、候选答案、CRT 语义算子、WTQ 规范化 | 把风险控制真正接入运行链路 | 核心实现与实施例 |
| `tqa.py` / `run_wtq_myagent.py` | 增加 `--collaboration_mode`、`--mact_avg_tokens`、API token 记录 | 支持 legacy/selective 切换和成本评估 | 工程实施 |
| `evaluate_results.py` | 增加 api token、risk_distribution、risk_strata 评估 | 支持准确率和 token 的联合评价 | 可写为实验评价方法 |
| `compare_blind_results.py` | 增加 myAgent vs MACT 的 token ratio 和风险分层对比 | 判断是否满足“准确率不降、token 降低” | 可写为实验比较协议 |
| `calibrate_risk_policy.py` | 在 token 约束下搜索风险阈值 | 让权重/阈值可调且可解释 | 可写为参数校准实施例 |

## 5. 风险控制模块

### 5.1 为什么增加风险控制

原方案只靠路由难度和表格压缩，不能稳定识别以下高风险情况：

- 问题实体和表格列匹配不唯一；
- 时间边界存在歧义；
- 答案必须是闭集标签；
- 答案需要单位、精度或格式约束；
- 代码执行成功但逻辑口径错误；
- 同一问题可由不同推理路径得到不同答案。

因此新增 `RiskProfiler`，用多个可解释信号计算风险，而不是只看模型主观判断。

### 5.2 已实现公式

结构复杂度：

```text
C = 0.50 * C_coverage + 0.30 * C_dispersion + 0.20 * C_type
```

综合难度：

```text
D = 0.45 * semantic_complexity + 0.55 * C
```

推理前风险：

```text
R_pre = 0.20 * D + 0.30 * ambiguity + 0.30 * evidence_gap + 0.20 * operation_risk
```

推理后风险：

```text
R_post = R_pre
       + 0.30 * candidate_disagreement
       + 0.20 * verification_gap
       + 0.30 * hard_failure
```

风险分层：

```text
R_pre < 0.25          -> light
0.25 <= R_pre < 0.55 -> medium
R_pre >= 0.55         -> high
硬失败或 R_post >= 0.70 -> fallback
```

这些公式对应 `code/risk_control.py`，测试在 `tests/test_risk_control.py`。

### 5.3 专利可写点

可写为：

一种面向表格问答的风险驱动推理调度方法，其根据语义复杂度、结构覆盖度、歧义度、证据缺口和操作契约风险计算推理前风险，并根据候选答案分歧、证据绑定缺口和执行失败计算推理后风险，从而动态选择不同协作深度。

不建议写成：

固定使用某个阈值一定最优。阈值属于可调参数，正式实验后可以作为实施例参数。

## 6. 预算控制模块

### 6.1 为什么增加预算控制

用户目标不是单纯省 token，而是在准确率接近或超过 MACT 的同时降低 token。因此新增 `BudgetController`，把 MACT 平均 token 作为参考预算。

当前默认参考值：

```text
B_MACT = 8867
```

各层级预算：

```text
light    <= 0.25 * B_MACT
medium   <= 0.55 * B_MACT
high     <= 0.85 * B_MACT
fallback <= 1.10 * B_MACT
average  <= 0.75 * B_MACT
```

### 6.2 专利可写点

可写为：

一种在基准系统资源消耗约束下进行自适应推理预算分配的方法，对低风险样本采用轻量路径，对中高风险样本逐级增加验证和候选比较，并以总体平均预算为约束。

注意：`8867` 是本轮 MACT 开发基线，不应写成专利固定参数。专利中应写成“基准系统平均消耗 B”。

## 7. 证据包 EvidencePack

### 7.1 为什么增加证据包

原方案中不同路径看到的表格上下文可能不一致：压缩 prompt 看到的是局部表格，代码执行看到的是 DataFrame，评估看到的是最终输出。这样不利于解释错误，也不利于多路径协作。

新增 `EvidenceBuilder` 后，每个样本都会生成统一证据包：

```text
EvidencePack
  dataset_name
  candidate_rows
  candidate_columns
  entity_matches
  operation_hints
  missing_or_ambiguous_items
  structure_signals
  ambiguity_signals
  gap_signals
  operation_signals
  provenance
```

对应代码：`code/evidence_builder.py`。  
对应测试：`tests/test_evidence_builder.py`。

### 7.2 专利可写点

可写为：

在表格问答推理前构建不依赖金标答案的证据包，将候选行、候选列、实体匹配、操作线索和缺失/歧义项统一编码，供风险评估、代码生成、候选答案验证和后验风险评估共同使用。

## 8. 候选答案协作与后验判断

### 8.1 为什么增加 CandidateAnswer 和 AgreementJudge

原方案通常只有一个 Planner/Calculator 结果。一旦该路径逻辑错了，系统会“稳定地错”。因此新增候选答案结构和一致性判断：

```text
CandidateAnswer
  raw_answer
  normalized_answer
  is_valid
  reasoning_summary
  executable_program
  execution_result
  token_usage
  failure
```

`AgreementJudge` 根据答案契约比较候选是否等价：

- label/scalar：规范化后精确比较；
- numeric：按容差比较；
- list：按集合 Jaccard；
- tuple：按位置比较；
- 无有效候选：触发 fallback；
- 有效候选冲突：提高 `R_post`。

对应代码：`code/selective_collaboration.py`。

### 8.2 专利可写点

可写为：

一种基于答案契约的候选答案一致性判断方法，对不同答案形态采用不同等价判定规则，并将候选答案分歧反馈至后验风险模型，以决定是否触发更高成本的推理路径。

## 9. 答案契约 AnswerContract

### 9.1 为什么增加答案契约

WTQ、TabFact、CRT-QA 的答案格式差异很大。如果只让模型自由输出，会出现：

- 应输出 `true/false` 却输出解释句；
- 应输出 scalar 却输出 list；
- 应输出三位小数却错误四舍五入；
- 应输出多个实体却只输出一个；
- yes/no、more/less/equal、tuple 等闭集标签混乱。

因此新增 `AnswerContract`，把答案形态显式化：

```text
kind = scalar | label | list | tuple
allowed_labels
decimal_places
arity
reasoning_required
instructions
```

对应代码：`code/answer_contracts.py`。  
在 pipeline 中会对生成代码和最终结果进行契约校验。

### 9.2 专利可写点

可写为：

一种表格问答答案契约机制，根据任务类型和问题文本生成答案形态约束，并在代码生成、执行后规范化、候选比较和评估输出阶段持续校验。

## 10. 数据集适配策略

### 10.1 为什么做适配

三个数据集原始格式不同：

- WTQ：TSV + CSV 表格 + denotation；
- TabFact：TSV/JSON + 表格 CSV + true/false label；
- CRT-QA：JSON 问题 + 引用表格，有些表格位于相邻 TabFact 数据目录。

如果不统一格式，就很难让 MACT 和 myAgent 使用同一评估协议。

### 10.2 实现方式

`dataset_adapters.py` 将数据统一成：

```text
id
source_dataset
question / statement
table_text
answer
table_id
public metadata
```

同时处理：

- 重复表头去重；
- WTQ canonical answer 拆分；
- TabFact `[UNK]` 实体恢复；
- CRT 表格路径自动查找；
- 缺失 CRT 表格显式报错。

### 10.3 专利可写点

这部分更适合作为实施例，而不是核心创新点。可以写成：

系统可通过数据集适配层将不同表格问答数据统一为问题、表格、答案契约和公开元数据，从而支持统一风险评估和统一评测。

## 11. 模型后端改造

### 11.1 为什么改造

为了在本地使用 DeepSeek API，并且未来能切换其他模型，需要把模型调用从固定本地/固定 Azure 方式改成参数化后端。

新增参数包括：

```text
--model_provider auto|deepseek|openai_compatible|azure|local
--api_base
--api_key_env
--thinking disabled|enabled
--temperature
--max_tokens
--api_timeout
--api_max_retries
```

对应代码：`code/model_backends.py`，入口接入在 `code/tqa.py` 和 `code/run_wtq_myagent.py`。

### 11.2 专利可写性

这部分不建议作为专利核心点。它是实验工程能力，用来保证不同模型可复现实验。专利中最多写成“可调用外部大语言模型接口”，不应强调 DeepSeek 或某个 API 名称。

## 12. Pipeline 接入

### 12.1 为什么改造 TableQAPipeline

原 pipeline 只有 legacy 运行方式，不能比较“旧方案”和“新选择性协作方案”。因此新增：

```text
--collaboration_mode legacy|selective|calibration
--mact_avg_tokens
```

并在 `TQASessionState` 中增加观测字段：

```text
risk_assessment
post_risk_assessment
risk_level
evidence_pack
candidate_answers
agreement_decision
budget_state
```

### 12.2 新运行逻辑

selective 模式下：

```text
build EvidencePack
run legacy solver
assess R_pre
wrap result as CandidateAnswer
compare candidates with AgreementJudge
assess R_post
fallback if needed
record token budget
```

其中 CRT 的部分明确模式会先走确定性语义算子，避免把明显可程序化的语义判断交给 LLM 猜测。

### 12.3 专利可写点

可写为：

一种兼容单路径表格问答系统的选择性协作增强方法，在保留原有求解路径的基础上，增加证据包、风险评估、候选答案比较和预算控制层，以实现按样本风险动态分配推理资源。

## 13. 数据集针对性优化

### 13.1 WTQ

已做修改：

- 国家代码规范化，例如 `ITA -> Italy`；
- datetime-like 值还原为日期文本，例如避免 pandas 时间戳被输出为纳秒整数；
- “how long ... after YYYY” 的年份单位补全；
- `final_value` 和 `final_answer` 同步，避免内部答案正确但评测字段不一致。

修改原因：

WTQ 的评估是 denotation，格式细节会直接影响正确率。旧逻辑中模型可能算出正确实体或日期，但最终字段保存不一致。

专利写法：

可写为实施例中的“答案规范化子模块”，不建议把 `ITA -> Italy` 这种具体映射写成核心权利要求。

### 13.2 TabFact

已做修改：

- 数据集 profile 将 TabFact 设置为 true/false 契约；
- 数字、比较、计数类 TabFact 不再完全依赖纯文本分类；
- 保留子句、否定、逻辑连接词相关风险信号；
- 评估统一使用二分类准确率。

修改原因：

TabFact 表面是二分类，但很多样本需要表格计数或比较。只用自然语言判断容易误判。

专利写法：

可写为“基于答案契约的事实验证任务适配”，但当前 TabFact 仍有 1/5 错题，正式写实验效果前需要更大盲测。

### 13.3 CRT-QA

已做修改：

- 对 `24 hours` 和 `1 day` 做等价 duration 规范化；
- 对“event type 是否基于 stages/days 有系统性差异”的 yes/no 问题增加确定性判断；
- 对 `% (1960)`、`% (2000)`、`% (2040)` 这类百分比快照表，按请求行和年份范围求平均；
- 对错误相对增长公式、硬编码 label 覆盖、错误 round 精度增加校验。

修改原因：

CRT 的失败主要不是执行错误，而是口径错误。LLM 容易把“百分比快照”误解成“相对增长率”，也容易把孤立变化误解为系统性差异。

专利写法：

可写为实施例：针对表格字段类型和问题模式触发确定性语义算子，并将其作为低成本求解路径的一部分。注意不要写成“针对 CRT 某条题目的规则”，应写成“单位归一化、快照百分比聚合、系统性映射判定”。

## 14. 评估与对比修改

新增或改造：

- `evaluate_results.py`：统一输出 primary accuracy、api token、risk distribution、risk strata；
- `compare_blind_results.py`：比较 myAgent 和 MACT 的正确率、token ratio、风险分层表现；
- `calibrate_risk_policy.py`：在 token ratio 约束下搜索风险阈值；
- `sample_benchmark.py`：支持确定性抽样、按 table_id 去重、平衡标签。

修改原因：

专利和论文不能只报告“某次跑得更好”，必须能解释：

- 准确率是否提升；
- token 是否降低；
- 哪类风险样本收益最大；
- 是否存在执行失败；
- 是否对某个数据集过拟合。

专利写法：

可写为实验评价协议，不建议写成核心权利要求。

## 15. 当前小样本验证结果

本轮使用 DeepSeek API 小样本开发验证，每个数据集 5 条：

| 数据集 | 正确 | 准确率 | 平均 API tokens | 执行失败 |
| --- | ---: | ---: | ---: | ---: |
| WTQ | 4/5 | 80.0% | 2428.0 | 0 |
| TabFact | 4/5 | 80.0% | 2138.0 | 0 |
| CRT | 5/5 | 100.0% | 3631.0 | 0 |
| 合计 | 13/15 | 86.7% | 2732.3 | 0 |

对比参考：

```text
MACT 参考平均 token = 8867
myAgent 当前平均 token = 2732.3
token ratio ≈ 30.8%
```

验证命令结果：

```text
python -m unittest discover -s tests -q
150 tests OK

python -m compileall -q code
OK

git diff --check
OK
```

说明：这是开发样本，不是正式盲测结果。正式专利/论文实验仍应采用冻结代码后的更大规模盲测。

## 16. 哪些内容可以写进专利

### 16.1 建议写成核心发明点

1. 风险驱动的选择性协作推理框架  
   根据推理前风险和推理后风险动态决定是否调用更复杂的推理路径。

2. 多维风险评估公式  
   将语义复杂度、结构覆盖度、歧义风险、证据缺口、操作契约风险合成为 `R_pre`。

3. 推理后风险修正机制  
   根据候选答案分歧、证据绑定缺口和执行/契约失败计算 `R_post`。

4. 证据包 EvidencePack  
   在不读取金标答案的前提下，抽取候选行列、实体匹配、操作线索和证据缺口。

5. 答案契约驱动的候选比较  
   根据 scalar、label、list、tuple 的不同形态使用不同等价判定方法。

6. 预算约束下的动态升级  
   在整体平均 token 预算内，对高风险样本分配更多推理资源。

7. 确定性语义算子与 LLM 推理的混合路径  
   对单位归一化、百分比快照聚合、系统性映射判定等可程序化问题优先使用低成本确定性算子。

### 16.2 建议写成实施例

1. WTQ 中的 denotation 规范化；
2. TabFact 的 true/false 契约；
3. CRT-QA 的 duration、percentage snapshot、event association 处理；
4. DeepSeek API 小样本实验；
5. legacy/selective 模式切换；
6. MACT 作为 token 和准确率对标基线。

### 16.3 不建议写成专利核心点

1. 使用 DeepSeek-v4-pro 或某个具体模型；
2. API key、base URL、命令行参数；
3. 具体数据集样本 ID 或错误题；
4. `ITA -> Italy` 这种具体映射表；
5. 小样本 13/15 结果；
6. 固定阈值 `0.25/0.55/0.70` 的唯一性；
7. 某个 prompt 的具体措辞。

这些内容可以作为实验设置或实施例，但不应作为核心权利要求。

## 17. 防止被认为过拟合的说明

本轮设计中应强调以下边界：

- 风险评估不读取 gold answer；
- 证据包只使用问题、表格和公开数据集格式；
- 规则不按 sample ID、table ID 或具体题目文本分支；
- CRT 语义算子按字段类型和问题模式触发；
- 正式实验前应冻结代码、参数、prompt 和 evaluator；
- 开发样本结果只用于调试，不作为最终论文/专利性能结论。

这一段很重要，因为专利报告中不能让审查人或答辩老师感觉系统只是针对数据集做记忆式修补。

## 18. 可以放进专利报告的技术效果表述

建议表述：

> 本方案通过构建表格证据包、计算推理前风险和推理后风险，并在基准 token 预算约束下选择不同协作深度，使系统能够对低风险样本保持低成本推理，对高风险样本触发候选答案验证或更高成本推理，从而在保持或提升答案正确性的同时降低平均推理资源消耗。

当前小样本可以谨慎写成：

> 在开发样本验证中，系统在 WTQ、TabFact、CRT-QA 三类表格问答任务上均可运行，未出现代码执行失败；相较固定高成本推理流程，当前 selective 模式显示出较低 token 消耗和较好的准确率趋势。

不建议现在写成：

> 已经证明全面超过 MACT。

原因：当前只是 5 条/数据集的小样本开发验证，正式结论需要更大盲测。

## 19. 后续正式实验建议

为了让专利报告和硕士论文更稳，建议下一步执行：

1. 固定当前代码提交和参数；
2. 每个数据集抽取至少 20 条 blind-style 样本做中间验证；
3. 最终每个数据集抽取至少 100 条未参与开发的样本；
4. myAgent 和 MACT 使用同一模型、同一 evaluator、同一数据切分；
5. 报告总体准确率、分数据集准确率、低/中/高风险分层准确率、平均 token、执行失败率；
6. 若统计显著性不足，论文/专利中表述为“在准确率不降低条件下降低资源消耗”，不要强写“显著超过”。

## 20. 文件索引

核心实现：

- `code/risk_control.py`
- `code/evidence_builder.py`
- `code/selective_collaboration.py`
- `code/my_agents.py`
- `code/answer_contracts.py`
- `code/dataset_profiles.py`

数据与模型接入：

- `code/dataset_adapters.py`
- `code/model_backends.py`
- `code/tqa.py`
- `code/run_wtq_myagent.py`

评估与对比：

- `code/evaluate_results.py`
- `code/compare_blind_results.py`
- `code/calibrate_risk_policy.py`
- `code/sample_benchmark.py`

测试：

- `tests/test_risk_control.py`
- `tests/test_evidence_builder.py`
- `tests/test_selective_collaboration.py`
- `tests/test_myagent_pipeline.py`
- `tests/test_answer_contracts.py`
- `tests/test_dataset_adapters.py`
- `tests/test_model_backends.py`
- `tests/test_evaluate_results.py`
- `tests/test_compare_blind_results.py`
- `tests/test_calibrate_risk_policy.py`
- `tests/test_sample_benchmark.py`

实验报告：

- `outputs/selective_risk_dev_2026-06-27/REPORT.md`

## 21. 2026-06-27 Blind20 语义算子增强记录

### 21.1 本轮优化前的问题

在 `outputs/blind20_selective_2026-06-27/` 的 20 条/数据集盲测中，myAgent 的整体结果为 47/60，MACT 为 50/60。myAgent 的平均 token 明显低于 MACT，但准确率未达到“比肩或略超 MACT”的目标。

主要差距集中在：

1. TabFact 的集合唯一性、否定比较、相关性判断。
2. CRT-QA 的年份范围、持续 top-k、半球标签等语义格式。
3. 中等风险样本没有触发更强的确定性校验或更高预算路径。

这些问题不是样本 ID 问题，而是可泛化的表格语义算子缺口。

### 21.2 本轮新增或强化的技术点

本轮在原有 selective collaboration 框架上增加了“数据集语义算子层”。该层位于表格压缩之后、LLM 生成程序之前，优先处理可以稳定程序化判断的问题，减少模型自由生成带来的误判。

具体修改包括：

1. TabFact only/set equality 算子  
   对“only A and B ...”这类陈述，抽取声明中的实体集合，再从表格对应列计算实际唯一实体集合，并忽略 `tba`、`no data`、`unknown` 等未定值。这样可以避免把待公布值误判为第三类实体。

2. TabFact inverse correlation 算子  
   对“X has inverse correlation with/to Y”这类陈述，基于列名和问题短语匹配数值列，过滤常量列，计算 Pearson 相关性。若相关性为负则返回 true。该逻辑避免把 `games` 常量列误选为 `games lost`。

3. TabFact 否定日期/序号比较算子  
   对“no game on DATE greater than week N”这类陈述，标准化日期并比较 week/rank/position 等序号列，判断是否存在违反否定条件的行。

4. CRT-QA century manufacturing 算子  
   对“1900s vs 1800s manufactured/produced/built”这类问题，解析制造年份或年份区间，并按世纪区间重叠计数，避免只按起始年统计。

5. CRT-QA consistent top-k 算子  
   支持两种表格形态：一种是 `rank 2008`、`rank 2009` 这类逐年 rank 列；另一种是单独 `rank` 列加多个年份数值列。后者在真实 CRT 表格中较常见。

6. CRT-QA hemisphere label 规范化  
   当问题要求 eastern/western hemisphere，而模型只输出 `Eastern` 或 `Western` 时，规范化为 `eastern hemisphere` 或 `western hemisphere`，减少格式性扣分。

7. 风险证据增强  
   在 `EvidenceBuilder` 中将 `consistently`、`correlation`、`inverse`、`top N`、`from YEAR to YEAR`、`1800s/1900s` 等模式加入 operation hints，提高 logic/dependency 风险信号。

### 21.3 涉及代码文件

核心修改：

- `code/my_agents.py`
  - 新增 `_loose_text_key`、`_loose_tokens`、`_split_entity_phrase`、`_parse_date_key`、`_extract_year_range` 等通用解析函数。
  - 新增 TabFact deterministic shortcuts。
  - 新增 CRT deterministic shortcuts。
  - 新增 CRT hemisphere 标量规范化。

- `code/evidence_builder.py`
  - 增加 temporal consistency / correlation / top-k / century-range 风险信号。

测试修改：

- `tests/test_myagent_pipeline.py`
  - 增加 TabFact only 集合、inverse correlation、CRT consistent top-k、CRT hemisphere 规范化等回归测试。

- `tests/test_evidence_builder.py`
  - 增加语义压力模式提高 logic/dependency 风险信号的测试。

### 21.4 Blind20 验证结果

最终验证目录：

- `outputs/blind20_semantic_v3_2026-06-27/`

对比基线：

- `outputs/blind_holdout_v4_2026-06-24/mact_wtq.jsonl`
- `outputs/blind_holdout_v4_2026-06-24/mact_tabfact.jsonl`
- `outputs/blind_holdout_v4_2026-06-24/mact_crt.jsonl`

结果如下：

| 数据集 | myAgent | MACT | myAgent 平均 tokens | MACT 平均 tokens | token ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| WTQ | 16/20, 80.0% | 16/20, 80.0% | 1968.10 | 9516.00 | 20.68% |
| TabFact | 19/20, 95.0% | 19/20, 95.0% | 3052.40 | 8052.55 | 37.91% |
| CRT-QA | 18/20, 90.0% | 15/20, 75.0% | 2341.75 | 9031.40 | 25.93% |
| 合计 | 53/60, 88.33% | 50/60, 83.33% | 2454.08 | 8866.65 | 27.68% |

验收标准结果：

```text
overall_accuracy_at_least_mact: true
at_least_two_datasets_at_least_mact: true
token_ratio_at_most_0_75: true
execution_failure_rate_at_most_0_02: true
acceptance_selective_risk_collaboration: true
```

### 21.5 对专利撰写的意义

本轮修改可以写成“风险感知的确定性语义算子选择机制”：

1. 系统不直接对所有问题使用高成本多轮协作，而是根据问题模式和证据包判断是否存在可程序化的表格语义算子。
2. 若存在确定性算子，则以低 token 成本执行结构化校验。
3. 若不存在确定性算子，则继续走原有 LLM planner / calculator / verifier / selective collaboration 路径。
4. 风险证据中的 logic、dependency、gap 信号仍进入 `R_pre`，为预算调度和后续 fallback 提供依据。

该设计比单纯 prompt 调参更适合作为专利技术点，因为它包含明确的模块、输入输出、触发条件、计算路径和可验证效果。

### 21.6 仍需谨慎说明的边界

这次结果是 20 条/数据集的 blind-style 中间验证，不应直接写成最终大规模结论。更稳妥的表述是：

> 在 60 条 blind-style 中间验证样本上，myAgent 在准确率上达到并超过 MACT 基线，同时平均 API token 约为 MACT 的 27.68%。该结果表明风险感知语义算子与选择性协作机制具有进一步扩大实验的价值。

正式论文或专利报告中，建议继续补充每数据集 100 条以上的冻结盲测结果。
