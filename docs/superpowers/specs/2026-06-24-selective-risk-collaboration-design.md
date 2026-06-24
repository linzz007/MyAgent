# Selective Risk Collaboration Design

日期：2026-06-24

## 1. 目标

在 WTQ、TabFact 和 CRT 三个数据集上，将 myAgent 从当前“低 Token 但总体准确率低于 MACT”的状态，升级为“准确率不低于 MACT，同时平均真实 API Token 不超过 MACT 的 75%”。

当前独立盲测基线为：

- myAgent：45/60，75.0%，平均 2349 Token。
- MACT：50/60，83.3%，平均 8867 Token。
- myAgent 在 WTQ 和 TabFact 落后，在 CRT 领先。
- 当前 Router 标记的 easy 样本并不是真正低风险样本，因此不能继续把 easy/medium/hard 当作推理预算依据。

本轮设计不以继续压低 Token 为首要目标，而是在平均预算约束下，把节省的 Token 重新投入高风险样本。

## 2. 验收标准

所有主要指标必须在相同模型、相同采样、相同 evaluator 和相同答案规范下与 MACT 配对比较。

正式盲测验收标准：

1. 三个数据集各随机冻结至少 100 条未参与开发的样本，并按 table ID 分组隔离。
2. myAgent 总体准确率不低于 MACT，目标为高出至少 1 个百分点。
3. 至少两个数据集的准确率不低于 MACT，任一数据集不得低于 MACT 超过 5 个百分点。
4. 低风险和高风险样本分别报告准确率；两个分层均不得低于 MACT 超过 2 个百分点，且至少一个分层不低于 MACT。
5. myAgent 平均真实 API Token 不超过 MACT 的 75%。当前基线对应约 6650 Token/样本，但正式阈值使用同批 MACT 实测值计算。
6. 执行失败率不超过 2%。
7. 报告 Wilson 置信区间、配对 McNemar 检验、准确率差、Token 差和每个组件的消融结果。

60 条 V4 盲测只作为问题诊断证据，不再作为调参集或最终论文结论。

## 3. 非目标与防过拟合边界

本轮不做以下事情：

- 不按 sample ID、table ID、具体实体名或完整题目文本分支。
- 不在运行时读取 gold answer、评测结果或历史错误标签。
- 不针对冻结盲测修改 prompt、阈值或规则。
- 不为每一种自然语言表述手写专用求解器。
- 不要求所有样本都执行多智能体协作。

允许按公开数据集协议选择输入清洗、证据构建和答案规范，但通用风险计算、预算控制、候选比较和升级机制必须共享。

## 4. 方案选择

采用“选择性双路径协同”方案：

- 低风险样本继续使用轻量确定性路径。
- 中风险样本使用 Code Solver 加 Evidence Verifier。
- 高风险样本并行生成 Code Solver 和 ReAct Solver 两个候选，由 Agreement Judge 比较。
- 双路径冲突、证据不足或运行失败时，才升级到 Thinking Solver。

未采用的方案：

- 全量双路径 MACT-lite：准确率可能接近 MACT，但 Token 优势和专利差异性不足。
- 单一强模型加长 prompt：实现简单，但难以稳定修复 WTQ 证据召回和多答案问题，也无法解释 Token 如何按风险分配。

## 5. 风险模型

### 5.1 压缩复杂度

保留现有专利中的语义与结构联合复杂度，但调整权重：

```text
D = 0.45 * S + 0.55 * C
```

- `S`：问题的语义复杂度。
- `C`：预计触达单元格比例及结构覆盖复杂度。
- `D`：主要控制表格压缩强度，不直接决定最终答案。

结构权重提高到 0.55，是因为 V4 的主要错误集中在实体覆盖、候选行遗漏和多列关系，而不是纯语言理解。

为避免大表格因“触达单元格占全表比例较小”被错误判断为简单，结构复杂度按以下方式计算：

```text
C = 0.50 * C_cov + 0.30 * C_disp + 0.20 * C_type
```

- `C_cov`：预计触达单元格数量的对数归一值，归一分母最多按 64 个单元格计算。
- `C_disp`：候选行和候选列在表格中的分散程度。
- `C_type`：相关列的混合类型、缺失值和单位不一致程度。

`S` 继续由现有语义评分器产生，但必须在测试中验证相同问题重复评分的稳定性；解析失败时使用可复现的操作词和依赖步数规则分数，不允许随机回填。

### 5.2 推理前风险

```text
R_pre = 0.20 * D + 0.30 * A + 0.30 * G + 0.20 * O
```

- `A`：歧义风险，包括多列匹配、多实体匹配、指代和相对时间无法唯一解析。
- `G`：证据缺口，包括候选行或列覆盖不足、答案所需实体未被检索、关键值缺失。
- `O`：操作与答案契约风险，包括聚合、排序、比例、日期、多跳、否定、多答案、单位和结构化输出。

所有特征只能来自问题、表格、公开数据集协议和中间执行状态，不能使用 gold answer。

三个风险量按固定子特征计算：

```text
A = 0.35 * a_entity + 0.25 * a_column + 0.20 * a_temporal + 0.20 * a_reference
G = 0.35 * g_entity + 0.25 * g_column + 0.20 * g_missing + 0.20 * g_stability
O = 0.25 * o_steps + 0.25 * o_dependency + 0.20 * o_contract
    + 0.15 * o_unit + 0.15 * o_logic
```

- `a_entity`：问题实体无匹配或存在多个同等候选的比例。
- `a_column`：问题字段可映射到多个列或无法映射的比例。
- `a_temporal`：相对时间、时间范围或排序方向未能唯一确定的程度。
- `a_reference`：代词、比较基准或省略对象未能唯一绑定的程度。
- `g_entity`：抽取实体中没有任何表格证据匹配的比例。
- `g_column`：推理操作要求的字段中未进入候选列的比例。
- `g_missing`：候选证据中关键单元格为空、TBA 或不可解析的比例。
- `g_stability`：精确匹配与归一/模糊匹配两种检索所得候选集合的差异，使用 `1 - Jaccard`。
- `o_steps`：显式操作数量除以 4 后截断到 `[0,1]`。
- `o_dependency`：存在两步及以上依赖计算时为 1，否则为 0。
- `o_contract`：scalar、label、list、tuple 等答案结构的规范化风险。
- `o_unit`：需要单位识别、换算或精度决策的程度。
- `o_logic`：否定、AND、OR、条件和量词组合的复杂度。

所有子特征先截断到 `[0,1]`。实体或列没有被问题要求时，对应风险记为 0，而不是因分母为空记为 1。

### 5.3 推理后风险

```text
R_post = min(1, R_pre + 0.30 * D_p + 0.20 * V_g + 0.30 * X)
```

- `D_p`：两条候选路径的答案分歧程度。
- `V_g`：答案原子无法绑定证据或证据相互冲突的程度。
- `X`：执行、答案契约、单位或规范化失败。

其中：

```text
D_p = 1 - answer_similarity(candidate_1, candidate_2)
V_g = 0.70 * ungrounded_answer_ratio + 0.30 * evidence_conflict
X = max(exec_failure, contract_failure, unit_failure, normalization_failure)
```

`answer_similarity` 由 AnswerContract 决定：标签和标量使用规范化等价，数值使用数据集容差，列表使用集合 Jaccard，tuple 按位置比较。失败标记均为 0 或 1。

执行异常、答案无法规范化、非法标签、必要单位缺失和答案完全无证据绑定属于硬升级条件，不受加权分数抵消。

### 5.4 风险阈值

- `R_pre < 0.25`：轻量确定性路径。
- `0.25 <= R_pre < 0.55`：Code Solver + Evidence Verifier。
- `R_pre >= 0.55`：Code Solver + ReAct Solver + Agreement Judge。
- 双路径冲突、硬升级条件成立或 `R_post >= 0.70`：Thinking Solver。

阈值只允许在开发集上通过 table-ID 分组交叉验证调整，盲测开始前冻结。

## 6. 架构与接口

### 6.1 DatasetAdapter

复用并扩展现有 `dataset_adapters.py`、`dataset_profiles.py` 和 `answer_contracts.py`，输出统一的 `NormalizedSample`：

```text
NormalizedSample
  question
  dataframe
  dataset_name
  answer_contract
  public_metadata
```

适配层只负责数据格式、公开协议和答案结构，不负责决定答案。

### 6.2 RiskProfiler

新增独立风险控制模块，输入 `NormalizedSample`、现有 Router 特征和压缩候选，输出：

```text
RiskAssessment
  D, A, G, O
  R_pre
  hard_triggers
  recommended_level
  feature_evidence
```

`feature_evidence` 保存每个风险分数的可审计依据，便于论文分析和阈值校准。

### 6.3 EvidenceBuilder

输入完整表格、问题和数据集 profile，输出不可变的 `EvidencePack`：

```text
EvidencePack
  schema_profile
  candidate_rows
  candidate_columns
  entity_matches
  operation_hints
  missing_or_ambiguous_items
  provenance
```

证据包统一供所有 Solver 使用，避免不同路径看到不一致的表格版本。压缩表格只是一种视图；完整表格索引始终可供代码执行和验证使用。

### 6.4 Solver 路径

所有路径返回统一的 `CandidateAnswer`：

```text
CandidateAnswer
  normalized_answer
  raw_answer
  reasoning_summary
  evidence_refs
  executable_program
  execution_result
  confidence
  token_usage
  failure
```

- `LowRiskExecutor`：直接查找、确定性比较和简单分类，失败时立即升级。
- `CodeSolver`：由 Planner 生成受限代码，通过现有 Calculator 执行，并保留程序与结果。
- `ReActSolver`：围绕 EvidencePack 进行有限轮次的“提出操作、执行、检查证据”，不允许自由读取完整 gold 信息。
- `ThinkingSolver`：只处理冲突或高后验风险样本，使用更完整证据和更高推理预算生成最终候选。

### 6.5 AgreementJudge

先执行确定性比较，再在必要时调用模型判断：

1. 使用 `AnswerContract` 规范化两个答案。
2. 检查答案是否相同、集合是否等价、数值是否在允许误差内。
3. 检查每个答案原子是否可绑定到表格证据或可复现执行结果。
4. 若答案一致且证据完整，直接接受。
5. 若答案不同但一方存在执行失败、非法契约或无证据，接受有效一方。
6. 若双方均有效但冲突，计算 `R_post` 并升级 Thinking Solver。

Judge 不允许仅凭语言流畅度选择答案。

### 6.6 BudgetController

以同批 MACT 的开发集平均 Token `B_M` 为基准，控制每个协作层级的最大预算：

- 轻量路径：不超过 `0.25 * B_M`。
- Code + Verify：不超过 `0.55 * B_M`。
- 双路径：不超过 `0.85 * B_M`。
- Thinking fallback：单样本不超过 `1.10 * B_M`，并记录升级原因。
- 整体平均预算必须满足 `AvgToken(myAgent) <= 0.75 * AvgToken(MACT)`。

当前 8867 Token 的 MACT 基线对应约 2200、4900、7540 和 9750 Token 的层级上限。正式实验按同模型实测基线重新计算，避免把 DeepSeek 的 Token 统计硬编码到其他模型。

BudgetController 不能在已经发现硬失败时强行输出无效答案；如果剩余预算不足，返回明确的 `budget_exhausted` 状态，并计入失败率。

## 7. 数据集策略

### 7.1 WTQ

WTQ 使用完整实体索引和精确 denotation 契约：

- 对所有列建立大小写归一、标点归一和唯一子串索引。
- 区分实体查询、逆向查询、时间条件、排序、聚合和多答案集合。
- 相对时间缺少参考年份时标记歧义，不使用当前年份猜测。
- 多答案在推理过程中保持列表结构，最终由 evaluator 按集合语义比较。
- EvidenceBuilder 不因压缩而删除完整实体候选；压缩只影响传给模型的上下文。

### 7.2 TabFact

TabFact 将复合陈述拆分为可验证子句：

- 抽取实体、比较、计数、比例、否定和逻辑连接词。
- 每个子句单独绑定证据或执行结果。
- 所有必要子句验证成功后再按 AND、OR、NOT 合并。
- 数字、计数、比较和比例默认进入 Code Solver，不由纯文本分类器猜测。

### 7.3 CRT

CRT 保留当前表现较好的统一 Planner/Calculator 路径：

- 解析题目给出的动态候选标签和 tuple 契约。
- 显式校验单位、精度、主客场等输出语义。
- 只有代码执行失败、单位冲突、答案契约失败或双路径不一致时升级。

数据集策略只能改变证据构建和输出契约，风险升级机制保持一致。

## 8. 数据流

```text
Raw Sample
  -> DatasetAdapter / AnswerContract
  -> Router features + full table schema
  -> EvidenceBuilder
  -> RiskProfiler computes R_pre
  -> BudgetController selects collaboration level
  -> LowRiskExecutor or CodeSolver or dual-path solvers
  -> AgreementJudge + Evidence Verifier
  -> RiskProfiler computes R_post
  -> optional ThinkingSolver
  -> FinalAnswerAgent normalizes output
  -> shared dataset evaluator
  -> trace, tokens, risk and failure metrics
```

`TableQAPipeline` 继续作为外部入口，但只负责编排。风险计算、证据构建、求解路径和候选比较拆到独立模块，避免继续扩大当前 `my_agents.py`。

## 9. 异常与降级处理

- API 暂时失败：指数退避重试，重试次数和 Token 单独记录；超过上限返回失败，不静默伪造答案。
- Planner 输出非法代码：先进行 AST/白名单校验，再执行；失败信息反馈给一次受限重规划。
- 代码执行失败：设置硬升级条件，保留真实异常，不能被“空答案”契约错误覆盖。
- EvidencePack 无候选：回退到完整 schema 和实体索引重新构建一次，仍为空则升级。
- 候选答案无法规范化：尝试一次基于 AnswerContract 的格式修复；内容不得在格式修复阶段改变。
- 双路径冲突：不得通过固定优先级盲选，必须经过证据与执行验证。
- Token 超限：停止新增推理调用，保留已有有效候选；没有有效候选则标记失败。
- 评测异常：原始预测保留，评测器失败不得改写为错误答案或正确答案。

所有异常都写入统一 trace：`stage`、`error_type`、`retry_count`、`risk_before`、`risk_after`、`tokens` 和 `selected_candidate`。

## 10. 参数调整方法

参数调整目标为：

```text
minimize 1 - Accuracy
subject to AvgToken(myAgent) <= 0.75 * AvgToken(MACT)
```

使用按 table ID 分组的开发集交叉验证调整：

- 风险权重与阈值。
- 每个层级的调用轮数和 Token 上限。
- EvidenceBuilder 的候选数量。
- AgreementJudge 的证据完整度阈值。

优先选择多个折上稳定、参数变化不敏感的配置，而不是开发集单点最高准确率。所有最终参数、随机种子、prompt 版本和代码提交哈希在盲测前冻结。

## 11. 测试策略

### 11.1 单元测试

- 风险公式边界、硬升级条件和阈值分层。
- 实体歧义、证据缺口、操作风险和答案契约风险特征。
- EvidencePack 的来源追踪与不可变性。
- WTQ 列表答案、TabFact 子句合并、CRT 动态标签和 tuple。
- AgreementJudge 的集合等价、数值容差、执行失败和证据冲突。
- BudgetController 的层级上限、总预算统计和超限行为。

### 11.2 集成测试

- 三个数据集分别覆盖低风险、中风险、高风险和硬失败样本。
- 验证每次升级都有可解释原因，低风险样本不会无故进入 Thinking Solver。
- 验证完整表格索引可用于执行，但 prompt 只接收预算允许的证据视图。
- 验证 API Token 来自 provider usage；缺失时标记 estimated，不能与真实 Token 混报。

### 11.3 回归与消融

至少报告以下版本：

1. 当前 V4 基线。
2. V4 + RiskProfiler。
3. V4 + RiskProfiler + EvidenceBuilder。
4. 上述版本 + 双路径 AgreementJudge。
5. 完整版本 + Thinking fallback。

每个版本报告总准确率、三个数据集准确率、低/高风险准确率、平均 Token、升级率和失败率。这样可以判断收益来自证据增强、双路径还是最终高预算回退。

## 12. 盲测协议

1. 从完整数据中排除所有开发样本、历史 benchmark18、blind36、V4 sample ID 和 table ID。
2. 使用预注册随机种子按 table ID 分组抽取，每个数据集至少 100 条。
3. 保存样本清单 SHA-256、代码提交哈希、参数文件哈希、prompt 哈希和模型配置。
4. 先冻结 evaluator，并用人工构造边界案例验证 evaluator，而不是用待测预测反向修改规则。
5. 在全部 myAgent 和 MACT 预测完成前，不计算分数据集正确率，不修改代码和参数。
6. 对两个项目使用同一个 evaluator；数据集规范化规则相同，禁止项目自带评分器产生不可比结果。
7. 输出配对明细，但错误分析只能在该轮盲测完全结束后进行。

风险分层由 `R_pre` 在推理前生成：

- 低风险：`R_pre < 0.55` 且无硬升级条件。
- 高风险：`R_pre >= 0.55` 或存在硬升级条件。

该二分仅用于论文和性能诊断；实际执行仍保留轻量、Code + Verify、双路径和 fallback 四级预算。

## 13. 专利技术点映射

现有“语义复杂度 + 单元格覆盖复杂度”的技术点继续用于自适应压缩。本轮增加第二层可解释控制：

- `D` 决定证据压缩强度。
- `R_pre` 决定初始协作深度。
- `R_post` 根据路径分歧、证据验证和执行状态决定是否追加推理。
- BudgetController 在准确率目标下约束平均资源消耗。

相较固定多智能体流程，核心差异是“风险驱动的协作深度与后验升级”；相较仅做简单/复杂路由，核心差异是风险由歧义、证据缺口、操作契约和运行时验证共同决定，并能在推理后动态修正。

论文和专利只能声称盲测实际支持的结论。若正式结果仅达到准确率持平和 Token 降低，应表述为“在准确率不降低条件下降低推理资源”，不能写成统计显著超越。

## 14. 实施范围

本轮实施只覆盖：

- 新增风险、证据、候选和预算数据结构。
- 将现有 Pipeline 接入选择性协同流程。
- 增加三个数据集的证据构建策略。
- 增加共享 evaluator、风险分层、Token 和消融报告。
- 增加对应单元、集成和小规模开发集测试。

正式 100 条/数据集盲测在代码与参数冻结后单独执行，不在开发阶段反复查看结果。
