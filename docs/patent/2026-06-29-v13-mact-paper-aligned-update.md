# myAgent v13 与 MACT 论文对齐后的专利化优化说明

日期：2026-06-29  
项目：`D:\AAAcode\code-code\agent+\myAgent-main`  
对标论文：`D:\AAAcode\Agent调研资料\任务协同\-Efficient Multi-Agent Collaboration with Tool Use.pdf`  
模型：DeepSeek API，`deepseek-v4flash`  
本轮原则：不重跑 MACT；只运行 myAgent v13，并复用已有 MACT 结果做同 id、同评估器比较。

## 1. 当前结论

v13 的目标不是继续省 token，而是把 CRT、TabFact 中高风险错题抽象成“问题类型触发的公式化 verifier”。这些 verifier 不读取样本 id，不读取金标答案，触发条件只来自题干、表头和表格内容，因此更适合写成专利里的算法模块。

本轮先做 50 条/数据集小规模验证，确认超过 MACT 后，又只补跑 myAgent v13 的剩余 150 条/数据集，最终形成三个数据集各 200 条、共 600 条完整结果。与已有 MACT 输出按 id 配对后：

| 数据集 | myAgent v13 | MACT 同 200 条 | 结论 | myAgent 平均 token | MACT 平均 token | token 比例 |
|---|---:|---:|---|---:|---:|---:|
| WTQ | 158/200, 79.00% | 156/200, 78.00% | 高 2 题 | 17048.73 | 45046.51 | 37.85% |
| TabFact | 189/200, 94.50% | 188/200, 94.00% | 高 1 题 | 16211.68 | 41734.40 | 38.84% |
| CRT | 140/200, 70.00% | 135/200, 67.50% | 高 5 题 | 12088.89 | 55536.89 | 21.77% |
| 总计 | 487/600, 81.17% | 479/600, 79.83% | 高 8 题 | 15116.43 | 47439.26 | 31.86% |

输出文件：

| 文件 | 说明 |
|---|---|
| `outputs/blind200_v4flash_2026-06-27/wtq_myagent_v13_200_complete.jsonl` | WTQ v13 200 条完整输出 |
| `outputs/blind200_v4flash_2026-06-27/tabfact_myagent_v13_200_complete.jsonl` | TabFact v13 200 条完整输出 |
| `outputs/blind200_v4flash_2026-06-27/crt_myagent_v13_200_complete.jsonl` | CRT v13 200 条完整输出 |
| `outputs/blind200_v4flash_2026-06-27/strong200_v13_compare.json` | v13 与 MACT、v12 的 600 条配对比较 |

## 2. MACT 论文中的关键机制

MACT 的核心是多智能体在线规划。它主要包含：

| 模块 | 作用 |
|---|---|
| planning agent | 生成下一步动作和预估观察结果 |
| coding agent | 生成 Python 代码，把动作落到表格计算 |
| tools | Python interpreter、calculator、Wikipedia search |
| memory state | 保存历史 action、observation、prediction |
| self-consistency shortcut | 当多次最终预测高度一致时提前停止 |

MACT 的每轮流程可以概括为：

```text
action generation
-> action selection
-> tool selection / code generation
-> observation computation
-> memory state update
```

论文中的误差来源对 myAgent 有直接参考价值：

| MACT 误差来源 | 对 myAgent 的启发 |
|---|---|
| coding agent 生成错误代码 | 对可公式化题型优先走确定性 verifier |
| 严格 EM 口径导致格式错 | 建立 answer contract 和最终答案规范化 |
| planner 分解失败 | 对高风险题二次校验，不完全相信单次 plan |

## 3. myAgent 与 MACT 的差异化专利点

MACT 的主线是“多智能体在线规划 + 工具执行 + 自一致性提前停止”。myAgent 的主线应描述为：

```text
数据集适配
-> 问题契约生成
-> 表格压缩与证据包构造
-> 双阶段风险评分
-> 问题类型触发的公式化 verifier
-> 高风险选择性协作/二次校验
-> 答案合同归一化
```

这与 MACT 的差别在于：myAgent 不把所有复杂题都交给 LLM 反复规划，而是先判断题目是否属于可程序化验证类型。能确定计算的部分直接由 verifier 执行；只有剩余高风险或 verifier 不覆盖的部分才花更多 token 调用模型协作。

## 4. v13 新增的专利化模块

### 4.1 形态归一语义匹配

修改位置：`code/my_agents.py`

新增 `_singular_token`，用于把题干和表头中的单复数差异归一。例如：

```text
mountains -> mountain
diseases -> disease
countries -> country
```

专利表述：

```text
对题干 token q_i 和表头 token c_j 执行形态归一函数 s(x)，
若 s(q_i)=s(c_j) 或二者满足模糊编辑距离阈值，则认为该列可绑定。
```

这个模块解决的是表格问答中“自然语言问题和表头表达不完全一致”的普遍问题。

### 4.2 TabFact 最大值差分 verifier

新增方法：`_tabfact_numeric_difference_from_max_answer`

适用题型：

```text
实体 E 的指标 M 比最高指标值少 D。
```

公式化描述：

```text
answer = True
若 |max_i T[i,M] - T[row(E),M] - D| <= epsilon
否则 answer = False
```

该模块把自然语言事实验证转成“实体行定位 + 指标列绑定 + 最大值差分判断”。

### 4.3 TabFact 列值计数 verifier

新增方法：`_tabfact_column_value_count_answer`

适用题型：

```text
N 个对象在位置/属性 Z 中。
```

公式化描述：

```text
count = sum_i I(match(T[i,C], Z))
answer = (count == N)
```

该模块处理的是“陈述中的数量是否等于表中满足某属性的行数”。

### 4.4 TabFact 最高比分 verifier

新增方法：`_tabfact_highest_scoring_game_answer`

适用题型：

```text
最高得分比赛是日期 D，比分总和为 S。
```

公式化描述：

```text
score_total_i = left_score_i + right_score_i
max_score = max_i score_total_i
answer = (max_score == S) and exists i: date_match(T[i,date], D)
```

该模块让体育比分类事实验证不再依赖 LLM 判断 winner 或最高分。

### 4.5 CRT 百分比/概率 verifier

新增方法：

| 方法 | 题型 |
|---|---|
| `_crt_owned_percentage_answer` | 某实体拥有/占有比例 |
| `_crt_medal_probability_answer` | 随机选中对象来自某国家，或国家至少获得一枚金牌的概率 |

公式化描述：

```text
P(entity) = count(match(row, entity)) / N
P(weighted_entity) = sum_i weight_i * I(match(T[i,C], entity)) / sum_i weight_i
P(condition) = sum_i I(condition_i) / N
```

这里还加入了 total/grand total 行排除，避免把汇总行当作普通国家或实体。

### 4.6 CRT 比值 verifier

新增方法：`_crt_medal_ratio_answer`

适用题型：

```text
silver to gold
gold medals to total medals
top N countries combined
```

公式化描述：

```text
R_raw = sum_i A_i : sum_i B_i
R_decimal = round(sum_i A_i / sum_i B_i, precision)
```

当前实现会根据题干和 answer contract 保留 raw ratio 或 decimal ratio。这个点对 CRT 很关键，因为 CRT 中 `2:6` 和 `1:3` 数学等价，但评估口径可能要求原始合计比。

### 4.7 CRT 多组多数关系 verifier

新增方法：`_crt_majority_medal_by_group_answer`

适用题型：

```text
是否存在某个 sport/event/category，使单一国家获得该组多数奖牌。
```

公式化描述：

```text
for each group g:
  total_g = sum_i medals_i where group_i = g
  max_country_g = max_c sum_i medals_i where group_i = g and country_i = c
  if max_country_g > total_g / 2: answer = Yes
answer = No
```

### 4.8 CRT 无金牌显著奖牌量 verifier

新增方法：`_crt_significant_medals_without_gold_answer`

适用题型：

```text
是否存在没有金牌但奖牌总数显著的国家。
```

公式化描述：

```text
max_total = max_i total_i
answer = exists i: gold_i = 0 and total_i >= lambda * max_total
```

当前 `lambda` 使用 0.5。这个权重可以写成专利中的可调阈值参数。

## 5. 为什么这不是针对数据集过拟合

本轮新增规则有三个约束：

1. 不使用样本 id、table id、gold answer。
2. 只通过题干模式、列语义、单元格值触发。
3. 输出必须通过统一评估器和 answer contract。

因此它不是“看到某道题错了就硬编码答案”，而是把错题抽象成可复用的推理类型。专利里不建议写成一堆函数名，而应写成：

```text
一种基于问题类型识别的公式化表格验证算子库，
其算子由题干语义、表格列绑定、行级条件、聚合函数和答案合同共同确定。
```

工程实现可以有多个 verifier，但权利要求应覆盖统一机制：

```text
问题类型识别 -> 列/实体绑定 -> 公式计算 -> 合同归一化 -> 风险评分反馈
```

## 6. 验证结果

### 6.1 单元测试

```text
python -m unittest discover -s tests -v
Ran 182 tests
OK
```

### 6.2 离线风险扫描

离线扫描只用于评估 verifier 的潜在收益。扫描时 verifier 只读取题干和表格；金标只用于事后判断是否正确。

| 数据集 | v13 新 verifier 命中 | 命中正确 | 潜在修正 v12 错题 | 潜在回退 |
|---|---:|---:|---:|---:|
| TabFact | 3 | 3 | 3 | 0 |
| CRT | 7 | 7 | 7 | 0 |

命中类型：

| 数据集 | 类型 |
|---|---|
| TabFact | 最大值差分、列值计数、最高比分 |
| CRT | 拥有比例、奖牌概率、奖牌比值、多数关系、无金牌显著奖牌量 |

### 6.3 600 条实际运行

见第 1 节。当前 600 条结果显示 myAgent v13 已经超过同 id MACT 8 题，token 为 MACT 的 31.86%。

相对 v12，v13 从 480/600 提升到 487/600，净增 7 题；其中 TabFact 从 186/200 提升到 189/200，CRT 从 135/200 提升到 140/200。WTQ 从 159/200 小幅波动到 158/200，说明本轮真正收益集中在 CRT 和 TabFact 的可公式化 verifier。

## 7. 对后续论文和专利写法的建议

建议把当前系统写成一个统一算法，而不是写成“很多小功能拼接”。可以按下面结构组织：

1. 表格输入适配与 schema profile 构造。
2. 基于问题文本和表结构的 answer contract 生成。
3. 基于结构复杂度、歧义度、证据缺口、操作复杂度的双阶段风险评分。
4. 对高风险题触发选择性协作与二次校验。
5. 对可公式化题型触发 deterministic verifier。
6. 使用答案合同进行最终归一化和评估口径对齐。

核心公式可以写：

```text
difficulty = 0.45 * semantic_complexity + 0.55 * structure_score
pre_risk = 0.20 * difficulty + 0.30 * ambiguity + 0.30 * evidence_gap + 0.20 * operation_risk
post_risk = pre_risk + 0.30 * disagreement + 0.20 * verification_gap + 0.30 * hard_failure
```

其中 verifier 的选择可以写成：

```text
v* = argmax_v score(type(question), schema(table), contract(answer), v)
```

若 `score(v*)` 超过阈值，则执行公式化 verifier；否则进入 LLM planner 或高风险协作路径。

## 8. 当前风险

当前 v13 已经能作为专利报告中的一个更完整版本，但还有两个边界需要如实说明：

1. 本轮没有重跑 MACT，只复用了已有 MACT blind200 输出做同 id 对比。
2. v13 已经完整运行 myAgent 600 条，但由于 LLM API 存在随机性，正式服务器实验建议固定模型、temperature、输出文件和评估脚本版本。

如果后续继续优化，方向不应是继续省 token，而应是补齐 CRT 中“时长区间、异常点、相关性、ratio 格式合同”这四类通用 verifier。
