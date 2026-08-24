# 2026-08-24 Patent Direction Freeze and Next Experiment PRD

本文档用于冻结当前 MyAgent 专利方向、约束后续代码优化范围，并给服务器端实验执行者提供新的最小必要实验计划。后续实验和专利说明书应以本文档为优先依据；若与旧 PRD 冲突，以本文档的“方向冻结”和“禁止事项”为准。

## 1. 当前目标

当前目标不是继续在固定 formal200 样本上追分，也不是复现所有表格问答论文。当前目标是把 MyAgent 固化为一个可写进专利和硕士论文的稳定方案：

1. 保留已经完成的 Qwen3-32B formal200 正结果。
2. 将方法描述统一为“成本感知两路径路由 + 双评分 + 表格压缩 + 答案契约 + 选择性协作验证”。
3. 后续优化只允许围绕现有模块做小范围机制修正。
4. 使用 Seed-E 错误簇做诊断，再用新的 Seed-F/G 做盲测验证。
5. 不再添加新的大 agent、新路径或样本级规则。

## 2. 当前代码方向

当前代码的专利核心流程应描述为：

1. 输入问题、表格和数据集类型。
2. 将不同来源表格标准化为 pandas DataFrame，并保留原始表副本。
3. 根据问题和数据集格式生成答案契约，约束答案类型、标签集合、数值精度、分数格式、缩写格式等。
4. 识别问题类型和问题标签，例如查值、计数、聚合、比较、时序、事实验证、否定逻辑等。
5. 通过语义复杂度和结构复杂度形成双评分，并得到任务难度。
6. 顶层路由只分为 SIMPLE 和 COMPLEX 两类。
7. 在进入求解前执行问题感知表格压缩，包括行列裁剪、邻域扩展、列画像、证据保留和必要时全局回退。
8. 若高置信确定性表格算子可解，则直接执行并按答案契约输出。
9. SIMPLE 路径执行轻量查值和答案规范化。
10. COMPLEX 路径执行计划生成、pandas 代码执行、结果校验和答案规范化。
11. 后验风险较高或候选不一致时，触发选择性协作验证。

注意：文档和专利中不要再把 Light / Tool / Collab 写成三条顶层路径。Tool 推理和 Collab 验证只能作为 COMPLEX 路径内部的子过程。

## 3. 当前实验状态

### 3.1 可作为主结果的 formal200

Qwen3-32B formal200 已达成当前阶段目标：

| Method | WTQ | TabFact | CRT | Overall | Avg token | Avg time |
|---|---:|---:|---:|---:|---:|---:|
| MyAgent | 157/200 | 190/200 | 133/200 | 480/600 = 0.8000 | 6293.12 | 16.749s |
| MACT | 156/200 | 185/200 | 124/200 | 465/600 = 0.7750 | 11318.89 | 126.861s |
| Direct-CoT | 126/200 | 149/200 | 111/200 | 386/600 = 0.6433 | 712.67 | 2.55s |
| Single-Agent Pandas | 138/200 | 159/200 | 124/200 | 421/600 = 0.7017 | 1074.24 | 7.60s |

可写结论：在 Qwen3-32B 与相同 formal200 样本下，本方法在 WTQ、TabFact、CRT 及总体准确率上均超过 MACT，并且平均 token 约为 MACT 的 55.6%。

### 3.2 不能过度宣称的 Seed-E

Seed-E Gate-50 是稳定性诊断，不是主正结果：

| Dataset | MyAgent | MACT | Delta |
|---|---:|---:|---:|
| WTQ | 31/50 | 37/50 | -6 |
| TabFact | 40/50 | 42/50 | -2 |
| CRT | 24/50 | 26/50 | -2 |
| Overall | 95/150 = 0.6333 | 105/150 = 0.7000 | -10 |

结论边界：

1. formal200 正结果仍然有效。
2. 不能宣称多随机种子稳定超过 MACT。
3. Seed-E 应作为错误簇诊断来源，用于指导后续可解释机制修正。

## 4. 当前代码问题与处理原则

### 4.1 已确认的问题

1. `code/my_agents.py` 过大，集中包含路由、压缩、求解、校验和大量数据集确定性算子，工程可维护性一般。
2. 确定性算子数量较多，若继续按错题追加规则，会产生“针对数据集堆补丁”的观感。
3. `robust_outputs.py` 属于实验运行保障层，不是专利核心算法。
4. `no_risk_scoring` 消融准确率不弱，因此风险评分不应被表述为直接提高准确率的模块，而应表述为成本、路径和验证预算控制模块。
5. Windows PowerShell 的 `Get-Content` 可能把 UTF-8 中文显示成乱码；判断源码编码时应使用 Python UTF-8 读取或 `rg`，不要根据终端乱码误判。

### 4.2 本轮轻量优化

本轮只做保守优化，不改变主流程：

1. 补强 `RouterAgent._rule_based_route` 的英文通用操作词，覆盖 average、sum、ratio、difference、highest、lowest、before、after 等常见结构操作。
2. 新增 `tests/test_patent_direction_guardrails.py`，用于防止后续改动偏离专利方向：
   - 顶层路由必须保持 SIMPLE / COMPLEX。
   - 关键 prompt 不能出现常见乱码或替换字符。
   - 表格压缩、问题路由、风险评分、确定性算子、强验证默认机制不能被无意关闭。
   - SIMPLE 查值输出模板应保持可读。

### 4.3 暂不做的清理

1. 暂不拆分 `my_agents.py`，避免在专利和实验临近阶段引入大范围回归。
2. 暂不删除已有确定性算子，因为它们支撑 formal200 正结果。
3. 暂不重写路由、压缩、风险评分的大结构。
4. 暂不新增新的 agent 层或第三条顶层路径。

## 5. 后续允许优化的范围

后续优化必须能归入以下已有模块之一：

1. 答案契约：数值精度、分数格式、yes/no、true/false、缩写、列表/元组格式。
2. 表格压缩：最大保留行数、邻域窗口、全局行列回退条件、列画像预算。
3. 路由阈值：SIMPLE/COMPLEX 的边界、难度阈值、明显复杂题强制进入 COMPLEX。
4. 风险评分：高风险题触发强验证的阈值、后验风险触发条件。
5. 确定性表格算子门控：只扩大共性操作类型，不扩大样本级规则；必须有置信门控。
6. 高风险二次校验：只对 WTQ/CRT/TabFact 的共性错误类型触发，不做全量无差别加验。

## 6. 禁止事项

1. 禁止基于样本 ID 写规则。
2. 禁止基于 fixed formal200 题干写规则。
3. 禁止新增大 agent、大路径、大层级。
4. 禁止把 robust runner 写成专利核心创新。
5. 禁止把 Seed-E 负结果隐藏为“已稳定泛化”。
6. 禁止为了单个数据集继续无限追加 shortcut。
7. 禁止只在 formal200 上验证修复后就宣称泛化成功。

## 7. 新实验计划

### 7.1 P0：本地与服务器同步检查

每次开始实验前执行：

```bash
git -C /home/ubuntu/lzz/MyAgent pull --ff-only
git -C /home/ubuntu/lzz/MACT pull --ff-only
```

MyAgent 本地检查：

```bash
cd /home/ubuntu/lzz/MyAgent
python -m py_compile code/my_agents.py code/tqa.py code/answer_contracts.py code/dataset_profiles.py code/robust_outputs.py
python -m unittest discover -s tests -p 'test_patent_direction_guardrails.py'
python -m unittest discover -s tests -p 'test_robust_outputs.py'
python -m unittest discover -s tests -p 'test_tqa_failure_exit.py'
```

### 7.2 P1：Seed-E 错误簇 focused validation

优先验证已准备的答案契约 focused 输入：

```bash
cd /home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_patent_seed_e_gate50_20260823
bash run_answer_contract_focused_validation.sh
```

通过条件：

1. 所有 focused 输入都有输出。
2. 不能出现新的格式错误或执行失败。
3. 修复必须能解释为答案契约、压缩、门控或验证机制变化。

### 7.3 P2：Seed-F Gate-50 盲测

当 P1 focused validation 通过后，生成一个新的 Seed-F Gate-50：

1. 每个数据集 50 条。
2. 排除 formal200、ablation50、Seed-C/D/E、focused targeted slices。
3. MyAgent 和 MACT 必须同样本、同模型、同评价脚本。
4. 先跑 MyAgent，再只在必要时跑 MACT paired。

通过条件：

1. MyAgent overall 不低于 MACT。
2. MyAgent token ratio 目标不超过 MACT 的 0.75。
3. 若单数据集略低，但总体更高，需要在报告中解释。

### 7.4 P3：Seed-F Gate-100

只有 Seed-F Gate-50 通过后才扩大到 Gate-100。若 Gate-50 未通过，回到错误簇诊断，不直接跑更大实验。

### 7.5 P4：是否重跑 formal200

只有在以下情况才重跑 MyAgent formal200：

1. 修改影响了答案契约、压缩策略、路由阈值、确定性算子或强验证触发。
2. Seed-F/G 表明机制修复稳定。
3. 需要生成最终论文表格。

MACT formal200 已有结果，不因 MyAgent 小修反复重跑；只有换模型、换样本或换评价脚本时才重跑 MACT。

## 8. 专利说明书写法边界

可以写：

1. 本方法基于问题类型识别、双评分机制和成本感知两路径路由。
2. 本方法在求解前对表格进行问题感知压缩，并保留关键证据。
3. 本方法通过答案契约约束输出格式，降低可评分错误。
4. 本方法对高风险样本触发选择性协作验证。
5. 在 Qwen3-32B formal200 设置下，本方法相比 MACT 取得更高准确率和更低 token 消耗。

暂时不要写：

1. 本方法在所有随机种子下稳定超过 MACT。
2. 本方法在所有模型上均超过 MACT。
3. 风险评分直接带来准确率提升。
4. robust fallback 是核心创新。
5. 当前确定性算子覆盖所有表格推理问题。

## 9. 交付与同步规则

每次服务器实验或代码修改后，必须同步以下内容：

1. MyAgent commit hash。
2. MACT commit hash。
3. 实验输入路径。
4. 输出路径。
5. 评价脚本路径。
6. 每个数据集行数、准确率、平均 token、平均耗时。
7. 是否使用 focused、Seed-F、formal200 或 ablation。
8. 是否改动核心逻辑。
9. 若改动核心逻辑，说明归属模块：答案契约、压缩、路由、风险、验证或确定性算子门控。

推荐每次实验结束写入：

```text
docs/server/server_codex_reports/YYYY-MM-DD-brief-experiment-summary.md
```

## 10. 当前结论

当前代码可以继续作为专利和论文实验版本，但后续必须冻结大方向。下一步不是继续增加功能层，而是用 Seed-E 错误簇做机制诊断，通过 focused validation 和新的 Seed-F/G 盲测证明修复是否具有泛化性。

