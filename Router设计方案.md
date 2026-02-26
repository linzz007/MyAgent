# Router 设计方案（研究生论文/专利用）

本文档给出一个清晰、可落地、实现难度不高的 Router 方案，可作为研究生毕业论文/专利的核心点。它基于“语义难度 + 结构/通信难度”双轴建模，通过 LLM 抽特征、规则计算分数、表压缩和路径路由，在保证正确率的前提下减少通信成本。

---

## 1. Router 总体目标与输出

- **输入**：问题文本 + 原始表格（支持单轴、双轴、多级表头）。
- **输出**：
  1) `sem_score`：语义难度分（0–1）。  
  2) `cell_score`：结构/通信复杂度分（0–1）。  
  3) `compressed_table`：压缩后的表（至少做列裁剪，后续可扩展行裁剪）。  
  4) `route`：`SIMPLE` / `COMPLEX`。

核心思想：语义轴（sem）衡量推理复杂度，结构/通信轴（cell）衡量需要触达的单元格规模与表结构复杂度，二者联合决定路由与压缩强度。

---

## 2. 模块拆分（沿用现有 2/3/4 模块框架）

1) **表结构分析器（Table Structure Analyzer）**  
   - 输入：原始 DataFrame。  
   - 输出：`table_type`（single_axis / double_axis / multi_header）、`n_rows`、`n_cols`，可选“是否有合计行/列”等。  
   - 实现：先用规则即可。初版可只区分 single_axis / other，后续再细化。

2) **语义特征提取器（Semantic Feature Extractor，现有 FeatureExtractor 升级）**  
   - 让 LLM 输出“语义特征 + 选列”，而不是直接给语义分数。示例 JSON：  
     ```json
     {
       "coarse_intent": "lookup | aggregation | comparison | ranking | trend | multi_step",
       "semantic_flags": {
         "has_aggregation": true,
         "has_comparison": false,
         "has_temporal_reasoning": true,
         "has_multi_step": false,
         "num_constraints": 2,
         "num_entities_mentioned": 1
       },
       "selected_columns": ["Year", "League"]
     }
     ```
   - 实现：修改 prompt 要求输出上述字段；解析 JSON 已有基础可复用。

3) **结构/通信复杂度计算器（Cell Complexity Calculator）**  
   - 输入：原始表形状 `(orig_rows, orig_cols)`，压缩后形状 `(reduced_rows, reduced_cols)`，`table_type`。  
   - 计算：`cell_ratio = (reduced_rows * reduced_cols) / (orig_rows * orig_cols)`。  
   - 定义 `cell_score`：  
     ```
     cell_score = min(1, cell_ratio * w(table_type))
     w(single_axis)=1.0, w(double_axis)=1.2, w(multi_header)=1.5
     ```
     直觉：双轴/多级表头更难，即使 cell 数相同。

4) **语义难度计算器（Semantic Difficulty Calculator）**  
   - 输入：`semantic_flags` + `coarse_intent`。  
   - 输出：`sem_score ∈ [0,1]`。  
   - 简单可解释公式（示例，可调权重）：  
     ```
     sem_score_raw = (
       0.3 * has_aggregation +
       0.2 * has_comparison +
       0.2 * has_temporal_reasoning +
       0.2 * has_multi_step +
       0.1 * tanh(num_constraints)
     ) / (0.3+0.2+0.2+0.2+0.1)
     if coarse_intent in {"multi_step","aggregation"}:
         sem_score_raw += 0.1
     sem_score = clip(sem_score_raw, 0, 1)
     ```
   - 实现：纯 Python 计算，无需训练。

5) **路由策略（Routing Policy）**  
   - 先计算 `sem_score` 与 `cell_score`，再融合：  
     `total_score = α * sem_score + (1 - α) * cell_score`（如 α=0.6）。  
   - 结合意图与表类型的规则示例：  
     ```
     if coarse_intent == "lookup" and table_type == "single_axis" and total_score < 0.4:
         route = SIMPLE
     else:
         route = COMPLEX
     ```
     可扩展：table_type != single_axis 且 sem_score > 0.3 → 优先 COMPLEX；  
     cell_score > 0.7 → 即使 sem 低也走 COMPLEX。

---

## 3. 论文/专利可讲的贡献点

- **双轴难度建模**：语义轴 (sem) + 结构/通信轴 (cell)，联合驱动路由。  
- **结构感知**：支持单轴、双轴、多级表头，结构类型进入 cell_score 的加权。  
- **通信成本优化**：在保证正确率前提下，通过列（未来可行）压缩，降低 LLM token 消耗。  
- **可解释 & 可消融**：分数全部由显式特征 + 规则得到，便于实验对比与专利审查。  
- **低实现门槛**：无需新大模型或复杂训练，LLM 只做特征抽取，其余是规则与简单公式。

---

## 4. 落地实现的最小改动列表（基于现有 my_agents.py）

1) **FeatureExtractor Prompt 调整**  
   - 要求输出：`coarse_intent`、`semantic_flags`（布尔/整数）、`selected_columns`。  
   - 不再要求 LLM 输出 `semantic_score` 数值。

2) **表结构分析器**  
   - 在 Router 前置一个小函数/方法：根据表头/形状判定 `table_type`（先实现 single_axis/other 即可）。

3) **sem_score 计算函数**  
   - 新增纯 Python 函数：根据 `semantic_flags` + `coarse_intent` 按上述公式算 `sem_score`。

4) **cell_score 计算函数**  
   - 在 `_reduce_df` 后拿到 `(orig_rows, orig_cols)` 和 `(reduced_rows, reduced_cols)`，按加权公式算 `cell_score`。

5) **路由决策**  
   - 用 `sem_score`、`cell_score`、`coarse_intent`、`table_type`，按上述规则判定 SIMPLE/COMPLEX。

---

## 5. 后续可选升级（不影响第一版）

- 精细的表结构识别（双轴 vs 多级表头 vs 汇总行/列）。  
- 引入行级压缩与更细的结构复杂度指标（访问路径长度、维度交叉度）。  
- 将规则权重（α、各特征系数）用一个轻量模型（Logistic Regression / XGBoost）做学习微调。  
- 在不同数据集上跑消融实验，画“正确率 vs token 消耗”曲线，丰富论文结果。

---

## 6. 结论

这个方案在实现上保持轻量：LLM 抽语义特征，规则计算 sem/cell 分数，表压缩减少通信，简单规则做路由。它足够清晰、可解释，易于写作与专利审查，同时又能落到你当前代码中快速迭代。***

