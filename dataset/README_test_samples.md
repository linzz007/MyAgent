# 三个数据集格式测试文件说明

本目录包含三个数据集的示例文件，用于测试代码是否能正确处理不同格式的数据集。

## 文件说明

### 1. WikiTableQuestions (TSV格式)
- **文件**: `test_samples_wtq.tsv`
- **格式**: TSV (Tab-Separated Values)
- **字段**:
  - `id`: 样本ID
  - `question`: 问题文本
  - `table_path`: CSV表格的相对路径（相对于TSV文件所在目录）
  - `answer`: 答案

**示例**:
```tsv
id	question	table_path	answer
test-wtq-1	what was the last year where this team was a part of the usl a-league?	test_csv/590.csv	2004
```

**特点**:
- 表格存储在独立的CSV文件中
- 需要从`table_path`读取CSV文件
- 支持字段别名：`utterance` → `question`, `context` → `table_path`, `targetValue` → `answer`

### 2. HiTab (JSONL格式)
- **文件**: `test_samples_hitab.jsonl`
- **格式**: JSONL (每行一个JSON对象)
- **字段**:
  - `id`: 样本ID
  - `table_id`: 表格ID
  - `question`: 问题文本
  - `table_text`: 表格数据（list-of-lists格式，即二维数组）
  - `answer`: 答案（数组格式）
  - `aggregation`: 聚合类型

**示例**:
```json
{
  "id": "test-hitab-1",
  "question": "if the pre-production development activities were to be included, how many dollars would the fy 2017 r&d budget authority have been?",
  "table_text": [
    ["", "2017 actual", "2017 proposed"],
    ["total", "125289.0", "154983.0"],
    ["pre-production development activities", "30000.0", "30000.0"]
  ],
  "answer": [154983.0]
}
```

**特点**:
- 表格直接嵌入在JSON中（`table_text`字段）
- 表格格式为list-of-lists（二维数组）
- 需要调用`build_df_from_table()`函数将list-of-lists转换为DataFrame

### 3. TAT-QA (JSON格式)
- **文件**: `test_samples_tatqa.json`
- **格式**: JSON（数组格式，每个元素包含一个表格和多个问题）
- **结构**:
  - 顶层是数组，每个元素是一个表格样本
  - 每个样本包含：
    - `table`: 表格对象，包含`uid`和`table`（list-of-lists）
    - `paragraphs`: 相关段落（可选）
    - `questions`: 问题数组，每个问题包含：
      - `question`: 问题文本
      - `answer`: 答案（数组）
      - `answer_from`: 答案来源（"table-text"或"text"）
      - 其他元数据

**示例**:
```json
[
  {
    "table": {
      "uid": "test-table-1",
      "table": [
        ["", "", "Years Ended September 30,", ""],
        ["", "2019", "2018", "2017"],
        ["Fixed Price", "$  1,452.4", "$  1,146.2", "$  1,036.9"]
      ]
    },
    "questions": [
      {
        "question": "What is the amount of total sales in 2019?",
        "answer": ["$1,496.5"],
        "answer_from": "table-text"
      }
    ]
  }
]
```

**特点**:
- 一个表格对应多个问题
- 表格格式为list-of-lists（二维数组）
- 需要调用`build_df_from_table()`函数将list-of-lists转换为DataFrame
- 支持从表格或段落中提取答案

## 使用方法

### 运行测试脚本

```bash
cd code
python test_three_datasets.py --plan_model_name deepseek-v3.1
```

### 测试选项

- `--plan_model_name`: LLM模型名称（默认: deepseek-v3.1）
- `--wtq_path`: WikiTableQuestions测试文件路径（默认: ../dataset/test_samples_wtq.tsv）
- `--hitab_path`: HiTab测试文件路径（默认: ../dataset/test_samples_hitab.jsonl）
- `--tatqa_path`: TAT-QA测试文件路径（默认: ../dataset/test_samples_tatqa.json）
- `--skip_wtq`: 跳过WikiTableQuestions测试
- `--skip_hitab`: 跳过HiTab测试
- `--skip_tatqa`: 跳过TAT-QA测试

### 示例命令

```bash
# 测试所有数据集
python test_three_datasets.py --plan_model_name deepseek-v3.1

# 只测试HiTab和TAT-QA
python test_three_datasets.py --plan_model_name deepseek-v3.1 --skip_wtq

# 使用自定义路径
python test_three_datasets.py \
  --plan_model_name deepseek-v3.1 \
  --wtq_path /path/to/wtq.tsv \
  --hitab_path /path/to/hitab.jsonl \
  --tatqa_path /path/to/tatqa.json
```

## 测试内容

测试脚本会：
1. 加载每个数据集的示例文件
2. 初始化Router、Planner、Calculator等组件
3. 对每个样本执行Router路由（不执行完整pipeline）
4. 输出路由结果、sem_score、cell_score、difficulty_level等信息
5. 统计成功处理的样本数量

## 预期输出

如果测试通过，你会看到：
- ✓ 成功加载样本
- ✓ 路由结果（SIMPLE/COMPLEX）
- ✓ sem_score和cell_score的值
- ✓ difficulty_level（easy/hard）
- ✓ 测试总结：所有数据集都显示"✓ 通过"

## 注意事项

1. **环境变量**: 确保设置了`DASHSCOPE_API_KEY`（如果使用DeepSeek）或其他LLM API密钥
2. **CSV文件**: WikiTableQuestions测试需要CSV文件，已复制到`test_csv/`目录
3. **数据格式**: 确保示例文件的数据格式与实际数据集一致
4. **路径**: 测试脚本中的路径是相对于`code/`目录的

## 数据集格式对比

| 数据集 | 格式 | 表格存储方式 | 问题数量 | 特点 |
|--------|------|--------------|----------|------|
| WikiTableQuestions | TSV | CSV文件（外部） | 1问题/样本 | 简单，表格独立存储 |
| HiTab | JSONL | list-of-lists（嵌入） | 1问题/样本 | 表格嵌入，支持复杂结构 |
| TAT-QA | JSON | list-of-lists（嵌入） | 多问题/表格 | 一个表格多个问题，支持段落 |

## 下一步

如果所有测试都通过，你可以：
1. 使用`run_wtq_myagent.py`运行完整的WikiTableQuestions实验
2. 扩展测试脚本以支持HiTab和TAT-QA的完整pipeline
3. 根据实际数据集调整数据加载逻辑
