# DeepSeek API Backend Design

## Goal

让 myAgent 的实验入口通过命令行参数选择模型后端，先稳定接通 DeepSeek 官方 OpenAI 兼容 API，并保留 Azure 与本地 vLLM 路径。

## Recommended Approach

新增单一的 `code/model_backends.py`，把模型初始化和 `prompt -> text` 调用统一封装。`code/tqa.py` 与 `code/run_wtq_myagent.py` 只负责解析参数并调用该模块，避免每个入口重复实现 provider 逻辑。

备选方案是直接在两个入口中分别修正 DeepSeek 调用，改动更少但会产生重复代码；另一种方案是立即引入 YAML provider 配置，扩展性更强但对当前需求过重。因此采用共享 Python 适配器。

## Interface

- `--model_provider`: `auto`、`deepseek`、`azure`、`local`；默认 `auto`，按模型名兼容旧命令。
- `--plan_model_name`: DeepSeek 默认推荐值由调用命令明确传入 `deepseek-v4-flash`。
- `--api_base`: 默认 `https://api.deepseek.com`，允许切换其他 OpenAI 兼容地址。
- `--api_key_env`: 默认 `DEEPSEEK_API_KEY`，只读取环境变量，不接受明文命令行 Key。
- `--thinking`: `disabled` 或 `enabled`，默认 `disabled`，便于先做成本稳定的基线实验。
- `--temperature`、`--max_tokens`: 统一控制生成参数。

## Runtime Behavior

DeepSeek 分支延迟导入 `openai`；本地分支延迟导入 `transformers`、`vllm` 与项目本地模型封装。这样仅使用 API 时不要求本机安装 CUDA/vLLM。API 异常直接抛出并带样本错误日志，禁止静默返回空字符串污染实验结果。

## Validation

使用假的 OpenAI 客户端验证 URL、Key、模型名、thinking 参数与响应解析，不访问真实 API、不消耗额度。另运行两个入口的 `--help`，确认当前无 vLLM 的 Python 3.13 环境仍可启动。
