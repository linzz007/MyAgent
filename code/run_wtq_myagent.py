"""基于 myAgent 流水线的 WikiTableQuestions/TSV 简化入口。

用法示例：
python run_wtq_myagent.py ^
  --plan_model_name your-model-name ^
  --model_path your-model-path ^
  --dataset_path "../dataset/WikiTableQuestions-master/WikiTableQuestions-master/data/myAgentdataset.tsv"

TSV 文件最小字段要求：
- 第一行是表头，至少包含：id, question, table_path, answer
- table_path 为相对于 TSV 文件所在目录的 CSV 路径
"""

import argparse
import json
import os
import traceback
from typing import Callable

import pandas as pd
from openai import OpenAI
from transformers import AutoTokenizer
from vllm import LLM

from agents import load_gpt_azure, get_completion
from llm import OpenSourceLLM
from my_agents import (
    RouterAgent,
    PlannerAgent,
    Calculator,
    CriticAgent,
    FinalAnswerAgent,
    TableQAPipeline,
    TQASessionState,
    load_csv_row_col_names,
)


def build_llm_fn(args) -> Callable[[str], str]:
    """根据命令行参数构造一个统一的 llm_fn(prompt:str)->str 接口。"""
    model_name = (args.plan_model_name or "").lower()

    # 1) 使用 DeepSeek / DashScope 兼容 OpenAI 的 API（例如 deepseek-v3.1）
    if "deepseek" in model_name:
        client = OpenAI(
            api_key="sk-079069c34e4741eb89f37d953def438b",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        ds_model = args.plan_model_name or "deepseek-v3.1"

        def llm_fn(prompt: str) -> str:
            try:
                completion = client.chat.completions.create(
                    model=ds_model,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={"enable_thinking": True},
                    stream=False,
                )
                msg = completion.choices[0].message
                return (msg.content or "") if msg else ""
            except Exception:  # noqa: BLE001
                return ""

        return llm_fn

    # 2) 使用 Azure GPT 系列模型
    if "gpt" in model_name:
        client = load_gpt_azure()

        def llm_fn(prompt: str) -> str:
            try:
                outputs = get_completion(prompt, client=client, n=1)
                return outputs[0] if outputs else ""
            except Exception:  # noqa: BLE001
                return ""

        return llm_fn

    # 3) 使用本地 / 开源模型（vLLM）
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LLM(model=args.model_path)
    os_llm = OpenSourceLLM(
        model_name=args.plan_model_name,
        model=model,
        vllm=model,
        tokenizer=tokenizer,
    )

    def llm_fn(prompt: str) -> str:
        try:
            outputs = os_llm(prompt, num_return_sequences=1, return_prob=False)
            return outputs[0] if outputs else ""
        except Exception:  # noqa: BLE001
            return ""

    return llm_fn


def load_tsv_dataset(tsv_path: str):
    """读取 WikiTableQuestions 风格的 TSV 数据。"""
    rows = []
    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return rows
    header = lines[0].split("\t")
    for line in lines[1:]:
        parts = line.split("\t")
        row = {k: v for k, v in zip(header, parts)}
        rows.append(row)
    return rows


def main(args):
    llm_fn = build_llm_fn(args)

    # 组装 myAgent 各个子模块
    router = RouterAgent(llm_fn=llm_fn)
    planner = PlannerAgent(llm_fn=llm_fn)
    calculator = Calculator()
    critic = CriticAgent(llm_fn=llm_fn)
    final_answer_agent = FinalAnswerAgent(llm_fn=llm_fn)
    pipeline = TableQAPipeline(
        router=router,
        planner=planner,
        calculator=calculator,
        critic=critic,
        final_answer_agent=final_answer_agent,
        max_replan=2,
    )

    # 读取 TSV 数据集
    dataset = load_tsv_dataset(args.dataset_path)

    plan_model_name = (args.plan_model_name or "").split("/")[-1].strip()
    output_path = args.output_path or f"wtq_{plan_model_name}_myAgent.jsonl"

    base_dir = os.path.dirname(args.dataset_path)

    trial = 0
    for idx, row in enumerate(dataset):
        try:
            question = row.get("question", "")
            table_rel_path = row.get("table_path") or row.get("table") or ""
            csv_path = os.path.join(base_dir, table_rel_path)

            # 读取完整 df（复杂路径用）
            df = pd.read_csv(csv_path)

            # 从 CSV 推出“行名/列名”字符串，供 Router 的 LLM 判断难度 & 单元格数量
            row_names_str, col_names_str = load_csv_row_col_names(csv_path)
            # 这里只保留最核心的结构信息：列名 + 行/列名称字符串
            table_schema = {
                "columns": [str(c) for c in df.columns.tolist()],
                "preview_text": "",
                "row_names_str": row_names_str,
                "col_names_str": col_names_str,
            }

            state = TQASessionState(question=question, df=df, table_schema=table_schema)

            if args.router_only:
                # 仅测试 Router：只跑路由，不进入 Planner/Calculator/Critic/FinalAnswer
                state = router.route(state)

                ctx = state.routing_context or {}
                print(
                    f"[RouterTest] id={row.get('id', idx)} "
                    f"route_type={state.route_type} "
                    f"sem_score={ctx.get('sem_score')} "
                    f"cell_score={ctx.get('cell_score')} "
                    f"total_score={ctx.get('total_score')} "
                    f"difficulty_level={state.difficulty_level}"
                )
            else:
                # 正常完整流水线
                state = pipeline.run(state)

            item = dict(row)
            item["route_type"] = state.route_type
            item["difficulty_score"] = state.difficulty_score
            item["difficulty_level"] = state.difficulty_level
            item["routing_context"] = state.routing_context
            item["planner_plan_steps"] = state.plan_steps
            item["planner_code"] = state.code_str
            item["exec_success"] = state.exec_success
            item["exec_error"] = state.exec_error
            item["final_value"] = state.final_value
            item["critic_verdict"] = state.critic_verdict
            item["critic_feedback"] = state.critic_feedback
            item["final_answer"] = state.final_answer

            with open(output_path, "a", encoding="utf-8") as fout:
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            trial += 1
            print(f"Finished sample {trial}/{len(dataset)}")
        except Exception:  # noqa: BLE001
            print(traceback.format_exc())
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan_model_name",
        default="",
        help="name of the planning model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="model path to the planning model.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../dataset/WikiTableQuestions-master/WikiTableQuestions-master/data/myAgentdataset.tsv",
        help="path to the WTQ-style tsv dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="output jsonl file path (optional).",
    )
    parser.add_argument(
        "--router_only",
        action="store_true",
        help="仅测试 Router：只跑路由并输出路由相关信息，不执行 Planner/Calculator/Critic/FinalAnswer。",
    )
    args = parser.parse_args()
    main(args)
