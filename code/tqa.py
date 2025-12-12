""" MACT（NAACL 2025）相关的工具类与函数
版权 (c) 2025 Robert Bosch GmbH

本程序是自由软件：你可以按照自由软件基金会发布的 GNU Affero 通用公共许可证第 3 版或（你可以选择的）更高版本的条款重新发布和/或修改。
本程序的发布目的是希望它有用，但不提供任何保证；甚至不包含适销性或特定用途适用性的默示保证。详情参考 GNU Affero 通用公共许可证。
你应该已经收到一份 GNU Affero 通用公共许可证副本；如果没有，请访问 <https://www.gnu.org/licenses/>。"""

import argparse
import json
import os
import traceback

import pandas as pd
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
    build_df_from_table,
    _build_table_schema,
    load_csv_row_col_names,
)
from utils import get_databench_table


def main(args):
    # 输入：args(argparse.Namespace) 命令行参数集
    # 输出：无
    # 作用：根据参数加载模型与数据集，构建自定义表格问答流水线，并记录输出结果。

    # ---------- 构建统一 LLM 函数（llm_fn: prompt -> str） ----------
    if "gpt" in args.plan_model_name.lower():
        client = load_gpt_azure()

        def llm_fn(prompt: str) -> str:
            try:
                return get_completion(prompt, client=client, n=1)[0]
            except Exception:
                return ""

    else:
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
            except Exception:
                return ""

    # ---------- 组装各个子 Agent ----------
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

    # ---------- 读取数据集（jsonl 或 WikiTableQuestions tsv） ----------
    if args.dataset_path.endswith(".tsv"):
        table_dataset = []
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f if line.strip()]
        # 假设首行为表头，字段至少包含：id, question, table_path, answer
        header = lines[0].split("\t")
        for line in lines[1:]:
            parts = line.split("\t")
            row = {k: v for k, v in zip(header, parts)}
            table_dataset.append(row)
    else:
        with open(args.dataset_path, "r", encoding="utf-8") as f:
            table_dataset = [json.loads(line) for line in f]

    # ---------- 输出文件路径 ----------
    plan_model_name = args.plan_model_name.split("/")[-1].strip()
    output_path = f"{args.task}_{plan_model_name}_myAgent.jsonl"

    # ---------- 遍历样本并运行流水线 ----------
    trial = 0
    for idx, row in enumerate(table_dataset):
        try:
            question = row["question"] if "question" in row else row.get("statement", "")

            if args.dataset_path.endswith(".tsv"):
                # TSV 模式（例如 WikiTableQuestions）：从相对路径读取 CSV 表格
                table_rel_path = row.get("table_path") or row.get("table") or ""
                base_dir = os.path.dirname(args.dataset_path)
                csv_path = os.path.join(base_dir, table_rel_path)

                # 读取完整 df（供复杂路径使用）
                df = pd.read_csv(csv_path)
                table_schema = _build_table_schema(df)

                # 额外抽取“行名/列名”两个字符串，供 Router 的语义评估使用
                row_names_str, col_names_str = load_csv_row_col_names(csv_path)
                table_schema["row_names_str"] = row_names_str
                table_schema["col_names_str"] = col_names_str

                state = TQASessionState(question=question, df=df, table_schema=table_schema)

            elif args.task == "databench":
                # DataBench：直接从 parquet 构造完整 DataFrame
                _, _, df_path = get_databench_table(args.table_dir, row["dataset"])
                df = pd.read_parquet(df_path, engine="pyarrow")
                table_schema = _build_table_schema(df)
                state = TQASessionState(question=question, df=df, table_schema=table_schema)

            else:
                # 原 MACT jsonl 模式：table_text 为 list-of-lists
                table = row["table_text"]
                df = build_df_from_table(table)
                table_schema = _build_table_schema(df)
                state = TQASessionState(question=question, df=df, table_schema=table_schema)

            state = pipeline.run(state)

            item = dict(row)
            item["route_type"] = state.route_type
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
            print(f"Finished sample {trial}/{len(table_dataset)}")
        except Exception:
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
        "--code_model_name",
        default="",
        help="(unused in myAgent) name of the coding model.",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        help="cache dir to load a model from.",
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
        default="../datasets/wtq.jsonl",
        help="dataset path.",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="../datasets/databench/data",
        help="databench table directory",
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=6,
        help="(unused in myAgent) maximum number for valid iterations.",
    )
    parser.add_argument(
        "--max_actual_step",
        type=int,
        default=6,
        help="(unused in myAgent) maximum number for all iterations.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="wtq",
        choices=["wtq", "crt", "tat", "scitab", "databench"],
    )
    parser.add_argument(
        "--as_reward",
        type=str,
        default="consistency",
        choices=["consistency", "llm", "logp", "rollout", "combined"],
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--long_table_op",
        type=str,
        default="ignore",
        choices=["code-agent", "ignore", "short-table"],
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--plan_sample",
        type=int,
        default=5,
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--code_sample",
        type=int,
        default=5,
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--use_pre_answer",
        type=bool,
        default=True,
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--answer_aggregate",
        type=float,
        default=1.0,
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--direct_reasoning",
        action="store_true",
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--without_tool",
        action="store_true",
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--code_endpoint",
        default="11039",
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--debugging",
        action="store_true",
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    parser.add_argument(
        "--code_as_observation",
        action="store_true",
        help="(unused in myAgent) kept for CLI compatibility.",
    )
    args = parser.parse_args()
    main(args)

