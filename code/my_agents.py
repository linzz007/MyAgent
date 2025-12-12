"""Custom table QA pipeline for myAgent.

This module defines a simplified agent architecture:
- TQASessionState: per-sample state container
- RouterAgent: decides SIMPLE vs COMPLEX based on semantic + structural scores
- PlannerAgent: generates [PLAN] + [CODE] from question + table schema
- Calculator: safely executes generated code on a pandas DataFrame
- CriticAgent: lightweight checker; can request REPLAN via feedback
- FinalAnswerAgent: formats final natural language answer
- TableQAPipeline: orchestrator that wires everything together
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from utils import table2df


class TQASessionState:
    """Unified state container for one table QA session.

    One question + one table => one state object.
    All intermediate information (routing scores, plan, code, execution
    results, critic feedback, final answer) is stored here.
    """

    def __init__(self, question: str, df: pd.DataFrame, table_schema: Dict[str, Any]):
        self.question: str = question
        self.df: pd.DataFrame = df
        self.table_schema: Dict[str, Any] = table_schema

        # Routing / 难度与路径信息
        self.route_type: Optional[str] = None  # "SIMPLE" or "COMPLEX"
        self.semantic_features: Dict[str, Any] = {}
        self.structural_features: Dict[str, Any] = {}
        self.difficulty_score: Optional[float] = None  # 0~1 综合难度分
        self.difficulty_level: Optional[str] = None  # "easy" / "hard"
        self.routing_context: Dict[str, Any] = {}

        # Planner
        self.planner_raw_output: str = ""
        self.plan_steps: List[str] = []
        self.code_str: str = ""

        # Calculator
        self.exec_success: bool = False
        self.exec_error: Optional[str] = None
        self.exec_locals: Dict[str, Any] = {}
        self.final_value: Any = None  # scalar or small object

        # Critic
        self.critic_raw_output: str = ""
        self.critic_verdict: Optional[str] = None  # "PASS" or "REPLAN"
        self.critic_feedback: str = ""

        # Final answer
        self.final_answer: Optional[str] = None


# ------------------------ helpers ------------------------


def _build_table_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a lightweight schema from a DataFrame.

    Currently includes:
    - columns: list of column names
    - preview_text: a short textual preview of the first few rows
    """
    columns = list(df.columns)
    preview_df = df.head(5)
    preview_text = preview_df.to_string(index=False)
    return {
        "columns": columns,
        "preview_text": preview_text,
    }


def build_df_from_table(table: Any) -> pd.DataFrame:
    """Convert MACT-style table_text (list-of-lists) to a DataFrame.

    Uses utils.table2df to get executable code that reconstructs df.
    """
    if isinstance(table, pd.DataFrame):
        return table
    if not isinstance(table, list):
        raise ValueError("Unsupported table format for DataFrame construction.")
    df_code = table2df(table)
    local_env: Dict[str, Any] = {}
    exec(df_code, {}, local_env)  # defines `df` in local_env
    df = local_env.get("df", None)
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("Failed to construct DataFrame from table.")
    return df


def load_csv_row_col_names(csv_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Read a CSV file and return row/column names as two strings (or None).

    - Column names: if there is at least one non-empty column label, join
      them with '##' into col_names_str.
    - Row names: only if df.index is not a RangeIndex and has at least one
      non-empty label; join them with '##' into row_names_str.

    Returns:
        row_names_str, col_names_str
    """
    df = pd.read_csv(csv_path)

    col_labels = [str(c).strip() for c in df.columns.tolist()]
    col_names_str: Optional[str] = "##".join(col_labels) if any(col_labels) else None

    if not isinstance(df.index, pd.RangeIndex):
        idx_labels = [str(i).strip() for i in df.index.tolist()]
        row_names_str: Optional[str] = "##".join(idx_labels) if any(idx_labels) else None
    else:
        row_names_str = None

    return row_names_str, col_names_str


def _between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    sub = text.split(start, 1)[1]
    if end in sub:
        sub = sub.split(end, 1)[0]
    return sub.strip()


def _find_line_startswith(text: str, prefix: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith(prefix):
            return line.strip()
    return ""


# ------------------------ Router ------------------------


ROUTER_PROMPT_TEMPLATE = """You are a classifier for table questions.
Given a Chinese question and a table schema, decide whether solving it
requires complex multi-step numerical reasoning.

Output exactly one token: "SIMPLE" or "COMPLEX".

[Question]
{question}

[Table Columns]
{col_names}

[Hints]
- If the question only asks for a single value lookup, classification or a direct comparison between 2 cells: SIMPLE.
- If the question requires sum/average/ratio, year-on-year/环比, difference across years or aggregations over multiple rows/columns: COMPLEX.

[Answer]
"""


class RouterAgent:
    """Router that combines semantic and structural scores.

    - sem_score: semantic complexity, based on question only (0=easy,1=hard)
    - cell_score: structural complexity, based on question + row/col names,
                  approximated by number of cells touched / total cells
    - total_score = w_sem * sem_score + w_cell * cell_score
      => SIMPLE / COMPLEX decision
    """

    def __init__(self, llm_fn: Callable[[str], str], router_prompt_template: str = ROUTER_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = router_prompt_template

    def _rule_based_route(self, question: str) -> Optional[str]:
        """Simple keyword-based fallback when scores are ambiguous."""
        complex_keywords = [
            "增长", "增幅", "占比", "比例", "同比", "环比", "平均",
            "总和", "合计", "总计", "变化", "差值", "增速", "下降",
        ]
        simple_triggers = ["是多少", "有多少", "是什么", "为多少"]

        has_complex = any(k in question for k in complex_keywords)
        has_simple = any(k in question for k in simple_triggers)

        if has_complex:
            return "COMPLEX"
        if has_simple and not has_complex:
            return "SIMPLE"
        return None

    def _score_difficulty(self, sem_score: float, cell_score: float) -> Tuple[float, str]:
        """DifficultyScorer: 根据语义/结构两个分数，给出总分 + 难度等级。

        - 输入：
          sem_score  ∈ [0,1]  语义复杂度（越大越复杂）
          cell_score ∈ [0,1]  结构复杂度 / 单元格覆盖比例
        - 输出：
          total_score ∈ [0,1]
          difficulty_level ∈ {"easy","hard"}
        """
        # 加权融合，总分 0~1
        w_sem, w_cell = 0.6, 0.4
        total_score = w_sem * sem_score + w_cell * cell_score
        total_score = max(0.0, min(1.0, total_score))

        # 对应你 README 中的 0.0–0.3 / 0.3–0.7 / 0.7–1.0 规则
        if total_score < 0.5:
            level = "easy"
        else:
            level = "hard"
        return total_score, level

    # -------- LLM scoring helpers --------

    def llm_semantic_score(self, question: str) -> float:
        """Return a semantic complexity score in [0,1] based only on the question."""
        prompt = f"""
You are an expert in semantic parsing for table-based QA. Evaluate the semantic complexity of this specific table question: {question}

Break down the required operations (e.g., direct lookup=simple, filter/aggregate=medium, multi-step/comparison/inference=complex). Rate on a [0-1] scale: 0=very simple (single-step retrieval), 1=very complex (multi-hop reasoning or verification).

Explain reasoning in 2-3 sentences, highlighting key operations. Then, output only the score as a decimal (e.g., 0.60).

[Answer]
"""
        raw = (self.llm_fn(prompt) or "").strip()
        m = re.search(r"\d+(\.\d+)?", raw)
        if not m:
            return 0.5
        try:
            val = float(m.group(0))
        except ValueError:
            val = 0.5
        return max(0.0, min(1.0, val))

    def llm_cell_score(
        self,
        question: str,
        row_names: Optional[str],
        col_names: Optional[str],
        df: pd.DataFrame,
    ) -> Tuple[float, int]:
        """Estimate which rows/columns are needed and how many cells will be touched.

        Returns:
            cell_score in [0,1]
            estimated_cells (int)
        """
        row_part = row_names or "N/A"
        col_part = col_names or "N/A"
        prompt = f"""
You are an assistant for estimating how many table cells are needed
to answer a question.

Given:
- A Chinese question.
- Candidate row names (separated by '##').
- Candidate column names (separated by '##').

1. Decide which row names and column names are actually needed.
2. Output ONLY a JSON object in the format:
   {{"rows": ["row_name1", ...], "cols": ["col_name1", ...]}}

[Question]
{question}

[Row Candidates]
{row_part}

[Column Candidates]
{col_part}

[Answer JSON]
"""
        print(f"表格row_names: {row_names}")
        print(f"表格col_names: {col_names}")
        raw = (self.llm_fn(prompt) or "").strip()
        try:
            data = json.loads(raw)
            sel_rows = list(set(data.get("rows", [])))
            sel_cols = list(set(data.get("cols", [])))
        except Exception:
            sel_rows, sel_cols = [], []

        n_rows, n_cols = df.shape
        # Columns: map selected names back to actual columns; if none selected, assume all columns
        if sel_cols:
            used_cols = [c for c in df.columns if str(c) in sel_cols]
        else:
            used_cols = list(df.columns)

        # Rows: if any selected row names, approximate by their count; otherwise use full table
        used_row_count = n_rows if not sel_rows else min(n_rows, len(sel_rows))

        estimated_cells = max(1, used_row_count * max(1, len(used_cols)))
        total_cells = max(1, n_rows * max(1, n_cols))

        cell_score = estimated_cells / total_cells
        cell_score = max(0.0, min(1.0, cell_score))
        return cell_score, estimated_cells

    def route(self, state: TQASessionState) -> TQASessionState:
        """综合计算 sem_score + cell_score → difficulty_score → SIMPLE/COMPLEX。"""
        q = state.question or ""

        # 1) semantic complexity score
        sem_score = self.llm_semantic_score(q)
        state.semantic_features["sem_score"] = sem_score

        # 2) structural complexity score
        row_names = state.table_schema.get("row_names_str")
        col_names = state.table_schema.get("col_names_str")
        cell_score, estimated_cells = self.llm_cell_score(
            question=q,
            row_names=row_names,
            col_names=col_names,
            df=state.df,
        )
        state.structural_features["cell_score"] = cell_score
        state.structural_features["estimated_cells_touched"] = estimated_cells

        # 3) DifficultyScorer：得到总分和难度等级
        total_score, difficulty_level = self._score_difficulty(sem_score, cell_score)
        state.difficulty_score = total_score
        state.difficulty_level = difficulty_level
        state.routing_context = {
            "sem_score": sem_score,
            "cell_score": cell_score,
            "estimated_cells_touched": estimated_cells,
            "total_score": total_score,
            "difficulty_level": difficulty_level,
        }

        # 4) 根据难度等级进行路由：easy→SIMPLE，hard→COMPLEX，
        if difficulty_level == "easy":
            state.route_type = "SIMPLE"
            return state
        if difficulty_level == "hard":
            state.route_type = "COMPLEX"
            return state



# ------------------------ Planner ------------------------


PLANNER_PROMPT_TEMPLATE = """You are a table reasoning planner and programmer.

You are given:
- A Chinese question about a governmental statistics table.
- A table represented as a pandas DataFrame `df`.

Your job:
1. First, write a step-by-step plan in natural language to solve the question.
2. Then, write executable Python code that uses ONLY the given DataFrame `df`
   (and standard Python/pandas operations) to compute the final answer.

Requirements:
- In the [PLAN] section, number the steps: Step1, Step2, ...
- In the [CODE] section, write valid Python code.
- Assume `df` is already defined as a pandas DataFrame with the following columns:
  {col_names}
- Use column names exactly as shown.
- At the end of your code, assign the final scalar answer to a variable named:
  final_answer_value

Question:
{question}

Table Schema (first few rows):
{table_preview}

If there was a previous critic feedback, consider it:
{critic_feedback}

Now produce your reasoning plan and code with the following format:

[PLAN]
Step1: ...
Step2: ...
...

[CODE]
# your python code here
...
final_answer_value = ...
"""


class PlannerAgent:
    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = PLANNER_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def plan(self, state: TQASessionState) -> TQASessionState:
        # Build prompt from question + table schema + critic feedback
        columns = state.table_schema.get("columns", [])
        col_names = ", ".join(str(c) for c in columns)
        table_preview = state.table_schema.get("preview_text", "")
        critic_feedback = state.critic_feedback or "无"

        prompt = self.prompt_tmpl.format(
            col_names=col_names,
            question=state.question,
            table_preview=table_preview,
            critic_feedback=critic_feedback,
        )
        raw = self.llm_fn(prompt)
        state.planner_raw_output = raw

        plan_text = _between(raw, "[PLAN]", "[CODE]")
        code_text = raw.split("[CODE]", 1)[-1] if "[CODE]" in raw else ""
        state.code_str = code_text.strip()

        plan_lines = []
        for line in plan_text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("step"):
                plan_lines.append(stripped)
        state.plan_steps = plan_lines

        return state


# ------------------------ Calculator ------------------------


class Calculator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def _safe_execute(code_str: str, df: pd.DataFrame) -> Dict[str, Any]:
        import pandas as pd  # local import for sandbox globals

        allowed_builtins = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "round": round,
            "sorted": sorted,
        }
        global_env: Dict[str, Any] = {
            "__builtins__": allowed_builtins,
            "pd": pd,
        }
        local_env: Dict[str, Any] = {"df": df}
        exec(code_str, global_env, local_env)
        return local_env

    def execute(self, state: TQASessionState) -> TQASessionState:
        """Execute planner-generated code on state.df and update execution fields."""
        code = state.code_str or ""
        if not code.strip():
            state.exec_success = False
            state.exec_error = "Empty code_str from planner."
            state.final_value = None
            state.exec_locals = {}
            return state

        try:
            local_env = self._safe_execute(code, state.df)
            state.exec_success = True
            state.exec_locals = local_env
            state.final_value = local_env.get("final_answer_value", None)
            state.exec_error = None
        except Exception as e:  # noqa: BLE001
            state.exec_success = False
            state.exec_error = str(e)
            state.final_value = None
            state.exec_locals = {}
        return state


# ------------------------ Critic ------------------------


CRITIC_PROMPT_TEMPLATE = """You are a careful auditor for table-based numerical reasoning.

You will be given:
- A question about a governmental statistics table.
- A reasoning plan (PLAN) written in steps.
- A Python code snippet (CODE) that was executed on the table DataFrame `df`.
- The execution result (RESULT), or an error if execution failed.

Your job:
1. Check whether the plan and the code correctly answer the question.
2. Check whether the result is reasonable and matches the question.
3. If you find serious issues, ask to REPLAN and describe what to fix.
4. Otherwise, PASS and briefly confirm correctness.

Output in the following format:

[VERDICT] PASS or REPLAN

[COMMENT]
(1-3 sentences explanation, in Chinese)
[/COMMENT]

[HINT_FOR_PLANNER]
(if REPLAN, give concrete suggestions how to modify PLAN or CODE; if PASS, you can say "保持当前方案")
[/HINT_FOR_PLANNER]


[QUESTION]
{question}

[PLAN]
{plan_text}

[CODE]
{code_str}

[RESULT]
success = {exec_success}
final_answer_value = {final_value}
error = {exec_error}
"""


class CriticAgent:
    def __init__(self, llm_fn: Callable[[str], str], prompt_template: str = CRITIC_PROMPT_TEMPLATE):
        self.llm_fn = llm_fn
        self.prompt_tmpl = prompt_template

    def review(self, state: TQASessionState) -> TQASessionState:
        # Check PLAN / CODE / execution result and decide PASS / REPLAN + feedback
        plan_text = "\n".join(state.plan_steps) if state.plan_steps else state.planner_raw_output
        code_str = state.code_str
        prompt = self.prompt_tmpl.format(
            question=state.question,
            plan_text=plan_text,
            code_str=code_str,
            exec_success=state.exec_success,
            final_value=state.final_value,
            exec_error=state.exec_error,
        )
        raw = self.llm_fn(prompt)
        state.critic_raw_output = raw

        verdict_line = _find_line_startswith(raw, "[VERDICT]")
        if "REPLAN" in verdict_line.upper():
            state.critic_verdict = "REPLAN"
        else:
            state.critic_verdict = "PASS"

        comment = _between(raw, "[COMMENT]", "[/COMMENT]")
        hint = _between(raw, "[HINT_FOR_PLANNER]", "[/HINT_FOR_PLANNER]")
        state.critic_feedback = hint or comment

        return state


# ------------------------ Final answer ------------------------


COMPLEX_ANSWER_PROMPT = """你是一名政务统计分析助手，请根据【问题】【推理步骤】【最终数值结果】生成简洁、正式的中文回答。

【问题】
{question}

【推理步骤】
{plan_steps}

【最终计算结果】
final_answer_value = {final_value}

要求：
1. 用 2–3 句话回答。
2. 首句直接给出数值答案，并说明单位（如果题目中能看出）。
3. 第二句简要说明是基于哪几年/哪些指标进行计算的（可参考推理步骤）。
4. 不要暴露 Python 代码。

现在给出回答：
"""

SIMPLE_ANSWER_PROMPT = """你是一个政务统计表格问答助手。

给定一个简单问题和表格（已在系统中解析），系统已经为你定位了目标单元格的数值：
value = {simple_answer_value}

【问题】
{question}

请用 1–2 句中文给出直接答案，并简单提及年份/地区/指标名称（如果题目中能看出）。
"""


class FinalAnswerAgent:
    def __init__(
        self,
        llm_fn: Callable[[str], str],
        complex_prompt: str = COMPLEX_ANSWER_PROMPT,
        simple_prompt: str = SIMPLE_ANSWER_PROMPT,
    ):
        self.llm_fn = llm_fn
        self.complex_prompt = complex_prompt
        self.simple_prompt = simple_prompt

    def respond(self, state: TQASessionState) -> TQASessionState:
        # SIMPLE: directly answer based on question + table preview
        # COMPLEX: use PLAN + final_value to format a more formal answer
        if state.route_type == "SIMPLE":
            table_preview = state.table_schema.get("preview_text", "")
            prompt = (
                "你是一个政务统计表格问答助手。\n\n"
                "【问题】\n"
                f"{state.question}\n\n"
                "【表格预览】\n"
                f"{table_preview}\n\n"
                "请根据表格内容，用 1–2 句中文给出直接答案，"
                "并尽量提及涉及的年份、地区和指标名称（如果题目中能看出）。"
            )
        else:
            plan_steps_text = "\n".join(state.plan_steps)
            prompt = self.complex_prompt.format(
                question=state.question,
                plan_steps=plan_steps_text,
                final_value=state.final_value,
            )
        answer = self.llm_fn(prompt)
        state.final_answer = answer.strip()
        return state


# ------------------------ Orchestrator ------------------------


class TableQAPipeline:
    """High-level orchestrator that wires all agents together.

    External interface: run(state) → update state in-place.
    Internally calls Router / Planner / Calculator / Critic / FinalAnswer.
    """

    def __init__(
        self,
        router: RouterAgent,
        planner: PlannerAgent,
        calculator: Calculator,
        critic: CriticAgent,
        final_answer_agent: FinalAnswerAgent,
        max_replan: int = 2,
    ) -> None:
        self.router = router
        self.planner = planner
        self.calculator = calculator
        self.critic = critic
        self.final_answer_agent = final_answer_agent
        self.max_replan = max_replan

    def build_state_from_table(self, question: str, table: Any) -> TQASessionState:
        df = build_df_from_table(table)
        schema = _build_table_schema(df)
        return TQASessionState(question=question, df=df, table_schema=schema)

    def run(self, state: TQASessionState) -> TQASessionState:
        # 1) routing
        state = self.router.route(state)

        # 2) simple path
        if state.route_type == "SIMPLE":
            state = self.final_answer_agent.respond(state)
            return state

        # 3) complex path with planner / calculator / critic (allowing limited REPLAN)
        for _ in range(self.max_replan):
            state = self.planner.plan(state)
            state = self.calculator.execute(state)
            state = self.critic.review(state)
            if state.critic_verdict == "PASS":
                break

        state = self.final_answer_agent.respond(state)
        return state
