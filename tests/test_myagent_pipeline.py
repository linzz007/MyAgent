from pathlib import Path
import sys
import unittest

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

from my_agents import (  # noqa: E402
    Calculator,
    CRITIC_PROMPT_TEMPLATE,
    CriticAgent,
    FinalAnswerAgent,
    LLMCallTracker,
    PLANNER_PROMPT_TEMPLATE,
    PlannerAgent,
    RouterAgent,
    TQASessionState,
    TableCompressor,
    TableQAPipeline,
    _build_table_schema,
    _canonicalize_wtq_scalar,
    _strip_entity_metadata,
    _strip_code_fence,
    build_df_from_table,
    validate_answer_contract_code_alignment,
    validate_generated_code_grounding,
)
from answer_contracts import infer_answer_contract  # noqa: E402


class FakePipelineLLM:
    def __init__(
        self,
        semantic_score: float,
        rows,
        cols,
        classification_output='{"label":"true"}',
        direct_answer_output='{"answer":"100,000"}',
        planner_outputs=None,
        verification_output='{"label":"true"}',
    ):
        self.semantic_score = semantic_score
        self.rows = rows
        self.cols = cols
        self.classification_output = classification_output
        self.direct_answer_output = direct_answer_output
        self.planner_outputs = list(planner_outputs or [])
        self.verification_output = verification_output
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if "semantic parsing" in prompt:
            return f"{self.semantic_score:.2f}"
        if "estimating how many table cells" in prompt:
            import json

            return json.dumps({"rows": self.rows, "cols": self.cols})
        if "closed-label table classifier" in prompt:
            return self.classification_output
        if "TabFact verification judge" in prompt:
            return self.verification_output
        if "direct table QA extractor" in prompt:
            return self.direct_answer_output
        if "table reasoning planner and programmer" in prompt:
            if self.planner_outputs:
                return self.planner_outputs.pop(0)
            return (
                "[PLAN]\n"
                "Step1: Select 2020 and 2021 profit values.\n"
                "Step2: Subtract the 2020 value from the 2021 value.\n"
                "[CODE]\n"
                "values = df.set_index('Year')['Profit']\n"
                "final_answer_value = float(values.loc['2021']) - float(values.loc['2020'])\n"
            )
        if "careful auditor" in prompt:
            return "[VERDICT] PASS\n[COMMENT]\ncorrect\n[/COMMENT]\n[HINT_FOR_PLANNER]\nkeep\n[/HINT_FOR_PLANNER]"
        if "Evidence Critic" in prompt:
            return "[VERDICT] PASS\n[COMMENT]\nevidence retained\n[/COMMENT]"
        if "Logic Critic" in prompt:
            return "[VERDICT] PASS\n[COMMENT]\nlogic correct\n[/COMMENT]"
        if "[MAIN_PLAN]" in prompt and "[MAIN_VALUE]" in prompt:
            return (
                "[CODE]\n"
                "subset = df[df['Year'].isin(['2020', '2021'])].set_index('Year')\n"
                "final_answer_value = float(subset.loc['2021', 'Profit']) - float(subset.loc['2020', 'Profit'])\n"
            )
        return "20"


class MyAgentPipelineSmokeTests(unittest.TestCase):
    @staticmethod
    def _pipeline(fake_llm, enable_multi_view_validation=False):
        tracker = LLMCallTracker(fake_llm)
        pipeline = TableQAPipeline(
            router=RouterAgent(tracker),
            planner=PlannerAgent(tracker),
            calculator=Calculator(),
            critic=CriticAgent(tracker),
            final_answer_agent=FinalAnswerAgent(tracker),
            enable_multi_view_validation=enable_multi_view_validation,
            max_replan=2,
        )
        return pipeline, tracker

    def test_simple_path_routes_compresses_and_returns_one_cell(self):
        df = pd.DataFrame(
            {
                "Year": ["2020", "2021"],
                "Revenue": [100, 120],
                "Profit": [10, 30],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.05, rows=["2021"], cols=["Revenue"])
        pipeline, tracker = self._pipeline(fake)
        state = TQASessionState(
            question="What was the Revenue in 2021?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        self.assertEqual(result.route_type, "SIMPLE")
        self.assertTrue(result.simple_lookup_success)
        self.assertEqual(result.simple_lookup_value, 120)
        self.assertEqual(result.final_value, 120)
        self.assertEqual(result.compression_info["strategy"], "strict_cell_block")
        self.assertLess(result.compression_info["compression_ratio"], 1.0)
        self.assertEqual(tracker.snapshot()["llm_call_count"], 2)

    def test_state_infers_entity_list_answer_contract(self):
        df = pd.DataFrame({"Team": ["Alpha", "Beta"], "Races": [13, 14]})

        state = TQASessionState(
            question="Which teams raced at least 13 races?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        self.assertEqual(state.answer_contract.kind, "list")
        self.assertTrue(state.answer_contract.reasoning_required)

    def test_wtq_scalar_entity_is_expanded_to_unique_exact_table_cell(self):
        df = pd.DataFrame({"Name": ["Ryan Dalziel", "Robert Doornbos"]})

        value = _canonicalize_wtq_scalar(
            "Dalziel",
            df,
            "Who was faster, Dalziel or Doornbos?",
        )

        self.assertEqual(value, "Ryan Dalziel")

    def test_entity_metadata_suffix_is_removed_from_requested_name(self):
        self.assertEqual(
            _strip_entity_metadata("monster release date : june 28, 2006"),
            "monster",
        )
        self.assertEqual(
            _strip_entity_metadata("The Greatest Story Never Told"),
            "The Greatest Story Never Told",
        )

    def test_planner_prompt_contains_answer_contract(self):
        df = pd.DataFrame({"Team": ["Alpha", "Beta"], "Races": [13, 14]})
        fake = FakePipelineLLM(semantic_score=0.8, rows=[], cols=["Team", "Races"])
        state = TQASessionState(
            question="Which teams raced at least 13 races?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        PlannerAgent(fake).plan(state)

        planner_prompt = fake.prompts[-1]
        self.assertIn("Return the entities, not their count", planner_prompt)

    def test_state_uses_supplied_contract_and_dataset_instructions(self):
        df = pd.DataFrame({"Venue": ["home", "away"], "Wins": [8, 3]})
        contract = infer_answer_contract(
            "Was the team more successful at home or away?",
            allowed_labels=("home", "away"),
            kind_override="label",
            reasoning_required=True,
        )
        state = TQASessionState(
            question="Was the team more successful at home or away?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_contract=contract,
            dataset_profile="crt",
            dataset_instructions="Compare like-for-like units.",
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=[], cols=["Venue", "Wins"])

        PlannerAgent(fake).plan(state)

        self.assertIs(state.answer_contract, contract)
        self.assertEqual(state.dataset_profile, "crt")
        self.assertIn("Compare like-for-like units.", fake.prompts[-1])

    def test_tuple_final_answer_is_machine_readable_json(self):
        df = pd.DataFrame({"Result": ["decision", "finish"]})
        contract = infer_answer_contract(
            "How many by decision and how many by finish?",
            kind_override="tuple",
            arity=2,
            reasoning_required=True,
        )
        state = TQASessionState(
            question="How many by decision and how many by finish?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_contract=contract,
        )
        state.route_type = "COMPLEX"
        state.final_value = [3, 12]

        result = FinalAnswerAgent(lambda _: "unused").respond(state)

        self.assertEqual(result.final_answer, "[3, 12]")

    def test_invalid_complex_answer_uses_structured_recovery(self):
        df = pd.DataFrame({"Title": ["Miss Malaga 2011"], "Winner": ["Ursula"]})
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["Winner"],
            direct_answer_output='{"answer":"Ursula"}',
        )
        state = TQASessionState(
            question="Who held the Miss Malaga 2011 title?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.route_type = "COMPLEX"
        state.final_value = ""
        state.contract_validation = {
            "valid": False,
            "reason": "Answer contract requires one non-empty scalar value.",
        }

        result = FinalAnswerAgent(fake).respond(state)

        self.assertEqual(result.final_value, "Ursula")
        self.assertEqual(result.final_answer, "Ursula")
        self.assertIn("structured recovery", fake.prompts[-1])

    def test_planner_prompt_contains_full_table_column_profiles(self):
        df = pd.DataFrame(
            {
                "Fate": ["Fell"] * 8 + ["Pulled Up", "Refused"],
                "Runner": [f"Horse {index}" for index in range(10)],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=[], cols=["Fate"])
        state = TQASessionState(
            question="How many runners did not finish?",
            df=df,
            table_schema=_build_table_schema(df, max_preview_rows=8),
        )

        PlannerAgent(fake).plan(state)

        planner_prompt = fake.prompts[-1]
        self.assertIn("[COLUMN PROFILES]", planner_prompt)
        self.assertIn("Pulled Up", planner_prompt)

    def test_planner_prompt_contains_table_grounding_rules(self):
        df = pd.DataFrame(
            {
                "Team": ["Alpha", "Beta"],
                "Coach": ["unknown", "known"],
                "Location": ["Adelaide", "Perth"],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=[], cols=["Coach"])
        state = TQASessionState(
            question="One of the teams in Australia has an unknown coach.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
            table_context="List of soccer clubs in Australia",
        )

        PlannerAgent(fake).plan(state)

        planner_prompt = fake.prompts[-1]
        self.assertIn("DataFrame shape: 2 rows x 3 columns", planner_prompt)
        self.assertIn("table-level context", planner_prompt)
        self.assertIn("distinct or unique", planner_prompt)
        self.assertIn("List of soccer clubs in Australia", planner_prompt)
        self.assertIn("tend", planner_prompt)
        self.assertIn("conserved", planner_prompt)
        self.assertIn("former titleholder", planner_prompt)
        self.assertIn("Do not drop summary, sum, total, or aggregate rows", planner_prompt)
        self.assertIn('asking "after 1936" excludes', planner_prompt)
        self.assertIn("percentage columns are snapshots", planner_prompt)
        self.assertIn("systematic pattern", planner_prompt)

    def test_wtq_country_code_scalar_expands_to_country_name(self):
        df = pd.DataFrame(
            {
                "Cyclist": [
                    "Mario Cipollini (ITA)",
                    "Paolo Bettini (ITA)",
                    "Tom Boonen (BEL)",
                ]
            }
        )

        self.assertEqual(
            _canonicalize_wtq_scalar(
                "ITA",
                df,
                "which country had the most cyclists finish within the top 10?",
            ),
            "Italy",
        )

    def test_wtq_country_code_pipeline_syncs_final_answer(self):
        df = pd.DataFrame(
            {
                "Rank": [1, 2, 3],
                "Cyclist": [
                    "Mario Cipollini (ITA)",
                    "Paolo Bettini (ITA)",
                    "Tom Boonen (BEL)",
                ],
            }
        )
        planner_output = (
            "[PLAN]\n"
            "Step1: Count country codes among top ranked cyclists.\n"
            "[CODE]\n"
            "final_answer_value = 'ITA'\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=["1", "2", "3"],
            cols=["Rank", "Cyclist"],
            planner_outputs=[planner_output],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="which country had the most cyclists finish within the top 10?",
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="wtq",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "Italy")
        self.assertEqual(result.final_answer, "Italy")

    def test_wtq_how_long_after_year_adds_year_unit(self):
        df = pd.DataFrame({"Year": ["1936/37", "1953/54"]})

        self.assertEqual(
            _canonicalize_wtq_scalar(
                17,
                df,
                "how long did it take to win after 1936?",
            ),
            "17 years",
        )

    def test_wtq_datetime_scalar_formats_as_date_text(self):
        df = pd.DataFrame({"Original air date": ["January 26, 1995"]})

        self.assertEqual(
            _canonicalize_wtq_scalar(
                pd.Timestamp("1995-01-26").to_datetime64(),
                df,
                "which episode aired on january 19 and what was the next airdate?",
            ),
            "January 26, 1995",
        )

    def test_build_df_removes_repeated_header_rows(self):
        table = [
            ["week", "record"],
            ["week", "record"],
            ["1", "0 - 1"],
            ["2", "1 - 1"],
        ]

        df = build_df_from_table(table)

        self.assertEqual(df["record"].tolist(), ["0 - 1", "1 - 1"])
        self.assertEqual(df["week"].tolist(), [1.0, 2.0])

    def test_build_df_handles_quotes_in_column_names_without_code_generation(self):
        table = [
            ["company", "group 's equity shareholding"],
            ["Example Air", "30%"],
        ]

        df = build_df_from_table(table)

        self.assertEqual(
            list(df.columns),
            ["company", "group 's equity shareholding"],
        )
        self.assertEqual(df.iloc[0].tolist(), ["Example Air", "30%"])

    def test_build_df_normalizes_blank_and_annotated_numeric_cells(self):
        table = [
            ["Name", "Points", "Sales"],
            ["A", "318.65", "1,336,150 +"],
            ["B", "", "401000 +"],
        ]

        df = build_df_from_table(table)

        self.assertEqual(df["Points"].min(), 318.65)
        self.assertTrue(pd.isna(df.loc[1, "Points"]))
        self.assertEqual(df["Sales"].tolist(), [1336150.0, 401000.0])

    def test_schema_profiles_include_values_beyond_preview_and_missing_markers(self):
        df = pd.DataFrame(
            {
                "Age": list(range(1, 13)),
                "Fate": ["Fell"] * 8
                + ["Pulled Up", "Brought Down", "Refused", "TBA"],
                "Mixed": [1, "two"] + [None] * 10,
            }
        )

        schema = _build_table_schema(df, max_preview_rows=8)

        fate = schema["column_profiles"]["Fate"]
        self.assertEqual(fate["semantic_type"], "text")
        self.assertEqual(
            fate["representative_values"],
            ["Fell", "Pulled Up", "Brought Down", "Refused", "TBA"],
        )
        self.assertEqual(fate["missing_marker_count"], 1)
        self.assertEqual(schema["column_profiles"]["Age"]["semantic_type"], "numeric")
        self.assertEqual(schema["column_profiles"]["Mixed"]["semantic_type"], "mixed")
        self.assertIn("Pulled Up", schema["column_profiles_text"])
        self.assertNotIn("Pulled Up", schema["preview_text"])

    def test_schema_profile_text_is_bounded_for_wide_tables(self):
        df = pd.DataFrame(
            {
                f"Column {column}": [f"value-{column}-{row}" for row in range(30)]
                for column in range(80)
            }
        )

        schema = _build_table_schema(df)

        self.assertLessEqual(len(schema["column_profiles_text"]), 8000)

    def test_generated_code_strips_standalone_xml_end_tags(self):
        raw = (
            "```python\n"
            "value = 1 < 2\n"
            "final_answer_value = 'true'\n"
            "</final_answer_value>\n"
            "```"
        )

        cleaned = _strip_code_fence(raw)

        self.assertIn("value = 1 < 2", cleaned)
        self.assertNotIn("</final_answer_value>", cleaned)

    def test_grounding_validator_rejects_unique_count_for_x_of_n(self):
        valid, reason = validate_generated_code_grounding(
            "Grass was the surface in 3 of the 14 championships.",
            "total = df['championship'].nunique()\nfinal_answer_value = total == 14",
        )

        self.assertFalse(valid)
        self.assertIn("Use row counts", reason)

    def test_grounding_validator_allows_explicit_distinct_count(self):
        valid, reason = validate_generated_code_grounding(
            "Were there 3 distinct championships?",
            "final_answer_value = df['championship'].nunique() == 3",
        )

        self.assertTrue(valid)
        self.assertEqual(reason, "")

    def test_grounding_validator_requires_reference_exclusion_for_same_group(self):
        invalid, reason = validate_generated_code_grounding(
            "How many elements are in the same group as Neon?",
            "group = df.loc[df['Name'] == 'Neon', 'Group'].iloc[0]\n"
            "final_answer_value = (df['Group'] == group).sum()",
        )
        invalid_with_unrelated_subtraction, _ = validate_generated_code_grounding(
            "How many elements are in the same group as Neon?",
            "offset = 5 - 1\n"
            "group = df.loc[df['Name'] == 'Neon', 'Group'].iloc[0]\n"
            "final_answer_value = (df['Group'] == group).sum()",
        )
        valid, _ = validate_generated_code_grounding(
            "How many elements are in the same group as Neon?",
            "group = df.loc[df['Name'] == 'Neon', 'Group'].iloc[0]\n"
            "final_answer_value = ((df['Group'] == group) & "
            "(df['Name'] != 'Neon')).sum()",
        )

        self.assertFalse(invalid)
        self.assertFalse(invalid_with_unrelated_subtraction)
        self.assertIn("Exclude the named reference item", reason)
        self.assertTrue(valid)

    def test_grounding_validator_rejects_unrequested_summary_exclusion(self):
        code = (
            "# Exclude summary rows if any\n"
            "df_counties = df[~df['county'].str.lower().eq('norway')].copy()\n"
            "final_answer_value = df_counties['county'].iloc[0]"
        )

        valid, reason = validate_generated_code_grounding(
            "Which county has had the most consistent percentage change?",
            code,
        )

        self.assertFalse(valid)
        self.assertIn("Do not exclude summary", reason)

    def test_grounding_validator_rejects_after_year_inclusive_boundary(self):
        code = "rows = df[df['start_year'] >= 1936]\nfinal_answer_value = rows.iloc[0]['Year']"

        valid, reason = validate_generated_code_grounding(
            "How long did it take to win after 1936?",
            code,
        )

        self.assertFalse(valid)
        self.assertIn("after 1936", reason)

    def test_grounding_validator_rejects_relative_formula_for_snapshot_average(self):
        code = (
            "top5 = df[df['rank'] <= 5]\n"
            "pct_change = ((top5['% (2040)'] - top5['% (1960)']) / top5['% (1960)']) * 100\n"
            "final_answer_value = round(pct_change.mean(), 3)"
        )

        valid, reason = validate_generated_code_grounding(
            "What is the average percentage change in population between 1960 and 2040?",
            code,
        )

        self.assertFalse(valid)
        self.assertIn("percentage snapshot", reason)

    def test_grounding_validator_rejects_hardcoded_label_after_condition(self):
        code = (
            "if df['days'].nunique() > 1:\n"
            "    final_answer_value = 'Yes'\n"
            "else:\n"
            "    final_answer_value = 'No'\n"
            "final_answer_value = 'Yes'\n"
        )

        valid, reason = validate_generated_code_grounding(
            "Is there a difference in the types of events based on stages?",
            code,
        )

        self.assertFalse(valid)
        self.assertIn("hard-coded", reason)

    def test_contract_code_alignment_rejects_wrong_rounding_precision(self):
        contract = infer_answer_contract(
            "What is the average percentage change?",
            decimal_places=3,
        )
        code = "final_answer_value = round(avg_change, 0)"

        valid, reason = validate_answer_contract_code_alignment(code, contract)

        self.assertFalse(valid)
        self.assertIn("3 decimal", reason)

    def test_grounding_validator_rejects_combining_all_peer_groups(self):
        valid, reason = validate_generated_code_grounding(
            "How does Mandsaur compare to the other districts?",
            "target = df[df['district'] == 'mandsaur']['count'].sum()\n"
            "others = df[df['district'] != 'mandsaur']['count'].sum()\n"
            "final_answer_value = 'more' if target > others else 'less'",
        )

        self.assertFalse(valid)
        self.assertIn("each peer group", reason)

    def test_reasoning_prompts_are_domain_neutral(self):
        self.assertNotIn("Chinese question", PLANNER_PROMPT_TEMPLATE)
        self.assertNotIn("governmental statistics", PLANNER_PROMPT_TEMPLATE)
        self.assertNotIn("governmental statistics", CRITIC_PROMPT_TEMPLATE)

    def test_simple_fallback_records_a_structured_answer_value(self):
        df = pd.DataFrame(
            {
                "Period": ["1939/40", "1940/41"],
                "Killed": ["80,000", "100,000"],
                "Missing": ["5,000", "6,000"],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.05,
            rows=["1940/41"],
            cols=["Killed", "Missing"],
            direct_answer_output='{"answer":"100,000"}',
        )
        pipeline, tracker = self._pipeline(fake)
        state = TQASessionState(
            question="What was the affected figure in 1940/41?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        self.assertFalse(result.simple_lookup_success)
        self.assertEqual(result.final_value, "100,000")
        self.assertEqual(result.final_answer, "100,000")
        self.assertEqual(tracker.snapshot()["llm_call_count"], 3)

    def test_simple_path_normalizes_format_only_noise(self):
        df = pd.DataFrame(
            {
                "Country": ["Canada/United States", "Australia"],
                "Box Office": ["$10.8 billion", "$1.2 billion"],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.05,
            rows=["Canada/United States", "Australia"],
            cols=["Box Office"],
            direct_answer_output='{"answer":"$12.0 billion"}',
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="How much box office revenue did they account for?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "$12 billion")
        self.assertEqual(result.contract_validation, {"valid": True, "reason": ""})

    def test_aggregate_compression_keeps_every_row_and_exposes_last_row(self):
        df = pd.DataFrame(
            {
                "Round": list(range(20)),
                "Home/Away": ["Away" if i % 2 else "Home" for i in range(20)],
            }
        )
        state = TQASessionState(
            question="How many total away games were played?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.difficulty_level = "easy"
        state.structural_features = {
            "selected_rows": [],
            "selected_cols": ["Home/Away"],
            "cell_score": 0.1,
        }

        result = TableCompressor(max_easy_rows=12).compress(state)

        self.assertEqual(result.compression_info["compressed_rows"], 20)
        self.assertIn("19", result.table_schema["preview_text"])

    def test_geographic_aggregation_keeps_column_containing_question_values(self):
        df = pd.DataFrame(
            {
                "Circuit": ["Sandown", "Bathurst", "Homebush"],
                "Location": [
                    "Melbourne, Victoria",
                    "Bathurst, New South Wales",
                    "Sydney, New South Wales",
                ],
            }
        )
        state = TQASessionState(
            question="How many races were held in Victoria or New South Wales?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.route_type = "COMPLEX"
        state.difficulty_level = "medium"
        state.structural_features = {
            "selected_rows": [],
            "selected_cols": ["Circuit"],
            "cell_score": 0.3,
        }

        result = TableCompressor().compress(state)

        self.assertIn("Location", result.compression_info["used_cols"])

    def test_how_often_before_date_keeps_all_rows(self):
        df = pd.DataFrame(
            {"Air date": [f"November {day}, 2007" for day in range(1, 21)]}
        )
        state = TQASessionState(
            question="Previous to November 15, 2007 how often was the rank in the 30s?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.route_type = "COMPLEX"
        state.difficulty_level = "medium"
        state.structural_features = {
            "selected_rows": ["November 15, 2007"],
            "selected_cols": ["Air date"],
            "cell_score": 0.1,
        }

        result = TableCompressor().compress(state)

        self.assertEqual(result.compression_info["compressed_rows"], 20)

    def test_relative_row_compression_keeps_neighbor_and_label_column(self):
        df = pd.DataFrame(
            {
                "Rank": [1, 2, 3],
                "Country": ["Alpha", "Canada", "Australia"],
                "Revenue": [10, 20, 30],
                "Notes": ["a", "b", "c"],
            }
        )
        state = TQASessionState(
            question="What country is next after Canada?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.difficulty_level = "easy"
        state.structural_features = {
            "selected_rows": ["Canada"],
            "selected_cols": ["Revenue"],
            "cell_score": 0.1,
        }

        result = TableCompressor(max_easy_rows=12).compress(state)

        self.assertEqual(result.compression_info["used_rows"], ["0", "1", "2"])
        self.assertIn("Country", result.compression_info["used_cols"])
        self.assertIn("Notes", result.compression_info["used_cols"])

    def test_complex_compression_keeps_full_data_but_short_planner_preview(self):
        df = pd.DataFrame(
            {
                "Round": list(range(20)),
                "Home/Away": ["Away" if i % 2 else "Home" for i in range(20)],
            }
        )
        state = TQASessionState(
            question="How many away games were played?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.route_type = "COMPLEX"
        state.difficulty_level = "medium"
        state.structural_features = {
            "selected_rows": [],
            "selected_cols": ["Home/Away"],
            "cell_score": 0.1,
        }

        result = TableCompressor().compress(state)

        self.assertEqual(result.compression_info["compressed_rows"], 20)
        self.assertNotIn("19", result.table_schema["preview_text"])

    def test_cross_column_superlative_keeps_all_year_columns(self):
        df = pd.DataFrame(
            {
                "Tournament": ["win - loss"],
                "2006": ["1 - 3"],
                "2007": ["8 - 4"],
                "2008": ["4 - 4"],
            }
        )
        state = TQASessionState(
            question="2007 was the year with the best win-loss ratio.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
        )
        state.original_df = df
        state.route_type = "COMPLEX"
        state.difficulty_level = "medium"
        state.structural_features = {
            "selected_rows": ["win - loss"],
            "selected_cols": ["Tournament", "2007"],
            "cell_score": 0.25,
        }

        result = TableCompressor().compress(state)

        self.assertEqual(result.compression_info["used_cols"], list(df.columns))
        self.assertIn("global_columns", result.compression_info["strategy"])

    def test_abstract_performance_comparison_keeps_metric_columns(self):
        df = pd.DataFrame(
            {
                "Year": [2002, 2003],
                "Competition": ["A", "B"],
                "Position": ["100 m", "100 m"],
                "Event": ["1st", "2nd"],
                "Notes": ["11.3 secs", "11.1 secs"],
            }
        )
        state = TQASessionState(
            question="How does her early performance compare to later competitions?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.original_df = df
        state.route_type = "COMPLEX"
        state.difficulty_level = "medium"
        state.structural_features = {
            "selected_rows": [],
            "selected_cols": ["Competition", "Position"],
            "cell_score": 0.2,
        }

        result = TableCompressor().compress(state)

        self.assertEqual(result.compression_info["used_cols"], list(df.columns))

    def test_aggregate_question_forces_complex_route(self):
        df = pd.DataFrame(
            {
                "Round": list(range(20)),
                "Home/Away": ["Away" if i % 2 else "Home" for i in range(20)],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.05,
            rows=[],
            cols=["Home/Away"],
        )
        state = TQASessionState(
            question="How many total away games were played?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = RouterAgent(fake).route(state)

        self.assertEqual(result.route_type, "COMPLEX")

    def test_single_entity_how_many_value_stays_simple(self):
        df = pd.DataFrame(
            {
                "Team": ["KR", "Alpha"],
                "Goals": [27, 12],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.05,
            rows=["KR"],
            cols=["Goals"],
        )
        state = TQASessionState(
            question="How many goals did KR score?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = RouterAgent(fake).route(state)

        self.assertEqual(result.route_type, "SIMPLE")

    def test_relative_row_question_skips_single_cell_shortcut(self):
        df = pd.DataFrame(
            {
                "Name": ["Alpha", "Canada", "Australia"],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.05,
            rows=["Canada"],
            cols=["Name"],
            direct_answer_output='{"answer":"Australia"}',
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="What is next after Canada?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        self.assertFalse(result.simple_lookup_success)
        self.assertEqual(result.final_value, "Australia")
        self.assertEqual(
            result.simple_lookup_evidence["reason"],
            "relative_row_question_requires_context",
        )

    def test_semantic_score_ignores_unrelated_step_counts(self):
        router = RouterAgent(lambda _: "This needs 2 operations. Final score: 0.25")

        score = router.llm_semantic_score("Compare two values.")

        self.assertEqual(score, 0.25)

    def test_calculator_allows_only_approved_numeric_imports(self):
        df = pd.DataFrame({"Value": ["10", "20"]})
        calculator = Calculator()
        allowed_state = TQASessionState(
            question="What is the total?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        allowed_state.code_str = (
            "import pandas as pd\n"
            "print(df.head())\n"
            "final_answer_value = int(pd.to_numeric(df['Value']).sum())"
        )

        allowed_result = calculator.execute(allowed_state)

        self.assertTrue(allowed_result.exec_success)
        self.assertEqual(allowed_result.final_value, 30)

        blocked_state = TQASessionState(
            question="Read a system file.",
            df=df,
            table_schema=_build_table_schema(df),
        )
        blocked_state.code_str = "import os\nfinal_answer_value = os.getcwd()"

        blocked_result = calculator.execute(blocked_state)

        self.assertFalse(blocked_result.exec_success)
        self.assertIn("not allowed", blocked_result.exec_error)

    def test_calculator_allows_regex_and_isinstance_without_opening_os(self):
        df = pd.DataFrame({"Result": ["Age 21", "Age 34"]})
        state = TQASessionState(
            question="How many ages are present?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.code_str = (
            "import re\n"
            "matches = df['Result'].apply(lambda value: re.search(r'\\d+', value))\n"
            "final_answer_value = sum(isinstance(match.group(0), str) for match in matches)\n"
        )

        result = Calculator().execute(state)

        self.assertTrue(result.exec_success)
        self.assertEqual(result.final_value, 2)

    def test_calculator_helper_function_is_visible_inside_apply(self):
        df = pd.DataFrame({"Score": ["1-2", "3-1"]})
        state = TQASessionState(
            question="What are the score differences?",
            df=df,
            table_schema=_build_table_schema(df),
        )
        state.code_str = (
            "def difference(score):\n"
            "    left, right = score.split('-')\n"
            "    return abs(int(left) - int(right))\n"
            "values = df['Score'].apply(lambda value: difference(value))\n"
            "final_answer_value = int(values.sum())\n"
        )

        result = Calculator().execute(state)

        self.assertTrue(result.exec_success)
        self.assertEqual(result.final_value, 3)

    def test_complex_path_executes_and_passes_multi_view_validation(self):
        df = pd.DataFrame(
            {
                "Year": ["2019", "2020", "2021"],
                "Revenue": [90, 100, 120],
                "Profit": [5, 10, 30],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.90, rows=["2020", "2021"], cols=["Profit"])
        pipeline, tracker = self._pipeline(fake, enable_multi_view_validation=True)
        state = TQASessionState(
            question="How much did Profit increase from 2020 to 2021?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_contract=infer_answer_contract(
                "How much did Profit increase from 2020 to 2021?",
                reasoning_required=True,
            ),
        )

        result = pipeline.run(state)

        self.assertEqual(result.route_type, "COMPLEX")
        self.assertTrue(result.exec_success)
        self.assertEqual(result.final_value, 20.0)
        self.assertEqual(result.critic_verdict, "PASS")
        self.assertEqual(result.multi_view_validation["verdict"], "PASS")
        self.assertTrue(result.alternative_exec_success)
        self.assertEqual(result.alternative_final_value, 20.0)
        self.assertEqual(result.cross_validation_verdict, "PASS")
        self.assertGreaterEqual(tracker.snapshot()["llm_call_count"], 8)

    def test_medium_complex_path_skips_llm_critic_after_deterministic_checks(self):
        df = pd.DataFrame(
            {"Year": ["2019", "2020", "2021"], "Profit": [5, 10, 30]}
        )
        fake = FakePipelineLLM(
            semantic_score=0.75,
            rows=["2020", "2021"],
            cols=["Profit"],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="How much did Profit increase from 2020 to 2021?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_contract=infer_answer_contract(
                "How much did Profit increase from 2020 to 2021?",
                reasoning_required=True,
            ),
        )

        result = pipeline.run(state)

        self.assertEqual(result.difficulty_level, "medium")
        self.assertTrue(result.critic_skipped)
        self.assertEqual(result.critic_verdict, "PASS")
        self.assertFalse(any("careful auditor" in prompt for prompt in fake.prompts))

    def test_true_false_task_routes_compresses_and_classifies_without_planner(self):
        df = pd.DataFrame(
            {
                "Country": ["Italy", "France"],
                "Winner": ["yes", "no"],
            }
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=["Italy"],
            cols=["Country", "Winner"],
            classification_output='{"label":"true"}',
        )
        pipeline, tracker = self._pipeline(fake)
        state = TQASessionState(
            question="The winning country was Italy.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "true")
        self.assertEqual(result.final_answer, "true")
        self.assertTrue(result.exec_success)
        self.assertEqual(result.planner_raw_output, "")
        self.assertEqual(tracker.snapshot()["llm_call_count"], 3)

    def test_numeric_true_false_task_uses_code_reasoning(self):
        df = pd.DataFrame(
            {
                "Championship": list(range(14)),
                "Surface": ["grass", "grass", "grass"] + ["clay"] * 11,
            }
        )
        planner_output = (
            "[PLAN]\n"
            "Step1: Count grass rows and compare with 3.\n"
            "[CODE]\n"
            "grass_count = (df['Surface'].str.lower() == 'grass').sum()\n"
            "final_answer_value = bool(grass_count == 3)\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["Surface"],
            classification_output='{"label":"false"}',
            planner_outputs=[planner_output],
        )
        pipeline, tracker = self._pipeline(fake)
        state = TQASessionState(
            question="Grass was the surface in 3 of 14 championships, or 21.43%.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
        )

        result = pipeline.run(state)

        self.assertTrue(result.risk_escalated)
        self.assertNotEqual(result.planner_raw_output, "")
        self.assertEqual(result.classification_raw_output, "")
        self.assertEqual(result.final_value, "true")
        self.assertEqual(result.contract_validation, {"valid": True, "reason": ""})
        self.assertGreaterEqual(tracker.snapshot()["llm_call_count"], 3)

    def test_tabfact_verifier_can_correct_code_label_using_all_clauses(self):
        df = pd.DataFrame(
            {"place": ["t7"], "player": ["peter"], "prize": [9000]}
        )
        planner_output = (
            "[PLAN]\nStep1: Check the place only.\n[CODE]\n"
            "final_answer_value = df.loc[0, 'place'] == 't7'\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["place", "player", "prize"],
            planner_outputs=[planner_output],
            verification_output='{"label":"false"}',
        )
        pipeline, _ = self._pipeline(fake)
        contract = infer_answer_contract(
            "T7 was Peter's place and his prize was 10875.",
            "true_false",
            reasoning_required=True,
        )
        state = TQASessionState(
            question="T7 was Peter's place and his prize was 10875.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
            answer_contract=contract,
            dataset_profile="tabfact",
            dataset_instructions="Verify every clause.",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "false")
        self.assertTrue(any("TabFact verification judge" in p for p in fake.prompts))

    def test_ungrounded_unique_count_replans_before_execution(self):
        df = pd.DataFrame(
            {
                "Championship": ["A", "A", "B", "C"],
                "Surface": ["grass", "clay", "grass", "grass"],
            }
        )
        wrong = (
            "[PLAN]\nStep1: Count unique values.\n[CODE]\n"
            "total = df['Championship'].nunique()\n"
            "final_answer_value = total == 4\n"
        )
        corrected = (
            "[PLAN]\nStep1: Count rows.\n[CODE]\n"
            "total = len(df)\n"
            "grass = (df['Surface'] == 'grass').sum()\n"
            "final_answer_value = total == 4 and grass == 3\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["Surface"],
            planner_outputs=[wrong, corrected],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="Grass was the surface in 3 of the 4 championships.",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="true_false",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "true")
        self.assertEqual(result.grounding_validation, {"valid": True, "reason": ""})
        planner_prompts = [p for p in fake.prompts if "table reasoning planner" in p]
        self.assertEqual(len(planner_prompts), 2)
        self.assertIn("Use row counts", planner_prompts[1])

    def test_contract_mismatch_replans_entity_list(self):
        df = pd.DataFrame(
            {
                "Team": ["Alpha", "Beta", "Gamma"],
                "Races": [13, 14, 10],
            }
        )
        wrong_shape = (
            "[PLAN]\nStep1: Count qualifying teams.\n[CODE]\n"
            "final_answer_value = int((df['Races'] >= 13).sum())\n"
        )
        corrected_shape = (
            "[PLAN]\nStep1: Return qualifying team names.\n[CODE]\n"
            "final_answer_value = df.loc[df['Races'] >= 13, 'Team'].tolist()\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["Team", "Races"],
            planner_outputs=[wrong_shape, corrected_shape],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="Which teams raced at least 13 races?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        planner_prompts = [
            prompt for prompt in fake.prompts if "table reasoning planner" in prompt
        ]
        self.assertEqual(len(planner_prompts), 2)
        self.assertEqual(result.final_value, ["Alpha", "Beta"])
        self.assertEqual(result.contract_validation, {"valid": True, "reason": ""})
        self.assertIn("requires a non-empty list", planner_prompts[1])

    def test_execution_error_is_preserved_for_replan_feedback(self):
        df = pd.DataFrame({"Position": [1, 2, 3]})
        failing = (
            "[PLAN]\nStep1: Inspect the positions.\n[CODE]\n"
            "final_answer_value = int(df['Position'].str.startswith('Lord').sum())\n"
        )
        corrected = (
            "[PLAN]\nStep1: Count the listed positions.\n[CODE]\n"
            "final_answer_value = int(df['Position'].notna().sum())\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=[],
            cols=["Position"],
            planner_outputs=[failing, corrected],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="How many positions are listed?",
            df=df,
            table_schema=_build_table_schema(df),
        )

        result = pipeline.run(state)

        planner_prompts = [
            prompt for prompt in fake.prompts if "table reasoning planner" in prompt
        ]
        self.assertEqual(len(planner_prompts), 2)
        self.assertIn("Execution failed:", planner_prompts[1])
        self.assertIn(".str accessor", planner_prompts[1])
        self.assertIn("verify that filters match", planner_prompts[1])
        self.assertEqual(result.final_value, 3)

    def test_yes_no_task_normalizes_verbose_label(self):
        df = pd.DataFrame({"Event": ["A", "B"], "Acts": [5, 60]})
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=["A", "B"],
            cols=["Event", "Acts"],
            classification_output="The answer is No.",
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="Are the event sizes similar?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="yes_no",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "No")
        self.assertEqual(result.final_answer, "No")

    def test_comparison_label_task_uses_declared_closed_set(self):
        df = pd.DataFrame({"Year": [2020, 2021], "Score": [5, 8]})
        planner_output = (
            "[PLAN]\nStep1: Compare the first and last score.\n[CODE]\n"
            "final_answer_value = 'better' if df['Score'].iloc[-1] > "
            "df['Score'].iloc[0] else 'worse'\n"
        )
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=["2020", "2021"],
            cols=["Score"],
            classification_output='{"label":"worse"}',
            planner_outputs=[planner_output],
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="Did performance get better, worse, or stay equal?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="better_worse_equal",
        )

        result = pipeline.run(state)

        self.assertTrue(result.risk_escalated)
        self.assertEqual(result.classification_raw_output, "")
        self.assertEqual(result.final_value, "better")
        self.assertEqual(result.final_answer, "better")

    def test_crt_duration_shortcut_treats_24_hours_as_one_day(self):
        df = pd.DataFrame(
            {
                "year": [1990, 1991, 1992],
                "event": ["a", "b", "c"],
                "days": ["1 day", "24 hours", "1 day"],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=["1990", "1991"], cols=["days"])
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question=(
                "Has the duration of festivals held at Donington Park changed over time? "
                "Answer with only 'Yes' or 'No' that is most accurate and nothing else."
            ),
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="crt",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "No")
        self.assertEqual(result.final_answer, "No")
        self.assertNotIn("table reasoning planner", "\n".join(fake.prompts))

    def test_crt_event_type_difference_shortcut_requires_systematic_mapping(self):
        df = pd.DataFrame(
            {
                "event": ["monsters of rock", "one step beyond", "monsters of rock"],
                "days": ["1 day", "24 hours", "1 day"],
                "stages": ["1 stage", "1 stage", "2 stages"],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=["monsters"], cols=["event"])
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question=(
                "Is there a difference in the types of events hosted at Donington Park "
                "based on the number of stages or days they have? Answer with only 'Yes' "
                "or 'No' that is most accurate and nothing else."
            ),
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="crt",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, "No")
        self.assertEqual(result.final_answer, "No")
        self.assertNotIn("table reasoning planner", "\n".join(fake.prompts))

    def test_crt_percentage_snapshot_shortcut_averages_requested_cells(self):
        df = pd.DataFrame(
            {
                "rank": [1, 2, 3, 4, 5, "sum"],
                "county": ["oslo", "akershus", "hordaland", "rogaland", "sor", "norway"],
                "% (1960)": [13.2, 6.3, 9.4, 6.6, 5.8, 100.0],
                "% (2000)": [11.3, 10.4, 9.7, 8.3, 5.8, 100.0],
                "% (2040)": [12.8, 11.9, 10.2, 9.9, 6.0, 100.0],
            }
        )
        fake = FakePipelineLLM(semantic_score=0.8, rows=["1", "2"], cols=["% (1960)"])
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question=(
                "What is the average percentage change in population for the top 5 "
                "ranked Norwegian counties between 1960 and 2040?"
            ),
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="crt",
        )

        result = pipeline.run(state)

        self.assertEqual(result.final_value, 9.173)
        self.assertEqual(result.final_answer, "9.173")
        self.assertNotIn("table reasoning planner", "\n".join(fake.prompts))

    def test_classification_rejects_output_without_allowed_label(self):
        df = pd.DataFrame({"Event": ["A"], "Acts": [5]})
        fake = FakePipelineLLM(
            semantic_score=0.8,
            rows=["A"],
            cols=["Event", "Acts"],
            classification_output="uncertain",
        )
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="Is this supported?",
            df=df,
            table_schema=_build_table_schema(df),
            answer_mode="yes_no",
        )

        with self.assertRaisesRegex(RuntimeError, "allowed label"):
            pipeline.run(state)

    def test_selective_mode_records_risk_and_budget(self):
        df = pd.DataFrame({"Name": ["Alpha"], "Value": [7]})
        fake = FakePipelineLLM(semantic_score=0.05, rows=["Alpha"], cols=["Value"])
        tracker = LLMCallTracker(fake)
        pipeline = TableQAPipeline(
            router=RouterAgent(tracker),
            planner=PlannerAgent(tracker),
            calculator=Calculator(),
            critic=CriticAgent(tracker),
            final_answer_agent=FinalAnswerAgent(tracker),
            enable_selective_collaboration=True,
            mact_avg_tokens=8867.0,
        )
        state = TQASessionState(
            question="What is the value for Alpha?",
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="wtq",
        )

        result = pipeline.run(state)

        self.assertIsNotNone(result.risk_assessment)
        self.assertIn(result.risk_level, {"light", "medium", "high", "fallback"})
        self.assertIsNotNone(result.evidence_pack)
        self.assertIn("avg_tokens", result.budget_state)

    def test_selective_high_risk_runs_candidate_judge(self):
        df = pd.DataFrame({"Year": ["2020", "2021"], "Profit": [10, 30]})
        fake = FakePipelineLLM(
            semantic_score=0.9,
            rows=["2020", "2021"],
            cols=["Year", "Profit"],
        )
        tracker = LLMCallTracker(fake)
        pipeline = TableQAPipeline(
            router=RouterAgent(tracker),
            planner=PlannerAgent(tracker),
            calculator=Calculator(),
            critic=CriticAgent(tracker),
            final_answer_agent=FinalAnswerAgent(tracker),
            enable_multi_view_validation=True,
            enable_selective_collaboration=True,
        )
        state = TQASessionState(
            question="What is the profit difference after comparing 2021 and 2020?",
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="wtq",
        )

        result = pipeline.run(state)

        self.assertIsNotNone(result.agreement_decision)
        self.assertIsInstance(result.candidate_answers, list)

    def test_legacy_mode_does_not_populate_selective_fields(self):
        df = pd.DataFrame({"Name": ["Alpha"], "Value": [7]})
        fake = FakePipelineLLM(semantic_score=0.05, rows=["Alpha"], cols=["Value"])
        pipeline, _ = self._pipeline(fake)
        state = TQASessionState(
            question="What is the value for Alpha?",
            df=df,
            table_schema=_build_table_schema(df),
            dataset_profile="wtq",
        )

        result = pipeline.run(state)

        self.assertIsNone(result.risk_assessment)
        self.assertIsNone(result.evidence_pack)


if __name__ == "__main__":
    unittest.main()
