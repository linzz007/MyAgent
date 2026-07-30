from pathlib import Path
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "server"))

from prepare_model_gate_run import GateRunConfig, prepare_gate_run, safe_slug  # noqa: E402


class PrepareModelGateRunTests(unittest.TestCase):
    def test_safe_slug_keeps_model_tags_path_safe(self):
        self.assertEqual(safe_slug("DeepSeek/R1 Distill Qwen-32B"), "DeepSeek_R1_Distill_Qwen-32B")
        self.assertEqual(safe_slug(""), "model")

    def test_prepare_gate_run_writes_executable_scripts_and_manifest(self):
        """Catches scaffold drift that would put new experiments outside MACT or use the wrong gates."""
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            myagent_root = tmp / "MyAgent"
            mact_root = tmp / "MACT"
            model_dir = tmp / "models" / "DeepSeek-R1-Distill-Qwen-32B"
            run_dir = mact_root / "outputs" / "server_runs" / "deepseek_r1_qwen32b_gate50_20260730_193500"
            model_dir.mkdir(parents=True)
            myagent_root.mkdir()

            manifest = prepare_gate_run(
                GateRunConfig(
                    myagent_root=myagent_root,
                    mact_root=mact_root,
                    model_id=model_dir,
                    model_tag="deepseek_r1_qwen32b",
                    served_model_name="deepseek-r1-qwen32b-local",
                    run_dir=run_dir,
                )
            )

            manifest_path = run_dir / "gate_run_manifest.json"
            manifest_json = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_dir"], str(run_dir))
            self.assertEqual(manifest_json["model_tag"], "deepseek_r1_qwen32b")
            self.assertEqual(manifest_json["gpu_groups"], "4,5;6,7")
            self.assertEqual(
                manifest_json["endpoints"],
                ["http://127.0.0.1:8000/v1", "http://127.0.0.1:8001/v1"],
            )
            self.assertEqual(manifest_json["gate_limits"], {"gate10": 10, "gate50": 50})

            env_values = subprocess.run(
                [
                    "bash",
                    "-lc",
                    f"source {str(run_dir / 'vllm.env')!r}; printf '%s\\n%s\\n%s\\n' \"$MODEL_ID\" \"$GPU_GROUPS\" \"$SERVED_MODEL_NAME\"",
                ],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.splitlines()
            self.assertEqual(env_values, [str(model_dir), "4,5;6,7", "deepseek-r1-qwen32b-local"])

            gate10 = run_dir / "run_gate10.sh"
            gate50 = run_dir / "run_gate50.sh"
            gate10_text = gate10.read_text(encoding="utf-8")
            gate50_text = gate50.read_text(encoding="utf-8")
            self.assertIn("--limit-per-task 10", gate10_text)
            self.assertIn("--limit-per-task 50", gate50_text)
            self.assertIn("--output-root \"$RUN_DIR/myagent_gate10\"", gate10_text)
            self.assertIn("--output-root \"$RUN_DIR/myagent_gate50\"", gate50_text)
            self.assertIn("http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1", gate50_text)
            self.assertIn("summarize_model_gate_results.py", gate50_text)
            self.assertIn("--output \"$RUN_DIR/gate50_summary.json\"", gate50_text)
            self.assertIn("--markdown-output \"$RUN_DIR/gate50_summary.md\"", gate50_text)

            for script_name in (
                "start_services.sh",
                "healthcheck_services.sh",
                "run_gate10.sh",
                "run_gate50.sh",
                "stop_services.sh",
            ):
                script_path = run_dir / script_name
                self.assertTrue(script_path.stat().st_mode & stat.S_IXUSR)
                subprocess.run(["bash", "-n", str(script_path)], check=True)

            readme = (run_dir / "README.md").read_text(encoding="utf-8")
            self.assertIn("Do not commit API keys", readme)
            self.assertIn("gate50_summary.json", readme)
            self.assertIn("git add -f", readme)


if __name__ == "__main__":
    unittest.main()
