# 2026-07-19 Model Gate Results

服务器路径：`/home/ubuntu/lzz/MyAgent`
分支：`codex/selective-risk-collaboration`
主目标：排查当前实验链路问题，并用小实验筛选是否有模型值得进入 myAgent vs MACT 配对扩大实验。

## 1. 当前可用本地模型

服务器 `/home/ubuntu/models` 当前只有两个可直接使用的本地模型：

| model path | served name | status |
|---|---|---|
| `/home/ubuntu/models/Qwen3-32B` | `qwen3-32b-local` | 已在 GPU 5/6，port 8000，healthcheck 正常 |
| `/home/ubuntu/models/Qwen2.5-3B-Instruct` | `qwen25-3b-local` | 本轮新增 GPU 0，port 8010，healthcheck 正常 |

Qwen2.5-14B 的配置文件只有 example，服务器当前没有对应本地模型目录。

## 2. 新发现并修复的问题

### 问题

用 Codex 非交互命令启动 Qwen2.5-3B vLLM 时，服务能完成一次 healthcheck，但命令返回后 8010 端口消失，导致 myAgent shard 报：

```text
openai.APIConnectionError: Connection error.
httpcore.ConnectError: [Errno 111] Connection refused
```

这和此前 Qwen3/TabFact 的“服务未就绪或连接被拒”属于同类运行链路风险，但这次具体根因不同：`start_vllm_pool.sh` 使用普通后台 `&` 启动，在 Codex 非交互执行器里进程组会被清理；只加 `nohup` 仍不够。

### 修复

已将 `scripts/server/start_vllm_pool.sh` 的启动方式改为：

```bash
setsid nohup env ... vllm serve ... > "${log_file}" 2>&1 < /dev/null &
```

并新增测试：

```text
tests/test_start_vllm_pool.py
```

验证：

```text
python tests/test_start_vllm_pool.py: OK
bash -n scripts/server/start_vllm_pool.sh: exit 0
```

修复后独立命令复查：

```text
port 8010: open
healthcheck: qwen25-3b-local returns ok
pid: 80805, status Ssl
```

## 3. Qwen2.5-3B myAgent Smoke 5

运行命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen25_3b_1gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8010/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen25_3b_policy_v5_smoke5_all \
  --limit-per-task 5 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

计时：

```text
START: 2026-07-19 21:31:05 CST
END:   2026-07-19 21:32:05 CST
real:  1m0.343s
```

完整性：

| dataset | raw rows | merged rows | log tail | failed | missing |
|---|---:|---:|---|---:|---:|
| WTQ | 5 | 5 | `Finished sample 5/5` | 0 | 0 |
| TabFact | 5 | 5 | `Finished sample 5/5` | 0 | 0 |
| CRT | 5 | 5 | `Finished sample 5/5` | 0 | 0 |

结果：

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg calls | avg seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 2/5 | 0.400 | 13,951.60 | 13,347.60 | 604.00 | 6.80 | 5.457 |
| TabFact | 3/5 | 0.600 | 1,183.20 | 1,055.40 | 127.80 | 3.40 | 1.161 |
| CRT | 4/5 | 0.800 | 13,880.40 | 13,395.80 | 484.60 | 6.00 | 4.547 |

## 4. Qwen2.5-3B myAgent 50 Gate

运行命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen25_3b_1gpu_local.env

time python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --endpoints http://127.0.0.1:8010/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen25_3b_policy_v5_50_all \
  --limit-per-task 50 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```

计时：

```text
START: 2026-07-19 21:32:33 CST
END:   2026-07-19 21:42:27 CST
real:  9m53.855s
```

完整性：

| dataset | raw rows | merged rows | eval samples | log tail | failed | missing |
|---|---:|---:|---:|---|---:|---:|
| WTQ | 50 | 50 | 50 | `Finished sample 50/50` | 0 | 0 |
| TabFact | 50 | 50 | 50 | `Finished sample 50/50` | 0 | 0 |
| CRT | 50 | 50 | 50 | `Finished sample 50/50` | 0 | 0 |

结果：

| dataset | correct | accuracy | avg tokens | avg prompt | avg completion | avg calls | avg seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 19/50 | 0.380 | 11,659.40 | 11,149.66 | 509.74 | 6.18 | 4.558 |
| TabFact | 39/50 | 0.780 | 2,351.08 | 2,046.12 | 304.96 | 3.58 | 2.489 |
| CRT | 26/50 | 0.520 | 11,630.20 | 11,093.86 | 536.34 | 5.70 | 4.731 |
| Overall | 84/150 | 0.560 | 8,546.89 | - | - | - | 3.926 |

判断：

- Qwen2.5-3B 可运行，速度快，失败数为 0。
- 但 overall accuracy 只有 56.0%，WTQ 和 CRT 明显偏低。
- WTQ/CRT token 仍接近 11k-12k，虽然 TabFact token 很低，但整体不是高性价比主模型。
- 本轮不建议继续跑 Qwen2.5-3B 的 MACT 50；它不适合作为专利正式主实验模型。

## 5. Qwen3 Policy v5 vs MACT 50 同 ID 配对

此前已有：

```text
myAgent policy v5 200:
outputs/server_runs/qwen3_32b_policy_v5_200_all/merged/*.jsonl

MACT Qwen3 50:
/home/ubuntu/lzz/MACT/outputs/server_runs/qwen3_32b_50/*_mact.jsonl
```

本轮核对 MACT 50 的所有 ID 都存在于 myAgent policy v5 200 输出中：

```text
WTQ missing_mact_ids_in_myagent200: 0
TabFact missing_mact_ids_in_myagent200: 0
CRT missing_mact_ids_in_myagent200: 0
```

因此可以对这 50/数据集做严格同 ID 配对比较。结果：

JSON artifact：

```text
outputs/server_runs/qwen3_32b_policy_v5_200_all/compare_policy_v5_matched_mact50.json
```

| dataset | myAgent correct | myAgent acc | myAgent avg tokens | MACT correct | MACT acc | MACT avg tokens | token ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| WTQ | 32/50 | 0.640 | 11,521.00 | 23/50 | 0.460 | 10,946.00 | 105.25% |
| TabFact | 43/50 | 0.860 | 2,273.30 | 44/50 | 0.880 | 11,051.98 | 20.57% |
| CRT | 38/50 | 0.760 | 11,791.96 | 33/50 | 0.660 | 12,384.26 | 95.22% |
| Overall | 113/150 | 0.753 | 8,528.75 | 100/150 | 0.667 | 11,460.75 | 74.42% |

配对分布：

| dataset | both correct | myAgent only | MACT only | both wrong | McNemar p |
|---|---:|---:|---:|---:|---:|
| WTQ | 16 | 16 | 7 | 11 | 0.0931 |
| TabFact | 40 | 3 | 4 | 3 | 1.0000 |
| CRT | 28 | 10 | 5 | 7 | 0.3018 |

判断：

- Qwen3 policy v5 在同 ID 50/数据集配对上通过当前 acceptance gate。
- Overall accuracy 高于 MACT：75.33% vs 66.67%。
- Overall token ratio 为 74.42%，刚好低于 75% 阈值，接近用户设定的 70% 目标。
- WTQ 和 CRT 准确率高于 MACT；TabFact 低 2 pp，但没有超过“单数据集不低于 5 pp”的风险线。
- 50/数据集样本量仍偏小，McNemar 结果不支持写成统计显著胜出；但足以支持进入 100/150 同 split 配对扩大实验。

## 6. 下一步建议

短期最有价值的下一步不是继续调 TabFact，也不是跑 3B MACT，而是扩大 Qwen3 同 split 配对：

1. 快速证据路线：跑 MACT Qwen3 first-100/数据集，然后用已有 myAgent policy v5 200 输出按同 ID 截取比较。优点是不需要重跑 myAgent；缺点是 first-N 不如 frozen table-diverse split 正式。
2. 正式路线：用 `code/freeze_blind_holdout.py` 冻结 100 或 150 条/数据集 table-diverse split，然后 myAgent 和 MACT 都重跑。优点是更适合专利/论文；缺点是耗时约 11.8h 到 17.6h。
3. 候选模型路线：除非服务器新增更强模型目录，否则当前本地只有 Qwen2.5-3B 和 Qwen3-32B；3B 已经 no-go，Qwen3 是当前主模型。

推荐执行顺序：

```text
Qwen3 first-100 MACT quick pair -> 若通过 -> frozen150 Qwen3 paired formal -> 少量消融
```

正式报告中应写：

> Qwen3 policy v5 在 50/数据集同 ID 配对 gate 上 overall accuracy 高于 MACT，overall token 为 MACT 的 74.42%，说明该模型和策略值得进入扩大配对实验。

不应写：

> 当前所有模型都已经超过 MACT，或 Qwen2.5-3B 也具备正式实验价值。

## 7. 已准备的 frozen150 正式输入

本轮已经冻结一个 150/数据集的 table-diverse split：

```text
datasets_ready/frozen_qwen3_eval_150_2026-07-19/
```

生成命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent

python code/freeze_blind_holdout.py \
  --wtq_input datasets_ready/full/wtq_unseen.jsonl \
  --tabfact_input datasets_ready/full/tabfact_test.jsonl \
  --crt_input datasets_ready/full/crt.jsonl \
  --output_dir datasets_ready/frozen_qwen3_eval_150_2026-07-19 \
  --history_root outputs \
  --history_root /home/ubuntu/lzz/MACT/outputs \
  --sample_size 150 \
  --seed 20260719
```

Manifest 摘要：

| dataset | records | unique tables | eligible records | category counts | prior id overlap | prior table overlap | sha256 |
|---|---:|---:|---:|---|---:|---:|---|
| WTQ | 150 | 150 | 2,466 | `all=150` | 0 | 0 | `bc12791bf6bdb26cc51e02486ba88d3a4123e058f08d82fddb6e8111609789ea` |
| TabFact | 150 | 150 | 10,702 | `false=75,true=75` | 0 | 0 | `3cbc3312929c6a23f5adf4a36321e4628046f794d0276700c91572d8a74ac46c` |
| CRT | 150 | 150 | 502 | `yes_no=60,closed_other=30,general=60` | 0 | 0 | `46daa3ccc221aa0d581552cba74e5ccb05ed4ab2274fcae77f1ccb15105ad82d` |

验证：

```text
wc -l datasets_ready/frozen_qwen3_eval_150_2026-07-19/*.jsonl
  150 crt.jsonl
  150 tabfact.jsonl
  150 wtq.jsonl

python -m json.tool datasets_ready/frozen_qwen3_eval_150_2026-07-19/manifest.json
  ok
```

`run_sharded_tqa.py` 的 dataset override dry-run 已验证可以读取该 split 并生成三任务命令。正式 myAgent 运行命令：

```bash
cd /home/ubuntu/lzz/MyAgent
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate lzz-agent
source configs/server/qwen3_32b_2gpu_local.env

python scripts/server/run_sharded_tqa.py \
  --repo-root . \
  --tasks wtq,tabfact,crt \
  --wtq-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/wtq.jsonl \
  --tabfact-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/tabfact.jsonl \
  --crt-dataset datasets_ready/frozen_qwen3_eval_150_2026-07-19/crt.jsonl \
  --endpoints http://127.0.0.1:8000/v1 \
  --model "$SERVED_MODEL_NAME" \
  --api-key-env LOCAL_VLLM_API_KEY \
  --output-root outputs/server_runs/qwen3_32b_policy_v5_frozen150_20260719 \
  --max-replan 2 \
  --mact-avg-tokens 11460
```
