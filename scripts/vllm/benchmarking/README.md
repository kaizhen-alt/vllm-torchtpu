# Benchmarking vllm-torchtpu

All configs target TPU v7x-8 (TP=8). KV cache is forced to fp8 in `run_benchmarks.sh`.

## Quick Start

```bash
# Guard configs (scheduled perf builds)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-fp8-tp8-ep

# Nightly sweeps
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp4-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-tp8-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3.5-35b-fp8-dp4-tp2-ep
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep

# Manual full 480B sweep, not run in CI
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep-full

# Pin a deterministic results dir (used by CI)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-30b-tp8-ep --results-dir /tmp/perf

# Dry run (print commands without executing)
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-coder-480b-fp8-tp8-ep --dry-run
```

## Available Configs

| Config | Model | MoE | EP | Weight quant | CI tier |
|--------|-------|-----|----|----|----|
| `qwen3-coder-30b-fp8-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct-FP8 | yes | on | fp8 | guard (scheduled) |
| `qwen3-coder-30b-tp8-ep` | Qwen3-Coder-30B-A3B-Instruct | yes | on | none | nightly |
| `qwen3.5-35b-fp8-tp4-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3.5-35b-fp8-tp8-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3.5-35b-fp8-dp4-tp2-ep` | Qwen3.5-35B-A3B-FP8 | yes | on | fp8 | nightly |
| `qwen3-coder-480b-fp8-tp8-ep` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | nightly short sweep |
| `qwen3-coder-480b-fp8-tp8-ep-full` | Qwen3-Coder-480B-A35B-Instruct-FP8 | yes | on | fp8 | manual only |

Configs are shell files in `configs/`. Add your own by copying an existing one.

## What It Does

1. Loads a config (model, TP, ISL/OSL sweep, concurrency levels)
2. Starts a `vllm serve` instance with the right flags
3. Optionally runs discarded full-traffic warmup passes
4. Runs `benchmark_serving.py` (from your vLLM install) for each ISL/OSL x concurrency combo
5. Saves JSON results to `benchmark_runs/<model>_tp<N>_<timestamp>/`

## Regression Checks

CI (`.buildkite/pipeline_perf.yml`) runs `check_regression.py` against the same
live server: once for performance, and once for each accuracy task.

### Performance (`--mode perf`)

Compares the benchmark JSON to `baselines/perf/<config>.baseline.json`, with a
5% tolerance. Gated metrics:

- Median TPOT
- Total token throughput
- Output token throughput
- Completed requests, as a floor relative to the baseline
- Median TTFT, only where the baseline entry carries it

Under closed-loop load beyond the prefill-admission capacity
(DP x max-num-batched-tokens / prompt-len), median TTFT is queue-position noise,
so nightly baselines are calibrated with `--gate-ttft-max-concurrency 0` to omit
it.

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode perf \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/perf/<config>.baseline.json
```

#### Roofline report (`--roofline-targets`)

`baselines/perf/roofline_targets.json` holds modeled output throughput per
physical chip. Its `targets` map has one entry per modeled config: the
`num_chips` it was modeled on, and for each cell an
`output_token_throughput_per_chip` and a `status` (`ok`, `unverified` or `oom`;
`ok` when absent). Its `not_modeled` map names the configs that the model does
not cover, each with a `reason` and a `detail`. The other fields describe the
model run and are not read.

Pass `--roofline-targets <file>` and `--config-name <config>` in perf check mode
to print each measured cell as a percentage of its target. `--baseline` is then
optional, so one run can check the baseline, print the report, or do both. The
report is advisory: a ratio never changes the exit code.

- The measured value is the cell's `output_throughput` divided by the physical
  chip count. That count is `--chips <N>`, or `--prefill-chips <P>` plus
  `--decode-chips <D>` for disaggregated serving, or else the config's
  `num_chips`.
- A cell whose target has `status` `unverified` is marked
  `ADVISORY (unverified)`.
- A measured cell is listed as `SKIP` when the config has no target for it,
  when its target has `status` `oom`, or when its result has no
  `output_throughput`. A target cell that was not measured is not listed.
- A config under `not_modeled` prints its reason instead of a table. Such a
  run needs no result files unless `--baseline` is also given.

The run exits `2` when the targets file is malformed, when the config is in
neither map, when a chip flag is invalid, or when no chip count is known.
`--roofline-targets` needs `--config-name`, and `--config-name` and the chip
flags need `--roofline-targets`. All five flags are usage errors outside perf
check mode.

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode perf \
  --results-dir <DIR> \
  --roofline-targets scripts/vllm/benchmarking/baselines/perf/roofline_targets.json \
  --config-name qwen3.5-397b-fp8-tp1-dp8-ep \
  --chips 4
```

### Accuracy (`--mode eval`)

Compares `lm_eval` accuracy to `baselines/eval/<config>.<task>.baseline.json`.
The default tolerance is 1.5pp; a config can override it with `EVAL_TOLERANCE`.
Only keys present in the baseline are gated.

Each task is compared on the accuracy metric of its baseline entry, the first
in the script's `ACC_KEYS` order when the entry has several. A task whose
results lack that metric fails as `(no result)`. No other accuracy metric of
the task stands in for it.

Where the baseline carries a standard error (`exact_match_stderr`), the allowed
drop becomes a one-sided 95% significance test:

```text
max(tolerance, 1.645 * hypot(stderr_baseline, stderr_run))
```

When only one side records a standard error, the two cases differ:

- **Baseline only**: it sets the scale for both sides, `sqrt(2) * stderr_baseline`.
- **Run only**: it covers only its own side, and the baseline is treated as
  exact, `stderr_run`.

The flat tolerance applies only where neither side records one.

> [!NOTE]
> **Why combine both standard errors?** The gate compares two independent runs,
> so the noisy quantity is their *difference*, whose standard error combines
> both sides in quadrature. Using one raw stderr would still fail ~24% of clean
> reruns on small benchmarks like `humaneval_plus_tpu` (N=164).

```bash
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode eval \
  --results-dir <DIR> \
  --baseline scripts/vllm/benchmarking/baselines/eval/<config>.<task>.baseline.json
```

`--calibrate` writes a baseline as the median of the runs, one `--results-dir`
per run. It exits `2` instead of writing a baseline when a run has no selected
task or lacks an accuracy metric that another run has.

### Baseline freeze policy (`--mode update`)

A baseline file is in one of two states, recorded in a top-level `_policy`
block. A file without the block, or with a block that has no `state`, is in
`development`.

- `development`: the setup is still below its target. Anyone may update the
  baseline.
- `frozen`: the setup met its target. The gated values, the state, and the owner
  change only when the same change adds a new `approval` record whose
  `approver` is the `owner` named in the file before the change.

```json
{
  "_policy": {
    "state": "frozen",
    "owner": "<GitHub handle>",
    "approval": {
      "approver": "<GitHub handle>",
      "reference": "<URL of the PR or issue where the owner approved>"
    }
  },
  "mmlu_pro": {"exact_match,custom-extract": 0.7664}
}
```

The rule in detail:

- The gated values are all top-level keys that do not start with `_`. Other
  `_` keys, such as `_source`, change freely.
- A `frozen` block must name an `owner`. `approval` is optional and needs both
  `approver` and `reference`. Each of the three is a non-empty string.
- An `approval` record counts once. A record that the file already carried
  before the change does not approve the change.
- A new baseline file and a `development` baseline need no approval, also to
  become `frozen`.

`--mode update` checks a change to a baseline file against this rule. Pass the
file before the change as `--previous-baseline`, and omit that flag for a new
baseline file. The check compares the two files and does not open `reference`,
so a reviewer confirms the approval there. No CI step runs the check yet.

It exits `0` when the change follows the policy and `1` when a frozen baseline
changed without a new approval from its owner. It exits `2` when a file cannot
be read, is not a JSON object, or has a malformed `_policy` block.

Update mode takes only `--baseline` and `--previous-baseline`, and `perf` and
`eval` modes do not take `--previous-baseline`. A flag given to a mode that
does not take it is a usage error (exit `2`).

```bash
git show origin/main:<baseline path> > /tmp/previous.baseline.json
python3 scripts/vllm/benchmarking/check_regression.py \
  --mode update \
  --previous-baseline /tmp/previous.baseline.json \
  --baseline <baseline path>
```

## Accuracy Evaluation (`lm_eval`)

### Tasks

| Task | Enabled by | Endpoint | Gated metric |
|------|------------|----------|--------------|
| `mmlu_llama`, `mmlu_pro` | `--run-lm-eval` | chat completions | `exact_match` |
| `humaneval_plus_tpu`, `mbpp_plus_tpu` | `--run-code-eval` (nightly only) | chat completions | pass@1 |

- `--run-lm-eval` walks `LM_EVAL_TASKS` (default `mmlu_llama mmlu_pro`; configs
  may narrow it).
- The code tasks use the full EvalPlus datasets, with evalplus-style instruct
  prompting and code-block extraction (see `lm_eval_tasks/`).
- Each task gates only when its baseline file exists.

### Tuning the `lm_eval` invocation

Two config vars tune the `lm_eval` invocation, for matching an external
reference run:

- `EXTRA_LM_EVAL_MODEL_ARGS`: `key=value` pairs merged into `--model_args`,
  last-wins per key.
- `LM_EVAL_GEN_KWARGS`: *replaces* the per-task `--gen_kwargs`, which is a
  merging action in lm-eval.

A generation cap goes in `LM_EVAL_GEN_KWARGS` as `max_gen_toks`: the
`--model_args` value is only a fallback for tasks whose yaml omits the key, and
`mmlu_pro` sets 2048 itself.

`LM_EVAL_GEN_KWARGS` takes either form lm-eval accepts, but not a mix:

- **JSON**, with double quotes: `{"chat_template_kwargs": {"thinking": false}}`.
  Any value containing `{` must be valid JSON, since lm-eval raises rather than
  falling back to its `key=value` parser once it sees a brace.
- **`key=value`**: `until=['<|im_end|>']`.

> [!TIP]
> **Suppressing reasoning is per-template, not universal.** The harness default
> sends `chat_template_kwargs: {"enable_thinking": false}`; a template keyed on
> a different flag ignores it silently and the model reasons anyway — Kimi-K2.6
> wants `thinking`. Check the checkpoint's `chat_template.jinja` before assuming
> the default lands.

## Speculative Decoding (`run_benchmarks.sh --mode spec_bench`)

`run_benchmarks.sh` supports two runner modes via `--mode serving|spec_bench`
(or `BENCHMARK_MODE=serving|spec_bench`; distinct from `check_regression.py
--mode perf|eval|update` above). In `spec_bench` mode, the runner sweeps batch
size (`SPEC_BS_LIST`) × speculative draft tokens (`SPEC_K_LIST`) on the
[Spec-Bench](https://github.com/hemingkx/Spec-Bench) dataset:

1. Restarts `vllm serve` per `(BS, K)` cell with `--max-num-seqs <BS>` and
   `--speculative-config` (`K=0` starts without `--speculative-config` as the
   no-speculation baseline).
2. Runs `SPEC_WARMUP_PROMPTS` warmup prompts and `SPEC_REPS` measured passes of
   `vllm bench serve --dataset-name spec_bench --max-concurrency <BS>`.
3. Saves per-cell results to `bs<BS>/k<K>_rep<N>.json` and aggregates mean E2E
   time, speedup vs. `K=0`, and acceptance rate via `summarize_spec_bench.py`
   into `spec_benchmark_summary.json` and `spec_benchmark_summary.md`.

```bash
./scripts/vllm/benchmarking/run_benchmarks.sh --config qwen3-0.6b-ngram
```

### Spec-Bench Configs

| Config | Model | Method | Draft | K swept | Topology |
|--------|-------|--------|-------|---------|----------|
| `qwen3-0.6b-ngram` | Qwen3-0.6B | `ngram` | - | `0 1 2 4` | TP=1 (smoke) |
| `llama-3-8b-ngram` | Meta-Llama-3-8B-Instruct | `ngram` | - | `0 1 2 4 8` | TP=1 |
| `llama-3.3-70b-eagle3` | Llama-3.3-70B-Instruct | `eagle3` | `Llama-3.3-70B-Instruct-Eagle3` | `0 1 2 3 4` | TP=8 |
| `gemma-4-31b-it-mtp` | gemma-4-31B-it | `mtp` | `gemma-4-31B-it-assistant` | `0 1 2 3 4 5` | TP=8 |
| `qwen3.5-397b-fp8-mtp` | Qwen3.5-397B-A17B-FP8 | `mtp` | built-in MTP head | `0 1 2 3 4 5` | TP=8, EP |
| `qwen3-4b-dflash` | Qwen3-4B | `dflash` | `Qwen3-4B-DFlash-b16` | `0 15` | TP=1 |
| `qwen3-4b-dspark` | Qwen3-4B | `dspark` | `dspark_qwen3_4b_block7` | `0 7` | TP=1 |
| `qwen3.5-35b-fp8-dflash` | Qwen3.5-35B-A3B-FP8 | `dflash` | `Qwen3.5-35B-A3B-DFlash` | `0 15` | DP=2 × TP=4, EP |

## Prerequisites

- vLLM installed (provides `benchmark_serving.py`)
- `vllm_torchtpu` installed (this repo)
- TPU device available
- Model weights downloaded (or use `--load-format dummy` via config)

## Results

Results are saved to `benchmark_runs/` (gitignored). Each run creates a directory containing:

- `config.json` — benchmark parameters
- `isl<N>_osl<N>_c<N>.json` — per-combo results from benchmark_serving.py (`serving` mode)
- `bs<BS>/k<K>_rep<N>.json`, `spec_benchmark_summary.json`, `spec_benchmark_summary.md` — per-cell results and aggregated speedup/acceptance tables (`spec_bench` mode)
- `server.log` — vLLM server output
- `benchmark.log` — run metadata

`publish_perf_dashboard.py` sends a results directory's
`isl<N>_osl<N>_c<N>.json` files to the Performance tab of
[ci.vllm.ai](https://ci.vllm.ai/perf), next to the GPU runs of the same model.
A perf step opts in with `UPLOAD_TO_VLLM_CI: "1"` in its `env`;
`run_eval_flow.sh` then publishes that step's own results after the database
upload, and skips it on PRs and non-main branches like that upload. `--dry-run`
prints the rows without posting them.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `BENCHMARK_MODE` | `serving` | Runner mode: `serving` or `spec_bench` (also `--mode`) |
| `BENCHMARK_WARMUP_RUNS` | `0` | Full benchmark passes to discard before writing each gated result |
| `EVAL_SOFT_FAIL` | `0` | `1` downgrades an eval regression to a `::warning` instead of failing the step. Perf regressions still fail. For a config whose eval baseline is new and whose run-to-run spread is not yet known — prefer it to `soft_fail` on the Buildkite step, which would mask the perf gate too. Forwarded into the pod by `.buildkite/kubernetes/common.sh` |
| `RANGE_RATIO_STYLE` (config var) | `symmetric` | How `RANDOM_RANGE_RATIO` is interpreted: `symmetric` = vllm bench serve native `[(1-r)L, (1+r)L]`; `min` = benchmark_serving.py-style `[rL, L]`, translated for vllm bench serve |
| `SPEC_METHOD` (config var) | (empty) | Speculative decoding method (`ngram`, `eagle3`, `mtp`, `dflash`, `dspark`); required when any `K > 0` |
| `SPEC_DRAFT_MODEL` / `SPEC_DRAFT_TP` (config var) | (empty) / `TENSOR_PARALLELISM` | Draft model and its TP size for methods that use a separate drafter |
| `SPEC_K_LIST` (config var) | `0 1 2 3 4 5` | Space-separated non-negative integer `K` values to sweep (`0` = baseline) |
| `SPEC_BS_LIST` (config var) | `1 4 8` | Space-separated positive integer batch sizes (`--max-num-seqs` and `--max-concurrency`) |
| `SPEC_BENCH_DATASET` / `SPEC_BENCH_CATEGORY` (config var) | auto-download / `summarization` | Path to Spec-Bench `question.jsonl` and category filter |
| `SPEC_NUM_PROMPTS` / `SPEC_WARMUP_PROMPTS` / `SPEC_REPS` (config var) | `100` / `20` / `2` | Prompts per measured pass, warmup prompts per cell, and measured repetitions per cell |
| `SERVED_MODEL_NAME` (config var) | (empty) | Optional `--served-model-name` passed in `spec_bench` mode |
