# Roofline Model and Hardware Onboarding

The `vllm_torchtpu.roofline` package computes Layer-1 (L1) physical lower
bounds and uncalibrated serial-phase candidate shortlists directly from a
Hugging Face `config.json` (`ModelGeometry` and `Dtypes`) and a TPU hardware
profile (`ChipSpec` and `SliceTopology`).

> **L1 Shortlist Scope (`bound_*` Pruning vs. `est_*` Ranking):**
> `solve` and `emit` run on CPU with zero TPU allocation. Lower bounds
> (`bound_ttft_ms`, `bound_tpot_ms`, `floor_throughput_per_chip`) depend only
> on hardware specs, HF model dimensions, and traffic inputs. Uncalibrated
> estimates (`est_ttft_ms`, `est_tpot_ms`) and layout rankings shortlist
> candidates (`bound_families_by_concurrency`) before silicon benchmarking.
> Supplying optional flags (`--hbm-reserve-gb`, `--collective-latency-us`,
> `--moe-activation hull:<path>`, dtype overrides) records explicit entries in
> `provenance.supplied_values` without altering uncalibrated defaults when
> omitted.

## 1. How to Onboard a New Model and Fail-Closed Contract

### Strict Fail-Closed Policy and Canonical 1:1 HF Identity

`vllm_torchtpu.roofline` rejects unsupported models instead of extrapolating
from partial config fields. All public entrypoints (`supports`,
`from_hf_config`, `audit_hf_config`, `resolve_model_identity`,
`load_hf_config_with_identity`, and CLI subcommands `inspect`, `solve`,
`emit`, `ceiling`, `plan`, and `explain`) fail closed with
`UnsupportedArchitectureError` (CLI exit code `2`) whenever a config:

- uses an unlisted or omitted `model_type`;
- declares mismatched `architectures` or a non-causal draft model
  (`is_causal: false`);
- activates an unmodeled structural control (for example, `ple_layer_ids` on
  `qwen4_exp`, `o_lora_rank > 0`, or unmodeled attention/SSM layer types).

No bypass flag exists. Internally, `audit_hf_config` runs a private dry-run
parse solely to report `unmapped_structural_keys` on blocked configs while
keeping `onboarding_tier != "tier0_ready"` and `geometry = None`.

### Currently Supported Canonical Models

The canonical registry (`SUPPORTED_MODELS` in `constants.py`) maps each
supported model ID 1:1 to public Hugging Face repositories, `(model_type,
architectures)` pairs, and allowed layer kinds:

| Canonical ID | Public HF Repositories | Architecture & Quantization |
| :--- | :--- | :--- |
| `qwen3_5_moe` | `Qwen/Qwen3.5-397B-A17B-FP8`, `nvidia/Qwen3.5-397B-A17B-NVFP4` | Hybrid GDN + GQA + MoE + MTP (FP8, ModelOpt NVFP4) |
| `glm_moe_dsa` | `zai-org/GLM-5.2-FP8`, `nvidia/GLM-5.2-NVFP4` | MLA + DSA + IndexShare + MoE (FP8, ModelOpt NVFP4) |
| `qwen4_exp` | `Qwen/Qwen3.8-Flash-Next-FP8` | Bring-up only: pass `--hf-overrides '{"text_config":{"ple_layer_ids":[]}}'`, context `<= 2048`, `spec=0` |

### Step-by-Step Onboarding Workflow (`inspect --json`)

Run `inspect` against an authentic public Hugging Face repository or local
`config.json`:

```bash
python -m vllm_torchtpu.roofline inspect \
  --model <org/repo_or_local_dir> \
  [--hf-overrides '{"text_config":{"ple_layer_ids":[]}}'] \
  --json
```

`inspect` emits `vllm_torchtpu.roofline.inspect/v1` JSON (`schema_version`,
`chip`, `onboarding_tier`, `canonical_id`, `hf_repo`, `model_type`,
`model_type_status`, `layer_counts`, `param_counts`, `dtypes`,
`unmapped_structural_keys`, `ignored_non_text_keys`, and `blocking_reasons`),
exiting `0` only for `tier0_ready` and `2` otherwise:

| Tier | `onboarding_tier` | Required Action |
| :--- | :--- | :--- |
| Tier 0 | `tier0_ready` | Canonical identity matched and zero unmapped structural keys; run `solve` / `emit` / `plan` on CPU directly. |
| Tier 1 | `tier1_config_needed` | Primitives already exist; map config or quantization keys in `geometry.py` / `hf_loader.py` and register `ModelSpec` in `SUPPORTED_MODELS`. |
| Tier 2 | `tier2_primitive_needed` | Unmodeled attention, recurrence, FFN, or table primitive; implement through the 4-file primitive seam below. |

### Tier 2 Four-File Primitive Seam

When a new architecture introduces an unmodeled primitive, keep all math inside
four strictly downward modules:

1. `constants.py`: register canonical `ModelSpec` in `SUPPORTED_MODELS` and
   structural constants.
2. `geometry.py`: parse `config.json` fields, track consumed keys, enforce
   `_validate_family_layer_kinds`, and emit `ModelGeometry`.
3. `memory.py`: account for weights, KV cache (separating linear KV from fixed
   sliding-window KV), recurrent state, and `param_counts` conservation.
4. `modules.py`: implement per-layer HBM, matmul, vector, and collective costs
   (with both strict unpadded lower bounds and estimate terms).

Wire runtime legality and launch flags in `solver.py` and `emit.py` so emitted
commands stay aligned with `vllm_torchtpu` validators.

### Mandatory Gate to Promote a Model into `SUPPORTED_MODELS`

Every promoted model must pass four CPU and container test suites:

1. Check in the authentic public HF `config.json` under
   `tests/roofline/fixtures/`.
2. Pass `test_roofline_fixtures.py`: `resolve_model_identity`,
   `audit_hf_config` returning `tier0_ready` with zero
   `unmapped_structural_keys`, `param_counts` conservation, and spoofed-repo
   negative controls.
3. Pass L1 min-step pruning invariants (`bound <= est`, `DP=1 == DP=2`,
   zero-think and full-hit continuity, monotonicity, `x_cap_pool` pruning, and
   no false prunes) in `test_roofline_invariants.py`,
   `test_roofline_occupancy.py`, `test_roofline_step.py`, and
   `test_roofline_solver.py`.
4. Pass live `vllm_torchtpu` config validator parity in
   `test_roofline_runtime_parity.py`.

## 2. Runtime Flag Parity for Disaggregation and KV Offload

`emit` and `solve --from-serve-cmd` match `TpuPlatform.check_and_update_config`
and configure KV transfer exclusively via `--kv-transfer-config`:

- **Disaggregated serving**: emits `--kv-transfer-config` with
  `TPURaidenConnector` (`vllm_torchtpu.distributed.kv_transfer.tpu_connector`)
  using `kv_producer` on prefill and `kv_consumer` on decode.
- **Standalone host KV offload** (`--kv-offload-gb <GiB>`): emits
  `--kv-transfer-config` with `TPURaidenOffloadingConnector`
  (`vllm_torchtpu.offload.raiden_connector`), `kv_role="kv_both"`, and
  `kv_connector_extra_config={"cpu_bytes_to_use": <bytes>}`.
- **Combined disaggregation + offload**: wraps both connectors inside
  `TPUMultiConnector`
  (`vllm_torchtpu.distributed.kv_transfer.tpu_multi_connector`) with explicit
  per-child `kv_role` entries.
- **Hybrid models with offload**: prepends
  `TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1`, and uneven PP partitions prepend
  `VLLM_PP_LAYER_PARTITION=...` while requiring at least one full-attention
  layer per pipeline stage on Raiden hybrid deployments.

## 3. Zero-Code Hardware Onboarding (`--chip-spec` and `from_jax`)

Hardware profiles describe per-core compute, memory, interconnect, and mesh
capabilities (`cores_per_chip`, `chips_per_host`, `ici_links_per_chip`,
`fp4_flops`, `int8_flops`, `int4_flops`, `vector_flops`,
`smem_page_table_bytes`, `physical_alignment_elems`,
`subchannel_block_k_floor`, `min_torus_dim`, `has_sparse_core`, and
`dma_granule_bytes`).

### Live JAX Hydration (`from_jax`)

When running in a JAX-enabled environment, `from_jax(name, chip_version,
overrides)` queries `jax.experimental.pallas.tpu.get_tpu_info_for_chip` for the
requested `pltpu.ChipVersion` to hydrate core count, tiling, and memory traits
before applying optional `ChipOverrides`.

### Declarative JSON Profile (`vllm_torchtpu.roofline.chip_spec/v1`)

To evaluate projected or custom chips without code edits, pass
`--chip-spec <path.json>`. Required fields are `schema_version`, `name`, and
`provenance` (`"public_spec"` or `"projected"`).

- **With `"base"`** (for example `"base": "tpu7x"`): omitted fields inherit
  from the base `ChipSpec`. Changing `ici_links_per_chip` or `ici_link_bw`
  recomputes both `ici_bw` and `d2d_bw` unless overridden; changing
  `cores_per_chip` recomputes `ici_bw` while leaving `d2d_bw` unchanged.
  Changing `dcn_host_gbit_per_s` or `chips_per_host` recomputes `dcn_bw`.
- **Without `"base"`** (standalone profile): `hbm_bytes`, `hbm_bw`,
  `bf16_flops`, `vmem_bytes`, `cores_per_chip`, and `chips_per_host` are
  required. Omitted `fp8_flops`, `int8_flops`, `int4_flops`, and `fp4_flops`
  default to `0.0`; omitted ICI and DCN fields use the package defaults; and
  omitted low-level capability fields fall back to `_DEFAULT_TPU_CAPS`
  (initialized once from `TPU_7X` at import time).

```json
{
  "schema_version": "vllm_torchtpu.roofline.chip_spec/v1",
  "provenance": "projected",
  "name": "example_1core_2d",
  "cores_per_chip": 1,
  "chips_per_host": 4,
  "hbm_bytes": 128000000000,
  "hbm_bw": 4000000000000,
  "bf16_flops": 2000000000000000,
  "fp8_flops": 4000000000000000,
  "vmem_bytes": 134217728,
  "ici_link_bw": 100000000000,
  "ici_links_per_chip": 4,
  "ici_hop_latency_s": 5e-7,
  "dcn_host_gbit_per_s": 800,
  "topologies": [
    {
      "name": "2x2",
      "chip_axes": [2, 2, 1],
      "wrapped_axes": [false, false, false]
    }
  ]
}
```

Run `solve` with the profile on any supported canonical model:

```bash
python -m vllm_torchtpu.roofline solve \
  --model Qwen/Qwen3.5-397B-A17B-FP8 \
  --chip-spec example_1core_2d.json \
  --example-scenario coding-agent \
  --max-chips 4 \
  --json
```

## 4. Runtime Profiles (`ParallelRules` / `RuntimeProfile`) and Open-Loop Scenarios

### Runtime Admission Profiles (`--runtime-profile main` vs. `hardware`)

Each canonical `ModelSpec` pairs declarative `ParallelRules` with allowed KV
cache dtypes inside `RuntimeProfile` (`constants.py`). Pass
`--runtime-profile` across `inspect`, `solve`, `plan`, `emit`, `ceiling`, and
`explain`:

| Profile | Intended Scope | Parallel & Feature Gates | Admitted KV Dtypes |
| :--- | :--- | :--- | :--- |
| `main` (default) | Current runtime parity | Enforces per-model `ParallelRules` (for example prefill `DCP` mode, `PP`/`PCP` caps, `qwen4_exp` bring-up rules) | `"bf16"`, `"fp8"` |
| `hardware` | Unbridled L1 hardware ceiling | Uses `HARDWARE_PHYSICS_PROFILE`: lifts experimental bring-up overrides, prefill `DCP`/`PCP`/`PP` caps, and speculative draft limits | `"bf16"`, `"fp8"`, `"fp4"` |

When `runtime_profile="main"` is active, passing `--kv-dtype fp4` or
`--from-serve-cmd "... --kv-cache-dtype turboquant_4bit_nc"` is rejected cleanly
with exit code `2`. Switching to `--runtime-profile hardware` admits 4-bit KV
dtypes (emitting `--kv-cache-dtype fp4` in `emit` and `solve` launch payloads)
and unlocks unmodified `qwen4_exp` configs without `--hf-overrides`:

```bash
python -m vllm_torchtpu.roofline solve \
  --model Qwen/Qwen3.8-Flash-Next-FP8 \
  --runtime-profile hardware \
  --kv-dtype fp4 \
  --example-scenario coding-agent \
  --max-chips 16 \
  --json
```

### Open-Loop Scenarios (`think_time_s = None`)

Omitting `--think-time-s` in either custom `solve` invocations or
`--from-serve-cmd` / `--from-bench-cmd` workflows keeps
`scenario.think_time_s = None` (open-loop mode) instead of assuming zero pause
(`0.0 s`):

- **Open-loop (`--think-time-s` omitted, `think_time_s = None`)**: L1 pruning
  checks per-request `bound_ttft_ms`, batch-1 `bound_tpot_ms`, and HBM/SMEM
  capacity bounds without enforcing closed-loop saturation capacity floors.
- **Closed-loop (`--think-time-s <seconds>`)**: enables closed-loop Little's
  law minimum throughput demand and Tarski residency estimation for saturation
  benchmarks (`0.0`) or fixed client think pauses.
