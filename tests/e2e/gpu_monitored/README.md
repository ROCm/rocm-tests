# GPU Monitored Tests

Each test runs a GPU stress workload under continuous `amd-smi monitor` sampling,
then applies a **5-layer validation**, monitoring-evidence gates, post-run
analysis, and HTML report generation.

## Layout

| Path | Role |
|------|------|
| `tests/common/gpu_monitored/` | Shared library (monitoring, validation, orchestrator, workloads) |
| `tests/e2e/gpu_monitored/` | Pytest entry points (one file per workload) |
| `tests/e2e/rvs/` | Session fixtures for RVS binary discovery / optional source build |

Logic lives under `tests/common/gpu_monitored/workloads/` and is orchestrated by
`MonitoredTestOrchestrator`.

## Framework integration

These tests follow rocm-tests conventions with one deliberate execution model:

| Framework rule | How gpu_monitored complies |
|----------------|---------------------------|
| ``target_executor`` for GPU workloads | ✅ Workloads via ``RunContext.target_executor`` |
| Monitor without ROCR mask | ✅ ``gpu_monitored_monitor_executor`` (framework pattern) |
| ``@pytest.mark.gpu_count("ALL")`` | ✅ Whole-node acquisition on every test |
| ``hw.multi_gpu`` profile | ✅ Category profile + markers |

### Execution model

| Component | Executor |
|-----------|----------|
| GPU workloads | ``target_executor`` (`NodeExecutorGroup`` → ``LocalExecutor`` / ``SshExecutor`` with acquired GPU indices) |
| ``amd-smi monitor`` / dmesg / pretest | ``gpu_monitored_monitor_executor`` (``CpuExecutor`` locally; unmasked ``SshExecutor`` remotely) |

Tests are profiled as ``hw.multi_gpu`` with ``@pytest.mark.gpu_count("ALL")`` so
``target_executor`` reserves every GPU on the node.  Do not set visibility masks
manually — the framework injects ``ROCR_VISIBLE_DEVICES`` for the acquired set.

RVS ``stdout_file`` workload output uses ``run_command_redirect`` via the
framework executor (shell redirect to ``console.log``); everything else
routes through framework executors directly.

## Tests (5)

| Pytest name | Workload |
|-------------|----------|
| `test_gpu_cudamemtest_monitored` | cuda_memtest sub-tests 0–5 |
| `test_gpu_transferbench_monitored` | TransferBench rsweep |
| `test_gpu_rvs_iet_stress_monitored` | RVS IET |
| `test_gpu_rvs_tst_monitored` | RVS TST |
| `test_gpu_hipblaslt_bench_monitored` | hipBLASLt GEMM sweep |

## Running

```bash
# From repo root — requires --rock-dir (or ROCK_DIR / rocm-test.toml)
pytest tests/e2e/gpu_monitored/test_gpu_rvs_tst_monitored.py -v --rock-dir=/opt/rocm

# Collect without GPU (framework lint)
pytest tests/e2e/gpu_monitored/ --collect-only -q --no-gpu

# All monitored tests (long — soak tests included)
pytest tests/e2e/gpu_monitored/ -v --rock-dir=/opt/rocm
```

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ROCM_TEST_RVS_REF` | Git ref when building RVS from source |
| `GPU_MONITOR_INTERVAL` | amd-smi sample interval (default 1s) |
| `GPU_MONITOR_STRICT_PRETEST` | Fail on dirty pre-test dmesg when `1` |
| `CUDAMEMTEST_DURATION` | cudamemtest time budget (default 1800s) |
| `CUDAMEMTEST_MAX_BLOCKS` | Override cuda_memtest memory sizing |

RVS config resolution uses the vendored `rvs_config_mapping.csv`.
When RVS is built from source, session fixtures export `ROCM_TEST_RVS_BIN`,
`ROCM_TEST_RVS_CONF_ROOT`, and `ROCM_TEST_TRANSFERBENCH_BIN`.

## Artifacts

Per test under `output/artifacts/gpu_monitored/<test_name>/`:

- `power_temp.csv` — monitoring telemetry
- `console.log` — workload output
- `summary.json`, `health_checks.txt`, `report.html`
- `pretest_health.json`, `dmesg.log`
