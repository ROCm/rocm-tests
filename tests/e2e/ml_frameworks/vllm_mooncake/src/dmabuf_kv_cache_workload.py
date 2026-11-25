# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Disaggregated vLLM + Mooncake KV-cache workload under VRAM pressure.

Prefill and decode vLLM instances exchange KV blocks through MooncakeConnector, which registers GPU memory
with ``ibv_reg_dmabuf_mr`` on ROCm. While requests run, side tensors lift every GPU to a target VRAM level and
the HIP and amdgpu VRAM counters are sampled. A JSON summary is printed between sentinels; the caller judges it.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from dataclasses import dataclass, field
import json
import os
import pathlib
import random
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

import httpx
import torch

GIB = 1 << 30
SUMMARY_START = "__DMABUF_KV_SUMMARY_START__"
SUMMARY_END = "__DMABUF_KV_SUMMARY_END__"

KERNEL_VRAM_USED = "/sys/bus/pci/devices/{pci}/mem_info_vram_used"
OOM_PATTERN = re.compile(r"out of memory|OutOfMemoryError|hipErrorOutOfMemory", re.IGNORECASE)
DMABUF_PATTERN = re.compile(r"Failed to register memory|ibv_reg_dmabuf_mr.*(fail|error)", re.IGNORECASE)
ERROR_LINE_PATTERN = re.compile(r"Traceback|Error|Exception|Failed|FATAL")

PRODUCER = "kv_producer"
CONSUMER = "kv_consumer"
PREFILL_PORT_BASE = 8010
DECODE_PORT_BASE = 8020
BOOTSTRAP_PORT_BASE = 8998
LOCALHOST = "127.0.0.1"

FAKE_OOM_MARGIN = GIB
PRESSURE_CHUNK = GIB
MIN_PRESSURE_CHUNK = 64 << 20
MONITOR_INTERVAL_S = 2.0
HEALTH_POLL_S = 5.0
HTTP_PROBE_TIMEOUT_S = 5.0
BOOTSTRAP_RETRIES = 12
SETTLE_TIMEOUT_S = 60.0
TERMINATE_TIMEOUT_S = 60.0
LOG_TAIL_LINES = 40
MAX_EXCERPT_LINES = 80
MAX_LOG_HITS = 5
MAX_ERROR_CHARS = 300

TOKEN_ID_RANGE = (1000, 30000)
WARMUP_SEQ_LEN = 512
WARMUP_OUTPUT_TOKENS = 8


@dataclass
class Server:
    role: str
    port: int
    bootstrap_port: int
    gpus: list[int]
    log_path: pathlib.Path
    proc: subprocess.Popen | None = None
    engine_id: str | None = None

    @property
    def url(self) -> str:
        return f"http://{LOCALHOST}:{self.port}"

    @property
    def bootstrap_addr(self) -> str:
        return f"http://{LOCALHOST}:{self.bootstrap_port}"


@dataclass
class VramSample:
    gpu: int
    hip_used: int
    kernel_used: int
    free: int
    total: int

    @property
    def drift(self) -> float:
        return abs(self.hip_used - self.kernel_used) / self.total


@dataclass
class PhaseResult:
    seq_len: int
    batch: int
    completed: int = 0
    errors: list[str] = field(default_factory=list)


def pci_address(gpu: int) -> str:
    props = torch.cuda.get_device_properties(gpu)
    return f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"


class VramMonitor(threading.Thread):
    """Samples HIP and amdgpu VRAM usage on every GPU until stopped."""

    def __init__(self, pci_addrs: list[str]):
        super().__init__(daemon=True)
        self._pci_addrs = pci_addrs
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self.samples: list[VramSample] = []

    def snapshot(self) -> list[VramSample]:
        result = []
        for gpu, pci in enumerate(self._pci_addrs):
            free, total = torch.cuda.mem_get_info(gpu)
            kernel_used = int(pathlib.Path(KERNEL_VRAM_USED.format(pci=pci)).read_text())
            result.append(VramSample(gpu, total - free, kernel_used, free, total))
        return result

    @contextlib.contextmanager
    def quiesce(self):
        """Pause sampling so a deliberate allocation is not read as counter drift."""
        with self._lock:
            yield

    def run(self) -> None:
        while not self._halt.wait(MONITOR_INTERVAL_S):
            with self._lock:
                self.samples.extend(self.snapshot())

    def stop(self) -> None:
        self._halt.set()
        self.join()


def plan_topology(gpu_count: int, tp: int, log_dir: pathlib.Path) -> tuple[list[Server], list[Server]]:
    """Split GPUs evenly into prefill and decode instances of ``tp`` GPUs each."""
    per_role = gpu_count // (2 * tp)
    if per_role < 1:
        raise SystemExit(f"{gpu_count} GPUs cannot host prefill and decode instances at tensor-parallel {tp}")
    prefills = [
        Server(
            PRODUCER,
            PREFILL_PORT_BASE + i,
            BOOTSTRAP_PORT_BASE + i,
            list(range(i * tp, (i + 1) * tp)),
            log_dir / f"prefill{i}.log",
        )
        for i in range(per_role)
    ]
    first_decode_gpu = per_role * tp
    decodes = [
        Server(
            CONSUMER,
            DECODE_PORT_BASE + i,
            0,
            list(range(first_decode_gpu + i * tp, first_decode_gpu + (i + 1) * tp)),
            log_dir / f"decode{i}.log",
        )
        for i in range(per_role)
    ]
    return prefills, decodes


def launch(server: Server, args: argparse.Namespace) -> None:
    devices = ",".join(str(g) for g in server.gpus)
    env = {**os.environ, "HIP_VISIBLE_DEVICES": devices, "CUDA_VISIBLE_DEVICES": devices}
    if server.role == PRODUCER:
        env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(server.bootstrap_port)
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--port",
        str(server.port),
        "--tensor-parallel-size",
        str(args.tp),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        "bfloat16",
        "--kv-transfer-config",
        json.dumps({"kv_connector": "MooncakeConnector", "kv_role": server.role}),
    ]
    with server.log_path.open("w") as log:
        server.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def diagnostics(server: Server) -> str:
    """Return every error-bearing line of the server log, then its final lines."""
    lines = server.log_path.read_text(errors="replace").splitlines()
    hits = [ln[:MAX_ERROR_CHARS] for ln in lines if ERROR_LINE_PATTERN.search(ln)][:MAX_EXCERPT_LINES]
    tail = "\n".join(lines[-LOG_TAIL_LINES:])
    return f"error lines in {server.log_path}:\n" + "\n".join(hits) + f"\n--- tail ---\n{tail}"


def wait_ready(server: Server, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=HTTP_PROBE_TIMEOUT_S) as client:
        while time.monotonic() < deadline:
            if server.proc.poll() is not None:
                raise RuntimeError(
                    f"{server.role} on port {server.port} exited ({server.proc.returncode}):\n{diagnostics(server)}"
                )
            with contextlib.suppress(httpx.HTTPError):
                if client.get(f"{server.url}/health").status_code == httpx.codes.OK:
                    return
            time.sleep(HEALTH_POLL_S)
    raise TimeoutError(f"{server.role} on port {server.port} not healthy after {timeout:.0f}s:\n{diagnostics(server)}")


def fetch_engine_id(server: Server) -> None:
    """Read the prefill engine id that decode requests must name as their KV source."""
    with httpx.Client(timeout=HTTP_PROBE_TIMEOUT_S) as client:
        for _ in range(BOOTSTRAP_RETRIES):
            with contextlib.suppress(httpx.HTTPError, KeyError, ValueError):
                server.engine_id = client.get(f"{server.bootstrap_addr}/query").json()["0"]["engine_id"]
                return
            time.sleep(HEALTH_POLL_S)
    raise RuntimeError(f"no Mooncake bootstrap entry on {server.bootstrap_addr}:\n{diagnostics(server)}")


def terminate(server: Server) -> None:
    if server.proc is None or server.proc.poll() is not None:
        return
    os.killpg(server.proc.pid, signal.SIGTERM)
    try:
        server.proc.wait(TERMINATE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        os.killpg(server.proc.pid, signal.SIGKILL)
        server.proc.wait()


def scan_logs(servers: list[Server], pattern: re.Pattern) -> list[str]:
    hits = []
    for server in servers:
        found = [ln for ln in server.log_path.read_text(errors="replace").splitlines() if pattern.search(ln)]
        hits.extend(f"{server.log_path.name}: {ln.strip()[:MAX_ERROR_CHARS]}" for ln in found[:MAX_LOG_HITS])
    return hits


async def send_request(client, model, prefill, decode, prompt, output_tokens, request_id) -> str | None:
    """Run one prefill/decode pair; return an error string or None on success."""
    transfer_id = f"xfer-{request_id}"
    common = {"model": model, "prompt": prompt, "temperature": 0.0, "ignore_eos": True, "stream": False}
    prefill_body = {
        **common,
        "max_tokens": 1,
        "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False, "transfer_id": transfer_id},
    }
    decode_body = {
        **common,
        "max_tokens": output_tokens,
        "kv_transfer_params": {
            "do_remote_decode": False,
            "do_remote_prefill": True,
            "remote_bootstrap_addr": prefill.bootstrap_addr,
            "remote_engine_id": prefill.engine_id,
            "transfer_id": transfer_id,
        },
    }
    headers = {"X-Request-Id": request_id}
    replies = await asyncio.gather(
        client.post(f"{prefill.url}/v1/completions", json=prefill_body, headers=headers),
        client.post(f"{decode.url}/v1/completions", json=decode_body, headers=headers),
        return_exceptions=True,
    )
    for name, reply in zip(("prefill", "decode"), replies, strict=True):
        if isinstance(reply, Exception):
            return f"{name}: {type(reply).__name__}: {reply}"[:MAX_ERROR_CHARS]
        if reply.status_code != httpx.codes.OK:
            return f"{name}: HTTP {reply.status_code}: {reply.text[:MAX_ERROR_CHARS]}"
    generated = replies[1].json()["usage"]["completion_tokens"]
    if generated != output_tokens:
        return f"decode generated {generated} of {output_tokens} tokens"
    return None


async def run_phase(args, prefills, decodes, seq_len, batch, rng, tag) -> PhaseResult:
    result = PhaseResult(seq_len, batch)
    prompt_len = seq_len - args.output_tokens
    async with httpx.AsyncClient(timeout=None, limits=httpx.Limits(max_connections=None)) as client:
        tasks = []
        for i in range(batch):
            prompt = [rng.randint(*TOKEN_ID_RANGE) for _ in range(prompt_len)]
            tasks.append(
                send_request(
                    client,
                    args.model,
                    prefills[i % len(prefills)],
                    decodes[i % len(decodes)],
                    prompt,
                    args.output_tokens,
                    f"{tag}-{i}",
                )
            )
        outcomes = await asyncio.wait_for(asyncio.gather(*tasks), args.request_timeout)
    result.errors = [o for o in outcomes if o]
    result.completed = batch - len(result.errors)
    return result


def inject_pressure(gpu_count: int, target: float, monitor: VramMonitor) -> tuple[list, list[dict]]:
    """Fill each GPU with side tensors up to ``target`` of its VRAM; flag allocations refused while memory is free."""
    holders = []
    report = []
    with monitor.quiesce():
        for gpu in range(gpu_count):
            pattern = gpu % 250 + 1
            tensors = []
            fake_oom = False
            while True:
                free, total = torch.cuda.mem_get_info(gpu)
                size = min(PRESSURE_CHUNK, int(total * target) - (total - free))
                if size < MIN_PRESSURE_CHUNK:
                    break
                try:
                    tensor = torch.empty(size, dtype=torch.uint8, device=f"cuda:{gpu}")
                except torch.cuda.OutOfMemoryError:
                    fake_oom = torch.cuda.mem_get_info(gpu)[0] >= size + FAKE_OOM_MARGIN
                    break
                tensor.fill_(pattern)
                tensors.append(tensor)
            free, total = torch.cuda.mem_get_info(gpu)
            holders.append((pattern, tensors))
            report.append({"gpu": gpu, "used_frac": (total - free) / total, "fake_oom": fake_oom})
    return holders, report


def verify_pressure(holders: list, report: list[dict]) -> None:
    for entry, (pattern, tensors) in zip(report, holders, strict=True):
        entry["intact"] = all(bool(torch.all(t == pattern)) for t in tensors)


def wait_for_baseline(monitor: VramMonitor, baseline: list[VramSample], limit: float) -> float:
    """Return the worst post-run used-VRAM deviation from baseline once it settles below ``limit``."""
    deadline = time.monotonic() + SETTLE_TIMEOUT_S
    while True:
        now = monitor.snapshot()
        leak = max(abs(n.hip_used - b.hip_used) / b.total for n, b in zip(now, baseline, strict=True))
        if leak < limit or time.monotonic() >= deadline:
            return leak
        time.sleep(MONITOR_INTERVAL_S)


def parse_phases(text: str) -> list[tuple[int, int]]:
    return [tuple(int(v) for v in item.split("x")) for item in text.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--phases", required=True, help="comma-separated SEQ_LENxBATCH items")
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--pressure-target", type=float, required=True)
    parser.add_argument("--drift-limit", type=float, required=True)
    parser.add_argument("--startup-timeout", type=float, required=True)
    parser.add_argument("--request-timeout", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=pathlib.Path)
    args = parser.parse_args()
    args.phase_list = parse_phases(args.phases)
    args.max_model_len = max(seq for seq, _ in args.phase_list)
    return args


def main() -> int:
    args = parse_args()
    gpu_count = torch.cuda.device_count()
    log_dir = args.log_dir or pathlib.Path(tempfile.mkdtemp(prefix="vllm_mooncake_"))
    log_dir.mkdir(parents=True, exist_ok=True)
    prefills, decodes = plan_topology(gpu_count, args.tp, log_dir)
    servers = prefills + decodes

    monitor = VramMonitor([pci_address(g) for g in range(gpu_count)])
    baseline = monitor.snapshot()
    rng = random.Random(args.seed)
    phases: list[PhaseResult] = []
    holders: list = []
    pressure: list[dict] = []

    if not os.path.isdir(args.model):
        from huggingface_hub import snapshot_download

        snapshot_download(args.model, ignore_patterns=["original/*"])

    monitor.start()
    try:
        for server in servers:
            launch(server, args)
        for server in servers:
            wait_ready(server, args.startup_timeout)
        for server in prefills:
            fetch_engine_id(server)

        warmup = asyncio.run(
            run_phase(
                argparse.Namespace(**{**vars(args), "output_tokens": WARMUP_OUTPUT_TOKENS}),
                prefills,
                decodes,
                WARMUP_SEQ_LEN,
                len(decodes),
                rng,
                "warmup",
            )
        )
        if warmup.errors:
            raise RuntimeError(f"warm-up request failed: {warmup.errors[0]}")

        holders, pressure = inject_pressure(gpu_count, args.pressure_target, monitor)
        for index, (seq_len, batch) in enumerate(args.phase_list):
            phases.append(asyncio.run(run_phase(args, prefills, decodes, seq_len, batch, rng, f"p{index}")))
        verify_pressure(holders, pressure)
        with monitor.quiesce():
            monitor.samples.extend(monitor.snapshot())
    except BaseException:
        for server in servers:
            if server.log_path.exists():
                print(f"--- {server.role} on port {server.port} ---\n{diagnostics(server)}", file=sys.stderr)
        print(f"logs kept in {log_dir}", file=sys.stderr)
        raise
    finally:
        monitor.stop()
        holders.clear()
        for gpu in range(gpu_count):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        for server in servers:
            terminate(server)

    leak = wait_for_baseline(monitor, baseline, args.drift_limit)
    summary = {
        "model": args.model,
        "gpu_count": gpu_count,
        "topology": {"prefill": [s.gpus for s in prefills], "decode": [s.gpus for s in decodes]},
        "phases": [vars(p) for p in phases],
        "pressure": pressure,
        "accounting": {
            "samples": len(monitor.samples),
            "max_drift_frac": max(s.drift for s in monitor.samples),
            "min_free_bytes": min(s.free for s in monitor.samples),
            "leak_frac": leak,
        },
        "log_errors": {"oom": scan_logs(servers, OOM_PATTERN), "dmabuf": scan_logs(servers, DMABUF_PATTERN)},
        "log_dir": str(log_dir),
    }
    print(f"{SUMMARY_START}\n{json.dumps(summary)}\n{SUMMARY_END}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
