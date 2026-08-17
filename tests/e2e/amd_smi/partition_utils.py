# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AMD SMI memory/compute partition orchestration for MI350X-class ASICs.

Drives ``amd-smi`` to switch memory and compute (accelerator) partitions,
reloads the amdgpu driver on memory changes, verifies the readback, runs GPU
workloads under each partition mode, and gates on dmesg faults. Focused on
MI350X POR values with scalable hooks for future ASICs.
"""

# pylint: disable=logging-fstring-interpolation,too-many-lines

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
import time

import yaml

WORKLOAD_CONFIG_PATH = Path(__file__).parent / "workload.yaml"

# Valid compute (accelerator) partition modes reported by amd-smi.
_VALID_COMPUTE_MODES = ["SPX", "DPX", "TPX", "QPX", "CPX"]


@dataclass
class _Outcome:
    """Terminal result of a partition test flow."""

    status: str  # "PASS", "FAIL", or "SKIP"
    message: str


@dataclass
class _TesterOp:
    """Command result exposing the fields the partition logic consumes."""

    exit_code: int
    output: str
    stderr: str = ""


def parse_test_filter(raw: str) -> dict[str, str]:
    """Parse a ``key=value,key=value`` filter string into a dict.

    Multiple workloads are separated with ``:`` inside a value, keeping ``,`` free
    as the pair delimiter.
    """
    result: dict[str, str] = {}
    for pair in (raw or "").split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result


class _AmdSmiRunner:
    """Thin command boundary: wraps an executor and applies sudo for privileged calls."""

    def __init__(self, executor, rock_dir: str | None):
        self.executor = executor
        self.rock_dir = rock_dir or ""
        self.amd_smi = f"{self.rock_dir}/bin/amd-smi" if self.rock_dir else "amd-smi"
        # rocm_agent_enumerator / rocminfo may not be on PATH; prefer the rock_dir copy.
        self.rocm_agent_enumerator = (
            f"{self.rock_dir}/bin/rocm_agent_enumerator" if self.rock_dir else "rocm_agent_enumerator"
        )
        self.rocminfo = f"{self.rock_dir}/bin/rocminfo" if self.rock_dir else "rocminfo"

    def run(  # pylint: disable=unused-argument
        self,
        cmd: str,
        privilege: bool = False,
        timeout: float | None = None,
        log_output: bool = True,
    ) -> _TesterOp:
        """Execute *cmd* (optionally under ``sudo -n``) and return a `_TesterOp`."""
        full = f"sudo -n {cmd}" if privilege else cmd
        result = self.executor.run(full, timeout=timeout)
        return _TesterOp(exit_code=result.exit_code, output=result.stdout, stderr=result.stderr)


class MemoryPartitionUtility:
    """Track GPU memory partition state via ``amd-smi partition --memory --csv``."""

    def __init__(self, runner: _AmdSmiRunner):
        self.logger = logging.getLogger(__name__)
        self._runner = runner
        self._partition_data: list[dict] | None = None
        self.refresh()

    def refresh(self) -> bool:
        """Refresh memory partition data. Returns True on success."""
        try:
            tester_op = self._runner.run("amd-smi partition --memory --csv", privilege=True, log_output=False)
            self.logger.debug(f"Memory partition CSV output:\n{tester_op.output}")
            return self._parse_csv_output(tester_op.output)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to fetch memory partition data: {err}")
            self._partition_data = None
            return False

    def _parse_csv_output(self, csv_output: str) -> bool:
        try:
            lines = csv_output.strip().split("\n")
            if len(lines) < 2:
                self.logger.error("Memory partition CSV output has insufficient data")
                return False

            header = self._parse_csv_line(lines[0])
            self._partition_data = []

            for line in lines[1:]:
                if not line.strip():
                    continue
                values = self._parse_csv_line(line)
                row = dict(zip(header, values, strict=False))
                gpu_id_str = row.get("gpu_id", "").strip()
                if not gpu_id_str:
                    continue
                self._partition_data.append(
                    {
                        "gpu_id": int(gpu_id_str),
                        "memory_partition_caps": row.get("memory_partition_caps", "").strip(),
                        "current_memory_partition": row.get("current_memory_partition", "").strip(),
                    }
                )

            self.logger.info(f"Memory partition data refreshed: {self._partition_data}")
            return True
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to parse memory partition CSV: {err}")
            self._partition_data = None
            return False

    @staticmethod
    def _parse_csv_line(line: str) -> list[str]:
        """Split a CSV line, honouring quoted fields that contain commas."""
        result: list[str] = []
        current_field = ""
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                result.append(current_field.strip().strip('"'))
                current_field = ""
            else:
                current_field += char
        result.append(current_field.strip().strip('"'))
        return result

    def get_gpu_count(self) -> int:
        if self._partition_data is None:
            return 0
        return len(self._partition_data)

    def get_all_current_partitions(self) -> dict:
        result = {}
        if self._partition_data is None:
            return result
        for gpu_data in self._partition_data:
            gpu_id = gpu_data.get("gpu_id")
            if gpu_id is not None:
                result[gpu_id] = gpu_data.get("current_memory_partition")
        return result

    def get_all_partition_caps(self) -> dict:
        result = {}
        if self._partition_data is None:
            return result
        for gpu_data in self._partition_data:
            gpu_id = gpu_data.get("gpu_id")
            if gpu_id is not None:
                caps_str = gpu_data.get("memory_partition_caps", "")
                result[gpu_id] = [cap.strip() for cap in caps_str.split(",")] if caps_str else []
        return result

    def __repr__(self) -> str:
        if self._partition_data is None:
            return "MemoryPartitionUtility(data=None)"
        return f"MemoryPartitionUtility(gpus={self.get_gpu_count()}, partitions={self.get_all_current_partitions()})"


class ComputePartitionUtility:
    """Track GPU compute partition profiles via ``amd-smi partition --accelerator --csv``.

    The current active profile is marked with ``*`` after the accelerator type
    (e.g. ``SPX*``); each profile lists its supported memory partitions.
    """

    def __init__(self, runner: _AmdSmiRunner):
        self.logger = logging.getLogger(__name__)
        self._runner = runner
        self._gpu_profiles: dict | None = None
        self._current_partitions: dict | None = None
        self.refresh()

    def refresh(self, current_mem_partitions: dict | None = None) -> bool:
        """Refresh compute partition data, reusing pre-fetched memory state when given."""
        try:
            if current_mem_partitions is None:
                current_mem_partitions = MemoryPartitionUtility(self._runner).get_all_current_partitions()
            tester_op = self._runner.run("amd-smi partition --accelerator --csv", privilege=True, log_output=False)
            self.logger.debug(f"Compute partition CSV output: {tester_op.output}")
            return self._parse_csv_output(tester_op.output, current_mem_partitions)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to fetch compute partition data: {err}")
            self._gpu_profiles = None
            self._current_partitions = None
            return False

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def _parse_csv_output(self, csv_output: str, current_mem_partitions: dict | None = None) -> bool:  # noqa: C901
        try:
            # Output may hold several CSVs separated by a blank line; only the
            # first (accelerator profiles) is needed here.
            first_csv = csv_output.strip().split("\n\n")[0].strip()
            lines = first_csv.split("\n")
            if len(lines) < 2:
                self.logger.error("CSV output has insufficient data")
                return False

            header = self._parse_csv_line(lines[0])
            self._gpu_profiles = {}
            self._current_partitions = {}
            current_gpu_id = None
            current_profile = None

            for line in lines[1:]:
                if not line.strip():
                    continue
                values = self._parse_csv_line(line)
                row = dict(zip(header, values, strict=False))

                gpu_id_str = row.get("gpu_id", "").strip()
                if gpu_id_str and gpu_id_str.upper() != "N/A":
                    parsed_gpu_id = self._safe_int(gpu_id_str)
                    if parsed_gpu_id is not None:
                        current_gpu_id = parsed_gpu_id
                        self._gpu_profiles.setdefault(current_gpu_id, [])

                if current_gpu_id is None:
                    continue

                profile_index_str = row.get("profile_index", "").strip()
                if profile_index_str and profile_index_str.upper() != "N/A":
                    profile_index = self._safe_int(profile_index_str)
                    if profile_index is None:
                        continue
                    accelerator_type = row.get("accelerator_type", "").strip()
                    if not accelerator_type or accelerator_type.upper() == "N/A":
                        continue

                    is_current = accelerator_type.endswith("*")
                    clean_accelerator_type = accelerator_type.rstrip("*")

                    mem_caps_str = row.get("memory_partition_caps", "").strip()
                    if mem_caps_str.upper() == "N/A":
                        mem_caps: list[str] = []
                    else:
                        mem_caps = [
                            cap.strip()
                            for cap in mem_caps_str.split(",")
                            if cap.strip() and cap.strip().upper() != "N/A"
                        ]

                    current_profile = {
                        "profile_index": profile_index,
                        "memory_partition_caps": mem_caps,
                        "accelerator_type": clean_accelerator_type,
                        "is_current": is_current,
                        "partition_id": row.get("partition_id", "N/A"),
                        "num_partitions": self._safe_int(row.get("num_partitions", "")),
                        "num_resources": self._safe_int(row.get("num_resources", "")),
                        "resources": [],
                    }
                    self._gpu_profiles[current_gpu_id].append(current_profile)

                    if is_current:
                        current_mem_partition = mem_caps[0] if mem_caps else None
                        if current_mem_partition is None and current_mem_partitions:
                            current_mem_partition = current_mem_partitions.get(current_gpu_id)
                        self._current_partitions[current_gpu_id] = {
                            "compute": clean_accelerator_type,
                            "memory": current_mem_partition,
                            "profile_index": profile_index,
                        }

                resource_type = row.get("resource_type", "").strip()
                if current_profile and resource_type and resource_type.upper() != "N/A":
                    current_profile["resources"].append(
                        {
                            "resource_index": self._safe_int(row.get("resource_index", "")),
                            "resource_type": resource_type,
                            "resource_instances": self._safe_int(row.get("resource_instances", "")),
                            "resources_shared": self._safe_int(row.get("resources_shared", "")),
                        }
                    )

            self.logger.info(f"Compute partition data refreshed: {len(self._gpu_profiles)} GPUs")
            return True
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to parse compute partition CSV: {err}")
            self._gpu_profiles = None
            self._current_partitions = None
            return False

    @staticmethod
    def _parse_csv_line(line: str) -> list[str]:
        result: list[str] = []
        current_field = ""
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == "," and not in_quotes:
                result.append(current_field.strip().strip('"'))
                current_field = ""
            else:
                current_field += char
        result.append(current_field.strip().strip('"'))
        return result

    @staticmethod
    def _safe_int(value: str) -> int | None:
        try:
            return int(value.strip()) if value.strip() and value.strip() != "N/A" else None
        except ValueError:
            return None

    def get_gpu_count(self) -> int:
        if self._gpu_profiles is None:
            return 0
        return len(self._gpu_profiles)

    def get_available_compute_partitions(self, gpu_id: int) -> list[str] | None:
        if self._gpu_profiles is None:
            return None
        profiles = self._gpu_profiles.get(gpu_id)
        if profiles is None:
            self.logger.warning(f"GPU ID {gpu_id} not found in partition data")
            return None
        return list({profile["accelerator_type"] for profile in profiles})

    def get_profiles_for_gpu(self, gpu_id: int) -> list[dict] | None:
        if self._gpu_profiles is None:
            return None
        return self._gpu_profiles.get(gpu_id)

    def get_memory_caps_for_compute_partition(self, gpu_id: int, compute_partition: str) -> list[str] | None:
        profiles = self.get_profiles_for_gpu(gpu_id)
        if profiles is None:
            return None
        for profile in profiles:
            if profile["accelerator_type"].upper() == compute_partition.upper():
                return profile["memory_partition_caps"]
        return None

    def get_all_current_compute_partitions(self) -> dict:
        result = {}
        if self._current_partitions is None:
            return result
        for gpu_id, partition_info in self._current_partitions.items():
            result[gpu_id] = partition_info.get("compute")
        return result

    def get_all_current_states(self) -> dict:
        if self._current_partitions is None:
            return {}
        return self._current_partitions.copy()

    def is_combination_supported(self, gpu_id: int, compute_partition: str, memory_partition: str) -> bool:
        mem_caps = self.get_memory_caps_for_compute_partition(gpu_id, compute_partition)
        if mem_caps is None:
            return False
        return memory_partition.upper() in [m.upper() for m in mem_caps]

    def __repr__(self) -> str:
        if self._gpu_profiles is None:
            return "ComputePartitionUtility(data=None)"
        return (
            f"ComputePartitionUtility(gpus={self.get_gpu_count()}, "
            f"partitions={self.get_all_current_compute_partitions()})"
        )


class AmdSmiMemoryPartition:  # pylint: disable=too-many-instance-attributes
    """Change memory partition using amd-smi and verify the result."""

    POR_PARTITION_MATRIX = {
        "MI350X": {"compute_partition": "DPX", "memory_partition": "NPS2"},
    }

    DMESG_FAULT_PATTERNS = ("page fault", "segfault")

    def __init__(self, executor, rock_dir: str | None, platform_name: str, test_filter: dict | None = None):
        self.test_name = "AMDSMI_memory_partition"
        self.logger = logging.getLogger(__name__)
        self.executor = executor
        self.platform_name = platform_name
        self._runner = _AmdSmiRunner(executor, rock_dir)
        self.test_filter = test_filter or {}
        self.memory_partition_util: MemoryPartitionUtility | None = None
        self.compute_partition_util: ComputePartitionUtility | None = None
        self._workload_jobs_cache: dict | None = None
        self._built_workloads: set = set()
        # Partition health gate: workloads are skipped after a failed partition set.
        self._partition_healthy = True
        self._last_partition_failure_label = ""

    # ------------------------------------------------------------------
    # Command boundary + reconstructed system helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: str, privilege: bool = False, timeout: float | None = None) -> _TesterOp:
        return self._runner.run(cmd, privilege=privilege, timeout=timeout)

    def _gpu_gfx_ids(self) -> list[str]:
        res = self._run(self._runner.rocm_agent_enumerator)
        ids = [ln.strip() for ln in (res.output or "").splitlines() if ln.strip()]
        gfx_ids = [gfx for gfx in ids if gfx and gfx != "gfx000"]
        # Fall back to the bare name if the rock_dir copy is absent / produced nothing.
        if not gfx_ids and self._runner.rocm_agent_enumerator != "rocm_agent_enumerator":
            res = self._run("rocm_agent_enumerator")
            ids = [ln.strip() for ln in (res.output or "").splitlines() if ln.strip()]
            gfx_ids = [gfx for gfx in ids if gfx and gfx != "gfx000"]
        return gfx_ids

    def _is_mi350(self) -> bool:
        ids = self._gpu_gfx_ids()
        return bool(ids) and ids[0] == "gfx950"

    def _is_mi355(self) -> bool:
        res = self._run(f"{self._runner.amd_smi} static --asic --json")
        return bool(re.search(r"MI355X", res.output or "", flags=re.IGNORECASE))

    def _get_asic_key(self) -> str | None:
        if self._is_mi350() or self._is_mi355():
            return "MI350X"
        return None

    def is_driver_loaded(self) -> bool:
        """True unless rocminfo reports the ROCk module is not loaded."""
        try:
            out = self._run(self._runner.rocminfo).output or ""
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.error(f"Failed to run rocminfo while checking driver status: {exc}")
            return False
        pattern = re.compile(r"\s*ROCk module is NOT (loaded|live),\s*possibly no GPU devices\s*", re.IGNORECASE)
        if pattern.search(out) or "Unable to open" in str(out):
            self.logger.error(out)
            return False
        return True

    def load_driver(self, timeout: int = 90) -> bool:
        """Load the amdgpu module and poll rocminfo until it reports loaded."""
        if self.is_driver_loaded():
            self.logger.warning("AMDGPU Driver is already loaded.")
            return True
        self.logger.info("Load AMDGPU Driver Begin")
        self._run("modprobe amdgpu", privilege=True)
        start_time = time.time()
        deadline = start_time + max(timeout, 0)
        while True:
            if self.is_driver_loaded():
                self.logger.info(f"Driver loaded in {int(time.time() - start_time)} seconds")
                return True
            if time.time() >= deadline:
                break
            time.sleep(1)
        self.logger.info(f"Driver did NOT load within {timeout} seconds")
        return False

    def get_compute_partition_type(self) -> str:
        """Read the current compute partition via ``amd-smi static --partition``."""
        partition_type = ""
        pattern = re.compile(rf"(?:ACCELERATOR|COMPUTE)_PARTITION:\s*({'|'.join(_VALID_COMPUTE_MODES)})")
        out = self._run(f"{self._runner.amd_smi} static --partition", privilege=True).output
        if out:
            found = pattern.findall(out)
            if found:
                partition_type = found[0]
        if not partition_type:
            self.logger.error(f"Partition_type not listed in valid partitions: {_VALID_COMPUTE_MODES}")
        return partition_type

    def get_memory_partition_type(self) -> str:
        """Read the current memory partition via ``amd-smi static --partition``."""
        result = ""
        pattern = re.compile(r"MEMORY_PARTITION:\s*(NPS\d+)")
        out = self._run(f"{self._runner.amd_smi} static --partition", privilege=True).output
        if out:
            found = pattern.findall(out)
            if found:
                result = found[0]
        self.logger.debug(f"Memory partition type: {result}")
        return result

    def get_dmesg_since(self, since: datetime) -> str:
        cmd = f"journalctl --since '{since.replace(tzinfo=None)}' --no-pager"
        return self._run(cmd, privilege=True).output

    def _set_memory_partition_type(self, memory_partition: str) -> None:
        self._run(f"echo Y | sudo -n {self._runner.amd_smi} set --memory-partition {memory_partition}")

    def _set_compute_partition_type(self, compute_partition: str) -> None:
        self._run(f"{self._runner.amd_smi} set --compute-partition {compute_partition}", privilege=True)

    # ------------------------------------------------------------------
    # Partition utilities
    # ------------------------------------------------------------------

    def _init_partition_utilities(self) -> bool:
        try:
            self.memory_partition_util = MemoryPartitionUtility(self._runner)
            self.compute_partition_util = ComputePartitionUtility(self._runner)
            self.logger.info(f"Memory partition utility: {self.memory_partition_util}")
            self.logger.info(f"Compute partition utility: {self.compute_partition_util}")
            return True
        except Exception as err:  # pylint: disable=broad-except
            self.logger.warning(f"Failed to initialize partition utilities: {err}")
            return False

    def _refresh_partition_utilities(self) -> None:
        if self.memory_partition_util:
            self.memory_partition_util.refresh()
        if self.compute_partition_util:
            mem_parts = self.memory_partition_util.get_all_current_partitions() if self.memory_partition_util else None
            self.compute_partition_util.refresh(current_mem_partitions=mem_parts)

    def _driver_reload(self) -> bool:
        """Reload amdgpu after a memory partition change (Linux only)."""
        if self.platform_name != "linux":
            self.logger.warning("Driver reload skipped: non-Linux platform detected.")
            return True
        self.logger.info("Reloading amdgpu driver after memory partition change.")
        self._run("modprobe -r ast", privilege=True)
        self._run("modprobe -r amdgpu", privilege=True)
        return self.load_driver(timeout=120)

    def _get_memory_partition_info(self) -> dict:
        info: dict[str, list[str]] = {"supported": [], "current": []}
        if not self.memory_partition_util:
            self.logger.warning("MemoryPartitionUtility not initialized")
            return info
        all_caps = self.memory_partition_util.get_all_partition_caps()
        all_current = self.memory_partition_util.get_all_current_partitions()
        supported_set: set = set()
        for caps in all_caps.values():
            supported_set.update(caps)
        info["supported"] = sorted(supported_set)
        info["current"] = sorted(set(all_current.values()))
        self.logger.info(f"Memory partition info from utility: {info}")
        return info

    def _get_compute_partition_info(self) -> dict:
        info: dict[str, list[str]] = {"supported": [], "current": []}
        if self.compute_partition_util:
            supported_set: set = set()
            for gpu_id in range(self.compute_partition_util.get_gpu_count()):
                available = self.compute_partition_util.get_available_compute_partitions(gpu_id)
                if available:
                    supported_set.update(available)
            info["supported"] = sorted(supported_set)
            all_current = self.compute_partition_util.get_all_current_compute_partitions()
            info["current"] = sorted(set(all_current.values()))
            self.logger.info(f"Compute partition info from utility: {info}")
        return info

    def _get_original_partition_state(self) -> dict:
        original_state: dict[str, str | None] = {"compute": None, "memory": None}
        if self.compute_partition_util:
            current_states = self.compute_partition_util.get_all_current_states()
            if current_states:
                first_gpu_state = next(iter(current_states.values()))
                original_state["compute"] = first_gpu_state.get("compute")
                original_state["memory"] = first_gpu_state.get("memory")
                self.logger.info(f"Original partition state from utility: {original_state}")
                return original_state
        original_state["compute"] = self.get_compute_partition_type()
        original_state["memory"] = self.get_memory_partition_type()
        self.logger.info(f"Original partition state from system API: {original_state}")
        return original_state

    def _is_valid_partition_combination(self, compute_partition: str, memory_partition: str) -> bool:
        """Only combinations present in the accelerator profiles are valid."""
        if not self.compute_partition_util:
            self.logger.warning("ComputePartitionUtility not initialized, cannot validate combination")
            return True
        is_valid = self.compute_partition_util.is_combination_supported(
            gpu_id=0, compute_partition=compute_partition, memory_partition=memory_partition
        )
        if is_valid:
            self.logger.info(f"Valid combination: {compute_partition} + {memory_partition}")
        else:
            valid_memory = self.compute_partition_util.get_memory_caps_for_compute_partition(
                gpu_id=0, compute_partition=compute_partition
            )
            self.logger.error(
                f"Invalid combination: {compute_partition} + {memory_partition}. "
                f"Valid memory partitions for {compute_partition}: {valid_memory}"
            )
        return is_valid

    def _set_memory_partition_with_compute(self, memory_partition: str, compute_partition: str) -> bool:
        """Memory change -> driver reload -> compute change, in that order."""
        self.logger.info(f"Setting memory partition to {memory_partition}")
        try:
            self._set_memory_partition_type(memory_partition)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to set memory partition to {memory_partition}: {err}")
            return False

        self.logger.info("Performing driver reload (memory reset)")
        if not self._driver_reload():
            self.logger.error(f"Driver reload failed after setting memory partition to {memory_partition}")
            return False

        self.logger.info(f"Setting accelerator/compute partition to {compute_partition}")
        try:
            self._set_compute_partition_type(compute_partition)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to set compute partition to {compute_partition}: {err}")
            return False
        return True

    def _get_reset_loop_count(self, default_count: int = 72) -> int:
        """Loop count from ``loop_count`` / ``loop`` in test_filter, else default."""
        loop_count = default_count
        raw_key = next((k for k in ("loop_count", "loop") if self.test_filter and k in self.test_filter), None)
        if raw_key:
            raw_value = self.test_filter.get(raw_key)
            try:
                loop_count = int(raw_value)
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid loop_count '{raw_value}'; using default {default_count}")
                loop_count = default_count
            if loop_count <= 0:
                self.logger.warning(f"Non-positive loop_count '{raw_value}'; using default {default_count}")
                loop_count = default_count
        self.logger.info(f"Memory partition reset loop count: {loop_count}")
        return loop_count

    def _log_amd_smi_partition_status(self, iteration: int, loop_count: int) -> None:
        self.logger.info(f"amd-smi partition status after iteration {iteration}/{loop_count}")
        try:
            tester_op = self._run("amd-smi", privilege=True)
            self.logger.info("amd-smi output:\n%s", tester_op.output)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.warning("Failed to fetch amd-smi status: %s", err)

    def _check_dmesg_health(self, since: datetime, label: str) -> list[tuple[str, str]]:
        """Scan journalctl since *since* for DMESG_FAULT_PATTERNS (case-insensitive)."""
        findings: list[tuple[str, str]] = []
        try:
            self.logger.debug(f"[{label}] dmesg health check: since={since}, patterns={self.DMESG_FAULT_PATTERNS}")
            dmesg_output = self.get_dmesg_since(since)
            if not dmesg_output:
                self.logger.info(f"[{label}] dmesg health check: no output")
                return findings
            for line in dmesg_output.splitlines():
                lower = line.lower()
                for pattern in self.DMESG_FAULT_PATTERNS:
                    if pattern in lower:
                        findings.append((pattern, line.strip()))
            if findings:
                self.logger.error(f"[{label}] dmesg health check FAILED -- {len(findings)} issue(s):")
                for pattern, raw in findings:
                    self.logger.error(f"  [{pattern}] {raw}")
            else:
                self.logger.info(f"[{label}] dmesg health check passed")
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.warning(f"[{label}] dmesg health check error: {exc}")
        return findings

    # ------------------------------------------------------------------
    # Workload config
    # ------------------------------------------------------------------

    def _load_workload_config(self) -> dict:
        """Load and cache workload jobs from the co-located YAML."""
        if self._workload_jobs_cache is not None:
            return self._workload_jobs_cache
        if not WORKLOAD_CONFIG_PATH.exists():
            self.logger.warning(f"Workload config file not found: {WORKLOAD_CONFIG_PATH}")
            self._workload_jobs_cache = {}
            return self._workload_jobs_cache
        try:
            with open(WORKLOAD_CONFIG_PATH, encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self._workload_jobs_cache = config.get("jobs", {})
            self.logger.debug(f"Loaded {len(self._workload_jobs_cache)} workload(s): {list(self._workload_jobs_cache)}")
            return self._workload_jobs_cache
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"Failed to load workload config: {err}")
            self._workload_jobs_cache = {}
            return self._workload_jobs_cache

    def _get_available_workloads(self) -> list:
        return list(self._load_workload_config().keys())

    def _get_workload_config(self, workload_name: str) -> dict | None:
        return self._load_workload_config().get(workload_name)

    def _get_workload_name(self) -> str:
        if self.test_filter and "workload_name" in self.test_filter:
            return self.test_filter.get("workload_name", "")
        return ""

    def _get_workload_command(self) -> str:
        """Workload command from test_filter override or the named config job."""
        if self.test_filter and "workload_cmd" in self.test_filter:
            return self.test_filter.get("workload_cmd", "")
        workload_name = self._get_workload_name()
        if workload_name:
            workload_config = self._get_workload_config(workload_name)
            if workload_config:
                self.logger.info(f"Using workload '{workload_name}' from config")
                return workload_config.get("run", "") or workload_config.get("steps", "")
            self.logger.warning(
                f"Workload '{workload_name}' not found in config. Available: {self._get_available_workloads()}"
            )
        return ""

    def _get_workload_timeout(self, default_timeout: int = 300) -> int:
        if self.test_filter and "workload_timeout" in self.test_filter:
            try:
                return int(self.test_filter.get("workload_timeout", default_timeout))
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid workload_timeout; using default {default_timeout}s")
        return default_timeout

    def _execute_workload(self, cmd: str, timeout: int, workload_name: str = "workload") -> tuple[bool, str]:
        """Run a (multi-line) shell workload; return (success, failure_reason)."""
        try:
            self.logger.info(f"[{workload_name}] Executing (timeout={timeout}s, cmd_length={len(cmd)} chars)")
            self.logger.debug(f"[{workload_name}] Contents:\n{cmd}")
            tester_op = self._run(cmd, privilege=False, timeout=timeout)
            self.logger.debug(
                f"[{workload_name}] exit_code={tester_op.exit_code}, "
                f"output_length={len(tester_op.output or '')} chars"
            )

            output_faults = []
            if tester_op.output:
                lower_output = tester_op.output.lower()
                for pattern in self.DMESG_FAULT_PATTERNS:
                    if pattern in lower_output:
                        output_faults.append(pattern)

            if tester_op.exit_code == 0 and not output_faults:
                self.logger.info(f"[{workload_name}] Completed successfully")
                self.logger.debug(f"[{workload_name}] Output:\n{tester_op.output}")
                return True, ""

            reasons = []
            if tester_op.exit_code != 0:
                reasons.append(f"exit_code={tester_op.exit_code}")
                self.logger.error(f"[{workload_name}] Failed (exit {tester_op.exit_code})")
                if tester_op.output:
                    tail_lines = [
                        ln.strip()
                        for ln in tester_op.output.strip().splitlines()[-10:]
                        if ln.strip() and not ln.startswith("+")
                    ]
                    if tail_lines:
                        last_error = tail_lines[-1][:120]
                        reasons.append(f'last_line="{last_error}"')
            if output_faults:
                reasons.append(f"fault_patterns={output_faults}")
                self.logger.error(f"[{workload_name}] Fault pattern(s) in output: {output_faults}")
            self.logger.error(f"[{workload_name}] Output:\n{tester_op.output}")
            return False, "; ".join(reasons)
        except Exception as err:  # pylint: disable=broad-except
            self.logger.error(f"[{workload_name}] Exception: {err}")
            return False, f"exception: {err}"

    def _cleanup_workload(self, workload_name: str) -> None:
        workload_config = self._get_workload_config(workload_name)
        if not workload_config:
            return
        cleanup_cmd = workload_config.get("cleanup", "")
        if not cleanup_cmd or not cleanup_cmd.strip():
            return
        self.logger.info(f"[{workload_name}] Running post-test cleanup")
        self._execute_workload(cmd=cleanup_cmd, timeout=120, workload_name=f"{workload_name}_cleanup")

    def _verify_partition_change(
        self, set_success: bool, expected_compute: str, expected_memory: str, label: str
    ) -> bool:
        """Refresh utilities, cross-check via the independent readback, gate workloads."""
        self._refresh_partition_utilities()
        actual_compute = self.get_compute_partition_type()
        actual_memory = self.get_memory_partition_type()

        self.logger.debug(
            f"[{label}] Partition readback: "
            f"compute={actual_compute!r} (expected {expected_compute!r}), "
            f"memory={actual_memory!r} (expected {expected_memory!r})"
        )

        compute_match = actual_compute == expected_compute
        memory_match = actual_memory == expected_memory
        status_ok = set_success and compute_match and memory_match

        self._partition_healthy = status_ok
        if status_ok:
            self.logger.debug(f"[{label}] Partition healthy, workloads unblocked")
        else:
            self._last_partition_failure_label = label
            self.logger.warning(f"[{label}] Partition set UNHEALTHY -- subsequent workloads will be skipped")
            if not set_success:
                self.logger.error(f"{label} partition change failed due to driver reload or set failure")
            if not compute_match:
                self.logger.error(f"{label} compute mismatch: expected={expected_compute!r}, actual={actual_compute!r}")
            if not memory_match:
                self.logger.error(f"{label} memory mismatch: expected={expected_memory!r}, actual={actual_memory!r}")
        return status_ok

    def _should_run_workload(self) -> bool:
        return bool(self._get_workload_command())

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def execute(self) -> _Outcome:  # noqa: C901
        """Toggle memory partition to POR and back for loop_count iterations."""
        self.logger.info("Executing AMD SMI memory partition functional test.")

        if not self.is_driver_loaded():
            self.logger.info("AMDGPU driver not loaded. Attempting to load...")
            if not self.load_driver(timeout=120):
                return _Outcome("FAIL", "AMDGPU driver failed to load. Cannot proceed with partition test.")

        asic_key = self._get_asic_key()
        if not asic_key:
            return _Outcome("SKIP", "Supported only on MI350X ASICs")

        por_matrix = self.POR_PARTITION_MATRIX.get(asic_key)
        if not por_matrix:
            return _Outcome("SKIP", f"No POR matrix defined for ASIC {asic_key}")

        self._init_partition_utilities()

        memory_info = self._get_memory_partition_info()
        compute_info = self._get_compute_partition_info()
        supported = {"memory": memory_info["supported"], "compute": compute_info["supported"]}
        self.logger.info(f"Supported partitions from utilities: {supported}")

        por_compute = por_matrix["compute_partition"]
        por_memory = por_matrix["memory_partition"]

        if not self._is_valid_partition_combination(por_compute, por_memory):
            valid_memory = (
                self.compute_partition_util.get_memory_caps_for_compute_partition(0, por_compute)
                if self.compute_partition_util
                else []
            )
            return _Outcome(
                "FAIL",
                f"POR combination {por_compute}+{por_memory} not valid. "
                f"Valid memory partitions for {por_compute}: {valid_memory}",
            )

        original_state = self._get_original_partition_state()
        original_compute = original_state["compute"]
        original_memory = original_state["memory"]
        if not original_compute:
            original_compute = self.get_compute_partition_type()
        if not original_memory:
            if memory_info["current"] and len(memory_info["current"]) == 1:
                original_memory = memory_info["current"][0]
            else:
                original_memory = self.get_memory_partition_type()

        self.logger.info(f"Current partitions - Compute: {original_compute}, Memory: {original_memory}")

        status_records: list[bool] = []
        workload_records: list[tuple] = []
        dmesg_issues: list[tuple[str, str]] = []
        loop_count = self._get_reset_loop_count()
        needs_por_toggle = original_compute != por_compute or original_memory != por_memory

        cached_wl_cmd = self._get_workload_command()
        workload_from_config = bool(cached_wl_cmd) and not (self.test_filter and "workload_cmd" in self.test_filter)
        cached_wl_name = (
            (self._get_workload_name() or "custom") if workload_from_config else ("custom" if cached_wl_cmd else "")
        )
        cached_wl_timeout = self._get_workload_timeout() if cached_wl_cmd else 0
        if cached_wl_cmd:
            self.logger.info(f"Workload: name='{cached_wl_name}', timeout={cached_wl_timeout}s")

        for iteration in range(1, loop_count + 1):
            iter_start = datetime.now()
            self.logger.info(f"Memory partition reset iteration {iteration}/{loop_count}")

            if cached_wl_cmd:
                if not self._partition_healthy:
                    failed_at = self._last_partition_failure_label or "unknown"
                    skip_reason = f"SKIPPED -- partition_set_failed at [{failed_at}]"
                    self.logger.warning(
                        f"Skipping workload in iteration {iteration}: partition set failed at [{failed_at}]"
                    )
                    workload_records.append((cached_wl_name, None, skip_reason))
                else:
                    self.logger.info(f"Running workload before partition reset (iteration {iteration})")
                    workload_success, wl_reason = self._execute_workload(
                        cached_wl_cmd, cached_wl_timeout, cached_wl_name
                    )
                    workload_records.append((cached_wl_name, workload_success, wl_reason))
                    if not workload_success:
                        self.logger.warning(f"Workload failed in iteration {iteration}: {wl_reason}")

            if needs_por_toggle:
                self.logger.info(
                    f"Toggling Action Triggered - Setting POR partitions:\n"
                    f"  Target POR     : Memory={por_memory}, Profile={por_compute}"
                )
                set_por_success = self._set_memory_partition_with_compute(por_memory, por_compute)
                status_records.append(self._verify_partition_change(set_por_success, por_compute, por_memory, "POR"))

            self.logger.info(
                f"Toggling Action Triggered - Restoring Original partitions:\n"
                f"  Target Original: Memory={original_memory}, Profile={original_compute}"
            )
            restore_success = self._set_memory_partition_with_compute(original_memory, original_compute)
            status_records.append(
                self._verify_partition_change(restore_success, original_compute, original_memory, "Original")
            )

            self._log_amd_smi_partition_status(iteration, loop_count)

            findings = self._check_dmesg_health(iter_start, f"Iter {iteration}")
            if findings:
                dmesg_issues.extend(findings)
                status_records.append(False)

        if workload_records:
            self._log_workload_summary(workload_records)
        if dmesg_issues:
            self.logger.error(f"dmesg fault summary: {len(dmesg_issues)} issue(s) across all iterations")

        if workload_from_config and cached_wl_name:
            self._cleanup_workload(cached_wl_name)

        wl_ok = all(ok is not False for _, ok, _ in workload_records) if workload_records else True
        passed = all(status_records) and wl_ok

        fail_parts = []
        if not all(status_records):
            fail_parts.append("partition checks failed")
        wl_failed_names = [n for n, ok, _ in workload_records if ok is False]
        wl_skipped_names = [n for n, ok, _ in workload_records if ok is None]
        if wl_failed_names:
            fail_parts.append(f"workloads_failed [{', '.join(wl_failed_names)}]")
        if wl_skipped_names:
            fail_parts.append(f"workloads_skipped [{', '.join(wl_skipped_names)}]")

        if passed:
            return _Outcome("PASS", "Memory partition switch to POR and restore completed.")
        return _Outcome("FAIL", f"FAILED: {'; '.join(fail_parts)}")

    def _log_workload_summary(self, workload_records: list[tuple]) -> None:
        wl_passed = [n for n, ok, _ in workload_records if ok is True]
        wl_failed = [(n, r) for n, ok, r in workload_records if ok is False]
        wl_skipped = [(n, r) for n, ok, r in workload_records if ok is None]
        self.logger.info(
            f"Workload summary: {len(wl_passed)} passed, {len(wl_failed)} failed, "
            f"{len(wl_skipped)} skipped / {len(workload_records)} total"
        )
        for name, ok, reason in workload_records:
            if ok is True:
                self.logger.info(f"  {name}: PASS")
            elif ok is None:
                self.logger.info(f"  {name}: SKIP -- {reason}")
            else:
                self.logger.info(f"  {name}: FAIL -- {reason}")


class AmdSmiMemoryPartitionChangePostWorkload(AmdSmiMemoryPartition):
    """Toggle stress: run all workloads under baseline (SPX/NPS1) and POR (DPX/NPS2)."""

    BASELINE_PARTITIONS = {
        "MI350X": {"compute_partition": "SPX", "memory_partition": "NPS1"},
    }

    # Partition changes trigger expected MODE1 GPU resets; only genuine crashes matter here.
    DMESG_FAULT_PATTERNS = ("page fault", "segfault")

    PARTITION_SET_REPETITIONS = 1
    WORKLOAD_FILTER: tuple | None = None

    def __init__(self, executor, rock_dir, platform_name, test_filter=None):
        super().__init__(executor, rock_dir, platform_name, test_filter=test_filter)
        self.test_name = "amdsmi_mem_partition_change_post_workload"

    def _get_reset_loop_count(self, default_count: int = 2) -> int:
        return super()._get_reset_loop_count(default_count=default_count)

    def _get_workload_timeout(self, default_timeout: int = 900) -> int:
        if self.test_filter and "workload_timeout" in self.test_filter:
            try:
                return int(self.test_filter.get("workload_timeout", default_timeout))
            except (TypeError, ValueError):
                self.logger.warning(f"Invalid workload_timeout; using default {default_timeout}s")
        return default_timeout

    def _get_workload_filter(self) -> tuple | None:
        """Colon-separated job keys from ``workload_filter`` / ``workload``, class default, or None."""
        raw_key = next((k for k in ("workload_filter", "workload") if self.test_filter and k in self.test_filter), None)
        if raw_key:
            raw = self.test_filter.get(raw_key, "")
            if raw:
                keys = tuple(k.strip() for k in raw.split(":") if k.strip())
                if keys:
                    available = self._get_available_workloads()
                    unknown = [k for k in keys if k not in available]
                    if unknown:
                        self.logger.warning(
                            f"workload_filter contains unknown job(s): {unknown}; available: {available}"
                        )
                    self.logger.info(f"workload_filter from test_filter: {keys}")
                    return keys
        if self.WORKLOAD_FILTER is not None:
            self.logger.debug(f"workload_filter from class constant: {self.WORKLOAD_FILTER}")
        return self.WORKLOAD_FILTER

    def _build_all_workloads(self) -> bool:
        """One-time build step per selected workload; skipped for workload_cmd overrides."""
        if self.test_filter and "workload_cmd" in self.test_filter:
            return True
        wl_filter = self._get_workload_filter()
        jobs = self._load_workload_config()
        if wl_filter is not None:
            jobs = {k: v for k, v in jobs.items() if k in wl_filter}
        if not jobs:
            return True

        build_timeout = self._get_workload_timeout(default_timeout=600)
        all_ok = True
        for name, config in jobs.items():
            if name in self._built_workloads:
                self.logger.debug(f"[{name}] Build already completed, skipping")
                continue
            build_cmd = config.get("build", "")
            if not build_cmd:
                self.logger.debug(f"[{name}] No build step defined, skipping")
                self._built_workloads.add(name)
                continue
            self.logger.info(f"[{name}] Running one-time build step (timeout={build_timeout}s)")
            ok, reason = self._execute_workload(build_cmd, build_timeout, f"{name}_build")
            if ok:
                self._built_workloads.add(name)
                self.logger.info(f"[{name}] Build succeeded")
            else:
                self.logger.error(f"[{name}] Build FAILED: {reason}")
                all_ok = False
        return all_ok

    # pylint: disable-next=too-many-branches
    def _run_all_workloads_sequential(self, partition_label: str) -> list:  # noqa: C901
        """Run every selected workload under the current partition; honour the health gate."""
        results: list[tuple] = []
        wl_filter = self._get_workload_filter()

        if not self._partition_healthy:
            failed_at = self._last_partition_failure_label or "unknown"
            skip_reason = f"SKIPPED -- partition_set_failed at [{failed_at}]"
            self.logger.warning(f"Skipping workloads under {partition_label}: partition set failed at [{failed_at}]")
            if self.test_filter and "workload_cmd" in self.test_filter:
                results.append(("custom", None, skip_reason))
            else:
                jobs = self._load_workload_config()
                if wl_filter is not None:
                    jobs = {k: v for k, v in jobs.items() if k in wl_filter}
                for name in jobs:
                    results.append((name, None, skip_reason))
            return results

        timeout = self._get_workload_timeout()

        if self.test_filter and "workload_cmd" in self.test_filter:
            cmd = self.test_filter.get("workload_cmd", "")
            if cmd:
                self.logger.info(f"Running custom workload under {partition_label}")
                ok, reason = self._execute_workload(cmd, timeout, "custom")
                results.append(("custom", ok, reason))
            return results

        jobs = self._load_workload_config()
        if not jobs:
            self.logger.warning("No workloads configured in workload.yaml")
            return results

        if wl_filter is not None:
            all_keys = list(jobs.keys())
            jobs = {k: v for k, v in jobs.items() if k in wl_filter}
            self.logger.debug(
                f"workload_filter={wl_filter}: {len(jobs)}/{len(all_keys)} jobs selected "
                f"(all={all_keys}, selected={list(jobs.keys())})"
            )
            if not jobs:
                self.logger.warning(
                    f"workload_filter={wl_filter} matched no jobs; available: {self._get_available_workloads()}"
                )
                return results

        self.logger.debug(f"Running {len(jobs)} workload(s) under {partition_label}: {list(jobs.keys())}")
        for name, config in jobs.items():
            run_cmd = config.get("run", "") or config.get("steps", "")
            if not run_cmd:
                self.logger.warning(f"Workload '{name}' has no run/steps, skipping")
                results.append((name, False, "no run/steps defined in workload.yaml"))
                continue
            self.logger.info(f"Running workload '{name}' under {partition_label} (timeout={timeout}s)")
            ok, reason = self._execute_workload(run_cmd, timeout, name)
            results.append((name, ok, reason))
            if not ok:
                self.logger.warning(f"Workload '{name}' failed under {partition_label}: {reason}")
        return results

    @staticmethod
    def _wl_status_label(ok) -> str:
        if ok is True:
            return "PASS"
        if ok is None:
            return "SKIP"
        return "FAIL"

    # pylint: disable-next=too-many-locals
    def _log_iteration_summary_table(self, rows: list[dict], base_label: str, por_label: str) -> None:
        """Log a per-iteration table of partition sets and per-workload statuses."""
        if not rows:
            return

        headers = (
            "Loop",
            f"Set {base_label}",
            f"Workloads ({base_label})",
            f"Set {por_label}",
            f"Workloads ({por_label})",
            f"Restore {base_label}",
        )

        cells = []
        for row in rows:
            loop_str = str(row["loop"])
            set_base_entry = "PASS" if row["set_base_entry"] else "FAIL"
            wl_base = ", ".join(f"{n}:{self._wl_status_label(ok)}" for n, ok, _ in row["wl_base"]) or "--"
            reps = row["set_por_reps"]
            por_pass = sum(1 for v in reps if v)
            set_por = "PASS" if all(reps) else f"FAIL ({por_pass}/{len(reps)})"
            wl_por = ", ".join(f"{n}:{self._wl_status_label(ok)}" for n, ok, _ in row["wl_por"]) or "--"
            reps_b = row["set_base_exit_reps"]
            base_pass = sum(1 for v in reps_b if v)
            set_base_exit = "PASS" if all(reps_b) else f"FAIL ({base_pass}/{len(reps_b)})"
            cells.append((loop_str, set_base_entry, wl_base, set_por, wl_por, set_base_exit))

        widths = [max(len(headers[i]), *(len(c[i]) for c in cells)) for i in range(len(headers))]

        def _fmt(vals):
            return "| " + " | ".join(v.ljust(w) for v, w in zip(vals, widths, strict=False)) + " |"

        sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
        lines = ["", "=== Iteration Summary ===", sep, _fmt(headers), sep]
        lines.extend(_fmt(c) for c in cells)
        lines.append(sep)
        self.logger.info("\n".join(lines))

    def _cleanup_all_workloads(self) -> None:
        if self.test_filter and "workload_cmd" in self.test_filter:
            return
        wl_filter = self._get_workload_filter()
        jobs = self._load_workload_config()
        if wl_filter is not None:
            jobs = {k: v for k, v in jobs.items() if k in wl_filter}
        for name in jobs:
            self._cleanup_workload(name)

    # pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
    def execute(self) -> _Outcome:  # noqa: C901
        """Set baseline, then loop: workloads @baseline -> set POR -> workloads @POR -> restore."""
        self.logger.info("Executing partition change post workload stress test.")

        if not self.is_driver_loaded():
            self.logger.info("AMDGPU driver not loaded. Attempting to load...")
            if not self.load_driver(timeout=120):
                return _Outcome("FAIL", "AMDGPU driver failed to load.")

        asic_key = self._get_asic_key()
        if not asic_key:
            return _Outcome("SKIP", "Supported only on MI350X ASICs")

        self._init_partition_utilities()

        por = self.POR_PARTITION_MATRIX.get(asic_key, {})
        baseline = self.BASELINE_PARTITIONS.get(asic_key, {})
        por_compute = por.get("compute_partition", "DPX")
        por_memory = por.get("memory_partition", "NPS2")
        base_compute = baseline.get("compute_partition", "SPX")
        base_memory = baseline.get("memory_partition", "NPS1")

        for label, cp, mp in [("Baseline", base_compute, base_memory), ("POR", por_compute, por_memory)]:
            if not self._is_valid_partition_combination(cp, mp):
                return _Outcome("FAIL", f"{label} combination {cp}+{mp} is not valid.")

        loop_count = self._get_reset_loop_count()
        status_records: list[bool] = []
        workload_records: list[tuple] = []

        base_label = f"{base_compute}/{base_memory}"
        por_label = f"{por_compute}/{por_memory}"
        self.logger.info(
            f"Config: baseline={base_label}, POR={por_label}, "
            f"loop_count={loop_count}, workloads={self._get_available_workloads()}"
        )

        build_ok = self._build_all_workloads()
        if not build_ok:
            self.logger.error("One or more workload builds failed")

        self.logger.info(f"[Init] Setting baseline: {base_label}")
        set_ok = self._set_memory_partition_with_compute(base_memory, base_compute)
        verify_ok = self._verify_partition_change(set_ok, base_compute, base_memory, "Baseline Init")
        status_records.append(verify_ok)
        if not verify_ok:
            self.logger.error("Failed to set initial baseline, continuing to report")

        dmesg_issues: list[tuple[str, str]] = []
        iteration_rows: list[dict] = []

        for iteration in range(1, loop_count + 1):
            iter_start = datetime.now()
            self.logger.info(f"=== Iteration {iteration}/{loop_count} ===")
            iter_row: dict = {"loop": iteration, "set_base_entry": status_records[-1]}

            self.logger.info(f"[Iter {iteration}] Workloads under {base_label}")
            wl_base = self._run_all_workloads_sequential(base_label)
            workload_records.extend(wl_base)
            iter_row["wl_base"] = wl_base

            por_reps = []
            for rep in range(1, self.PARTITION_SET_REPETITIONS + 1):
                rep_tag = f" rep {rep}/{self.PARTITION_SET_REPETITIONS}" if self.PARTITION_SET_REPETITIONS > 1 else ""
                self.logger.info(f"[Iter {iteration}] Setting POR: {por_label}{rep_tag}")
                set_ok = self._set_memory_partition_with_compute(por_memory, por_compute)
                verify_ok = self._verify_partition_change(
                    set_ok, por_compute, por_memory, f"POR (iter {iteration}{rep_tag})"
                )
                status_records.append(verify_ok)
                por_reps.append(verify_ok)
            iter_row["set_por_reps"] = por_reps

            self.logger.info(f"[Iter {iteration}] Workloads under {por_label}")
            wl_por = self._run_all_workloads_sequential(por_label)
            workload_records.extend(wl_por)
            iter_row["wl_por"] = wl_por

            base_exit_reps = []
            for rep in range(1, self.PARTITION_SET_REPETITIONS + 1):
                rep_tag = f" rep {rep}/{self.PARTITION_SET_REPETITIONS}" if self.PARTITION_SET_REPETITIONS > 1 else ""
                self.logger.info(f"[Iter {iteration}] Restoring baseline: {base_label}{rep_tag}")
                set_ok = self._set_memory_partition_with_compute(base_memory, base_compute)
                verify_ok = self._verify_partition_change(
                    set_ok, base_compute, base_memory, f"Baseline (iter {iteration}{rep_tag})"
                )
                status_records.append(verify_ok)
                base_exit_reps.append(verify_ok)
            iter_row["set_base_exit_reps"] = base_exit_reps

            iteration_rows.append(iter_row)
            self._log_amd_smi_partition_status(iteration, loop_count)

            findings = self._check_dmesg_health(iter_start, f"Iter {iteration}")
            if findings:
                dmesg_issues.extend(findings)
                status_records.append(False)

        self._log_iteration_summary_table(iteration_rows, base_label, por_label)
        if workload_records:
            self._log_workload_summary(workload_records)
        if dmesg_issues:
            self.logger.error(f"dmesg fault summary: {len(dmesg_issues)} issue(s) across all iterations")

        self._cleanup_all_workloads()

        failed_indices = [i for i, ok in enumerate(status_records) if not ok]
        if failed_indices:
            self.logger.info(f"status_records: {len(status_records)} entries, failures at indices {failed_indices}")

        wl_ok = all(ok is not False for _, ok, _ in workload_records) if workload_records else True
        passed = all(status_records) and wl_ok and build_ok

        fail_reasons = []
        if not build_ok:
            fail_reasons.append("workload_build_failed")
        if failed_indices:
            fail_reasons.append(f"partition_checks_failed={len(failed_indices)}")
        wl_failed_names = [n for n, ok, _ in workload_records if ok is False]
        wl_skipped_names = [n for n, ok, _ in workload_records if ok is None]
        if wl_failed_names:
            fail_reasons.append(f"workloads_failed={len(wl_failed_names)} [{', '.join(wl_failed_names)}]")
        if wl_skipped_names:
            fail_reasons.append(f"workloads_skipped={len(wl_skipped_names)} [{', '.join(wl_skipped_names)}]")
        if dmesg_issues:
            fail_reasons.append(f"dmesg_faults={len(dmesg_issues)}")

        if passed:
            return _Outcome("PASS", "Partition toggle with workload stress completed successfully.")
        return _Outcome("FAIL", f"FAILED: {'; '.join(fail_reasons)}")


class AmdSmiMemoryPartitionChangeThreeTimes(AmdSmiMemoryPartitionChangePostWorkload):
    """As post-workload, but each partition set+verify pair executes three times."""

    PARTITION_SET_REPETITIONS = 3

    def __init__(self, executor, rock_dir, platform_name, test_filter=None):
        super().__init__(executor, rock_dir, platform_name, test_filter=test_filter)
        self.test_name = "amdsmi_mem_partition_change_3x"


class AmdSmiMemoryPartitionMultipleHipbone(AmdSmiMemoryPartitionChangePostWorkload):
    """Partition toggle with the hipbone workload selected."""

    WORKLOAD_FILTER = ("hipbone",)

    def __init__(self, executor, rock_dir, platform_name, test_filter=None):
        super().__init__(executor, rock_dir, platform_name, test_filter=test_filter)
        self.test_name = "amdsmi_mem_partition_change_multiple_hipbone"


class AmdSmiMemoryPartitionMultipleAmgSolve(AmdSmiMemoryPartitionChangePostWorkload):
    """Partition toggle with the amgsolve workload selected."""

    WORKLOAD_FILTER = ("amgsolve",)

    def __init__(self, executor, rock_dir, platform_name, test_filter=None):
        super().__init__(executor, rock_dir, platform_name, test_filter=test_filter)
        self.test_name = "amdsmi_mem_partition_change_multiple_amgSolve"


@dataclass
class _DispatchOutcome:
    """Aggregate of one or more per-workload outcomes."""

    outcome: _Outcome
    per_case: dict[str, _Outcome] = field(default_factory=dict)


class AmdSmiMemoryPartitionSingleWorkloadOnly(AmdSmiMemoryPartitionChangePostWorkload):
    """Partition toggle with a single workload resolved from test_filter or the class default."""

    WORKLOAD_TEST_MAP = {
        "hipbone": "amdsmi_mem_partition_change_multiple_hipbone",
        "transferbench": "amdsmi_mem_partition_change_transferbench",
        "amgsolve": "amdsmi_mem_partition_change_multiple_amgSolve",
    }

    WORKLOAD_FILTER = ("hipbone",)

    def __init__(self, executor, rock_dir, platform_name, test_filter=None):
        super().__init__(executor, rock_dir, platform_name, test_filter=test_filter)
        self.test_name = self.WORKLOAD_TEST_MAP[self.WORKLOAD_FILTER[0]]

    def execute(self) -> _Outcome:
        wl_filter = self._get_workload_filter()

        if not wl_filter or len(wl_filter) <= 1:
            wl_key = wl_filter[0] if wl_filter else self.WORKLOAD_FILTER[0]
            self.test_name = self.WORKLOAD_TEST_MAP.get(wl_key, f"amdsmi_mem_partition_change_{wl_key}")
            self.WORKLOAD_FILTER = (wl_key,)  # pylint: disable=invalid-name
            self.logger.info(f"Single workload: test_name='{self.test_name}', workload='{wl_key}'")
            return super().execute()

        combined: dict[str, _Outcome] = {}
        self.logger.info(f"Multi-workload dispatch: {len(wl_filter)} test case(s) for workloads {list(wl_filter)}")
        for wl_key in wl_filter:
            self.test_name = self.WORKLOAD_TEST_MAP.get(wl_key, f"amdsmi_mem_partition_change_{wl_key}")
            self.WORKLOAD_FILTER = (wl_key,)  # pylint: disable=invalid-name
            self._partition_healthy = True
            self._last_partition_failure_label = ""
            self.logger.info(f"=== Dispatching test case: {self.test_name} (workload={wl_key}) ===")
            combined[self.test_name] = super().execute()

        failed = {name: res for name, res in combined.items() if res.status == "FAIL"}
        if failed:
            detail = "; ".join(f"{name}: {res.message}" for name, res in failed.items())
            return _Outcome("FAIL", f"FAILED cases: {detail}")
        if all(res.status == "SKIP" for res in combined.values()):
            return _Outcome("SKIP", "; ".join(f"{name}: {res.message}" for name, res in combined.items()))
        return _Outcome("PASS", "All dispatched workload test cases passed.")
