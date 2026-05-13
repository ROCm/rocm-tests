# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
node_spec.py -- Typed descriptor for a single compute node (local or remote).

NodeSpec holds the SSH credentials and GPU affinity for one node in the fleet.
HostConfigLoader parses host.yaml files into ordered lists of NodeSpec.

host.yaml format::

    HOST_IDX_1:
      HOSTNAME: gpu-node-01.example.com
      USERNAME: ci
      SSH_KEY:  ~/.ssh/ci_rsa
      # PASSWORD: secret     # prefer SSH_KEY for automated pipelines
      # GPU_ARCH: gfx942     # optional architecture filter
    HOST_IDX_2:
      HOSTNAME: gpu-node-02.example.com
      USERNAME: ci
      SSH_KEY:  ~/.ssh/ci_rsa

Nodes are always processed in HOST_IDX_N ascending order.
"""

from __future__ import annotations

from dataclasses import dataclass
import socket


@dataclass(frozen=True)
class NodeSpec:
    """Immutable descriptor for one compute node in the test fleet.

    For local execution, ``is_local`` returns True and no SSH is required.
    For remote execution, SSH credentials are used by ``NodePool`` to open
    a ``SshExecutor`` session on first use.

    Attributes:
        hostname:  DNS name or IP address of the node.
        username:  SSH login name (default: ``$USER`` at connection time).
        password:  SSH password — prefer ``ssh_key`` for automated pipelines.
        ssh_key:   Path to SSH private key; ``~`` is expanded by SshExecutor.
        gpu_arch:  Optional GFX architecture filter (e.g. ``"gfx942"``).
                   When set, only GPUs with matching arch are allocated.
        label:     Human-readable node identifier used in log prefixes and
                   file-lock names (e.g. ``"HOST_IDX_1"`` or ``"localhost"``).
    """

    hostname: str
    username: str | None = None
    password: str | None = None
    ssh_key: str | None = None
    gpu_arch: str | None = None
    label: str = "localhost"

    @property
    def is_local(self) -> bool:
        """True when this node is the machine running pytest.

        Checks against ``localhost``, ``127.0.0.1``, and the current
        machine's hostname so that a host.yaml that lists the runner's own
        hostname is treated transparently as a local node.

        Returns:
            bool: True if this node refers to the local machine.
        """
        try:
            local_names = {"localhost", "127.0.0.1", socket.gethostname()}
        except OSError:
            local_names = {"localhost", "127.0.0.1"}
        return self.hostname in local_names


class HostConfigLoader:
    """Parse a host.yaml file into an ordered list of ``NodeSpec`` objects.

    The YAML file uses ``HOST_IDX_N`` keys (N = 1, 2, …) to define nodes
    in execution priority order.  Each entry must have a ``HOSTNAME`` key;
    all other fields are optional.

    Example::

        nodes = HostConfigLoader.load("host.yaml")
        # nodes[0].label == "HOST_IDX_1"
        # nodes[1].label == "HOST_IDX_2"
    """

    @staticmethod
    def load(path: str) -> list[NodeSpec]:
        """Parse *path* and return nodes ordered by ``HOST_IDX_N`` key.

        Args:
            path: Path to the host YAML configuration file.

        Returns:
            List of ``NodeSpec`` in ``HOST_IDX_N`` ascending order.

        Raises:
            FileNotFoundError: If *path* does not exist.
            KeyError:          If a ``HOST_IDX_N`` entry has no ``HOSTNAME``.
            ValueError:        If the YAML structure is invalid.
            ImportError:       If ``PyYAML`` is not installed.
        """
        try:
            import yaml  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("HostConfigLoader requires PyYAML: pip install pyyaml") from exc

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not isinstance(data, dict):
            raise ValueError(f"host.yaml must be a YAML mapping at the top level, " f"got {type(data).__name__}")

        # Sort by HOST_IDX_N numerically so order is deterministic
        host_keys = sorted(
            (k for k in data if str(k).startswith("HOST_IDX_")),
            key=lambda k: int(str(k).rsplit("_", maxsplit=1)[-1]),
        )

        nodes: list[NodeSpec] = []
        for label in host_keys:
            entry = data[label]
            if not isinstance(entry, dict):
                raise ValueError(f"{label}: expected a YAML mapping, got {type(entry).__name__}")
            if "HOSTNAME" not in entry:
                raise KeyError(f"{label}: missing required key 'HOSTNAME'")

            nodes.append(
                NodeSpec(
                    hostname=entry["HOSTNAME"],
                    username=entry.get("USERNAME"),
                    password=entry.get("PASSWORD"),
                    ssh_key=entry.get("SSH_KEY"),
                    gpu_arch=entry.get("GPU_ARCH"),
                    label=str(label),
                )
            )

        return nodes
