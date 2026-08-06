# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""conftest.py -- Fixtures for tests/e2e/transformer_engine/ (TE JAX suite).

Provides the session-scoped ``te_jax_config`` fixture. The wheel filenames,
artifact base URLs, branch, and GPU family are supplied through ``ROCM_TEST_*``
environment variables so no build coordinates are hard-coded (per the CLAUDE.md
secrets/config rule). Missing values leave the corresponding config fields empty;
each test skips gracefully when a value it requires is absent.
"""

from __future__ import annotations

import pytest

from tests.e2e.transformer_engine._transformer_engine_jax import TeJaxConfig


@pytest.fixture(scope="session")
def te_jax_config() -> TeJaxConfig:
    """Return the Transformer Engine JAX suite configuration from the environment.

    Returns:
        A ``TeJaxConfig`` populated from ``ROCM_TEST_*`` environment variables.
    """
    return TeJaxConfig.from_env()
