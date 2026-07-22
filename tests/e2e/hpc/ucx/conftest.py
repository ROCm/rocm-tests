# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os

import pytest

from tests.e2e.hpc.ucx._workload import UCX_GIT_REF, UCX_GIT_URL

logger = logging.getLogger(__name__)

_GTEST_REL = "test/gtest/gtest"


@pytest.fixture(scope="session")
def ucx_build(rock_dir, framework_config, external_build, cmake_executor) -> str:
    if not rock_dir:
        pytest.skip("UCX build requires a ROCm install; pass --rock-dir / set ROCK_DIR")

    rocm_path = os.path.realpath(rock_dir) if cmake_executor is None else rock_dir
    build_timeout = float(framework_config.therock.build_timeout_secs)
    env_prefix = f"ROCM_PATH={rocm_path} LD_LIBRARY_PATH={rocm_path}/lib:{rocm_path}/lib64:$LD_LIBRARY_PATH"
    log_dir = os.path.join(framework_config.framework.artifact_dir, "ucx")

    with external_build.build_lock("hpc-ucx", timeout=build_timeout):
        source_dir = os.path.realpath(str(external_build.clone_repo(UCX_GIT_URL, "ucx/ucx", ref=UCX_GIT_REF)))
        external_build.assert_license_present(source_dir)
        build_dir = f"{source_dir}/build"
        configure_args = [
            "--disable-logging",
            "--disable-debug",
            "--disable-assertions",
            "--enable-params-check",
            f"--prefix={build_dir}/ucx",
            "--without-knem",
            "--without-cuda",
            f"--with-rocm={rocm_path}",
            "--enable-gtest",
            "--without-gdrcopy",
            "--without-java",
        ]
        logger.info("UCX %s: autogen/configure/make/install (rocm=%s)", UCX_GIT_REF, rocm_path)
        return external_build.configure_make_build(
            source_dir,
            build_dir,
            bootstrap_script="./autogen.sh",
            configure_script="../contrib/configure-release",
            configure_args=configure_args,
            env_prefix=env_prefix,
            make_install=True,
            sentinel=_GTEST_REL,
            log_dir=log_dir,
            use_lock=False,
            timeout=build_timeout,
        )


@pytest.fixture(scope="session")
def ucx_gtest_binary(ucx_build) -> str:
    return f"{ucx_build}/{_GTEST_REL}"
