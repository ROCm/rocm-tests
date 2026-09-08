# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts clone, build, and run an
external project, and install third-party build and runtime dependencies on the
test node. Each carries its own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the `rocm-examples` source tree, its dependencies,
or any built binaries; the test fixtures obtain them at runtime into the gitignored
`output/` build tree or install them onto the node. If any downstream packaging
flow redistributes the cloned source or built binaries, that redistribution must
retain the upstream copyright notices, license terms, and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. rocm-examples

The `rocm_examples_repo` fixture clones
[ROCm/rocm-examples](https://github.com/ROCm/rocm-examples) at runtime (ref
defaults to `amd-mainline`, overridable via `ROCM_TEST_ROCM_EXAMPLES_REF`), and
`rocm_examples_build_dir` configures and builds its CTest suite. The build is run
with `tolerate_build_failure` because the upstream tree contains samples for
optional ROCm components that not every install ships; samples whose binaries are
not produced are reported as skips.

- **Copyright:** Copyright (c) Advanced Micro Devices, Inc.
- **License:** MIT License
- **Upstream repository:** https://github.com/ROCm/rocm-examples
- **Upstream license file:** https://github.com/ROCm/rocm-examples/blob/amd-staging/LICENSE.md

---

### 2. System build dependencies installed on the test node

`_install_system_deps` installs the Tier-1 packages that the upstream
`rocm-examples` CMake looks for, because the minimal OSSCI container does not ship
them. These are installed from the distribution's own package repositories onto the
test node; they are **not** vendored or redistributed by this repository, and each
retains its own upstream license:

| Package group | Project | Typical license |
|---|---|---|
| `libavcodec-dev`, `libavformat-dev`, `libavutil-dev`, `libswscale-dev`, `libavdevice-dev` | [FFmpeg](https://ffmpeg.org/) | LGPL-2.1-or-later (GPL for some builds) |
| `libopencv-dev` | [OpenCV](https://opencv.org/) | Apache-2.0 (4.5+) |
| `libglfw3-dev` | [GLFW](https://www.glfw.org/) | Zlib |
| `libvulkan-dev`, `glslang-tools` | [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader), [glslang](https://github.com/KhronosGroup/glslang) | Apache-2.0 / BSD-3-Clause |
| `libglew-dev` | [GLEW](https://glew.sourceforge.net/) | BSD-3-Clause / MIT |
| `libglm-dev` | [GLM](https://github.com/g-truc/glm) | MIT (Happy Bunny / MIT) |
| `libva-dev` | [libva (VAAPI)](https://github.com/intel/libva) | MIT |
| `libdw-dev` | [elfutils](https://sourceware.org/elfutils/) | LGPL-3.0-or-later / GPL-2.0-or-later |

FFmpeg and elfutils are copyleft-licensed. They are consumed as pre-existing
shared libraries installed on the test node by the distribution's package manager,
and no `rocm-tests` code is derived from them. Any distribution flow that bundles
these libraries or binaries linked against them must satisfy the corresponding
LGPL/GPL obligations independently; consult the license text shipped with each
distribution package for the authoritative terms.

---

### 3. Python runtime dependencies

`_install_runtime_python_deps` performs a best-effort
`pip install -r <rock_dir>/libexec/rocprofiler-compute/requirements.txt`, needed by
the optional `tools` samples. The requirements file ships with the ROCm profiler
artifact, so the exact package set and versions are determined by the installed
ROCm build, not by this repository. The resulting packages are installed from PyPI
onto the test node and are neither vendored nor redistributed here; each retains
its own upstream license.

---

### 4. ROCm toolchain and runtime

The suite is compiled with the ROCm installation on the target host and links
against that install's HIP runtime and libraries. The ROCm stack is not distributed
by this repository and retains the license terms shipped with the installation.

---

## Redistribution Guidance

The `rocm-tests` files in this directory are MIT-licensed first-party test code.
The `rocm-examples` checkout, the apt/PyPI packages, and everything built from them
are runtime artifacts under `output/` or node-level installs, not vendored source
in this repository. If a release, container image, build cache, or test artifact
bundle includes the cloned source, the built binaries, or the third-party libraries
listed above, include the corresponding upstream license files with that
distributed material and satisfy the copyleft obligations noted in section 2.

---

## First-Party Test Code

`conftest.py` and `test_rocm_examples.py` in this directory are original
AMD-authored code, copyright Advanced Micro Devices, Inc., licensed under the MIT
License (the same license as the parent repository). They do not derive from
`rocm-examples` source code; they clone, configure, build, and execute it and
assert on the results.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. It is an engineering-compliance summary, not
legal advice; final sign-off for any product distribution should come from AMD
OSS/legal review. For questions about licensing, consult the upstream projects
linked above.*
