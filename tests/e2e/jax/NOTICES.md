# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent repository
([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)) and are governed by its
primary MIT license. During execution, these scripts clone, build, and run
external projects that carry their own licensing and copyright terms.

The purpose of this notice is attribution and provenance clarity. The repository
does not vendor or redistribute the `TransformerEngine` source tree, the JAX
wheels, or any built binaries; the test fixtures obtain them from upstream at
runtime. The upstream checkouts and wheels retain their own license files. If any
downstream packaging flow redistributes the cloned source or built artifacts,
that redistribution must retain the upstream copyright notices, license terms,
and disclaimers.

---

## Third-Party Runtime Dependencies

### 1. Transformer Engine (ROCm/TransformerEngine)

This test suite clones [ROCm/TransformerEngine](https://github.com/ROCm/TransformerEngine)
at runtime, builds it, and runs its own JAX and C++ unit-test suites
(`ci/jax.sh` and `tests/cpp`).

`ROCm/TransformerEngine` is a fork of
[NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine). The
upstream project is licensed under the Apache License 2.0; AMD's ROCm port is
dual-licensed, adding an MIT license for AMD modifications.

- **Original Work:** Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **Modifications:** Copyright (c) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
- **Licenses:** Apache License 2.0 (upstream) and MIT License (AMD modifications)
- **Upstream repository:** https://github.com/ROCm/TransformerEngine
- **Upstream license file:** https://github.com/ROCm/TransformerEngine/blob/dev/LICENSE
- **Original project:** https://github.com/NVIDIA/TransformerEngine

#### License — Apache 2.0 (upstream, summary)

```
Copyright (c) NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The full Apache 2.0 text and the AMD MIT license are reproduced in the upstream
`LICENSE` file linked above.

---

### 2. JAX and jaxlib (ROCm plugins)

The `te_jax_ut` suite installs `jaxlib`, the JAX ROCm plugin, and the JAX ROCm
plugin native wheels, then installs `jax` from a matching index, to run the
Transformer Engine JAX tests.

[JAX](https://github.com/jax-ml/jax) is licensed under the Apache License 2.0.
The ROCm plugin wheels are distributed by AMD and follow the same license terms.
The tests do not redistribute these wheels; they are downloaded from the
configured artifact index at runtime.

- **Copyright:** Copyright 2018 The JAX Authors.
- **License:** Apache License 2.0
- **Upstream repository:** https://github.com/jax-ml/jax
- **Upstream license file:** https://github.com/jax-ml/jax/blob/main/LICENSE

---

### 3. ROCm Runtime

The built Transformer Engine binaries and the JAX ROCm plugins link against
libraries from the installed ROCm stack (HIP runtime, LLVM device libraries,
communication and math libraries). These are separate host-level runtime
dependencies supplied by the ROCm installation on the target node and are **not**
distributed within this repository. Refer to the ROCm component licenses for
compliance requirements.

---

## Redistribution Guidance

The `rocm-tests` source files in this directory are MIT-licensed first-party test
code. The `TransformerEngine` checkout, the JAX/TE wheels, and any built binaries
are runtime artifacts, not vendored source in this repository. If a release,
container image, cache, or test-artifact bundle includes the cloned
`TransformerEngine` source, the downloaded wheels, or their build outputs, include
the corresponding upstream license and notice files with that distributed
material.

---

## First-Party Test Code

The Python source files in this directory (`conftest.py`,
`_transformer_engine_jax.py`, `test_transformer_engine_jax.py`) are original
AMD-authored code, copyright Advanced Micro Devices, Inc., and are licensed under
the MIT License (the same license as the parent repository). They do not derive
from `TransformerEngine` or JAX source code.

---

*This file is provided for compliance with the attribution clauses of the external
dependencies used by this module. For questions about licensing, consult the
upstream repositories linked above.*
