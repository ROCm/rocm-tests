# Notices and Attributions

The test scripts in this directory are governed by the primary MIT license of
the parent repository ([ROCm/rocm-tests](https://github.com/ROCm/rocm-tests)).

## Vendored source

`src/hipFile_system_test.cpp` is an AMD-authored test program (Copyright (c)
2026 Advanced Micro Devices, Inc., MIT-style license; the full permission
notice is preserved in the file header). `src/CMakeLists.txt` is adapted from
the same suite.

## Runtime dependency

During execution the test fixtures sparse-clone `projects/hipfile` from
[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems) (MIT) into the
gitignored `output/` build tree and build it into a private prefix. The
repository does not redistribute that source or its built binaries.
