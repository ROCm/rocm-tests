# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

This directory does not vendor decode media. At configure time CMake downloads:

`AMD_driving_virtual_20-H264.264`

from:

https://github.com/ROCm/rocm-systems/blob/develop/projects/rocdecode/data/videos/AMD_driving_virtual_20-H264.264

Git blob SHA-1: `3ea8eb4998bb37016fdbe385bbd7c2d9e1a9d1cb`.
File SHA-1: `c89d1c82e38b962e64d7a980265fb22b252f702c`.

The file stays in the build directory. It is the rocDecode H.264 elementary
stream, the same bytes previously copied here as `data/sample_stream.h264`.

rocDecode is licensed under the MIT License, with an additional notice that
AMD does not grant rights in third-party codec standards such as AVC/H.264.

https://github.com/ROCm/rocm-systems/blob/develop/projects/rocdecode/LICENSE

https://rocm.docs.amd.com/projects/rocDecode/en/latest/license.html
