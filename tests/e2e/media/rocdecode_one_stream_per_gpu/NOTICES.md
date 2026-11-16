# Notices and Attributions

The end-to-end test scripts in this directory are part of the parent
`rocm-tests` repository and are governed by the repository's MIT license.

`data/sample_stream.h264` is a copy of a rocDecode sample elementary stream.
rocDecode is third-party software relative to this test tree and retains its
own upstream license terms, including the codec-standards notice below.

All third-party software is provided "as is," without warranty of any kind,
express or implied, by the authors or copyright holders of `rocm-tests`.

## rocDecode sample stream

This directory vendors:

`data/sample_stream.h264`

copied from:

https://github.com/ROCm/rocDecode/blob/597b29a7c9695277f768759bd377dbfa8ad98236/data/videos/AMD_driving_virtual_20-H264.264

Source commit: `597b29a7c9695277f768759bd377dbfa8ad98236` (ROCm/rocDecode#473).
Git blob SHA-1: `3ea8eb4998bb37016fdbe385bbd7c2d9e1a9d1cb`.

Used by: `tests/e2e/media/rocdecode_one_stream_per_gpu/` (one H.264 stream per
compute partition).

rocDecode has since moved to `ROCm/rocm-systems` (`projects/rocdecode`). The
blob above is byte-identical to the file vendored here.

rocDecode is licensed under the MIT License, with an additional notice that
AMD does not grant rights in third-party codec standards such as AVC/H.264.

Upstream license:

https://github.com/ROCm/rocDecode/blob/develop/LICENSE

https://rocm.docs.amd.com/projects/rocDecode/en/latest/license.html

## Redistribution Guidance

The `rocm-tests` source files in this directory are AMD-authored test code under
the repository MIT license.

`data/sample_stream.h264` is redistributed rocDecode test media. Keep the
rocDecode copyright notice, MIT terms, and codec-standards notice with any
redistribution of that file.

## rocDecode license (MIT, with codec-standards notice)

```
MIT License

Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved

NOTICE REGARDING STANDARDS

AMD does not provide a license or sublicense to any Intellectual Property Rights
relating to any standards, including but not limited to any audio and/or video
codec technologies such as MPEG-2, MPEG-4; AVC/H.264; HEVC/H.265; AAC decode/FFMPEG;
AAC encode/FFMPEG; VC-1; and MP3 (collectively, the "Media Technologies").
For clarity, you will pay any royalties due for such third party technologies,
which may include the Media Technologies that are owed as a result of
AMD providing the Software to you.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
