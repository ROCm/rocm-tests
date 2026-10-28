# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""tf32_matmul_workload.py -- TF32 matrix multiplication workload for hipBLASLt validation.

Runs a warm-up followed by a timed nn.Linear forward pass at dimensions
(M=8196, K=512, N=3456) with TF32 enabled, then exports a Chrome profiler trace.
Exits 0 on success; non-zero on any exception.
"""

import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True

# Define dimensions
M, K, N = 8196, 512, 3456

# Create input tensors on GPU
input_tensor = torch.randn((M, K), dtype=torch.float32, device="cuda")
linear_layer = nn.Linear(K, N, dtype=torch.float32, device="cuda")

# Warm up
for _ in range(10):
    output = linear_layer(input_tensor)
torch.cuda.synchronize()

# Timed forward passes
start_time = time.time()
for _ in range(10):
    output = linear_layer(input_tensor)
    torch.cuda.synchronize()
end_time = time.time()

average_time = (end_time - start_time) / 10.0
print(f"Average time taken for matrix multiplication over 10 runs: {average_time * 1000.0} ms")

# Profiling with Chrome trace export
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        output = linear_layer(input_tensor)
    s.synchronize()
    torch.cuda.current_stream().wait_stream(s)

prof.export_chrome_trace(f"trace_forward_matmul_2p6_{M}_{N}_{K}.json")
print(f"Profiler trace written: trace_forward_matmul_2p6_{M}_{N}_{K}.json")
