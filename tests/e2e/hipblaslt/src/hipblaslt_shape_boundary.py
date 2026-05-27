#!/usr/bin/env python3
# **************************************************************************
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
# *************************************************************************/


import sys

import torch
import torch.nn as nn

torch.set_default_device("cuda:0")


def run_test(layer, x):
    """
    Execute forward pass and capture result.

    Returns:
        tuple: (success: bool, error: Exception or None)
    """
    try:
        layer(x)
        return True, None
    except RuntimeError as e:
        return False, e


def create_test_cases():
    """
    Define all test cases with expected outcomes.

    Returns:
        list: Test case tuples (id, in_features, tokens, batch, expected, description)
    """
    return [
        # Single batch tests - token count boundaries
        (1, 2048, 32767, 1, True),
        (2, 2047, 32784, 1, True),
        (3, 2047, 32785, 1, True),
        (4, 1024, 65535, 1, True),
        (5, 1024, 65536, 1, True),
        # Multi-batch tests
        (6, 2048, 32767, 16, True),
        (7, 2048, 32768, 16, True),
    ]


def test_shape_boundaries():
    """Test matrix dimension boundaries for hipBLASLt operations."""
    test_cases = create_test_cases()
    results = []

    print("=" * 80)
    print("Test Group: Matrix Shape Boundaries")
    print("=" * 80)

    for test_id, in_features, tokens, batch, expected in test_cases:
        layer = nn.Linear(in_features, in_features)
        x = torch.randn((batch, tokens, in_features))[:, 0]

        success, _error = run_test(layer, x)
        passed = success == expected

        status = "PASS" if passed else "FAIL"
        result_str = "OK" if success else "ERROR"

        print(f"Test {test_id}: [{status}]")
        print(f"         Shape: ({batch}, {tokens}, {in_features}) -> {result_str}")

        if not passed:
            print(
                f"         Expected: {'SUCCESS' if expected else 'FAILURE'},"
                f" Got: {'SUCCESS' if success else 'FAILURE'}"
            )

        results.append(passed)

    return results


def test_activation_ordering():
    """Test that operation ordering affects boundary behavior."""
    results = []

    print("\n" + "=" * 80)
    print("Test Group: Activation Function Ordering")
    print("=" * 80)

    layer = nn.Sequential(nn.ReLU(), nn.Linear(2048, 2048))
    x = torch.randn((1, 32768, 2048))[:, 0]
    success, _error = run_test(layer, x)
    expected = True
    passed = success == expected

    status = "PASS" if passed else "FAIL"
    print(f"Test 8: [{status}] ReLU->Linear at boundary (32768 tokens)")
    print(f"         Expected: SUCCESS, Got: {'SUCCESS' if success else 'FAILURE'}")
    results.append(passed)

    return results


def print_summary(all_results):
    """Print test execution summary."""
    total = len(all_results)
    passed = sum(all_results)
    failed = total - passed

    print("\n" + "=" * 80)
    print("Test Execution Summary")
    print("=" * 80)
    print(f"Total:  {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)

    if failed > 0:
        print("\nResult: FAILED - Some tests did not meet expectations")
        return 1
    else:
        print("\nResult: PASSED - All tests met expectations")
        return 0


def main():
    """Execute all test groups and report results."""
    print("\nhipBLASLt Shape Boundary Test Suite")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}\n")

    all_results = []

    # Run test groups
    all_results.extend(test_shape_boundaries())
    all_results.extend(test_activation_ordering())

    # Print summary and exit with appropriate code
    exit_code = print_summary(all_results)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
