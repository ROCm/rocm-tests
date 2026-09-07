// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// This test is a self-contained diagnostic for MIOpen non-packed and
// non-trivially-strided tensor handling during 3-D convolution.
//
// The test does two things:
//
//  1. It confirms that the MIOpen convolution is correct when the input
//     tensor is non-trivially strided (non-packed, non-trivially-strided,
//     or NP-NTS).
//
//  2. It confirms that MIOpen correctly distinguishes a strided input
//     tensor descriptor from a packed one, producing output that matches
//     a packed-buffer/packed-descriptor reference convolution.
//
// The input tensor is a 5-D blob with logical shape {4, 4, 16, 9, 16}
// and non-trivial strides {10240, 2560, 160, 16, 1}. The reference
// (fully packed) tensor has strides {9216, 2304, 144, 16, 1}. A 3-D
// convolution with a 3x3x3 filter (8 output channels) is run on both
// descriptors. If MIOpen handles the non-trivial strides correctly the
// two convolutions produce identical output and the binary exits 0,
// printing "W00t!" to stdout.
//
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <miopen/miopen.h>

#define CHECK_HIP(hip_call)                                               \
    do                                                                    \
    {                                                                     \
        const hipError_t status_check_hip = (hip_call);                   \
        if (status_check_hip != hipSuccess)                               \
        {                                                                 \
            std::cerr << "HIP error ("                                    \
                      << hipGetErrorName(status_check_hip) << ") at "     \
                      << __FILE__ << ":" << __LINE__ << ": "              \
                      << hipGetErrorString(status_check_hip)              \
                      << std::endl;                                       \
            static_cast<void>(hipDeviceReset());                          \
            std::terminate();                                             \
        }                                                                 \
    } while (0)

#define CHECK_MIOPEN(miopen_call)                                         \
    do                                                                    \
    {                                                                     \
        miopenStatus_t const status_check_miopen = (miopen_call);         \
        if (status_check_miopen != miopenStatusSuccess)                   \
        {                                                                 \
            std::cerr << "MIOpen error at "                               \
                      << __FILE__ << ":" << __LINE__ << ": "              \
                      << miopenGetErrorString(status_check_miopen)        \
                      << std::endl;                                       \
            static_cast<void>(hipDeviceReset());                          \
            std::terminate();                                             \
        }                                                                 \
    } while (0)

int main()
{
    CHECK_HIP(hipSetDevice(0));

    // Input tensor: non-packed layout (NP-NTS).
    std::vector<int> in_dims    = {4, 4, 16, 9, 16};
    std::vector<int> in_strides = {10240, 2560, 160, 16, 1};

    // Reference input: minimal packed layout with the same logical shape.
    std::vector<int> packed_in_strides = {9216, 2304, 144, 16, 1};

    // Output tensor.
    std::vector<int> out_dims    = {4, 8, 8, 4, 8};
    std::vector<int> out_strides = {2048, 256, 32, 8, 1};

    int const& in_channels  = in_dims[1];   // 4
    int const& out_channels = out_dims[1];  // 8

    std::vector<int> filter_dims    = {out_channels, in_channels, 3, 3, 3};
    std::vector<int> filter_strides = {108, 27, 9, 3, 1};
    std::vector<int> pad            = {1, 0, 1};
    std::vector<int> stride_conv    = {2, 2, 2};
    std::vector<int> dilation       = {1, 1, 1};

    // --- MIOpen setup ---
    miopenHandle_t handle;
    CHECK_MIOPEN(miopenCreate(&handle));

    miopenConvolutionDescriptor_t conv_desc;
    CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&conv_desc));
    CHECK_MIOPEN(miopenInitConvolutionNdDescriptor(conv_desc,
                                                   /*spatial_dim=*/3,
                                                   pad.data(),
                                                   stride_conv.data(),
                                                   dilation.data(),
                                                   miopenConvolution));
    CHECK_MIOPEN(miopenSetConvolutionGroupCount(conv_desc, 1));

    miopenTensorDescriptor_t in_desc, packed_in_desc, filter_desc, out_desc;

    // Non-packed (NP-NTS) input descriptor.
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&in_desc));
    CHECK_MIOPEN(miopenSetTensorDescriptor(in_desc,
                                           miopenFloat,
                                           5,
                                           in_dims.data(),
                                           in_strides.data()));

    // Packed input descriptor (same logical shape, minimal strides).
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&packed_in_desc));
    CHECK_MIOPEN(miopenSetTensorDescriptor(packed_in_desc,
                                           miopenFloat,
                                           5,
                                           in_dims.data(),
                                           packed_in_strides.data()));

    // Filter and output are fully packed.
    CHECK_MIOPEN(miopenCreateTensorDescriptor(&filter_desc));
    CHECK_MIOPEN(miopenSetTensorDescriptor(filter_desc,
                                           miopenFloat,
                                           5,
                                           filter_dims.data(),
                                           filter_strides.data()));

    CHECK_MIOPEN(miopenCreateTensorDescriptor(&out_desc));
    CHECK_MIOPEN(miopenSetTensorDescriptor(out_desc,
                                           miopenFloat,
                                           5,
                                           out_dims.data(),
                                           out_strides.data()));

    size_t workspace_size, workspace_size_packed;
    CHECK_MIOPEN(miopenConvolutionForwardGetWorkSpaceSize(handle,
                                                          filter_desc,
                                                          in_desc,
                                                          conv_desc,
                                                          out_desc,
                                                          &workspace_size));
    CHECK_MIOPEN(miopenConvolutionForwardGetWorkSpaceSize(handle,
                                                          filter_desc,
                                                          packed_in_desc,
                                                          conv_desc,
                                                          out_desc,
                                                          &workspace_size_packed));

    // --- Memory allocation ---
    size_t const input_size        = in_dims[0] * in_strides[0];
    size_t const packed_input_size = in_dims[0] * packed_in_strides[0];
    size_t const filter_size       = filter_dims[0] * filter_strides[0];
    size_t const output_size       = out_dims[0] * out_strides[0];

    size_t const input_bytes        = input_size * sizeof(float);
    size_t const packed_input_bytes = packed_input_size * sizeof(float);
    size_t const filter_bytes       = filter_size * sizeof(float);
    size_t const output_bytes       = output_size * sizeof(float);

    float *d_input, *d_packed_input, *d_filter, *d_output, *d_workspace;
    CHECK_HIP(hipMalloc(&d_input, input_bytes));
    CHECK_HIP(hipMalloc(&d_packed_input, packed_input_bytes));
    CHECK_HIP(hipMalloc(&d_filter, filter_bytes));
    CHECK_HIP(hipMalloc(&d_output, output_bytes));
    CHECK_HIP(hipMalloc(&d_workspace, std::max(workspace_size, workspace_size_packed)));

    // --- Fill inputs ---
    {
        float *h_input, *h_packed_input;
        CHECK_HIP(hipHostMalloc(&h_input, input_bytes, hipHostMallocDefault));
        CHECK_HIP(hipHostMalloc(&h_packed_input, packed_input_bytes, hipHostMallocDefault));

        std::fill_n(h_input, input_size, 1.f);
        std::fill_n(h_packed_input, packed_input_size, 1.f);

        // Zero out the H=0 slice so that non-packed vs packed strides differ.
        int const H = 0;
        for (int N = 0; N < in_dims[0]; ++N)
            for (int C = 0; C < in_dims[1]; ++C)
                for (int D = 0; D < in_dims[2]; ++D)
                    for (int W = 0; W < in_dims[4]; ++W)
                    {
                        h_input[N * in_strides[0]
                                + C * in_strides[1]
                                + D * in_strides[2]
                                + H * in_strides[3]
                                + W * in_strides[4]] = 0.f;

                        h_packed_input[N * packed_in_strides[0]
                                       + C * packed_in_strides[1]
                                       + D * packed_in_strides[2]
                                       + H * packed_in_strides[3]
                                       + W * packed_in_strides[4]] = 0.f;
                    }

        CHECK_HIP(hipMemcpy(d_input, h_input, input_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_packed_input, h_packed_input, packed_input_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipHostFree(h_packed_input));
        CHECK_HIP(hipHostFree(h_input));
    }
    {
        float* h_filter;
        CHECK_HIP(hipHostMalloc(&h_filter, filter_bytes, hipHostMallocDefault));
        std::fill_n(h_filter, filter_size, 1.f);
        CHECK_HIP(hipMemcpy(d_filter, h_filter, filter_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipHostFree(h_filter));
    }

    // --- Algorithm selection ---
    miopenConvFwdAlgorithm_t algo, algo_packed;
    {
        miopenConvAlgoPerf_t results[10];
        int result_count = -1;
        CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(
                         handle,
                         in_desc, d_input,
                         filter_desc, d_filter,
                         conv_desc,
                         out_desc, d_output,
                         /*requestAlgoCount=*/10, &result_count, results,
                         d_workspace, workspace_size,
                         /*exhaustiveSearch=*/true));
        algo = results[0].fwd_algo;

        CHECK_MIOPEN(miopenFindConvolutionForwardAlgorithm(
                         handle,
                         packed_in_desc, d_input,
                         filter_desc, d_filter,
                         conv_desc,
                         out_desc, d_output,
                         /*requestAlgoCount=*/10, &result_count, results,
                         d_workspace, workspace_size_packed,
                         /*exhaustiveSearch=*/true));
        algo_packed = results[0].fwd_algo;
    }

    float *h_output, *h_output_packed, *h_output_correct;
    CHECK_HIP(hipHostMalloc(&h_output, output_bytes));
    CHECK_HIP(hipHostMalloc(&h_output_packed, output_bytes));
    CHECK_HIP(hipHostMalloc(&h_output_correct, output_bytes));

    float const alpha = 1.f;
    float const beta  = 0.f;

    // Convolution 1: non-packed buffer with non-packed descriptor (target path).
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_MIOPEN(miopenConvolutionForward(handle,
                                          &alpha,
                                          in_desc, d_input,
                                          filter_desc, d_filter,
                                          conv_desc,
                                          algo,
                                          &beta,
                                          out_desc, d_output,
                                          d_workspace, workspace_size));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_output, d_output, output_bytes, hipMemcpyDeviceToHost));

    // Convolution 2: non-packed buffer with packed descriptor (intentional mismatch).
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_MIOPEN(miopenConvolutionForward(handle,
                                          &alpha,
                                          packed_in_desc, d_input,
                                          filter_desc, d_filter,
                                          conv_desc,
                                          algo,
                                          &beta,
                                          out_desc, d_output,
                                          d_workspace, workspace_size_packed));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_output_packed, d_output, output_bytes, hipMemcpyDeviceToHost));

    // Convolution 3: packed buffer with packed descriptor (correct reference).
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_MIOPEN(miopenConvolutionForward(handle,
                                          &alpha,
                                          packed_in_desc, d_packed_input,
                                          filter_desc, d_filter,
                                          conv_desc,
                                          algo_packed,
                                          &beta,
                                          out_desc, d_output,
                                          d_workspace, workspace_size_packed));
    CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(h_output_correct, d_output, output_bytes, hipMemcpyDeviceToHost));
    CHECK_HIP(hipDeviceSynchronize());

    // Verify the reference output against known-correct values.
    for (int N = 0; N < out_dims[0]; ++N)
        for (int C = 0; C < out_dims[1]; ++C)
            for (int D = 0; D < out_dims[2]; ++D)
            {
                float const n_ones_D = (D == 0 ? 2.f : 3.f);
                for (int H = 0; H < out_dims[3]; ++H)
                {
                    float const n_ones_H = (H == 0 ? 2.f : 3.f);
                    for (int W = 0; W < out_dims[4]; ++W)
                    {
                        float const n_ones_W    = (W == 0 ? 2.f : 3.f);
                        float const expected    = n_ones_D * n_ones_H * n_ones_W * in_channels;
                        size_t const idx        = N * out_strides[0]
                                                  + C * out_strides[1]
                                                  + D * out_strides[2]
                                                  + H * out_strides[3]
                                                  + W * out_strides[4];
                        if (h_output_correct[idx] != expected)
                        {
                            std::cerr << "Reference (packed buf, packed desc) produced unexpected value.\n"
                                      << "  Index: (" << N << "," << C << "," << D << "," << H << "," << W << ")\n"
                                      << "  Value: " << h_output_correct[idx]
                                      << "  Expected: " << expected << std::endl;
                            std::terminate();
                        }
                    }
                }
            }

    // Count differences between the NP-NTS result and the two references.
    size_t diff_to_packed  = 0UL;
    size_t diff_to_correct = 0UL;
    for (size_t i = 0; i < output_size; ++i)
    {
        if (h_output[i] != h_output_packed[i])  ++diff_to_packed;
        if (h_output[i] != h_output_correct[i]) ++diff_to_correct;
    }

    std::cout << "The output tensor has " << output_size << " total entries.\n\n"
              << "Comparing (non-packed buf, non-packed desc) to "
              << "(non-packed buf, packed desc)...\n\nDetected "
              << diff_to_packed << " differences.\n\n";

    if (diff_to_packed == 0)
        std::cout << " --> NOTE!!! This is a BAD thing. It means that MIOpen is ignoring the\n"
                  << "     non-packed strides set in the input tensor's descriptor!\n\n";
    else
        std::cout << " --> This *might* be ok. This means that MIOpen is doing something different\n"
                  << "     when non-packed strides are passed to the input tensor descriptor.\n"
                  << "     See below for more information.\n\n";

    std::cout << "Comparing (non-packed buf, non-packed desc) to "
              << "(packed buf, packed desc)...\n\nDetected "
              << diff_to_correct << " differences.\n\n";
    if (diff_to_correct == 0)
        std::cout << " --> W00t! The output of the convolution when the input tensor is non-trivially\n"
                  << "     strided appears to be correct and this issue can (probably) be closed!\n"
                  << std::endl;
    else
        std::cout << " --> Yes, this is BAD. The output of the convolution when the input tensor is\n"
                  << "     non-trivially strided is not correct.\n"
                  << std::endl;

    // Cleanup
    CHECK_HIP(hipHostFree(h_output_correct));
    CHECK_HIP(hipHostFree(h_output_packed));
    CHECK_HIP(hipHostFree(h_output));

    CHECK_HIP(hipFree(d_workspace));
    CHECK_HIP(hipFree(d_output));
    CHECK_HIP(hipFree(d_filter));
    CHECK_HIP(hipFree(d_packed_input));
    CHECK_HIP(hipFree(d_input));

    CHECK_MIOPEN(miopenDestroyTensorDescriptor(out_desc));
    CHECK_MIOPEN(miopenDestroyTensorDescriptor(filter_desc));
    CHECK_MIOPEN(miopenDestroyTensorDescriptor(packed_in_desc));
    CHECK_MIOPEN(miopenDestroyTensorDescriptor(in_desc));
    CHECK_MIOPEN(miopenDestroyConvolutionDescriptor(conv_desc));
    CHECK_MIOPEN(miopenDestroy(handle));

    CHECK_HIP(hipDeviceReset());
    std::cout << "Goodbye!" << std::endl;
    return diff_output_to_correct;

    return static_cast<int>(diff_to_correct);
}
