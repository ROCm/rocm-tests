#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cassert>
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>

// HIP Runtime
#include <hip/hip_runtime.h>

// Using updated interfaces for hipBLAS and hipSPARSE
#include <hipblas.h>
#include <hipsparse.h>

const char *sSDKname = "conjugateGradientPrecond_ROCm_MultiGPU";

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s:%d - %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HIPBLAS_CHECK(call) \
    do { \
        hipblasStatus_t status = call; \
        if (status != HIPBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "hipBLAS error in %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define HIPSPARSE_CHECK(call) \
    do { \
        hipsparseStatus_t status = call; \
        if (status != HIPSPARSE_STATUS_SUCCESS) { \
            fprintf(stderr, "hipSPARSE error in %s:%d - %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// STRESS TESTING KERNELS
// ============================================================================

// Memory stress test - intensive memory access pattern
__global__ void memoryStressKernel(float *data, size_t size, int iterations) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = (float)(idx % 1000) + 1.0f;  // Initialize with non-zero value
        for (int i = 0; i < iterations; i++) {
            val = val * 1.0001f + 0.0001f;
            val = sqrtf(fabsf(val));
            if (val > 1000.0f) val = 1.0f;  // Prevent overflow
        }
        data[idx] = val;
    }
}

// Compute stress test - intensive compute operations
__global__ void computeStressKernel(float *data, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = (float)(idx % 100) / 100.0f + 0.1f;  // Initialize with valid value
        for (int i = 0; i < 1000; i++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
            if (fabsf(val) > 10.0f) val = 0.5f;  // Keep in reasonable range
        }
        data[idx] = val;
    }
}

// Bandwidth test kernel - sequential memory access
__global__ void bandwidthTestKernel(const float *src, float *dst, size_t size) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// MULTI-GPU CONTEXT STRUCTURE
// ============================================================================

struct GPUContext {
    int deviceId;
    hipblasHandle_t blasHandle;
    hipsparseHandle_t sparseHandle;
    hipStream_t stream;
    
    // Device pointers
    int *d_col, *d_row;
    float *d_val, *d_x, *d_y, *d_r, *d_p, *d_omega;
    float *d_valsILU0;
    float *d_zm1, *d_zm2, *d_rm2;
    
    // Stress test buffers
    float *d_stress_buffer;
    size_t stress_buffer_size;
    
    // Descriptors
    hipsparseDnVecDescr_t vecp, vecX, vecY, vecR, vecZM1, vecomega;
    hipsparseSpMatDescr_t matA, matM_lower, matM_upper;
    hipsparseSpSVDescr_t spsvDescrL, spsvDescrU;
    hipsparseMatDescr_t descr, matLU;
    csrilu02Info_t infoILU;
    
    // Buffers
    void *d_bufferMV;
    
    // Local matrix info for distribution
    int local_N;
    int local_nz;
    int row_offset;
    
    // Timing
    hipEvent_t start_event, stop_event;
};

// ============================================================================
// LAPLACE MATRIX GENERATION (Same as CUDA version)
// ============================================================================

void genLaplace(int *row_ptr, int *col_ind, float *val, int M, int N, int nz, float *rhs)
{
    (void)M;
    (void)nz;
    assert(M == N);
    int n = (int)sqrt((double)N);
    assert(n * n == N);
    printf("Laplace dimension = %d\n", n);
    int idx = 0;

    // Loop over degrees of freedom
    for (int i = 0; i < N; i++) {
        int ix = i % n;
        int iy = i / n;

        row_ptr[i] = idx;

        // up
        if (iy > 0) {
            val[idx]     = 1.0;
            col_ind[idx] = i - n;
            idx++;
        }
        else {
            rhs[i] -= 1.0;
        }

        // left
        if (ix > 0) {
            val[idx]     = 1.0;
            col_ind[idx] = i - 1;
            idx++;
        }
        else {
            rhs[i] -= 0.0;
        }

        // center
        val[idx]     = -4.0;
        col_ind[idx] = i;
        idx++;

        // right
        if (ix < n - 1) {
            val[idx]     = 1.0;
            col_ind[idx] = i + 1;
            idx++;
        }
        else {
            rhs[i] -= 0.0;
        }

        // down
        if (iy < n - 1) {
            val[idx]     = 1.0;
            col_ind[idx] = i + n;
            idx++;
        }
        else {
            rhs[i] -= 0.0;
        }
    }

    row_ptr[N] = idx;
}

// ============================================================================
// STRESS TEST FUNCTIONS
// ============================================================================

void runMemoryStressTest(std::vector<GPUContext>& contexts, int stress_level) {
    printf("\n>>> Running Memory Stress Test (Level %d)...\n", stress_level);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t gpu = 0; gpu < contexts.size(); gpu++) {
        HIP_CHECK(hipSetDevice(contexts[gpu].deviceId));
        
        if (contexts[gpu].d_stress_buffer && contexts[gpu].stress_buffer_size > 0) {
            size_t total_size = contexts[gpu].stress_buffer_size / sizeof(float);
            unsigned int threads = 256;
            
            // Use 2D grid to handle large buffers
            size_t total_threads_needed = total_size;
            unsigned int blocks_x = 65535;  // Max x dimension
            unsigned int blocks_y = (unsigned int)((total_threads_needed + (blocks_x * threads) - 1) / (blocks_x * threads));
            
            // Limit y dimension as well
            if (blocks_y > 65535) blocks_y = 65535;
            
            size_t actual_size = (size_t)blocks_x * blocks_y * threads;
            if (actual_size > total_size) actual_size = total_size;
            
            printf("  GPU %zu: Launching %ux%u blocks x %u threads (%.2f M elements, %.2f GB)\n", 
                   gpu, blocks_x, blocks_y, threads, actual_size / 1e6, 
                   (actual_size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
            
            // Launch multiple kernels to cover the entire buffer
            size_t offset = 0;
            int num_launches = (int)((total_size + actual_size - 1) / actual_size);
            
            for (int launch = 0; launch < num_launches && offset < total_size; launch++) {
                size_t launch_size = (offset + actual_size <= total_size) ? actual_size : (total_size - offset);
                
                hipLaunchKernelGGL(memoryStressKernel, dim3(blocks_x, blocks_y), dim3(threads), 0, 
                                  contexts[gpu].stream,
                                  contexts[gpu].d_stress_buffer + offset, launch_size, stress_level * 100);
                
                // Check for kernel launch errors immediately
                hipError_t err = hipGetLastError();
                if (err != hipSuccess) {
                    fprintf(stderr, "Kernel launch error on GPU %zu: %s\n", gpu, hipGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
                
                offset += launch_size;
            }
            
            printf("  GPU %zu: Launched %d kernel batches to process %.2f GB\n", 
                   gpu, num_launches, (total_size * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
        }
    }
    
    // Synchronize all GPUs with error checking
    for (size_t gpu = 0; gpu < contexts.size(); gpu++) {
        HIP_CHECK(hipSetDevice(contexts[gpu].deviceId));
        HIP_CHECK(hipStreamSynchronize(contexts[gpu].stream));
        printf("  GPU %zu: Synchronized successfully\n", gpu);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf(">>> Memory Stress Test completed in %ld ms\n\n", duration.count());
}

void runComputeStressTest(std::vector<GPUContext>& contexts) {
    printf("\n>>> Running Compute Stress Test...\n");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t gpu = 0; gpu < contexts.size(); gpu++) {
        HIP_CHECK(hipSetDevice(contexts[gpu].deviceId));
        
        if (contexts[gpu].d_stress_buffer && contexts[gpu].stress_buffer_size > 0) {
            size_t size = contexts[gpu].stress_buffer_size / sizeof(float);
            unsigned int threads = 256;
            unsigned int blocks = (unsigned int)((size + threads - 1) / threads);
            
            // Limit blocks to avoid launch issues
            if (blocks > 65535) {
                blocks = 65535;
                size = (size_t)blocks * threads;
            }
            
            printf("  GPU %zu: Launching %u blocks x %u threads (%.2f M elements)\n", 
                   gpu, blocks, threads, size / 1e6);
            
            hipLaunchKernelGGL(computeStressKernel, dim3(blocks), dim3(threads), 0,
                              contexts[gpu].stream,
                              contexts[gpu].d_stress_buffer, size);
            
            // Check for kernel launch errors
            hipError_t err = hipGetLastError();
            if (err != hipSuccess) {
                fprintf(stderr, "Compute kernel launch error on GPU %zu: %s\n", gpu, hipGetErrorString(err));
                exit(EXIT_FAILURE);
            }
        }
    }
    
    // Synchronize all GPUs
    for (size_t gpu = 0; gpu < contexts.size(); gpu++) {
        HIP_CHECK(hipSetDevice(contexts[gpu].deviceId));
        HIP_CHECK(hipStreamSynchronize(contexts[gpu].stream));
        printf("  GPU %zu: Compute stress synchronized\n", gpu);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf(">>> Compute Stress Test completed in %ld ms\n\n", duration.count());
}

void runBandwidthTest(std::vector<GPUContext>& contexts) {
    printf("\n>>> Running Bandwidth Test...\n");
    
    for (size_t gpu = 0; gpu < contexts.size(); gpu++) {
        HIP_CHECK(hipSetDevice(contexts[gpu].deviceId));
        
        if (contexts[gpu].d_stress_buffer && contexts[gpu].stress_buffer_size > 0) {
            size_t size = contexts[gpu].stress_buffer_size / sizeof(float);
            unsigned int threads = 256;
            unsigned int blocks = (unsigned int)((size + threads - 1) / threads);
            
            // Limit blocks to avoid launch issues
            if (blocks > 65535) {
                blocks = 65535;
                size = (size_t)blocks * threads;
            }
            
            float *d_temp;
            HIP_CHECK(hipMalloc(&d_temp, size * sizeof(float)));
            HIP_CHECK(hipMemset(d_temp, 0, size * sizeof(float)));
            
            HIP_CHECK(hipEventRecord(contexts[gpu].start_event, contexts[gpu].stream));
            
            // Multiple iterations for accurate measurement
            for (int i = 0; i < 10; i++) {
                hipLaunchKernelGGL(bandwidthTestKernel, dim3(blocks), dim3(threads), 0,
                                  contexts[gpu].stream,
                                  contexts[gpu].d_stress_buffer, d_temp, size);
                
                // Check for errors
                hipError_t err = hipGetLastError();
                if (err != hipSuccess) {
                    fprintf(stderr, "Bandwidth kernel launch error on GPU %zu: %s\n", 
                            gpu, hipGetErrorString(err));
                    exit(EXIT_FAILURE);
                }
            }
            
            HIP_CHECK(hipEventRecord(contexts[gpu].stop_event, contexts[gpu].stream));
            HIP_CHECK(hipStreamSynchronize(contexts[gpu].stream));
            
            float elapsed_ms;
            HIP_CHECK(hipEventElapsedTime(&elapsed_ms, 
                                         contexts[gpu].start_event, 
                                         contexts[gpu].stop_event));
            
            double bandwidth_gb_s = (2.0 * size * sizeof(float) * 10) / (elapsed_ms * 1e6);
            printf("  GPU %zu: Bandwidth = %.2f GB/s (%.2f M elements)\n", gpu, bandwidth_gb_s, size / 1e6);
            
            HIP_CHECK(hipFree(d_temp));
        }
    }
    printf("\n");
}

// ============================================================================
// MULTI-GPU SETUP AND INITIALIZATION
// ============================================================================

void setupMultiGPU(std::vector<GPUContext>& contexts, int num_gpus, 
                   int N, int nz, bool enable_stress, float stress_ratio) {
    
    int available_gpus;
    HIP_CHECK(hipGetDeviceCount(&available_gpus));
    
    if (num_gpus > available_gpus) {
        printf("Warning: Requested %d GPUs but only %d available. Using %d GPUs.\n",
               num_gpus, available_gpus, available_gpus);
        num_gpus = available_gpus;
    }
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║   Multi-GPU Preconditioned Conjugate Gradient     ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Number of GPUs:    %2d                         ║\n", num_gpus);
    printf("║ Matrix Size:       %-10d                 ║\n", N);
    printf("║ Non-zeros:         %-10d                 ║\n", nz);
    printf("║ Stress Testing:    %-5s                      ║\n", enable_stress ? "ON" : "OFF");
    printf("╚════════════════════════════════════════════════╝\n\n");
    
    contexts.resize(num_gpus);
    
    // Print GPU information
    for (int i = 0; i < num_gpus; i++) {
        hipDeviceProp_t prop;
        HIP_CHECK(hipGetDeviceProperties(&prop, i));
        printf("GPU %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Clock Rate: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("  Multiprocessors: %d\n\n", prop.multiProcessorCount);
    }
    
    // Enable peer-to-peer access
    printf("Enabling P2P Access:\n");
    for (int i = 0; i < num_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                int can_access;
                HIP_CHECK(hipDeviceCanAccessPeer(&can_access, i, j));
                if (can_access) {
                    hipError_t err = hipDeviceEnablePeerAccess(j, 0);
                    if (err == hipSuccess) {
                        printf("  ✓ GPU %d <-> GPU %d\n", i, j);
                    } else if (err != hipErrorPeerAccessAlreadyEnabled) {
                        fprintf(stderr, "HIP error in %s:%d - %s\n", __FILE__, __LINE__,
                                hipGetErrorString(err));
                        exit(EXIT_FAILURE);
                    }
                }
            }
        }
    }
    printf("\n");
    
    // Initialize each GPU context
    for (int i = 0; i < num_gpus; i++) {
        HIP_CHECK(hipSetDevice(i));
        
        contexts[i].deviceId = i;
        contexts[i].row_offset = (i * N) / num_gpus;
        int row_end = ((i + 1) * N) / num_gpus;
        contexts[i].local_N = row_end - contexts[i].row_offset;
        contexts[i].local_nz = nz / num_gpus; // Simplified distribution
        
        // Create handles
        HIPBLAS_CHECK(hipblasCreate(&contexts[i].blasHandle));
        HIPSPARSE_CHECK(hipsparseCreate(&contexts[i].sparseHandle));
        HIP_CHECK(hipStreamCreate(&contexts[i].stream));
        
        HIPBLAS_CHECK(hipblasSetStream(contexts[i].blasHandle, contexts[i].stream));
        HIPSPARSE_CHECK(hipsparseSetStream(contexts[i].sparseHandle, contexts[i].stream));
        
        // Create timing events
        HIP_CHECK(hipEventCreate(&contexts[i].start_event));
        HIP_CHECK(hipEventCreate(&contexts[i].stop_event));
        
        // Allocate stress test buffer if enabled
        if (enable_stress) {
            size_t free_mem, total_mem;
            HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
            contexts[i].stress_buffer_size = (size_t)(free_mem * stress_ratio);
            HIP_CHECK(hipMalloc(&contexts[i].d_stress_buffer, contexts[i].stress_buffer_size));
            // Initialize buffer to avoid segfaults
            HIP_CHECK(hipMemset(contexts[i].d_stress_buffer, 0, contexts[i].stress_buffer_size));
            printf("GPU %d: Allocated stress buffer = %.2f MB\n", i,
                   contexts[i].stress_buffer_size / (1024.0 * 1024.0));
        } else {
            contexts[i].d_stress_buffer = nullptr;
            contexts[i].stress_buffer_size = 0;
        }
    }
    printf("\n");
}

// ============================================================================
// MAIN FUNCTION
// ============================================================================

int main(int argc, char **argv)
{
    // Configuration parameters
    int num_gpus = 1;  // Default single GPU
    int max_iter = 1000;  // Maximum CG iterations
    bool enable_stress = false;
    bool enable_gpu_switching = false;  // Switch GPUs during computation
    int switch_interval = 10;  // Switch GPU every N iterations
    float stress_ratio = 0.3f;  // Use 30% of available memory for stress testing
    int stress_level = 1;
    bool run_bandwidth_test = false;
    bool run_compute_stress = false;
    int num_stress_runs = 1;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--ngpus") == 0 && i + 1 < argc) {
            num_gpus = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--max-iter") == 0 && i + 1 < argc) {
            max_iter = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--switch-gpu") == 0) {
            enable_gpu_switching = true;
        } else if (strcmp(argv[i], "--switch-interval") == 0 && i + 1 < argc) {
            switch_interval = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--stress") == 0) {
            enable_stress = true;
        } else if (strcmp(argv[i], "--stress-level") == 0 && i + 1 < argc) {
            stress_level = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--stress-ratio") == 0 && i + 1 < argc) {
            stress_ratio = atof(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--bandwidth") == 0) {
            run_bandwidth_test = true;
        } else if (strcmp(argv[i], "--compute-stress") == 0) {
            run_compute_stress = true;
        } else if (strcmp(argv[i], "--stress-runs") == 0 && i + 1 < argc) {
            num_stress_runs = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --ngpus N           Use N GPUs (default: 1)\n");
            printf("  --size N            Grid size (N x N matrix = N^2 elements, default: 128 = 16384 elements)\n");
            printf("  --max-iter N        Max CG iterations (default: 1000)\n");
            printf("  --switch-gpu        Enable dynamic GPU switching during CG solve\n");
            printf("  --switch-interval N Switch GPU every N iterations (default: 10)\n");
            printf("  --stress            Enable stress testing\n");
            printf("  --stress-level N    Stress intensity level (default: 1)\n");
            printf("  --stress-ratio F    Memory ratio for stress (default: 0.3)\n");
            printf("  --bandwidth         Run bandwidth test\n");
            printf("  --compute-stress    Run compute stress test\n");
            printf("  --stress-runs N     Number of stress test runs (default: 1)\n");
            printf("\nExamples:\n");
            printf("  %s --size 512 --ngpus 4 --max-iter 5000\n", argv[0]);
            printf("  %s --size 512 --ngpus 8 --switch-gpu --switch-interval 20\n", argv[0]);
            printf("  %s --size 1024 --ngpus 8 --stress --stress-level 10 --stress-ratio 0.5\n", argv[0]);
            printf("\nNote: For size > 512, increase --max-iter (e.g., --size 1024 --max-iter 10000)\n");
            return 0;
        }
    }
    
    printf("%s starting...\n\n", sSDKname);
    
    int         k, M = 0, N = 0, nz = 0;
    int        *I = NULL, *J = NULL;
    const float tol = 1e-4f;  // Convergence tolerance
    float      *x, *rhs;
    float       r0, r1, alpha, beta;
    float      *val = NULL;
    float       rsum, diff, err = 0.0;
    float       qaerr1 = 0.0f, qaerr2 = 0.0f;
    float       dot, numerator, denominator, nalpha;
    const float floatone  = 1.0;
    const float floatzero = 0.0;
    int nErrors = 0;
    
    // Generate Laplace matrix
    // Parse --size parameter if provided
    M = N = 16384;  // Default size
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            int n_side = atoi(argv[i + 1]);
            M = N = n_side * n_side;
            break;
        }
    }
    nz    = 5 * N - 4 * (int)sqrt((double)N);
    I     = (int *)malloc(sizeof(int) * (N + 1));
    J     = (int *)malloc(sizeof(int) * nz);
    val   = (float *)malloc(sizeof(float) * nz);
    x     = (float *)malloc(sizeof(float) * N);
    rhs   = (float *)malloc(sizeof(float) * N);
    
    for (int i = 0; i < N; i++) {
        rhs[i] = 0.0;
        x[i]   = 0.0;
    }
    
    genLaplace(I, J, val, M, N, nz, rhs);
    
    // Setup multi-GPU contexts
    std::vector<GPUContext> contexts;
    setupMultiGPU(contexts, num_gpus, N, nz, enable_stress, stress_ratio);
    
    // Run stress tests if enabled
    if (enable_stress) {
        for (int run = 0; run < num_stress_runs; run++) {
            printf("╔════════════════════════════════════════════════╗\n");
            printf("║          STRESS TEST RUN %d/%d                   ║\n", run + 1, num_stress_runs);
            printf("╚════════════════════════════════════════════════╝\n");
            
            runMemoryStressTest(contexts, stress_level);
            
            if (run_compute_stress) {
                runComputeStressTest(contexts);
            }
            
            if (run_bandwidth_test) {
                runBandwidthTest(contexts);
            }
        }
    }
    
    // For demonstration, we'll use GPU 0 for the actual CG solve
    // (Full multi-GPU distribution would require more complex matrix partitioning)
    // If GPU switching is enabled, we'll allocate data on all GPUs
    int primary_gpu = 0;
    int num_active_gpus = enable_gpu_switching ? contexts.size() : 1;
    
    HIP_CHECK(hipSetDevice(primary_gpu));
    GPUContext& ctx = contexts[primary_gpu];
    
    // Allocate device memory on primary GPU (and all GPUs if switching enabled)
    for (int gpu_idx = 0; gpu_idx < num_active_gpus; gpu_idx++) {
        HIP_CHECK(hipSetDevice(contexts[gpu_idx].deviceId));
        GPUContext& gpu_ctx = contexts[gpu_idx];
        
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_col, nz * sizeof(int)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_row, (N + 1) * sizeof(int)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_val, nz * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_x, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_y, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_r, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_p, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_omega, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_valsILU0, nz * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_zm1, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_zm2, N * sizeof(float)));
        HIP_CHECK(hipMalloc((void **)&gpu_ctx.d_rm2, N * sizeof(float)));
        
        // Copy matrix data to this GPU
        HIP_CHECK(hipMemcpy(gpu_ctx.d_col, J, nz * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_ctx.d_row, I, (N + 1) * sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_ctx.d_val, val, nz * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_ctx.d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(gpu_ctx.d_r, rhs, N * sizeof(float), hipMemcpyHostToDevice));
    }
    
    HIP_CHECK(hipSetDevice(primary_gpu));
    
    // Create descriptors for all active GPUs
    for (int gpu_idx = 0; gpu_idx < num_active_gpus; gpu_idx++) {
        HIP_CHECK(hipSetDevice(contexts[gpu_idx].deviceId));
        GPUContext& gpu_ctx = contexts[gpu_idx];
        
        // Create vector descriptors
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecp, N, gpu_ctx.d_p, HIP_R_32F));
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecX, N, gpu_ctx.d_x, HIP_R_32F));
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecY, N, gpu_ctx.d_y, HIP_R_32F));
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecR, N, gpu_ctx.d_r, HIP_R_32F));
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecZM1, N, gpu_ctx.d_zm1, HIP_R_32F));
        HIPSPARSE_CHECK(hipsparseCreateDnVec(&gpu_ctx.vecomega, N, gpu_ctx.d_omega, HIP_R_32F));
        
        // Create matrix descriptor
        HIPSPARSE_CHECK(hipsparseCreateMatDescr(&gpu_ctx.descr));
        HIPSPARSE_CHECK(hipsparseSetMatType(gpu_ctx.descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
        HIPSPARSE_CHECK(hipsparseSetMatIndexBase(gpu_ctx.descr, HIPSPARSE_INDEX_BASE_ZERO));
        
        // Create CSR matrix
        HIPSPARSE_CHECK(hipsparseCreateCsr(&gpu_ctx.matA, N, N, nz,
                                           gpu_ctx.d_row, gpu_ctx.d_col, gpu_ctx.d_val,
                                           HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                           HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    }
    
    HIP_CHECK(hipSetDevice(primary_gpu));
    hipsparseDnVecDescr_t vecomega = ctx.vecomega;  // Use primary GPU's vecomega
    
    // Note: Matrix data already copied in allocation loop above
    
    // Copy A data to ILU(0) vals as input
    HIP_CHECK(hipMemcpy(ctx.d_valsILU0, ctx.d_val, nz * sizeof(float), hipMemcpyDeviceToDevice));
    
    // Create lower and upper triangular matrices for preconditioner
    hipsparseFillMode_t fill_lower = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t diag_unit = HIPSPARSE_DIAG_TYPE_UNIT;
    hipsparseFillMode_t fill_upper = HIPSPARSE_FILL_MODE_UPPER;
    hipsparseDiagType_t diag_non_unit = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    
    HIPSPARSE_CHECK(hipsparseCreateCsr(&ctx.matM_lower, N, N, nz,
                                       ctx.d_row, ctx.d_col, ctx.d_valsILU0,
                                       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseSpMatSetAttribute(ctx.matM_lower, HIPSPARSE_SPMAT_FILL_MODE,
                                                &fill_lower, sizeof(fill_lower)));
    HIPSPARSE_CHECK(hipsparseSpMatSetAttribute(ctx.matM_lower, HIPSPARSE_SPMAT_DIAG_TYPE,
                                                &diag_unit, sizeof(diag_unit)));
    
    HIPSPARSE_CHECK(hipsparseCreateCsr(&ctx.matM_upper, N, N, nz,
                                       ctx.d_row, ctx.d_col, ctx.d_valsILU0,
                                       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseSpMatSetAttribute(ctx.matM_upper, HIPSPARSE_SPMAT_FILL_MODE,
                                                &fill_upper, sizeof(fill_upper)));
    HIPSPARSE_CHECK(hipsparseSpMatSetAttribute(ctx.matM_upper, HIPSPARSE_SPMAT_DIAG_TYPE,
                                                &diag_non_unit, sizeof(diag_non_unit)));
    
    // Create ILU(0) info
    size_t bufferSizeMV, bufferSizeL, bufferSizeU, bufferSizeILU;
    void *d_bufferL, *d_bufferU, *d_bufferILU;
    
    HIPSPARSE_CHECK(hipsparseCreateCsrilu02Info(&ctx.infoILU));
    HIPSPARSE_CHECK(hipsparseCreateMatDescr(&ctx.matLU));
    HIPSPARSE_CHECK(hipsparseSetMatType(ctx.matLU, HIPSPARSE_MATRIX_TYPE_GENERAL));
    HIPSPARSE_CHECK(hipsparseSetMatIndexBase(ctx.matLU, HIPSPARSE_INDEX_BASE_ZERO));
    
    // Allocate workspace for all active GPUs
    for (int gpu_idx = 0; gpu_idx < num_active_gpus; gpu_idx++) {
        HIP_CHECK(hipSetDevice(contexts[gpu_idx].deviceId));
        GPUContext& gpu_ctx = contexts[gpu_idx];
        
        HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(gpu_ctx.sparseHandle,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 &floatone, gpu_ctx.matA, gpu_ctx.vecp, &floatzero, gpu_ctx.vecomega,
                                                 HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufferSizeMV));
        HIP_CHECK(hipMalloc(&gpu_ctx.d_bufferMV, bufferSizeMV));
    }
    
    HIP_CHECK(hipSetDevice(primary_gpu));
    
    int bufferSizeLU_int;
    HIPSPARSE_CHECK(hipsparseScsrilu02_bufferSize(ctx.sparseHandle, N, nz, ctx.matLU,
                                                   ctx.d_val, ctx.d_row, ctx.d_col, ctx.infoILU,
                                                   &bufferSizeLU_int));
    bufferSizeILU = bufferSizeLU_int;
    HIP_CHECK(hipMalloc(&d_bufferILU, bufferSizeILU));
    
    HIPSPARSE_CHECK(hipsparseSpSV_createDescr(&ctx.spsvDescrL));
    HIPSPARSE_CHECK(hipsparseSpSV_bufferSize(ctx.sparseHandle,
                                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                             &floatone, ctx.matM_lower, ctx.vecR, ctx.vecY,
                                             HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                             ctx.spsvDescrL, &bufferSizeL));
    HIP_CHECK(hipMalloc(&d_bufferL, bufferSizeL));
    
    HIPSPARSE_CHECK(hipsparseSpSV_createDescr(&ctx.spsvDescrU));
    HIPSPARSE_CHECK(hipsparseSpSV_bufferSize(ctx.sparseHandle,
                                             HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                             &floatone, ctx.matM_upper, ctx.vecY, ctx.vecZM1,
                                             HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                             ctx.spsvDescrU, &bufferSizeU));
    HIP_CHECK(hipMalloc(&d_bufferU, bufferSizeU));
    
    // ========================================================================
    // GPU-SWITCHING CONJUGATE GRADIENT (if enabled)
    // ========================================================================
    
    if (enable_gpu_switching && num_active_gpus > 1) {
        printf("\n╔════════════════════════════════════════════════╗\n");
        printf("║   CG with Dynamic GPU Switching               ║\n");
        printf("║   Switching every %d iterations across %d GPUs    ║\n", switch_interval, num_active_gpus);
        printf("╚════════════════════════════════════════════════╝\n\n");
        
        auto switch_cg_start = std::chrono::high_resolution_clock::now();
        
        int current_gpu_idx = 0;
        HIP_CHECK(hipSetDevice(contexts[current_gpu_idx].deviceId));
        GPUContext* active_ctx = &contexts[current_gpu_idx];
        
        k = 0;
        r0 = 0;
        HIPBLAS_CHECK(hipblasSdot(active_ctx->blasHandle, N, active_ctx->d_r, 1, active_ctx->d_r, 1, &r1));
        
        printf("Starting GPU-switching CG on GPU 0, residual = %e\n", sqrt(r1));
        
        while (r1 > tol * tol && k <= max_iter) {
            k++;
            
            // Check if we need to switch GPUs
            if (k % switch_interval == 0 && k > 1) {
                int next_gpu_idx = (current_gpu_idx + 1) % num_active_gpus;
                GPUContext* next_ctx = &contexts[next_gpu_idx];
                
                printf("  [GPU %d->%d] Switching at iteration %d (residual=%e)...\n", 
                       current_gpu_idx, next_gpu_idx, k, sqrt(r1));
                
                // Transfer state from current GPU to next GPU.
                // Prefer P2P copies when peer access is available; otherwise fall back
                // to host-staged copies to support topologies without full connectivity.
                int canAccessPeer = 0;
                HIP_CHECK(hipDeviceCanAccessPeer(&canAccessPeer,
                                                 next_ctx->deviceId,
                                                 active_ctx->deviceId));
                
                if (canAccessPeer) {
                    // Peer-to-peer path
                    HIP_CHECK(hipSetDevice(next_ctx->deviceId));
                    HIP_CHECK(hipMemcpyPeerAsync(next_ctx->d_x, next_ctx->deviceId,
                                                 active_ctx->d_x, active_ctx->deviceId,
                                                 N * sizeof(float), next_ctx->stream));
                    HIP_CHECK(hipMemcpyPeerAsync(next_ctx->d_r, next_ctx->deviceId,
                                                 active_ctx->d_r, active_ctx->deviceId,
                                                 N * sizeof(float), next_ctx->stream));
                    HIP_CHECK(hipMemcpyPeerAsync(next_ctx->d_p, next_ctx->deviceId,
                                                 active_ctx->d_p, active_ctx->deviceId,
                                                 N * sizeof(float), next_ctx->stream));
                    HIP_CHECK(hipStreamSynchronize(next_ctx->stream));
                } else {
                    // Fallback: host-staged copies when peer access is not available
                    printf("    Peer access between GPU %d and GPU %d is not available; "
                           "falling back to host-staged copies for state migration.\n",
                           active_ctx->deviceId, next_ctx->deviceId);

                    // Temporary host buffer for transfers
                    std::vector<float> h_buffer(N);

                    // Copy d_x
                    HIP_CHECK(hipSetDevice(active_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(h_buffer.data(), active_ctx->d_x,
                                        N * sizeof(float), hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(next_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(next_ctx->d_x, h_buffer.data(),
                                        N * sizeof(float), hipMemcpyHostToDevice));

                    // Copy d_r
                    HIP_CHECK(hipSetDevice(active_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(h_buffer.data(), active_ctx->d_r,
                                        N * sizeof(float), hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(next_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(next_ctx->d_r, h_buffer.data(),
                                        N * sizeof(float), hipMemcpyHostToDevice));

                    // Copy d_p
                    HIP_CHECK(hipSetDevice(active_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(h_buffer.data(), active_ctx->d_p,
                                        N * sizeof(float), hipMemcpyDeviceToHost));
                    HIP_CHECK(hipSetDevice(next_ctx->deviceId));
                    HIP_CHECK(hipMemcpy(next_ctx->d_p, h_buffer.data(),
                                        N * sizeof(float), hipMemcpyHostToDevice));
                }
                
                current_gpu_idx = next_gpu_idx;
                active_ctx = next_ctx;
            }
            
            if (k == 1) {
                HIPBLAS_CHECK(hipblasScopy(active_ctx->blasHandle, N, active_ctx->d_r, 1, active_ctx->d_p, 1));
            }
            else {
                if (fabs(r0) < 1e-30f) {
                    printf("  [GPU %d] CG breakdown: r0 ~ 0 at iteration %d\n", current_gpu_idx, k);
                    break;
                }
                beta = r1 / r0;
                HIPBLAS_CHECK(hipblasSscal(active_ctx->blasHandle, N, &beta, active_ctx->d_p, 1));
                HIPBLAS_CHECK(hipblasSaxpy(active_ctx->blasHandle, N, &floatone, active_ctx->d_r, 1, active_ctx->d_p, 1));
            }
            
            HIPSPARSE_CHECK(hipsparseSpMV(active_ctx->sparseHandle,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          &floatone, active_ctx->matA, active_ctx->vecp, &floatzero, active_ctx->vecomega,
                                          HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, active_ctx->d_bufferMV));
            HIPBLAS_CHECK(hipblasSdot(active_ctx->blasHandle, N, active_ctx->d_p, 1, active_ctx->d_omega, 1, &dot));
            if (fabs(dot) < 1e-30f) {
                printf("  [GPU %d] CG breakdown: p^T*A*p ~ 0 at iteration %d\n", current_gpu_idx, k);
                break;
            }
            alpha = r1 / dot;
            HIPBLAS_CHECK(hipblasSaxpy(active_ctx->blasHandle, N, &alpha, active_ctx->d_p, 1, active_ctx->d_x, 1));
            nalpha = -alpha;
            HIPBLAS_CHECK(hipblasSaxpy(active_ctx->blasHandle, N, &nalpha, active_ctx->d_omega, 1, active_ctx->d_r, 1));
            r0 = r1;
            HIPBLAS_CHECK(hipblasSdot(active_ctx->blasHandle, N, active_ctx->d_r, 1, active_ctx->d_r, 1, &r1));
            
            if (k % 100 == 0 || k == 1) {
                printf("  iteration = %3d [GPU %d], residual = %e\n", k, current_gpu_idx, sqrt(r1));
            }
        }
        
        auto switch_cg_end = std::chrono::high_resolution_clock::now();
        auto switch_cg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(switch_cg_end - switch_cg_start);
        
        printf("\nGPU-Switching CG completed in %d iterations (%.2f ms total, %.4f ms/iter)\n",
               k, (double)switch_cg_duration.count(), k > 0 ? (double)switch_cg_duration.count() / k : 0.0);
        printf("Final GPU: %d, Final residual = %e\n", current_gpu_idx, sqrt(r1));
        printf("Number of GPU switches: %d\n", k / switch_interval);
        
        // Copy result back
        HIP_CHECK(hipMemcpy(x, active_ctx->d_x, N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Verify result
        err = 0.0;
        for (int i = 0; i < N; i++) {
            rsum = 0.0;
            for (int j = I[i]; j < I[i + 1]; j++) {
                rsum += val[j] * x[J[j]];
            }
            diff = fabs(rsum - rhs[i]);
            if (diff > err) {
                err = diff;
            }
        }
        
        printf("Max error = %e\n", err);
        printf("Convergence Test: %s\n\n", (k <= max_iter) ? "✓ PASSED" : "✗ FAILED");
        nErrors += (k > max_iter) ? 1 : 0;
        qaerr1 = err;
    }
    
    // ========================================================================
    // CONJUGATE GRADIENT WITHOUT PRECONDITIONING
    // ========================================================================
    
    if (!enable_gpu_switching) {
        printf("\n╔════════════════════════════════════════════════╗\n");
        printf("║   Conjugate Gradient WITHOUT Preconditioning  ║\n");
        printf("╚════════════════════════════════════════════════╝\n\n");
    
        auto cg_start = std::chrono::high_resolution_clock::now();
        
        k  = 0;
        r0 = 0;
        HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_r, 1, &r1));
        
        while (r1 > tol * tol && k <= max_iter) {
            k++;
            
            if (k == 1) {
                HIPBLAS_CHECK(hipblasScopy(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_p, 1));
            }
            else {
                if (fabs(r0) < 1e-30f) {
                    printf("  CG breakdown: r0 ~ 0 at iteration %d\n", k);
                    break;
                }
                beta = r1 / r0;
                HIPBLAS_CHECK(hipblasSscal(ctx.blasHandle, N, &beta, ctx.d_p, 1));
                HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &floatone, ctx.d_r, 1, ctx.d_p, 1));
            }
            
            HIPSPARSE_CHECK(hipsparseSpMV(ctx.sparseHandle,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          &floatone, ctx.matA, ctx.vecp, &floatzero, vecomega,
                                          HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, ctx.d_bufferMV));
            HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_p, 1, ctx.d_omega, 1, &dot));
            if (fabs(dot) < 1e-30f) {
                printf("  CG breakdown: p^T*A*p ~ 0 at iteration %d\n", k);
                break;
            }
            alpha = r1 / dot;
            HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &alpha, ctx.d_p, 1, ctx.d_x, 1));
            nalpha = -alpha;
            HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &nalpha, ctx.d_omega, 1, ctx.d_r, 1));
            r0 = r1;
            HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_r, 1, &r1));
            
            if (k % 100 == 0 || k == 1) {
                printf("  iteration = %3d, residual = %e\n", k, sqrt(r1));
            }
        }
        
        auto cg_end = std::chrono::high_resolution_clock::now();
        auto cg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cg_end - cg_start);
        
        printf("\nCompleted in %d iterations (%.2f ms total, %.4f ms/iter)\n",
               k, (double)cg_duration.count(), k > 0 ? (double)cg_duration.count() / k : 0.0);
        printf("Final residual = %e\n", sqrt(r1));
        
        HIP_CHECK(hipMemcpy(x, ctx.d_x, N * sizeof(float), hipMemcpyDeviceToHost));
        
        // Verify result
        err = 0.0;
        for (int i = 0; i < N; i++) {
            rsum = 0.0;
            for (int j = I[i]; j < I[i + 1]; j++) {
                rsum += val[j] * x[J[j]];
            }
            diff = fabs(rsum - rhs[i]);
            if (diff > err) {
                err = diff;
            }
        }
    
        printf("Max error = %e\n", err);
        printf("Convergence Test: %s\n", (k <= max_iter) ? "✓ PASSED" : "✗ FAILED");
        nErrors += (k > max_iter) ? 1 : 0;
        qaerr1 = err;
    }  // End of non-switching CG
    
    // ========================================================================
    // PRECONDITIONED CONJUGATE GRADIENT WITH ILU(0)
    // ========================================================================
    
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║   Conjugate Gradient WITH ILU(0) Precond      ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");
    
    // Ensure we're on the primary GPU for PCG
    HIP_CHECK(hipSetDevice(primary_gpu));
    HIP_CHECK(hipDeviceSynchronize());
    
    // If GPU switching was enabled, verify we're on the right device
    if (enable_gpu_switching && num_active_gpus > 1) {
        int current_device;
        HIP_CHECK(hipGetDevice(&current_device));
        printf("Starting PCG on GPU %d (primary)\n", current_device);
    }
    
    // Perform ILU(0) analysis
    HIPSPARSE_CHECK(hipsparseScsrilu02_analysis(ctx.sparseHandle, N, nz, ctx.descr,
                                                ctx.d_valsILU0, ctx.d_row, ctx.d_col, ctx.infoILU,
                                                HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferILU));
    
    // Generate ILU(0) factors
    HIPSPARSE_CHECK(hipsparseScsrilu02(ctx.sparseHandle, N, nz, ctx.matLU,
                                       ctx.d_valsILU0, ctx.d_row, ctx.d_col, ctx.infoILU,
                                       HIPSPARSE_SOLVE_POLICY_USE_LEVEL, d_bufferILU));
    
    // Perform triangular solve analysis (descriptors must match the solve calls)
    HIPSPARSE_CHECK(hipsparseSpSV_analysis(ctx.sparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &floatone, ctx.matM_lower, ctx.vecR, ctx.vecY,
                                           HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                           ctx.spsvDescrL, d_bufferL));
    
    HIPSPARSE_CHECK(hipsparseSpSV_analysis(ctx.sparseHandle,
                                           HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                           &floatone, ctx.matM_upper, ctx.vecY, ctx.vecZM1,
                                           HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                           ctx.spsvDescrU, d_bufferU));
    
    // Reset initial guess
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    HIP_CHECK(hipMemcpy(ctx.d_r, rhs, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ctx.d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
    
    auto pcg_start = std::chrono::high_resolution_clock::now();
    
    k = 0;
    HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_r, 1, &r1));
    
    while (r1 > tol * tol && k <= max_iter) {
        // Apply preconditioner: d_zm1 = U^-1 L^-1 d_r
        HIPSPARSE_CHECK(hipsparseSpSV_solve(ctx.sparseHandle,
                                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                            &floatone, ctx.matM_lower, ctx.vecR, ctx.vecY,
                                            HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                            ctx.spsvDescrL));
        
        HIPSPARSE_CHECK(hipsparseSpSV_solve(ctx.sparseHandle,
                                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                            &floatone, ctx.matM_upper, ctx.vecY, ctx.vecZM1,
                                            HIP_R_32F, HIPSPARSE_SPSV_ALG_DEFAULT,
                                            ctx.spsvDescrU));
        k++;
        
        if (k == 1) {
            HIPBLAS_CHECK(hipblasScopy(ctx.blasHandle, N, ctx.d_zm1, 1, ctx.d_p, 1));
        }
        else {
            HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_zm1, 1, &numerator));
            HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_rm2, 1, ctx.d_zm2, 1, &denominator));
            if (fabs(denominator) < 1e-30f) {
                printf("  PCG breakdown: r_old^T*z_old ~ 0 at iteration %d\n", k);
                break;
            }
            beta = numerator / denominator;
            HIPBLAS_CHECK(hipblasSscal(ctx.blasHandle, N, &beta, ctx.d_p, 1));
            HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &floatone, ctx.d_zm1, 1, ctx.d_p, 1));
        }
        
        HIPSPARSE_CHECK(hipsparseSpMV(ctx.sparseHandle,
                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                      &floatone, ctx.matA, ctx.vecp, &floatzero, vecomega,
                                      HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, ctx.d_bufferMV));
        HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_zm1, 1, &numerator));
        HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_p, 1, ctx.d_omega, 1, &denominator));
        if (fabs(denominator) < 1e-30f) {
            printf("  PCG breakdown: p^T*A*p ~ 0 at iteration %d\n", k);
            break;
        }
        alpha = numerator / denominator;
        HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &alpha, ctx.d_p, 1, ctx.d_x, 1));
        HIPBLAS_CHECK(hipblasScopy(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_rm2, 1));
        HIPBLAS_CHECK(hipblasScopy(ctx.blasHandle, N, ctx.d_zm1, 1, ctx.d_zm2, 1));
        nalpha = -alpha;
        HIPBLAS_CHECK(hipblasSaxpy(ctx.blasHandle, N, &nalpha, ctx.d_omega, 1, ctx.d_r, 1));
        HIPBLAS_CHECK(hipblasSdot(ctx.blasHandle, N, ctx.d_r, 1, ctx.d_r, 1, &r1));
        
        if (k % 100 == 0 || k == 1) {
            printf("  iteration = %3d, residual = %e\n", k, sqrt(r1));
        }
    }
    
    auto pcg_end = std::chrono::high_resolution_clock::now();
    auto pcg_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pcg_end - pcg_start);
    
    printf("\nCompleted in %d iterations (%.2f ms total, %.4f ms/iter)\n",
           k, (double)pcg_duration.count(), k > 0 ? (double)pcg_duration.count() / k : 0.0);
    printf("Final residual = %e\n", sqrt(r1));
    
    HIP_CHECK(hipMemcpy(x, ctx.d_x, N * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify result
    err = 0.0;
    for (int i = 0; i < N; i++) {
        rsum = 0.0;
        for (int j = I[i]; j < I[i + 1]; j++) {
            rsum += val[j] * x[J[j]];
        }
        diff = fabs(rsum - rhs[i]);
        if (diff > err) {
            err = diff;
        }
    }
    
    printf("Max error = %e\n", err);
    printf("Convergence Test: %s\n", (k <= max_iter) ? "✓ PASSED" : "✗ FAILED");
    nErrors += (k > max_iter) ? 1 : 0;
    qaerr2 = err;
    
    // ========================================================================
    // CLEANUP
    // ========================================================================
    
    // Destroy descriptors
    HIPSPARSE_CHECK(hipsparseDestroyCsrilu02Info(ctx.infoILU));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(ctx.matLU));
    HIPSPARSE_CHECK(hipsparseSpSV_destroyDescr(ctx.spsvDescrL));
    HIPSPARSE_CHECK(hipsparseSpSV_destroyDescr(ctx.spsvDescrU));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(ctx.matM_lower));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(ctx.matM_upper));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(ctx.matA));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(ctx.vecp));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecomega));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(ctx.vecR));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(ctx.vecX));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(ctx.vecY));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(ctx.vecZM1));
    HIPSPARSE_CHECK(hipsparseDestroyMatDescr(ctx.descr));
    
    // Cleanup descriptors for non-primary GPUs (fix memory leak)
    if (enable_gpu_switching && num_active_gpus > 1) {
        for (int gpu_idx = 1; gpu_idx < num_active_gpus; gpu_idx++) {
            HIP_CHECK(hipSetDevice(contexts[gpu_idx].deviceId));
            GPUContext& gpu_ctx = contexts[gpu_idx];
            HIPSPARSE_CHECK(hipsparseDestroySpMat(gpu_ctx.matA));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecp));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecX));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecY));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecR));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecZM1));
            HIPSPARSE_CHECK(hipsparseDestroyDnVec(gpu_ctx.vecomega));
            HIPSPARSE_CHECK(hipsparseDestroyMatDescr(gpu_ctx.descr));
        }
    }
    
    // Free device memory for all active GPUs
    for (int gpu_idx = 0; gpu_idx < num_active_gpus; gpu_idx++) {
        HIP_CHECK(hipSetDevice(contexts[gpu_idx].deviceId));
        GPUContext& gpu_ctx = contexts[gpu_idx];
        
        HIP_CHECK(hipFree(gpu_ctx.d_bufferMV));
        HIP_CHECK(hipFree(gpu_ctx.d_col));
        HIP_CHECK(hipFree(gpu_ctx.d_row));
        HIP_CHECK(hipFree(gpu_ctx.d_val));
        HIP_CHECK(hipFree(gpu_ctx.d_x));
        HIP_CHECK(hipFree(gpu_ctx.d_y));
        HIP_CHECK(hipFree(gpu_ctx.d_r));
        HIP_CHECK(hipFree(gpu_ctx.d_p));
        HIP_CHECK(hipFree(gpu_ctx.d_omega));
        HIP_CHECK(hipFree(gpu_ctx.d_valsILU0));
        HIP_CHECK(hipFree(gpu_ctx.d_zm1));
        HIP_CHECK(hipFree(gpu_ctx.d_zm2));
        HIP_CHECK(hipFree(gpu_ctx.d_rm2));
    }
    
    HIP_CHECK(hipSetDevice(primary_gpu));
    HIP_CHECK(hipFree(d_bufferILU));
    HIP_CHECK(hipFree(d_bufferL));
    HIP_CHECK(hipFree(d_bufferU));
    
    // Cleanup multi-GPU contexts
    for (auto& context : contexts) {
        HIP_CHECK(hipSetDevice(context.deviceId));
        HIPBLAS_CHECK(hipblasDestroy(context.blasHandle));
        HIPSPARSE_CHECK(hipsparseDestroy(context.sparseHandle));
        HIP_CHECK(hipStreamDestroy(context.stream));
        HIP_CHECK(hipEventDestroy(context.start_event));
        HIP_CHECK(hipEventDestroy(context.stop_event));
        if (context.d_stress_buffer) {
            HIP_CHECK(hipFree(context.d_stress_buffer));
        }
    }
    
    // Free host memory
    free(I);
    free(J);
    free(val);
    free(x);
    free(rhs);
    
    // Final summary
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║              SUMMARY                      ║\n");
    printf("╠════════════════════════════════════════════════╣\n");
    printf("║ Total Errors:      %2d                          ║\n", nErrors);
    printf("║ CG Error:          %.2e                    ║\n", fabs(qaerr1));
    printf("║ PCG Error:         %.2e                    ║\n", fabs(qaerr2));
    printf("║ Number of GPUs:    %2d                          ║\n", (int)contexts.size());
    printf("║ Stress Testing:    %-5s                      ║\n", enable_stress ? "ON" : "OFF");
    printf("╚════════════════════════════════════════════════╝\n");
    
    return (nErrors == 0 && fabs(qaerr1) < 1e-5 && fabs(qaerr2) < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE;
}