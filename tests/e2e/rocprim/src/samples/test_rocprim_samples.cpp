/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
// SPDX-License-Identifier: MIT

#include "../include/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

void safe_free(void *p) {
  if (p)
    HIP_CHECK(hipFree(p));
}

bool nearly_equal(float a, float b, float abs_tol, float rel_tol) {
  const float diff = std::fabs(a - b);
  if (diff <= abs_tol)
    return true;
  const float denom = std::max(std::fabs(a), std::fabs(b));
  return diff <= rel_tol * denom;
}

std::vector<unsigned int>
cpu_histogram_even(const std::vector<float> &normalized, int num_bins,
                   float lower_level, float upper_level) {
  std::vector<unsigned int> histogram(static_cast<size_t>(num_bins), 0);
  const float range = upper_level - lower_level;
  const float bin_width = range / static_cast<float>(num_bins);

  for (float value : normalized) {
    if (value < lower_level)
      value = lower_level;
    if (value >= upper_level)
      value = std::nextafter(upper_level, lower_level);

    int bin = static_cast<int>((value - lower_level) / bin_width);
    bin = std::max(0, std::min(bin, num_bins - 1));
    histogram[static_cast<size_t>(bin)]++;
  }
  return histogram;
}

__global__ void compute_running_mean_kernel(const float *running_sum,
                                            float *running_mean, size_t size) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size)
    running_mean[idx] = running_sum[idx] / static_cast<float>(idx + 1);
}

__global__ void normalize_kernel(const float *input, float *output, float min_v,
                                 float max_v, size_t size) {
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    const float denom = max_v - min_v;
    output[idx] = (denom == 0.0f) ? 0.0f : (input[idx] - min_v) / denom;
  }
}

}  // namespace

TEST(RocprimSampleTests, RunningStatistics) {
  const size_t size = 100000;
  std::vector<float> input(size);
  for (size_t i = 0; i < size; i++) {
    input[i] = 50.0f + 0.01f * static_cast<float>(i) +
               (std::sin(static_cast<float>(i) * 0.001f) * 10.0f);
  }

  double cpu_total = 0.0;
  for (float value : input)
    cpu_total += static_cast<double>(value);

  auto cpu_prefix_sum_at = [&](size_t pos) -> double {
    double sum = 0.0;
    for (size_t i = 0; i <= pos; i++)
      sum += static_cast<double>(input[i]);
    return sum;
  };

  float *d_input = nullptr;
  float *d_cumsum = nullptr;
  float *d_running_mean = nullptr;
  HIP_CHECK(hipMalloc(&d_input, size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_cumsum, size * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_running_mean, size * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_input, input.data(), size * sizeof(float),
                      hipMemcpyHostToDevice));

  void *d_temp_scan = nullptr;
  size_t temp_scan_bytes = 0;
  HIP_CHECK(rocprim::inclusive_scan(nullptr, temp_scan_bytes, d_input, d_cumsum,
                                    size, rocprim::plus<float>(),
                                    hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_scan, temp_scan_bytes));
  HIP_CHECK(rocprim::inclusive_scan(d_temp_scan, temp_scan_bytes, d_input,
                                    d_cumsum, size, rocprim::plus<float>(),
                                    hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  std::vector<float> cumsum(size);
  HIP_CHECK(hipMemcpy(cumsum.data(), d_cumsum, size * sizeof(float),
                      hipMemcpyDeviceToHost));

  for (size_t i = 1; i < size; i++)
    ASSERT_GE(cumsum[i], cumsum[i - 1]) << "Cumulative sum decreased at " << i;

  const std::vector<size_t> sample_positions = {0, 1, 2, 99, 999, 9999,
                                                size - 1};
  const float scan_abs_tol = 0.5f;
  const float scan_rel_tol = 1e-5f;
  for (size_t pos : sample_positions) {
    const double cpu_ps = cpu_prefix_sum_at(pos);
    ASSERT_TRUE(nearly_equal(static_cast<float>(cpu_ps), cumsum[pos],
                             scan_abs_tol, scan_rel_tol))
        << "inclusive_scan mismatch at pos=" << pos << " CPU=" << cpu_ps
        << " GPU=" << cumsum[pos];
  }
  ASSERT_TRUE(nearly_equal(static_cast<float>(cpu_total), cumsum[size - 1],
                           scan_abs_tol, scan_rel_tol))
      << "inclusive_scan final sum mismatch";

  const int threads = 256;
  const int blocks = static_cast<int>((size + threads - 1) / threads);
  hipLaunchKernelGGL(compute_running_mean_kernel, dim3(blocks), dim3(threads),
                     0, hipStreamDefault, d_cumsum, d_running_mean, size);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  std::vector<float> running_mean(size);
  HIP_CHECK(hipMemcpy(running_mean.data(), d_running_mean, size * sizeof(float),
                      hipMemcpyDeviceToHost));

  const float mean_abs_tol = 1e-2f;
  const float mean_rel_tol = 1e-5f;
  for (uint64_t pos : {0ULL, 99ULL, 999ULL, 9999ULL,
                       static_cast<unsigned long long>(size - 1)}) {
    const double cpu_ps = cpu_prefix_sum_at(pos);
    const double cpu_mean = cpu_ps / static_cast<double>(pos + 1);
    ASSERT_TRUE(nearly_equal(static_cast<float>(cpu_mean), running_mean[pos],
                             mean_abs_tol, mean_rel_tol))
        << "running mean mismatch at pos=" << pos;
  }

  const double cpu_global_mean = cpu_total / static_cast<double>(size);
  ASSERT_TRUE(nearly_equal(static_cast<float>(cpu_global_mean),
                           running_mean[size - 1], mean_abs_tol, mean_rel_tol))
      << "final running mean mismatch";

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_cumsum));
  HIP_CHECK(hipFree(d_running_mean));
  HIP_CHECK(hipFree(d_temp_scan));
}

TEST(RocprimSampleTests, TopKFrequency) {
  const size_t size = 100000;
  std::vector<int> input(size);
  for (size_t i = 0; i < size; i++) {
    if (i < 30000)
      input[i] = 42;
    else if (i < 50000)
      input[i] = 17;
    else if (i < 65000)
      input[i] = 99;
    else if (i < 75000)
      input[i] = 5;
    else if (i < 82000)
      input[i] = 73;
    else
      input[i] = static_cast<int>((i % 100) + 200);
  }

  std::mt19937 rng(12345);
  std::shuffle(input.begin(), input.end(), rng);

  std::unordered_map<int, int> cpu_counts;
  cpu_counts.reserve(1024);
  for (int value : input)
    cpu_counts[value]++;

  std::vector<std::pair<int, int>> cpu_items;
  cpu_items.reserve(cpu_counts.size());
  for (const auto &kv : cpu_counts)
    cpu_items.emplace_back(kv.first, kv.second);

  auto top_cmp = [](const auto &a, const auto &b) {
    if (a.second != b.second)
      return a.second > b.second;
    return a.first < b.first;
  };
  std::sort(cpu_items.begin(), cpu_items.end(), top_cmp);

  const int top_k = std::min<int>(10, static_cast<int>(cpu_items.size()));
  std::vector<int> cpu_top_keys(top_k);
  std::vector<int> cpu_top_counts(top_k);
  for (int i = 0; i < top_k; i++) {
    cpu_top_keys[i] = cpu_items[i].first;
    cpu_top_counts[i] = cpu_items[i].second;
  }

  int *d_input = nullptr;
  int *d_sorted = nullptr;
  HIP_CHECK(hipMalloc(&d_input, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sorted, size * sizeof(int)));
  HIP_CHECK(
      hipMemcpy(d_input, input.data(), size * sizeof(int), hipMemcpyHostToDevice));

  void *d_temp_sort = nullptr;
  size_t temp_sort_bytes = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_sort_bytes, d_input,
                                     d_sorted, size, 0, sizeof(int) * 8,
                                     hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_sort, temp_sort_bytes));
  HIP_CHECK(rocprim::radix_sort_keys(d_temp_sort, temp_sort_bytes, d_input,
                                     d_sorted, size, 0, sizeof(int) * 8,
                                     hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  int *d_unique_out = nullptr;
  int *d_counts_out = nullptr;
  size_t *d_runs_count = nullptr;
  HIP_CHECK(hipMalloc(&d_unique_out, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_counts_out, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_runs_count, sizeof(size_t)));

  void *d_temp_rle = nullptr;
  size_t temp_rle_bytes = 0;
  HIP_CHECK(rocprim::run_length_encode(nullptr, temp_rle_bytes, d_sorted, size,
                                       d_unique_out, d_counts_out, d_runs_count,
                                       hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_rle, temp_rle_bytes));
  HIP_CHECK(rocprim::run_length_encode(d_temp_rle, temp_rle_bytes, d_sorted,
                                       size, d_unique_out, d_counts_out,
                                       d_runs_count, hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  size_t num_unique = 0;
  HIP_CHECK(hipMemcpy(&num_unique, d_runs_count, sizeof(size_t),
                      hipMemcpyDeviceToHost));
  ASSERT_GT(num_unique, 0U);

  std::vector<int> gpu_unique(num_unique);
  std::vector<int> gpu_counts(num_unique);
  HIP_CHECK(hipMemcpy(gpu_unique.data(), d_unique_out, num_unique * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gpu_counts.data(), d_counts_out, num_unique * sizeof(int),
                      hipMemcpyDeviceToHost));

  int64_t sum_counts = 0;
  for (int count : gpu_counts)
    sum_counts += count;
  ASSERT_EQ(sum_counts, static_cast<int64_t>(size));

  std::unordered_map<int, int> gpu_map;
  gpu_map.reserve(num_unique * 2);
  for (size_t i = 0; i < num_unique; i++)
    gpu_map[gpu_unique[i]] = gpu_counts[i];
  ASSERT_EQ(gpu_map.size(), cpu_counts.size());
  for (const auto &kv : cpu_counts) {
    auto it = gpu_map.find(kv.first);
    ASSERT_NE(it, gpu_map.end()) << "GPU missing key " << kv.first;
    ASSERT_EQ(it->second, kv.second) << "count mismatch for key " << kv.first;
  }

  int *d_unique_sorted = nullptr;
  int *d_counts_sorted = nullptr;
  HIP_CHECK(hipMalloc(&d_unique_sorted, num_unique * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_counts_sorted, num_unique * sizeof(int)));

  void *d_temp_sort_pairs = nullptr;
  size_t temp_sort_pairs_bytes = 0;
  HIP_CHECK(rocprim::radix_sort_pairs_desc(
      nullptr, temp_sort_pairs_bytes, d_counts_out, d_counts_sorted,
      d_unique_out, d_unique_sorted, num_unique, 0, sizeof(int) * 8,
      hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_sort_pairs, temp_sort_pairs_bytes));
  HIP_CHECK(rocprim::radix_sort_pairs_desc(
      d_temp_sort_pairs, temp_sort_pairs_bytes, d_counts_out, d_counts_sorted,
      d_unique_out, d_unique_sorted, num_unique, 0, sizeof(int) * 8,
      hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  std::vector<int> top_keys(top_k);
  std::vector<int> top_counts(top_k);
  HIP_CHECK(hipMemcpy(top_keys.data(), d_unique_sorted, top_k * sizeof(int),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(top_counts.data(), d_counts_sorted, top_k * sizeof(int),
                      hipMemcpyDeviceToHost));

  for (int i = 1; i < top_k; i++)
    ASSERT_LE(top_counts[i], top_counts[i - 1]);

  std::vector<std::pair<int, int>> gpu_top(top_k);
  std::vector<std::pair<int, int>> cpu_top(top_k);
  for (int i = 0; i < top_k; i++) {
    gpu_top[i] = {top_keys[i], top_counts[i]};
    cpu_top[i] = {cpu_top_keys[i], cpu_top_counts[i]};
  }
  std::sort(gpu_top.begin(), gpu_top.end(), top_cmp);
  std::sort(cpu_top.begin(), cpu_top.end(), top_cmp);
  ASSERT_EQ(gpu_top, cpu_top);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_sorted));
  HIP_CHECK(hipFree(d_unique_out));
  HIP_CHECK(hipFree(d_counts_out));
  HIP_CHECK(hipFree(d_runs_count));
  HIP_CHECK(hipFree(d_unique_sorted));
  HIP_CHECK(hipFree(d_counts_sorted));
  safe_free(d_temp_sort);
  safe_free(d_temp_rle);
  safe_free(d_temp_sort_pairs);
}

TEST(RocprimSampleTests, MlFeatureEngineering) {
  const size_t num_samples = 100000;
  std::vector<float> features(num_samples);

  std::mt19937 gen(42);
  std::normal_distribution<float> dis(50.0f, 15.0f);
  for (size_t i = 0; i < num_samples; i++)
    features[i] = dis(gen);

  auto [min_it, max_it] = std::minmax_element(features.begin(), features.end());
  const float cpu_min = *min_it;
  const float cpu_max = *max_it;

  std::vector<float> cpu_norm(num_samples);
  const float denom = cpu_max - cpu_min;
  for (size_t i = 0; i < num_samples; i++)
    cpu_norm[i] = (denom == 0.0f) ? 0.0f : (features[i] - cpu_min) / denom;

  const int num_bins = 10;
  const float lower_level = 0.0f;
  const float upper_level = 1.0001f;
  std::vector<unsigned int> cpu_hist =
      cpu_histogram_even(cpu_norm, num_bins, lower_level, upper_level);

  float *d_features = nullptr;
  float *d_normalized = nullptr;
  HIP_CHECK(hipMalloc(&d_features, num_samples * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_normalized, num_samples * sizeof(float)));
  HIP_CHECK(hipMemcpy(d_features, features.data(), num_samples * sizeof(float),
                      hipMemcpyHostToDevice));

  float *d_min = nullptr;
  float *d_max = nullptr;
  HIP_CHECK(hipMalloc(&d_min, sizeof(float)));
  HIP_CHECK(hipMalloc(&d_max, sizeof(float)));

  void *d_temp_min = nullptr;
  size_t temp_min_bytes = 0;
  HIP_CHECK(rocprim::reduce(nullptr, temp_min_bytes, d_features, d_min,
                            num_samples, rocprim::minimum<float>(),
                            hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_min, temp_min_bytes));
  HIP_CHECK(rocprim::reduce(d_temp_min, temp_min_bytes, d_features, d_min,
                            num_samples, rocprim::minimum<float>(),
                            hipStreamDefault));

  void *d_temp_max = nullptr;
  size_t temp_max_bytes = 0;
  HIP_CHECK(rocprim::reduce(nullptr, temp_max_bytes, d_features, d_max,
                            num_samples, rocprim::maximum<float>(),
                            hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_max, temp_max_bytes));
  HIP_CHECK(rocprim::reduce(d_temp_max, temp_max_bytes, d_features, d_max,
                            num_samples, rocprim::maximum<float>(),
                            hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  float gpu_min = 0.0f;
  float gpu_max = 0.0f;
  HIP_CHECK(hipMemcpy(&gpu_min, d_min, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&gpu_max, d_max, sizeof(float), hipMemcpyDeviceToHost));

  ASSERT_TRUE(nearly_equal(cpu_min, gpu_min, 1e-4f, 1e-6f));
  ASSERT_TRUE(nearly_equal(cpu_max, gpu_max, 1e-4f, 1e-6f));

  const int threads = 256;
  const int blocks = static_cast<int>((num_samples + threads - 1) / threads);
  hipLaunchKernelGGL(normalize_kernel, dim3(blocks), dim3(threads), 0,
                     hipStreamDefault, d_features, d_normalized, gpu_min,
                     gpu_max, num_samples);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  std::vector<float> gpu_norm(num_samples);
  HIP_CHECK(hipMemcpy(gpu_norm.data(), d_normalized,
                      num_samples * sizeof(float), hipMemcpyDeviceToHost));

  const float eps = 1e-4f;
  for (float value : gpu_norm)
    ASSERT_TRUE(value >= -eps && value <= 1.0f + eps);

  for (uint64_t pos : {0ULL, 1ULL, 2ULL, 99ULL, 999ULL,
                       static_cast<unsigned long long>(num_samples - 1)}) {
    ASSERT_TRUE(nearly_equal(cpu_norm[pos], gpu_norm[pos], 5e-4f, 1e-5f))
        << "normalized value mismatch at pos=" << pos;
  }

  unsigned int *d_histogram = nullptr;
  HIP_CHECK(hipMalloc(&d_histogram, num_bins * sizeof(unsigned int)));
  HIP_CHECK(hipMemset(d_histogram, 0, num_bins * sizeof(unsigned int)));

  void *d_temp_hist = nullptr;
  size_t temp_hist_bytes = 0;
  HIP_CHECK(rocprim::histogram_even(
      nullptr, temp_hist_bytes, d_normalized, num_samples, d_histogram,
      num_bins + 1, lower_level, upper_level, hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_hist, temp_hist_bytes));
  HIP_CHECK(rocprim::histogram_even(
      d_temp_hist, temp_hist_bytes, d_normalized, num_samples, d_histogram,
      num_bins + 1, lower_level, upper_level, hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  std::vector<unsigned int> gpu_hist(num_bins);
  HIP_CHECK(hipMemcpy(gpu_hist.data(), d_histogram,
                      num_bins * sizeof(unsigned int), hipMemcpyDeviceToHost));

  uint64_t total_binned = 0;
  for (unsigned int count : gpu_hist)
    total_binned += count;
  ASSERT_EQ(total_binned, num_samples);
  ASSERT_EQ(gpu_hist, cpu_hist);

  HIP_CHECK(hipFree(d_features));
  HIP_CHECK(hipFree(d_normalized));
  HIP_CHECK(hipFree(d_min));
  HIP_CHECK(hipFree(d_max));
  HIP_CHECK(hipFree(d_histogram));
  HIP_CHECK(hipFree(d_temp_min));
  HIP_CHECK(hipFree(d_temp_max));
  HIP_CHECK(hipFree(d_temp_hist));
}

TEST(RocprimSampleTests, EtlWorkflow) {
  const size_t size = 100000;
  std::vector<int> input(size);
  for (size_t i = 0; i < size; i++)
    input[i] = static_cast<int>((i % 1000) * 0.5);

  std::vector<int> cpu_scaled(size);
  std::transform(input.begin(), input.end(), cpu_scaled.begin(),
                 [](int x) { return x * 2; });

  std::vector<int> cpu_filtered;
  cpu_filtered.reserve(size);
  for (int value : cpu_scaled) {
    if (value > 100)
      cpu_filtered.push_back(value);
  }
  const size_t cpu_num_filtered = cpu_filtered.size();

  std::sort(cpu_filtered.begin(), cpu_filtered.end());
  cpu_filtered.erase(std::unique(cpu_filtered.begin(), cpu_filtered.end()),
                     cpu_filtered.end());
  const size_t cpu_num_unique = cpu_filtered.size();
  ASSERT_GT(cpu_num_filtered, 0U);
  ASSERT_GT(cpu_num_unique, 0U);

  int *d_input = nullptr;
  int *d_scaled = nullptr;
  int *d_filtered = nullptr;
  int *d_sorted = nullptr;
  int *d_unique = nullptr;
  HIP_CHECK(hipMalloc(&d_input, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_scaled, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_filtered, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sorted, size * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_unique, size * sizeof(int)));
  HIP_CHECK(
      hipMemcpy(d_input, input.data(), size * sizeof(int), hipMemcpyHostToDevice));

  HIP_CHECK(rocprim::transform(
      d_input, d_scaled, size, [] __device__(int x) { return x * 2; },
      hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  size_t *d_num_selected = nullptr;
  HIP_CHECK(hipMalloc(&d_num_selected, sizeof(size_t)));

  void *d_temp_select = nullptr;
  size_t temp_select_bytes = 0;
  auto pred = [] __device__(int x) { return x > 100; };
  HIP_CHECK(rocprim::select(nullptr, temp_select_bytes, d_scaled, d_filtered,
                            d_num_selected, size, pred, hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_select, temp_select_bytes));
  HIP_CHECK(rocprim::select(d_temp_select, temp_select_bytes, d_scaled,
                            d_filtered, d_num_selected, size, pred,
                            hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  size_t gpu_num_filtered = 0;
  HIP_CHECK(hipMemcpy(&gpu_num_filtered, d_num_selected, sizeof(size_t),
                      hipMemcpyDeviceToHost));
  ASSERT_EQ(gpu_num_filtered, cpu_num_filtered);

  void *d_temp_sort = nullptr;
  size_t temp_sort_bytes = 0;
  HIP_CHECK(rocprim::radix_sort_keys(nullptr, temp_sort_bytes, d_filtered,
                                     d_sorted, gpu_num_filtered, 0,
                                     sizeof(int) * 8, hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_sort, temp_sort_bytes));
  HIP_CHECK(rocprim::radix_sort_keys(d_temp_sort, temp_sort_bytes, d_filtered,
                                     d_sorted, gpu_num_filtered, 0,
                                     sizeof(int) * 8, hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  size_t *d_num_unique = nullptr;
  HIP_CHECK(hipMalloc(&d_num_unique, sizeof(size_t)));

  void *d_temp_unique = nullptr;
  size_t temp_unique_bytes = 0;
  HIP_CHECK(rocprim::unique(nullptr, temp_unique_bytes, d_sorted, d_unique,
                            d_num_unique, gpu_num_filtered,
                            rocprim::equal_to<int>{}, hipStreamDefault));
  HIP_CHECK(hipMalloc(&d_temp_unique, temp_unique_bytes));
  HIP_CHECK(rocprim::unique(d_temp_unique, temp_unique_bytes, d_sorted,
                            d_unique, d_num_unique, gpu_num_filtered,
                            rocprim::equal_to<int>{}, hipStreamDefault));
  HIP_CHECK(hipStreamSynchronize(hipStreamDefault));

  size_t gpu_num_unique = 0;
  HIP_CHECK(hipMemcpy(&gpu_num_unique, d_num_unique, sizeof(size_t),
                      hipMemcpyDeviceToHost));
  ASSERT_EQ(gpu_num_unique, cpu_num_unique);

  std::vector<int> gpu_output(gpu_num_unique);
  HIP_CHECK(hipMemcpy(gpu_output.data(), d_unique, gpu_num_unique * sizeof(int),
                      hipMemcpyDeviceToHost));

  ASSERT_TRUE(std::is_sorted(gpu_output.begin(), gpu_output.end()));
  ASSERT_EQ(std::adjacent_find(gpu_output.begin(), gpu_output.end()),
            gpu_output.end());
  ASSERT_EQ(gpu_output, cpu_filtered);

  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_scaled));
  HIP_CHECK(hipFree(d_filtered));
  HIP_CHECK(hipFree(d_sorted));
  HIP_CHECK(hipFree(d_unique));
  HIP_CHECK(hipFree(d_num_selected));
  HIP_CHECK(hipFree(d_num_unique));
  safe_free(d_temp_select);
  safe_free(d_temp_sort);
  safe_free(d_temp_unique);
}
