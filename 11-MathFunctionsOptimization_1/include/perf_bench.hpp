#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>
#include <x86intrin.h>

namespace perf {

struct BenchResult final {
  double mean;
  double stddev;
};

#define YEL "\x1B[33m"
#define RST "\x1B[0m"

struct Stats final {
  static BenchResult calculate(std::vector<uint64_t> &cycles,
                               size_t batch_size) {
    if (cycles.empty())
      return {0, 0};

    std::sort(cycles.begin(), cycles.end());

    size_t keep_size = static_cast<size_t>(cycles.size() * 0.95);

    std::vector<double> cpe_data;
    for (size_t i = 0; i < keep_size; ++i)
      cpe_data.push_back(static_cast<double>(cycles[i]) / batch_size);

    double sum = std::accumulate(cpe_data.begin(), cpe_data.end(), 0.0);
    double mean = sum / cpe_data.size();

    double sq_sum = std::inner_product(cpe_data.begin(), cpe_data.end(),
                                       cpe_data.begin(), 0.0);
    double stddev = std::sqrt(sq_sum / cpe_data.size() - mean * mean);

    return {mean, stddev};
  }
};

template <typename Func, typename Ty>
BenchResult measure_latency(Func &&func, const std::vector<Ty> &data) {
  const size_t iterations = 20000;
  const size_t n = data.size();
  std::vector<uint64_t> samples;
  samples.reserve(iterations);

  Ty warmup = 0;
  for (const auto &val : data)
    warmup = func(val);
  (void)warmup;

  for (size_t i = 0; i < iterations; ++i) {
    Ty dependency = 0;

    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t j = 0; j < n; ++j) {
      Ty input = data[j] + dependency * static_cast<Ty>(1e-20);
      dependency = func(input);
    }

    unsigned int aux;
    uint64_t end = __rdtscp(&aux);
    _mm_lfence();

    samples.push_back(end - start);

    auto volatile sink = dependency;
    (void)sink;
  }

  return Stats::calculate(samples, n);
}

template <typename Func, typename Ty>
BenchResult measure_throughput(Func &&func, const std::vector<Ty> &data) {
  const size_t iterations = 20000;
  const size_t n = data.size();
  std::vector<uint64_t> samples;
  samples.reserve(iterations);

  Ty warmup = 0;
  for (const auto &val : data)
    warmup = func(val);
  (void)warmup;

  for (size_t i = 0; i < iterations; ++i) {
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t j = 0; j < n; ++j) {
      auto volatile res = func(data[j]);
      (void)res;
    }

    unsigned int aux;
    uint64_t end = __rdtscp(&aux);
    _mm_lfence();

    samples.push_back(end - start);
  }

  return Stats::calculate(samples, n);
}

template <typename Func, typename Ty>
BenchResult measure_vector(Func &&func, const std::vector<Ty> &data) {
  const size_t outer_iterations = 10000;
  const size_t inner_batch = 10;
  const size_t n = data.size();

  std::vector<Ty> results(n);
  std::vector<uint64_t> samples;
  samples.reserve(outer_iterations);

  func(data.data(), results.data(), (int)n);

  for (size_t i = 0; i < outer_iterations; ++i) {
    _mm_mfence();
    uint64_t start = __rdtsc();

    for (size_t j = 0; j < inner_batch; ++j) {
      func(data.data(), results.data(), (int)n);
      asm volatile("" : : "g"(results.data()) : "memory");
    }

    unsigned int aux;
    uint64_t end = __rdtscp(&aux);
    _mm_lfence();

    samples.push_back(end - start);
  }

  return Stats::calculate(samples, n * inner_batch);
}

template <typename Func, typename Ty>
static void print(const std::string &label, Func &&func,
                  const std::vector<Ty> &data) {
  constexpr bool is_vector = std::is_invocable_v<Func, const Ty *, Ty *, int>;

  std::cout << YEL << label << RST << std::endl;
  if constexpr (is_vector) {
    BenchResult res = measure_vector(func, data);
    std::cout << "[THROUGHPUT] " << res.mean << " CPE (+- " << res.stddev
              << ")\n\n";
  } else {
    BenchResult lat = measure_latency(func, data);
    BenchResult thr = measure_throughput(func, data);
    std::cout << "[LATENCY]    " << lat.mean << " CPE (+- " << lat.stddev
              << ")\n";
    std::cout << "[THROUGHPUT] " << thr.mean << " CPE (+- " << thr.stddev
              << ")\n\n";
  }
}

} // namespace perf
