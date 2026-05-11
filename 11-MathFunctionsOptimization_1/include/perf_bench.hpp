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
static void print(const std::string &label, Func &&func,
                  const std::vector<Ty> &data) {
  // Определяем, является ли функция векторной (принимает 3 аргумента)
  // или скалярной (принимает 1 аргумент)
  constexpr bool is_vector = !std::is_invocable_v<Func, Ty>;

  std::cout << YEL << label << RST << std::endl;

  auto run_latency = [&]() {
    if constexpr (is_vector) {
      auto wrapper = [&](Ty x) {
        alignas(32) Ty src[8];
        alignas(32) Ty dst[8];
        for (int i = 0; i < 8; ++i)
          src[i] = x;
        func(src, dst, 8);
        return dst[0];
      };
      return measure_latency(wrapper, data).mean / 8.0;
    } else {
      return measure_latency(func, data).mean;
    }
  };

  auto run_throughput = [&]() {
    if constexpr (is_vector) {
      auto wrapper = [&](Ty x) {
        alignas(32) Ty src[8];
        alignas(32) Ty dst[8];
        for (int i = 0; i < 8; ++i)
          src[i] = x;
        func(src, dst, 8);
        return dst[0];
      };
      return measure_throughput(wrapper, data).mean / 8.0;
    } else {
      return measure_throughput(func, data).mean;
    }
  };

  double lat_mean = run_latency();
  double thr_mean = run_throughput();

  std::cout << "[LATENCY]    " << lat_mean << " CPE\n";
  std::cout << "[THROUGHPUT] " << thr_mean << " CPE\n\n";
}

} // namespace perf
