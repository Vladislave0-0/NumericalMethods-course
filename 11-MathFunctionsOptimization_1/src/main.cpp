#include "../include/perf_bench.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_SLEEF
#include <sleef.h>
#endif

extern "C" {
float logf_scalar(float x);
void logf_avx2(const float *src, float *dst, int n);
}

int main() {
  constexpr size_t size = 100000;
  std::vector<float> data(size);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(0.5f, 100.0f);
  std::generate(data.begin(), data.end(), [&]() { return dist(gen); });

  perf::print("std::log", [](float x) { return std::log(x); }, data);
  perf::print("logf_scalar", logf_scalar, data);
  perf::print("logf_avx2", logf_avx2, data);
  std::cout << "\n";

#ifdef USE_SLEEF
  auto sleef_simd = [](const float *src, float *dst, int n) {
    for (int i = 0; i < n; i += 8) {
      __m256 vx = _mm256_loadu_ps(&src[i]);
      __m256 vy = Sleef_logf8_u10(vx);
      _mm256_storeu_ps(&dst[i], vy);
    }
  };
  perf::print("SLEEF_AVX2", sleef_simd, data);
#endif

#ifdef USE_MKL
  auto mkl_ha = [](const float *src, float *dst, int n) {
    vmsLn(n, src, dst, VML_HA);
  };
  perf::print("MKL_HA", mkl_ha, data);
#endif
}
