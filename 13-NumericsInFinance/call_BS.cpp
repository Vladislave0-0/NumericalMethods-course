#include <chrono>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <sleef.h>
#include <vector>

extern "C" void logf_avx2(const float *src, float *dst, size_t size);

double runExperiment(int numSteps, uint64_t numPaths) {
  const float S0 = 100.0f, K = 100.0f, r = 0.05f, sigma = 0.1f, T = 1.0f;
  const float dt = T / numSteps;
  const float drift = (r - 0.5f * sigma * sigma) * dt;
  const float volSqrtDt = sigma * std::sqrt(dt);

  __m256 v_drift = _mm256_set1_ps(drift);
  __m256 v_vol = _mm256_set1_ps(volSqrtDt);
  __m256 v_S0 = _mm256_set1_ps(S0);
  __m256 v_K = _mm256_set1_ps(K);
  __m256 v_zero = _mm256_setzero_ps();
  __m256 v_two_pi = _mm256_set1_ps(2.0f * M_PI);
  __m256 v_minus_two = _mm256_set1_ps(-2.0f);

  double totalSum = 0.0;
  auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel reduction(+ : totalSum)
  {
    unsigned int seed = 123 + omp_get_thread_num();
    alignas(32) float u1[8], u2[8], l1[8];

#pragma omp for
    for (uint64_t i = 0; i < numPaths / 8; ++i) {
      __m256 v_S = v_S0;

      for (int s = 0; s < numSteps; ++s) {
        for (int j = 0; j < 8; ++j) {
          u1[j] = (float)rand_r(&seed) / RAND_MAX;
          u2[j] = (float)rand_r(&seed) / RAND_MAX;

          if (u1[j] < 1e-12f)
            u1[j] = 1e-12f;
        }

        logf_avx2(u1, l1, 8);

        __m256 radius =
            _mm256_sqrt_ps(_mm256_mul_ps(v_minus_two, _mm256_load_ps(l1)));
        __m256 angle = _mm256_mul_ps(v_two_pi, _mm256_load_ps(u2));
        __m256 z = _mm256_mul_ps(radius, Sleef_cosf8_u10(angle));

        // S = S * exp(drift + vol*z)
        v_S = _mm256_mul_ps(
            v_S, Sleef_expf8_u10(_mm256_fmadd_ps(v_vol, z, v_drift)));
      }

      __m256 payoff = _mm256_max_ps(_mm256_sub_ps(v_S, v_K), v_zero);
      alignas(32) float res[8];
      _mm256_store_ps(res, payoff);
      for (int j = 0; j < 8; ++j)
        totalSum += res[j];
    }
  }

  auto end = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration<double, std::milli>(end - start).count();

  double price = std::exp(-r * T) * (totalSum / numPaths);
  std::cout << numSteps << "\t" << price << "\t\t" << duration << " ms"
            << std::endl;

  return price;
}

int main() {
  std::vector<int> steps = {10, 50, 100, 500, 1000, 5000};
  std::cout << "Steps\tPrice\t\tTime" << std::endl;

  for (int s : steps)
    runExperiment(s, 1000000);
}
