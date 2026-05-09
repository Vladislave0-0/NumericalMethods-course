#include "minstd_vec.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>
#include <x86intrin.h>

static inline size_t count_inside_circle(const float *__restrict__ xy,
                                         size_t count) {
  size_t hits = 0;

#pragma omp simd reduction(+ : hits)
  for (size_t i = 0; i < count; ++i) {
    float x = xy[2 * i];
    float y = xy[2 * i + 1];
    hits += (x * x + y * y <= 1.0f) ? 1 : 0;
  }

  return hits;
}

double compute_pi_parallel(size_t total_points, uint32_t seed,
                           size_t block_size) {
  const int num_threads = omp_get_max_threads();
  std::vector<size_t> local_hits(num_threads, 0);

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    const size_t points_per_thread = total_points / num_threads;
    const size_t remainder =
        (tid == num_threads - 1) ? (total_points % num_threads) : 0;
    const size_t my_points = points_per_thread + remainder;
    const size_t rand_needed = my_points * 2;

    VecLCG_FP32 rng(block_size);
    rng.init_with_offset(seed, tid * points_per_thread * 2);

    std::vector<float> block_buf(2 * block_size);
    size_t generated = 0;
    size_t my_hits = 0;

    while (generated < rand_needed) {
      const size_t to_gen = (rand_needed - generated >= 2 * block_size)
                                ? 2 * block_size
                                : (rand_needed - generated);
      const size_t pairs = to_gen / 2;

      rng.fill_float(block_buf.data());              // X-координаты
      rng.fill_float(block_buf.data() + block_size); // Y-координаты

      size_t batch_hits = 0;

#pragma omp simd reduction(+ : batch_hits)
      for (size_t i = 0; i < pairs; ++i) {
        float x = block_buf[i];
        float y = block_buf[block_size + i];

        if (x * x + y * y <= 1.0f)
          batch_hits++;
      }

      my_hits += batch_hits;
      generated += 2 * pairs;
    }

    local_hits[tid] = my_hits;
  }

  size_t total_hits = 0;
  for (size_t h : local_hits)
    total_hits += h;

  return 4.0 * static_cast<double>(total_hits) /
         static_cast<double>(total_points);
}

int main() {
  constexpr size_t TOTAL_POINTS = 100'000'000; // 100 млн 2D-векторов
  constexpr size_t BLOCK_SIZE = 256;
  constexpr uint32_t SEED = 987654321;

  std::cout << "Monte Carlo Pi: " << TOTAL_POINTS
            << " points, block=" << BLOCK_SIZE
            << ", threads=" << omp_get_max_threads() << "\n\n";

  const unsigned long long start_ticks = __rdtsc();
  const auto t0 = std::chrono::high_resolution_clock::now();

  const double pi_est = compute_pi_parallel(TOTAL_POINTS, SEED, BLOCK_SIZE);

  const auto t1 = std::chrono::high_resolution_clock::now();
  const unsigned long long end_ticks = __rdtsc();

  const double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
  const unsigned long long cycles = end_ticks - start_ticks;
  const double cycles_per_random =
      static_cast<double>(cycles) / (TOTAL_POINTS * 2);

  constexpr double PI_REF = 3.14159265358979323846;
  const double abs_err = std::abs(pi_est - PI_REF);
  const double std_err = std::sqrt(PI_REF * (4.0 - PI_REF) / TOTAL_POINTS);

  std::cout.precision(10);
  std::cout << std::fixed;
  std::cout << "Estimated pi : " << pi_est << "\n"
            << "Reference pi : " << PI_REF << "\n\n"
            << "Abs error    : " << abs_err << "\n"
            << "Std error    : " << std_err << "\n\n"
            << "Time         : " << elapsed_sec << " s\n"
            << "Cycles/rand  : " << cycles_per_random << "\n\n";

  // Ошибка должна быть в пределах 3 сигма
  if (abs_err <= 3.0 * std_err) {
    std::cout << "Result within 3sigma confidence interval\n";
  } else {
    std::cout << "Warning: result outside 3sigma\n";
  }
}
