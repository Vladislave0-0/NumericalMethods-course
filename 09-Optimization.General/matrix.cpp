#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <string>
#include <time.h>
#include <tuple>
#include <vector>

#define BLU "\x1B[34m"
#define GRN "\x1B[32m"
#define RED "\x1B[31m"
#define YEL "\x1B[33m"
#define RST "\x1B[0m"

const int BLOCK_SIZE = 32;

struct Result {
  std::string name;
  double avg_ticks;
  double std_dev;
  bool correct;
};

void fill_random(float *mat, int size) {
  for (int i = 0; i < size; ++i)
    mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void print_table(const char *size_label, const std::vector<Result> &results) {
  printf("\n" YEL "Matrix size: %s" RST "\n", size_label);
  printf(BLU "┌─────────────────────┬─────────────┬────────────┬─────────┐" RST
             "\n");
  printf(BLU "│" RST " %-19s " BLU "│" RST " %-11s " BLU "│" RST " %-9s " BLU
             "│" RST " %-7s " BLU "│" RST "\n",
         "Method", "Avg Ticks", "Std Dev, %", "Correct");
  printf(BLU "├─────────────────────┼─────────────┼────────────┼─────────┤" RST
             "\n");

  for (const auto &res : results) {
    double std_dev_p = (res.std_dev / res.avg_ticks) * 100.0;
    const char *std_dev_color = (std_dev_p > 5.0 ? RED : GRN);

    printf(BLU "│" RST " %-19s " BLU "│" RST " %-11.2f " BLU "│"
               "%s %-9.2fx " RST BLU "│" RST " %-7s " BLU "│" RST "\n",
           res.name.c_str(), res.avg_ticks, std_dev_color, res.std_dev,
           res.correct ? GRN "  YES  " RST : RED "  FAIL " RST);
  }
  printf(BLU "└─────────────────────┴─────────────┴────────────┴─────────┘" RST
             "\n");
}

void matmul_naive(int M, int K, int N, const float *A, const float *B,
                  float *C) {
  // for (int i = 0; i < M * N; ++i)
  //   C[i] = 0.0f;

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float part_sum = 0.0f;

      for (int k = 0; k < K; ++k)
        part_sum += A[i * K + k] * B[k * N + j];

      C[i * N + j] = part_sum;
    }
  }
}

void matmul_optimized(int M, int K, int N, const float *__restrict A,
                      const float *__restrict B, float *__restrict C) {
  // for (int i = 0; i < M * N; ++i)
  //   C[i] = 0.0f;

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float fixed_a = A[i * K + k];

      for (int j = 0; j < N; ++j)
        C[i * N + j] += fixed_a * B[k * N + j];
    }
  }
}

void matmul_avx2(int M, int K, int N, const float *__restrict A,
                 const float *__restrict B, float *__restrict C) {
  // for (int i = 0; i < M * N; ++i)
  //   C[i] = 0.0f;

  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      // Load element A[i][k] and propagate it by the entire register
      __m256 vec_a = _mm256_set1_ps(A[i * K + k]);

      // Go through the row of the matrix B and C
      for (int j = 0; j < N; j += 8) {
        // Load 8 elements from B[k][j], ..., B[k][j+7]
        __m256 vec_b = _mm256_loadu_ps(&B[k * N + j]);

        // Load 8 elements from C[i][j], ..., C[i][j+7]
        __m256 vec_c = _mm256_loadu_ps(&C[i * N + j]);

        // C = A * B + C
        vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);

        // Saving the result back to the C memory
        _mm256_storeu_ps(&C[i * N + j], vec_c);
      }
    }
  }
}

void matmul_multi_reg(int M, int K, int N, const float *__restrict A,
                      const float *__restrict B, float *__restrict C) {
  // Go through the rows of matrix A with step 4
  for (int i = 0; i < M; i += 4) {
    // Go through the cols of matrix B with step 16 (2 registers of 8 floats)
    for (int j = 0; j < N; j += 16) {

      // 8 accumulators for the 4x16 block
      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc01 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc11 = _mm256_setzero_ps();
      __m256 acc20 = _mm256_setzero_ps();
      __m256 acc21 = _mm256_setzero_ps();
      __m256 acc30 = _mm256_setzero_ps();
      __m256 acc31 = _mm256_setzero_ps();

      for (int k = 0; k < K; ++k) {
        // Load elements from A
        __m256 a0 = _mm256_set1_ps(A[(i + 0) * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);
        __m256 a2 = _mm256_set1_ps(A[(i + 2) * K + k]);
        __m256 a3 = _mm256_set1_ps(A[(i + 3) * K + k]);

        // Load 16 elements from B
        __m256 b0 = _mm256_load_ps(&B[k * N + j]);
        __m256 b1 = _mm256_load_ps(&B[k * N + j + 8]);

        // Multiply and accumulate the sum
        acc00 = _mm256_fmadd_ps(a0, b0, acc00);
        acc01 = _mm256_fmadd_ps(a0, b1, acc01);
        acc10 = _mm256_fmadd_ps(a1, b0, acc10);
        acc11 = _mm256_fmadd_ps(a1, b1, acc11);
        acc20 = _mm256_fmadd_ps(a2, b0, acc20);
        acc21 = _mm256_fmadd_ps(a2, b1, acc21);
        acc30 = _mm256_fmadd_ps(a3, b0, acc30);
        acc31 = _mm256_fmadd_ps(a3, b1, acc31);
      }

      // Upload 4 registers into memory
      _mm256_store_ps(&C[(i + 0) * N + j], acc00);
      _mm256_store_ps(&C[(i + 0) * N + j + 8], acc01);
      _mm256_store_ps(&C[(i + 1) * N + j], acc10);
      _mm256_store_ps(&C[(i + 1) * N + j + 8], acc11);
      _mm256_store_ps(&C[(i + 2) * N + j], acc20);
      _mm256_store_ps(&C[(i + 2) * N + j + 8], acc21);
      _mm256_store_ps(&C[(i + 3) * N + j], acc30);
      _mm256_store_ps(&C[(i + 3) * N + j + 8], acc31);
    }
  }
}

void matmul_tiled_reg(int M, int K, int N, const float *__restrict A,
                      const float *__restrict B, float *__restrict C) {
  for (int bi = 0; bi < M; bi += BLOCK_SIZE) {
    for (int bk = 0; bk < K; bk += BLOCK_SIZE) {
      for (int bj = 0; bj < N; bj += BLOCK_SIZE) {
        for (int i = bi; i < bi + BLOCK_SIZE && i < M; i += 4) {
          for (int j = bj; j < bj + BLOCK_SIZE && j < N; j += 16) {
            __m256 acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
            acc00 = _mm256_load_ps(&C[(i + 0) * N + j]);
            acc01 = _mm256_load_ps(&C[(i + 0) * N + j + 8]);
            acc10 = _mm256_load_ps(&C[(i + 1) * N + j]);
            acc11 = _mm256_load_ps(&C[(i + 1) * N + j + 8]);
            acc20 = _mm256_load_ps(&C[(i + 2) * N + j]);
            acc21 = _mm256_load_ps(&C[(i + 2) * N + j + 8]);
            acc30 = _mm256_load_ps(&C[(i + 3) * N + j]);
            acc31 = _mm256_load_ps(&C[(i + 3) * N + j + 8]);

            for (int k = bk; k < bk + BLOCK_SIZE && k < K; ++k) {
              __m256 a0 = _mm256_set1_ps(A[(i + 0) * K + k]);
              __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);
              __m256 a2 = _mm256_set1_ps(A[(i + 2) * K + k]);
              __m256 a3 = _mm256_set1_ps(A[(i + 3) * K + k]);

              __m256 b0 = _mm256_load_ps(&B[k * N + j]);
              __m256 b1 = _mm256_load_ps(&B[k * N + j + 8]);

              acc00 = _mm256_fmadd_ps(a0, b0, acc00);
              acc01 = _mm256_fmadd_ps(a0, b1, acc01);
              acc10 = _mm256_fmadd_ps(a1, b0, acc10);
              acc11 = _mm256_fmadd_ps(a1, b1, acc11);
              acc20 = _mm256_fmadd_ps(a2, b0, acc20);
              acc21 = _mm256_fmadd_ps(a2, b1, acc21);
              acc30 = _mm256_fmadd_ps(a3, b0, acc30);
              acc31 = _mm256_fmadd_ps(a3, b1, acc31);
            }

            _mm256_store_ps(&C[(i + 0) * N + j], acc00);
            _mm256_store_ps(&C[(i + 0) * N + j + 8], acc01);
            _mm256_store_ps(&C[(i + 1) * N + j], acc10);
            _mm256_store_ps(&C[(i + 1) * N + j + 8], acc11);
            _mm256_store_ps(&C[(i + 2) * N + j], acc20);
            _mm256_store_ps(&C[(i + 2) * N + j + 8], acc21);
            _mm256_store_ps(&C[(i + 3) * N + j], acc30);
            _mm256_store_ps(&C[(i + 3) * N + j + 8], acc31);
          }
        }
      }
    }
  }
}

template <typename F>
Result run_test(std::string name, F &&func, int M, int K, int N, const float *A,
                const float *B, const float *ref_C, int outer_loops,
                int inner_loops) {
  int size_C = M * N;
  float *C = (float *)aligned_alloc(64, size_C * sizeof(float));

  // Warming up the cache
  std::fill(C, C + size_C, 0.0f);
  func(M, K, N, A, B, C);

  // Correctness check
  bool all_ok = true;
  for (int i = 0; i < size_C; ++i) {
    if (std::abs(ref_C[i] - C[i]) > 1e-3f) {
      all_ok = false;
      break;
    }
  }

  std::vector<unsigned long long> results;
  results.reserve(outer_loops * inner_loops);

  for (int out = 0; out < outer_loops; ++out) {
    std::fill(C, C + size_C, 0.0f);

    unsigned long long start = __rdtsc();
    for (int in = 0; in < inner_loops; ++in)
      func(M, K, N, A, B, C);
    unsigned long long end = __rdtsc();

    results.push_back(static_cast<double>(end - start) / inner_loops);
  }

  double sum = 0;
  for (double r : results)
    sum += r;
  double avg_ticks = sum / outer_loops;

  double sq_sum = 0;
  for (double r : results)
    sq_sum += (r - avg_ticks) * (r - avg_ticks);
  double std_dev = std::sqrt(sq_sum / outer_loops);

  free(C);
  return {std::move(name), avg_ticks, std_dev, all_ok};
}

int main() {
  srand(static_cast<unsigned>(time(NULL)));

  std::vector<std::tuple<int, int, int, int, int>> dims = {
      {32, 32, 32, 200, 2000},   {64, 64, 64, 200, 800},
      {128, 128, 128, 100, 250}, {256, 256, 256, 50, 250},
      {512, 512, 512, 50, 10},   {1024, 1024, 1024, 25, 10}};

  for (const auto &[M, K, N, outer_loops, inner_loops] : dims) {
    float *A = (float *)aligned_alloc(64, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(64, K * N * sizeof(float));
    float *ref_C = (float *)aligned_alloc(64, M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    // Reference
    std::fill(ref_C, ref_C + (M * N), 0.0f);
    matmul_naive(M, K, N, A, B, ref_C);

    std::vector<Result> results;
    unsigned long long base_avg_ticks = 0;

    auto run_and_store = [&](std::string name, auto func) {
      Result res =
          run_test(name, func, M, K, N, A, B, ref_C, outer_loops, inner_loops);
      results.push_back(std::move(res));
    };

    run_and_store("MatMul naive", matmul_naive);
    run_and_store("MatMul opt", matmul_optimized);
    run_and_store("MatMul AVX2 + FMA", matmul_avx2);
    run_and_store("MatMul reg", matmul_multi_reg);
    run_and_store("MatMul reg + tiling", matmul_tiled_reg);

    char size_str[32];
    sprintf(size_str, "%dx%dx%d", M, K, N);
    print_table(size_str, results);

    free(A);
    free(B);
    free(ref_C);
  }
}
