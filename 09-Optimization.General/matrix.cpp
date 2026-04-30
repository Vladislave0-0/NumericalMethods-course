#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <immintrin.h> // For AVX2, FMA
#ifdef _MSC_VER
#include <intrin.h> // For MSVC
#else
#include <x86intrin.h> // For GCC/Clang (Linux/macOS)
#endif

#define BLU "\x1B[34m"
#define GRN "\x1B[32m"
#define RED "\x1B[31m"
#define YEL "\x1B[33m"
#define RST "\x1B[0m"

struct matmul_result {
  char method[24];
  unsigned long long ticks;
  double speedup;
  int correct;
};

struct BenchmarkResult {
  std::string name;
  unsigned long long ticks;
  bool correct;
};

using MatMulFunc =
    std::function<void(int, int, int, const float *, const float *, float *)>;

void fill_random(float *mat, int size) {
  for (int i = 0; i < size; ++i)
    mat[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

void print_matmul_table(const char *size_label, matmul_result *results,
                        int count) {
  printf("\n" YEL "Testing Size: %s" RST "\n", size_label);
  printf(BLU "┌──────────────┬────────────┬──────────┬─────────┐" RST "\n");
  printf(BLU "│" RST " %-12s " BLU "│" RST " %-10s " BLU "│" RST " %-7s " BLU
             "│" RST " %-7s " BLU "│" RST "\n",
         "   Method", "   Ticks", " Speedup", "Correct");
  printf(BLU "├──────────────┼────────────┼──────────┼─────────┤" RST "\n");

  for (int i = 0; i < count; ++i) {
    const char *speed_color = (results[i].speedup > 1.05)
                                  ? GRN
                                  : (results[i].speedup < 0.95 ? RED : RST);

    printf(BLU "│" RST " %-12s " BLU "│" RST " %-10llu " BLU "│"
               "%s"
               " %-7.2fx " RST BLU "│" RST " %-5s  " BLU "│" RST "\n",
           results[i].method, results[i].ticks, speed_color, results[i].speedup,
           results[i].correct ? GRN "  YES " RST : RED "  FAIL" RST);
  }

  printf(BLU "└──────────────┴────────────┴──────────┴─────────┘" RST "\n");
}

BenchmarkResult run_test(const std::string &name, MatMulFunc func, int M, int K,
                         int N, const float *A, const float *B,
                         const float *ref_C) {

  int size_C = M * N;
  float *C = (float *)aligned_alloc(64, size_C * sizeof(float));

  // Matrix zeroing
  for (int i = 0; i < size_C; ++i)
    C[i] = 0.0f;

  // Warming up the cache
  func(M, K, N, A, B, C);
  for (int i = 0; i < size_C; ++i)
    C[i] = 0.0f;

  // Measure tacts
  unsigned long long start = __rdtsc();
  func(M, K, N, A, B, C);
  unsigned long long end = __rdtsc();

  // Correctness check
  bool ok = true;
  for (int i = 0; i < size_C; ++i) {
    if (std::abs(ref_C[i] - C[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }

  unsigned long long total_ticks = end - start;
  free(C);

  return {name, total_ticks, ok};
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
  // Go through the rows of matrix A with step 2
  for (int i = 0; i < M; i += 2) {
    // Go through the cols of matrix B with step 16 (2 registers of 8 floats)
    for (int j = 0; j < N; j += 16) {

      // 4 accumulators for the 2x16 block
      __m256 acc00 = _mm256_setzero_ps(); // Row i,   cols j  , ..., j+7
      __m256 acc01 = _mm256_setzero_ps(); // Row i,   cols j+8, ..., j+15
      __m256 acc10 = _mm256_setzero_ps(); // Row i+1, cols j  , ..., j+7
      __m256 acc11 = _mm256_setzero_ps(); // Row i+1, cols j+8, ..., j+15

      for (int k = 0; k < K; ++k) {
        // Load elements from A
        __m256 a0 = _mm256_set1_ps(A[i * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(i + 1) * K + k]);

        // Load 16 elements from B
        __m256 b0 = _mm256_loadu_ps(&B[k * N + j]);
        __m256 b1 = _mm256_loadu_ps(&B[k * N + j + 8]);

        // Multiply and accumulate the sum
        acc00 = _mm256_fmadd_ps(a0, b0, acc00);
        acc01 = _mm256_fmadd_ps(a0, b1, acc01);
        acc10 = _mm256_fmadd_ps(a1, b0, acc10);
        acc11 = _mm256_fmadd_ps(a1, b1, acc11);
      }

      // Upload 4 registers into memory once
      _mm256_storeu_ps(&C[i * N + j], acc00);
      _mm256_storeu_ps(&C[i * N + j + 8], acc01);
      _mm256_storeu_ps(&C[(i + 1) * N + j], acc10);
      _mm256_storeu_ps(&C[(i + 1) * N + j + 8], acc11);
    }
  }
}

int main() {
  std::vector<std::pair<int, int>> dim_pairs = {
      {32, 32}, {64, 64}, {128, 128}, {256, 256}, {512, 512}, {1024, 1024}};

  std::vector<std::pair<std::string, MatMulFunc>> methods = {
      {"MatMul naive", matmul_naive},
      {"MatMul opt", matmul_optimized},
      {"AVX2 + FMA", matmul_avx2},
      {"MatMul reg", matmul_multi_reg}};

  for (auto &dims : dim_pairs) {
    int M = dims.first, K = dims.first, N = dims.second;

    float *A = (float *)aligned_alloc(64, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(64, K * N * sizeof(float));
    float *ref_C = (float *)aligned_alloc(64, M * N * sizeof(float));

    fill_random(A, M * K);
    fill_random(B, K * N);

    // Reference
    matmul_naive(M, K, N, A, B, ref_C);

    matmul_result table_data[4];
    unsigned long long naive_ticks = 0;

    for (size_t i = 0; i < methods.size(); ++i) {
      auto res =
          run_test(methods[i].first, methods[i].second, M, K, N, A, B, ref_C);

      if (i == 0)
        naive_ticks = res.ticks;

      strncpy(table_data[i].method, res.name.c_str(), 23);
      table_data[i].ticks = res.ticks;
      table_data[i].speedup =
          (naive_ticks > 0) ? (double)naive_ticks / res.ticks : 1.0;
      table_data[i].correct = res.correct;
    }

    char size_str[32];
    sprintf(size_str, "%dx%dx%d", M, K, N);
    print_matmul_table(size_str, table_data, 4);

    free(A);
    free(B);
    free(ref_C);
  }
}
