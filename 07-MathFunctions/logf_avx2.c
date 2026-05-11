#include <immintrin.h>
#include <math.h>
#include <stdint.h>

extern float logf_scalar(float x);

#define TABLE_SIZE 256
#define TABLE_INDEX_BITS 8
#define TABLE_INDEX_SHIFT (23 - TABLE_INDEX_BITS)

#define MASK_EXP_FULL 0xff800000u
#define MASK_MANT 0x007FFFFFu
#define MIN_NORMAL 0x00800000u
#define REDUCE_OFFSET 0x3F2A2000u

#define LOG2_HI 0.693359375f
#define LOG2_LO -2.12194440e-4f

#define POLY_C1 0.9999999403953552f
#define POLY_C2 -0.4999999701976776f
#define POLY_C3 0.3333333432674408f

static float R_TABLE[TABLE_SIZE] __attribute__((aligned(64)));
static float T_TABLE[TABLE_SIZE] __attribute__((aligned(64)));
static int tables_initialized = 0;

void logf_init_tables(void) {
  if (tables_initialized)
    return;

  const int num0 = 341; // Центр редукции ~0.664
  const int den = 512;  // 2 * TABLE_SIZE

  for (int i = 0; i < TABLE_SIZE; ++i) {
    double x = (double)(num0 + i) / (double)den;
    double C = 1.0 / x; // C_i ≈ 1 / x_i

    if ((float)C < 1.0f) {
      x = (double)(num0 + i + (i - (den - num0))) / (double)den;
      C = 1.0 / x;
    }

    // Заполянем таблицы
    float C_f = (float)C;
    R_TABLE[i] = C_f;
    T_TABLE[i] = (float)(-log((double)C_f));
  }

  tables_initialized = 1;
}

void logf_avx2(const float *__restrict__ src, float *__restrict__ dst,
               size_t size) {
  if (!tables_initialized)
    logf_init_tables();

  const __m256i RED_CONST = _mm256_set1_epi32(REDUCE_OFFSET);
  const __m256i EXP_MASK = _mm256_set1_epi32(MASK_EXP_FULL);
  const __m256i IDX_MASK = _mm256_set1_epi32(MASK_MANT);
  const __m256 POLYv1 = _mm256_set1_ps(POLY_C1);
  const __m256 POLYv2 = _mm256_set1_ps(POLY_C2);
  const __m256 POLYv3 = _mm256_set1_ps(POLY_C3);
  const __m256 L2_HI = _mm256_set1_ps(LOG2_HI);
  const __m256 L2_LO = _mm256_set1_ps(LOG2_LO);
  const __m256 V_ZERO = _mm256_setzero_ps();
  const __m256 V_ABS_MASK = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
  const __m256 V_INF = _mm256_castsi256_ps(_mm256_set1_epi32(0x7F800000));

  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 vx = _mm256_loadu_ps(src + i);

    __m256 bad = _mm256_or_ps(
        _mm256_cmp_ps(vx, V_ZERO, _CMP_LE_OQ),
        _mm256_cmp_ps(_mm256_and_ps(vx, V_ABS_MASK), V_INF, _CMP_GE_OQ));

    __m256i ix = _mm256_castps_si256(vx);
    __m256i ix_norm = _mm256_sub_epi32(ix, RED_CONST);
    __m256 vn = _mm256_cvtepi32_ps(_mm256_srai_epi32(ix_norm, 23));

    __m256 vx_norm = _mm256_castsi256_ps(
        _mm256_sub_epi32(ix, _mm256_and_si256(ix_norm, EXP_MASK)));
    __m256i idx = _mm256_srli_epi32(_mm256_and_si256(ix_norm, IDX_MASK),
                                    TABLE_INDEX_SHIFT);

    __m256 ri = _mm256_i32gather_ps(R_TABLE, idx, 4);
    __m256 ti = _mm256_i32gather_ps(T_TABLE, idx, 4);

    __m256 r = _mm256_fmadd_ps(ri, vx_norm, _mm256_set1_ps(-1.0f));
    __m256 poly =
        _mm256_fmadd_ps(r, _mm256_fmadd_ps(r, POLYv3, POLYv2), POLYv1);

    __m256 res = _mm256_add_ps(_mm256_mul_ps(r, poly), ti);
    res = _mm256_fmadd_ps(vn, L2_HI, res);
    res = _mm256_fmadd_ps(vn, L2_LO, res);

    _mm256_storeu_ps(dst + i, res);

    int exc_mask = _mm256_movemask_ps(bad);
    if (exc_mask) {
      for (int j = 0; j < 8; ++j) {
        if (exc_mask & (1 << j))
          dst[i + j] = logf_scalar(src[i + j]);
      }
    }
  }

  for (; i < size; ++i)
    dst[i] = logf_scalar(src[i]);
}
