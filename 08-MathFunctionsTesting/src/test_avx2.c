#include "tests.h"

#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void logf_avx2(const float *src, float *dst, size_t n);

float logf_vector_wrapper(float x) {
  // Выравнивание по 32 байта для AVX
  alignas(32) float src[8];
  alignas(32) float dst[8];

  for (int i = 0; i < 8; i++)
    src[i] = x;

  logf_avx2(src, dst, 8);
  return dst[0];
}

double get_ulp(float actual, double expected) {
  if (isnan(actual) && isnan(expected))
    return 0.0;

  if (isinf(actual) && isinf(expected))
    return 0.0;

  if (expected == 0.0)
    return (double)actual == 0.0 ? 0.0 : INFINITY;

  // Находим экспоненту эталонного значения A
  int expected_exp = 0;
  frexp(expected, &expected_exp);

  // Отнимаем единицу, потому что frexp дает мантиссу в [0.5, 1). Поэтому
  // реальная экспонента на 1 меньше
  double a_prime = ldexp(1.0, expected_exp - 1 - 23);

  return fabs((double)actual - expected) / a_prime;
}

test_result verify_vector(float x, float exp_res, const char *label) {
  test_result res;
  memset(res.label, 0, sizeof(res.label));
  strncpy(res.label, label, sizeof(res.label) - 1);

  float actual = logf_vector_wrapper(x);

  if (isnan(actual))
    sprintf(res.output, "NaN");
  else if (isinf(actual))
    sprintf(res.output, actual > 0 ? "+inf" : "-inf");
  else
    sprintf(res.output, "%.6f", actual);

  // Не меняют errno/flags ради скорости. Их можно в частном порядке проверить
  res.errno_ok = 1;
  res.flag_ok = 1;

  // Проверяем на примерное совпадение
  int val_ok = isnan(exp_res)
                   ? isnan(actual)
                   : (fabsf(actual - exp_res) < 1e-5f || actual == exp_res);
  res.status_ok = val_ok;

  return res;
}

void test_vector_precision() {
  printf(BLU "\nRunning Vector Precision Tests:" RST);

  float test_values[] = {0.5f,  0.707f, 1.414f,          2.0f,
                         10.0f, 1.0f,   1.17549435e-38f, FLT_MAX};

  int num_tests = sizeof(test_values) / sizeof(float);
  test_result results[num_tests];

  for (int i = 0; i < num_tests; ++i) {
    float x = test_values[i];
    double ref = log((double)x);
    char label[24];
    sprintf(label, "x = %.4e", x);
    results[i] = verify_vector(x, (float)ref, label);

    double ulp = get_ulp(logf_vector_wrapper(x), ref);
    if (ulp > 3.5) {
      printf(RED "\n[DEBUG] x: %e, actual: %f, ref: %f, ULP: %.2f" RST, x,
             logf_vector_wrapper(x), ref, ulp);
    }
  }

  print_table(results, num_tests);
}

void test_vector_intervals() {
  printf(BLU "\nRunning Vector Interval Tests (3.5 ULP limit):\n" RST);

  const int POINTS_PER_INTERVAL = 10000;
  const float MAX_RELEVANT_ULP = 3.5;
  int total_intervals = 0, failed_intervals = 0;
  double max_ulp = 0.0;

  for (int e = -126; e < 127; ++e) {
    float start = ldexpf(1.0f, e);
    float end = ldexpf(1.0f, e + 1);
    float step = (end - start) / POINTS_PER_INTERVAL;

    int interval_passed = 1;
    for (int i = 0; i < POINTS_PER_INTERVAL; ++i) {
      float x = start + i * step;
      float actual = logf_vector_wrapper(x);
      double expected = log((double)x);
      double ulp = get_ulp(actual, expected);

      if (ulp > max_ulp)
        max_ulp = ulp;
      if (ulp > MAX_RELEVANT_ULP)
        interval_passed = 0;
    }

    total_intervals++;
    if (!interval_passed)
      failed_intervals++;
  }

  printf("    Max ULP observed: %5.2f %s %.1f %s\n" RST, max_ulp,
         max_ulp <= MAX_RELEVANT_ULP ? "<=" : ">", MAX_RELEVANT_ULP,
         max_ulp <= MAX_RELEVANT_ULP ? GRN "[PASS]" : RED "[FAIL]");
  printf("    Intervals Summary: %d total %s[%d failed]" RST "\n",
         total_intervals, (failed_intervals > 0 ? RED : GRN), failed_intervals);
}

int main() {
  test_vector_precision();
  test_vector_intervals();
}
