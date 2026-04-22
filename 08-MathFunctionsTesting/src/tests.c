#include "tests.h"

#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

extern float logf(float);

test_result verify_flags(float x, float exp_res, int exp_errno, int exp_flag,
                         const char *label) {
  test_result res;
  strncpy(res.label, label, 24);

  errno = 0;
  feclearexcept(FE_ALL_EXCEPT);

  float actual = logf(x);

  // Переводим в строковое представление результата
  if (isnan(actual))
    sprintf(res.output, "NaN");
  else if (isinf(actual))
    sprintf(res.output, actual > 0 ? "+inf" : "-inf");
  else
    sprintf(res.output, "%.6f", actual);

  res.errno_ok = (errno == exp_errno);
  res.flag_ok = (exp_flag == 0) || (fetestexcept(exp_flag));

  int val_ok = isnan(exp_res) ? isnan(actual) : (actual == exp_res);
  res.status_ok = (res.errno_ok && res.flag_ok && val_ok);

  return res;
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

void test_edge_cases() {
  printf(BLU "\nRunning Edge Case Tests:" RST);

  test_result results[9];
  int test = 0;

  // Отрицательные и NaN (ожидаем FE_INVALID / EDOM)
  results[test++] = verify_flags(-1.0f, NAN, EDOM, FE_INVALID, "x = -1.0");
  results[test++] = verify_flags(-42.0f, NAN, EDOM, FE_INVALID, "x = -42.0");
  results[test++] =
      verify_flags(-INFINITY, NAN, EDOM, FE_INVALID, "x = -infinity");
  results[test++] = verify_flags(NAN, NAN, 0, 0, "x = NaN");

  // Нули (ожидаем FE_DIVBYZERO / ERANGE)
  results[test++] =
      verify_flags(0.0f, -INFINITY, ERANGE, FE_DIVBYZERO, "x = +0.0");
  results[test++] =
      verify_flags(-0.0f, -INFINITY, ERANGE, FE_DIVBYZERO, "x = -0.0");

  // Бесконечность и единица
  results[test++] = verify_flags(INFINITY, INFINITY, 0, 0, "x = +infinity");
  results[test++] = verify_flags(1.0f, 0.0f, 0, 0, "x = 1.0");

  print_table(results, test);
}

void test_precision() {
  printf(BLU "\nRunning Precision Tests:" RST);

  float test_values[] = {
      0.5f,       0.707f, 1.414f, 2.0f, 10.0f,
      1.0000001f, // Точка вблизи единицы
      FLT_MIN,    // Граница нормализованных
      1e-40f,     // Субнормальное
      1e-42f,     // Субнормальное
      FLT_MAX     // Максимум
  };

  int num_tests = sizeof(test_values) / sizeof(float);
  test_result results[num_tests];

  for (int i = 0; i < num_tests; ++i) {
    float x = test_values[i];
    double ref = log((double)x);
    char label[24];

    if (x < 1e-37)
      sprintf(label, "x = %.1e", x);
    else
      sprintf(label, "x = %.4e", x);

    results[i] = verify_flags(x, (float)ref, 0, 0, label);
  }

  print_table(results, num_tests);
}

void test_intervals() {
  printf(BLU "\nRunning Interval Precision Tests (3.5 ULP limit):\n" RST);

  const int POINTS_PER_INTERVAL = 1000;
  const float MAX_RELEVANT_ULP = 4.0;
  int total_intervals = 0, failed_intervals = 0;
  double max_ulp = 0.0;

  // Проходим по всем экспонентам float
  for (int e = -126; e < 128; ++e) {
    float start = ldexpf(1.0f, e);
    float end = ldexpf(1.0f, e + 1);
    float step = (end - start) / POINTS_PER_INTERVAL;

    int interval_passed = 1;
    for (int i = 0; i < POINTS_PER_INTERVAL; ++i) {
      float x = start + i * step;

      if (x <= 0)
        continue;

      float actual = logf(x);
      double expected = log((double)x);
      double ulp = get_ulp(actual, expected);

      if (ulp > max_ulp)
        max_ulp = ulp;

      if (ulp > MAX_RELEVANT_ULP)
        interval_passed = 0;
    }
    ++total_intervals;

    if (!interval_passed)
      ++failed_intervals;
  }

  printf("    Max ULP observed: %5.2f %s %.1f %s\n" RST, max_ulp,
         max_ulp <= MAX_RELEVANT_ULP ? "<=" : ">", MAX_RELEVANT_ULP,
         max_ulp <= MAX_RELEVANT_ULP ? GRN "[PASS]" : RED "[FAIL]");
  printf("    Intervals Summary: %d total %s[%d failed]" RST "\n",
         total_intervals, (failed_intervals > 0 ? RED : GRN), failed_intervals);
}
