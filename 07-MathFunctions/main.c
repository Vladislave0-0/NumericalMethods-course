#include <errno.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define BLU "\x1B[34m"
#define RST "\x1B[0m"

float get_ulp_diff(float actual, double expected) {
  if (isnan(actual) && isnan(expected))
    return 0.0f;
  if (isinf(actual) && isinf(expected))
    return 0.0f;

  float expected_f = (float)expected;
  typedef union {
    float f;
    int32_t i;
  } f32_pun;
  f32_pun a = {.f = actual};
  f32_pun e = {.f = expected_f};

  return (float)abs(a.i - e.i);
}

void test_edge_cases() {
  printf(BLU "Running Edge Case Tests:\n" RST);

  // Тест на ноль
  errno = 0;
  feclearexcept(FE_ALL_EXCEPT);
  float res_zero = logf(0.0f);
  printf("    logf(0.0):  res=%f, errno=%s, divbyzero=%s\n", res_zero,
         (errno == ERANGE ? GRN "[OK]" RST : RED "[FAIL]" RST),
         (fetestexcept(FE_DIVBYZERO) ? GRN "[OK]" RST : RED "[FAIL]" RST));

  // Тест на отрицательное число
  errno = 0;
  feclearexcept(FE_ALL_EXCEPT);
  float res_neg = logf(-1.0f);
  printf("    logf(-1.0): res=%f, errno=%s, invalid=%s\n", res_neg,
         (errno == EDOM ? GRN "[OK]" RST : RED "[FAIL]" RST),
         (fetestexcept(FE_INVALID) ? GRN "[OK]" RST : RED "[FAIL]" RST));

  // Тест на бесконечность
  float res_inf = logf(INFINITY);
  printf("    logf(+inf): res=%f      %s\n", res_inf,
         (isinf(res_inf) ? GRN "[OK]" RST : RED "[FAIL]" RST));

  // Тест на единицу
  float res_one = logf(1.0f);
  printf("    logf(1.0):  res=%f %s\n", res_one,
         (res_one == 0.0f ? GRN "[OK]" RST : RED "[FAIL]" RST));
}

void test_precision() {
  printf(BLU "\nRunning Precision Tests:\n" RST);
  float test_values[] = {
      0.5f,    0.707f, 1.414f, 2.0f, 10.0f,
      FLT_MIN,         // Минимальное нормализованное
      1e-40f,  1e-42f, // Субнормальные числа
      FLT_MAX          // Максимальное значение
  };

  for (int i = 0; i < sizeof(test_values) / sizeof(float); i++) {
    float x = test_values[i];
    float actual = logf(x);
    double expected = log((double)x);
    float ulp = get_ulp_diff(actual, expected);

    int pass = (ulp <= 3.5f);

    printf(
        "    x = %10.4e | logf = %12.7f | ref = %12.7f | ULP diff: %3.1f %s\n",
        x, actual, (float)expected, ulp,
        (pass ? GRN "[PASS]" RST : RED "[FAIL]" RST));
  }
}

int main() {
  test_edge_cases();
  test_precision();
}
