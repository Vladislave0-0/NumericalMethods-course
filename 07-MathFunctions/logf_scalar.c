#include <errno.h>
#include <fenv.h>
#include <stdint.h>

typedef union {
  float f;
  uint32_t i;
} float_conv;

float logf_scalar(float x) {
  // Константы из fdlibm
  static const float ln2_hi = 6.9313812256e-01f; /* 0x3f317180 */
  static const float ln2_lo = 9.0580006145e-06f; /* 0x3717f7d1 */

  static const float Lg1 = 6.6666668653e-01f; /* 0x3F2AAAAB */
  static const float Lg2 = 4.0000000596e-01f; /* 0x3ECCCCCD */
  static const float Lg3 = 2.8571429849e-01f; /* 0x3E924925 */
  static const float Lg4 = 2.2222198546e-01f; /* 0x3E638E29 */
  static const float Lg5 = 1.8183572590e-01f; /* 0x3E3A3325 */
  static const float Lg6 = 1.5313838422e-01f; /* 0x3E1CD04F */
  static const float Lg7 = 1.4798198640e-01f; /* 0x3E178897 */

  float_conv conv;
  conv.f = x;
  uint32_t ix = conv.i;
  uint32_t abs_ix = ix & 0x7fffffff;

  // 1. Фильтрация
  if (abs_ix == 0) { // Если x = +0.0 или x = -0.0
    errno = ERANGE;
    feraiseexcept(FE_DIVBYZERO);
    return -1.0f / 0.0f;
  }

  if (ix >= 0x80000000) { // Если x <= 0
    errno = EDOM;
    feraiseexcept(FE_INVALID);
    return (x - x) / 0.0f; // NaN
  }

  // Если x = +inf или NaN
  if (ix >= 0x7f800000)
    return x;

  // Обработка субнормальных чисел
  int subnormal_bits = 0;
  if (ix < 0x00800000) {
    x *= 3.355443200e+07f; // умножаем на 2^25 для нормализации
    conv.f = x;
    ix = conv.i;
    subnormal_bits -= 25;
  }

  // 2. Редукция
  subnormal_bits += (int)(ix >> 23) - 127;
  // Делаем экспоненту нулевой (число в [1, 2))
  ix = (ix & 0x007fffff) | 0x3f800000;
  conv.i = ix;
  float m = conv.f;

  // Сдвигаем мантиссу в диапазон [sqrt(2)/2, sqrt(2)]
  if (m > 1.41421356f) {
    m *= 0.5f;
    ++subnormal_bits;
  }
  float f = m - 1.0f;

  // 3. Аппроксимация
  double df = (double)f;
  double s = df / (2.0 + df);
  double z = s * s;
  double R =
      z * (Lg1 +
           z * (Lg2 + z * (Lg3 + z * (Lg4 + z * (Lg5 + z * (Lg6 + z * Lg7))))));

  // 4. Реконструкция
  double hfsq = 0.5 * df * df;
  double res =
      (double)subnormal_bits * ln2_hi +
      ((double)subnormal_bits * ln2_lo + (s * (hfsq + R) + (df - hfsq)));

  return (float)res;
}
