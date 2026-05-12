#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>
#include <sleef.h>

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"
#define BOLD "\033[1m"

struct OptionParams {
  float S0;    // Текущая цена
  float K;     // Цена исполнения
  float T;     // Время (в годах)
  float r;     // Безрисковая ставка
  float sigma; // Волатильность
};

extern "C" void logf_avx2(const float *src, float *dst, size_t size);

double normalCDF(double x) { return std::erfc(-x / std::sqrt(2.0)) / 2.0; }

double blackScholesAnalytic(OptionParams p) {
  if (p.T <= 0)
    return std::max(0.0f, p.S0 - p.K);
  double d1 = (std::log(p.S0 / p.K) + (p.r + 0.5 * p.sigma * p.sigma) * p.T) /
              (p.sigma * std::sqrt(p.T));
  double d2 = d1 - p.sigma * std::sqrt(p.T);
  return p.S0 * normalCDF(d1) - p.K * std::exp(-p.r * p.T) * normalCDF(d2);
}

struct VectorXorshift {
  __m256i s0, s1;

  VectorXorshift(uint64_t seed, int tid) {
    // Начальное состояние
    s0 = _mm256_set_epi64x(seed + tid + 3, seed + tid + 2, seed + tid + 1,
                           seed + tid);
    s1 = _mm256_set_epi64x(0xDA34, 0xBF12, 0x7856, 0x55AA);
  }

  __m256 next_float() {
    __m256i x = s0;
    __m256i y = s1;

    s0 = y;
    // x ^= x << 23
    x = _mm256_xor_si256(x, _mm256_slli_epi64(x, 23));

    // s1 = x ^ y ^ (x >> 17) ^ (y >> 26)
    __m256i t1 = _mm256_xor_si256(x, y);
    __m256i t2 = _mm256_srli_epi64(x, 17);
    __m256i t3 = _mm256_srli_epi64(y, 26);
    s1 = _mm256_xor_si256(t1, _mm256_xor_si256(t2, t3));

    // s1 + y
    __m256i res_int = _mm256_add_epi64(s1, y);

    // Преобразование в float [0, 1)
    const __m256i mask = _mm256_set1_epi32(0x007FFFFF);
    const __m256i exp = _mm256_set1_epi32(0x3F800000); // Экспонента для 1.0f

    __m256i f_bits = _mm256_or_si256(_mm256_and_si256(res_int, mask), exp);
    __m256 f = _mm256_castsi256_ps(f_bits);

    // f в диапазоне [1.0, 2.0). Вычитаем 1.0, чтобы получить [0, 1)
    return _mm256_sub_ps(f, _mm256_set1_ps(1.0f));
  }
};

void calculateSingleOption(const OptionParams &opt, uint64_t numPaths,
                           double &callPrice) {
  // Размер блока подобран под L1 (32KB)
  const uint64_t BLOCK_SIZE = 1024;
  double totalPayoff = 0.0;

  float drift = (opt.r - 0.5f * opt.sigma * opt.sigma) * opt.T;
  float volSqrtT = opt.sigma * std::sqrt(opt.T);

  __m256 v_drift = _mm256_set1_ps(drift);
  __m256 v_vol = _mm256_set1_ps(volSqrtT);
  __m256 v_S0 = _mm256_set1_ps(opt.S0);
  __m256 v_K = _mm256_set1_ps(opt.K);
  __m256 v_zero = _mm256_setzero_ps();
  __m256 v_half = _mm256_set1_ps(0.5f);

// Внутренний параллелизм по траекториям
#pragma omp parallel reduction(+ : totalPayoff)
  {
    int tid = omp_get_thread_num();
    // Смещения сида для каждого потока (как в skip-ahead)
    VectorXorshift rng(123456789ULL, tid);

    alignas(32) float u1_buf[8];
    alignas(32) float log_u1_buf[8];

#pragma omp for
    for (uint64_t p = 0; p < numPaths; p += BLOCK_SIZE) {
      for (uint64_t i = 0; i < BLOCK_SIZE; i += 8) {
        // Генерируем равномерно распределенные числа
        __m256 u1 = rng.next_float();
        __m256 u2 = rng.next_float();

        // Защита от логарифма нуля
        u1 = _mm256_max_ps(u1, _mm256_set1_ps(1e-12f));

        _mm256_store_ps(u1_buf, u1);
        logf_avx2(u1_buf, log_u1_buf, 8);
        __m256 v_log_u1 = _mm256_load_ps(log_u1_buf);

        // Трансформация Бокса-Мюллера
        __m256 r =
            _mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2.0f), v_log_u1));
        __m256 theta = _mm256_mul_ps(_mm256_set1_ps(2.0f * 3.1415926535f), u2);
        __m256 z = _mm256_mul_ps(r, Sleef_cosf8_u10(theta));

        // Расчет двух зеркальных траекторий
        __m256 ST_a = _mm256_mul_ps(
            v_S0, Sleef_expf8_u10(_mm256_fmadd_ps(v_vol, z, v_drift)));
        __m256 ST_b = _mm256_mul_ps(
            v_S0, Sleef_expf8_u10(_mm256_fnmadd_ps(v_vol, z, v_drift)));

        __m256 payoff_a = _mm256_max_ps(_mm256_sub_ps(ST_a, v_K), v_zero);
        __m256 payoff_b = _mm256_max_ps(_mm256_sub_ps(ST_b, v_K), v_zero);
        __m256 avg_payoff =
            _mm256_mul_ps(_mm256_add_ps(payoff_a, payoff_b), v_half);

        // Накапливаем сумму в double
        __m128 lo_f = _mm256_extractf128_ps(avg_payoff, 0);
        __m128 hi_f = _mm256_extractf128_ps(avg_payoff, 1);

        __m256d d_lo = _mm256_cvtps_pd(lo_f);
        __m256d d_hi = _mm256_cvtps_pd(hi_f);

        alignas(32) double res_block[8];
        _mm256_store_pd(&res_block[0], d_lo);
        _mm256_store_pd(&res_block[4], d_hi);

        for (int k = 0; k < 8; ++k)
          totalPayoff += res_block[k];
      }
    }
  }

  // Дисконтирование среднего значения к текущему моменту времени
  double discount = std::exp(-opt.r * opt.T);
  callPrice = (totalPayoff / numPaths) * discount;
}

int main() {
  const int N_OPTIONS = 100;
  const uint64_t N_PATHS = 100000;

  std::vector<OptionParams> portfolio(N_OPTIONS);
  std::vector<double> results(N_OPTIONS);

  // Генерируем случайный портфель в рамках разумного
  std::mt19937 gen(123);
  std::uniform_real_distribution<float> distS0(90.0f, 110.0f);
  std::uniform_real_distribution<float> distK(90.0f, 110.0f);
  std::uniform_real_distribution<float> distT(0.2f, 2.0f);
  std::uniform_real_distribution<float> distSigma(0.1f, 0.4f);
  std::uniform_real_distribution<float> distR(0.01f, 0.05f);

  for (int i = 0; i < N_OPTIONS; ++i) {
    portfolio[i] = {distS0(gen), distK(gen), distT(gen), distR(gen),
                    distSigma(gen)};
  }

  double start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < N_OPTIONS; ++i)
    calculateSingleOption(portfolio[i], N_PATHS, results[i]);

  double end = omp_get_wtime();

  std::cout << std::fixed << std::setprecision(4);
  std::cout << BOLD << std::string(82, '=') << RESET << std::endl;
  std::cout << std::left << std::setw(5) << "ID" << std::setw(10) << "S0"
            << std::setw(10) << "K" << std::setw(8) << "T" << std::setw(8)
            << "Vol" << std::setw(15) << "Price" << std::setw(12) << "Analytic"
            << "Error / Status" << std::endl;
  std::cout << BOLD << std::string(82, '=') << RESET << std::endl;

  double maxErr = 0, avgErr = 0;

  for (int i = 0; i < N_OPTIONS; ++i) {
    double analytic = blackScholesAnalytic(portfolio[i]);
    double mcPrice = results[i];
    double err = std::abs(mcPrice - analytic);

    avgErr += err;
    if (err > maxErr)
      maxErr = err;

    std::cout << std::left << std::setw(5) << i + 1;
    std::cout << CYAN << std::setw(10) << portfolio[i].S0 << std::setw(10)
              << portfolio[i].K << std::setw(8) << portfolio[i].T
              << std::setw(8) << portfolio[i].sigma << RESET;

    // Выделение дешевых опционов
    if (mcPrice < 2.0)
      std::cout << MAGENTA << std::setw(15) << mcPrice << RESET;
    else
      std::cout << std::setw(15) << mcPrice;

    std::cout << std::setw(15) << analytic;

    if (err < 0.05)
      std::cout << GREEN << "[OK] " << err << RESET;
    else if (err < 0.7)
      std::cout << YELLOW << "[?]  " << err << RESET;
    else
      std::cout << RED << "[!]  " << err << RESET;

    std::cout << std::endl;
  }

  std::cout << "\n";
  std::cout << BOLD << std::string(82, '=') << RESET << std::endl;
  std::cout << "\n";
  std::cout << BOLD << "CALCULATION RESULTS:" << RESET << std::endl;
  std::cout << "Total time         : " << YELLOW << (end - start) * 1000.0
            << " ms" << RESET << std::endl;
  std::cout << "Average per option : " << (end - start) * 1000.0 / N_OPTIONS
            << " ms" << std::endl;
  std::cout << "Max. error         : " << (maxErr > 0.1 ? RED : GREEN) << maxErr
            << RESET << std::endl;
  std::cout << "Average error      : " << avgErr / N_OPTIONS << std::endl;
}
