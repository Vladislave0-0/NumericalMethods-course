#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <random>

double run_monte_carlo(long long points_num) {
  long long hits = 0;

#pragma omp parallel
  {
    std::mt19937 gen(std::random_device{}() ^ omp_get_thread_num());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    long long local_hits = 0;

#pragma omp for
    for (long long i = 0; i < points_num; ++i) {
      double x = dist(gen);
      double y = dist(gen);
      
      if (x * x + y * y <= 1.0)
        ++local_hits;
    }

#pragma omp atomic
    hits += local_hits;
  }

  return 4.0 * hits / points_num;
}

void benchmark_errors(const double &PI_REF) {
  std::ofstream out("benchmark_errors.csv");
  out << "N,AbsoluteError,StandardError\n";

  for (int p = 1; p <= 8; ++p) {
    long long N = std::pow(10, p);
    if (N > 100000000)
      break;

    double pi_est = run_monte_carlo(N);
    double abs_error = std::abs(pi_est - PI_REF);
    double std_error = std::sqrt(PI_REF * (4.0 - PI_REF) / N);

    out << N << "," << abs_error << "," << std_error << "\n";
  }

  out.close();
  std::cout << "Результаты бенчмаркинга для N от 10 до 10^8 сохранены в файл "
               "benchmark_errors.csv" << std::endl;
}

int main() {
  const long long N = 100'000'000;
  const double PI_REF = 3.14159265358979323846;

  auto start = std::chrono::high_resolution_clock::now();
  double pi_est = run_monte_carlo(N);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;

  double SE = std::sqrt(pi_est * (4.0 - pi_est) / N);
  double AE = std::abs(pi_est - PI_REF);

  std::cout << std::fixed << std::setprecision(10);
  std::cout << "Вычисленное pi: " << pi_est << "\n";
  std::cout << "Эталонное pi:   " << PI_REF << "\n\n";
  std::cout << "Стандартная ошибка: " << SE << "\n";
  std::cout << "Абсолютная ошибка:  " << AE << "\n\n";
  std::cout << "Время выполнения: " << elapsed.count() << " c" << "\n\n";

  benchmark_errors(PI_REF);
}
