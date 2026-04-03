#include <iomanip>
#include <iostream>
#include <random>

const int NUMBERS = 1000;
#define RESET "\033[0m"
#define CYAN "\033[36m"
#define MAGENTA "\033[35m"

template <typename T> double relative_error__(T measured, double actual) {
  if (actual == 0)
    return 0.0;
  return std::abs(static_cast<double>(measured) - actual) / std::abs(actual) *
         100.0;
}

template <typename T> T sum__(const std::vector<T> &data) {
  T sum = 0;
  for (const auto &x : data)
    sum += x;

  return sum;
}

template <typename T> T fast_variance(const std::vector<T> &data) {
  size_t n = data.size();
  if (n < 1)
    return 0;

  T sum_x = sum__(data);
  T sum_x_sqr = 0;
  for (const auto &x : data)
    sum_x_sqr += x * x;

  return (sum_x_sqr - (sum_x * sum_x) / static_cast<T>(n)) / static_cast<T>(n);
}

template <typename T> T two_pass_variance(const std::vector<T> &data) {
  size_t n = data.size();
  if (n < 1)
    return 0;

  T mean = sum__(data) / static_cast<T>(n);

  T sum_diff_sqr = 0;
  for (const auto &x : data)
    sum_diff_sqr += (x - mean) * (x - mean);

  return sum_diff_sqr / static_cast<T>(n);
}

template <typename T> T one_pass_variance(const std::vector<T> &data) {
  size_t n = 0;
  T mean = 0;
  T var = 0;

  for (const auto &x : data) {
    n++;
    if (n == 1) {
      mean = x;
      var = 0;
      continue;
    }

    T old_mean = mean;
    mean = old_mean + (x - old_mean) / static_cast<T>(n);

    T term1 = (x - old_mean) * (x - mean);
    var = var + (term1 - var) / static_cast<T>(n);
  }

  return var;
}

void run_test(const std::pair<int, float> &&sample) {
  auto [mu, sigma] = sample;
  std::cout << CYAN << "Тест: mu = " << mu
            << ", sigma = " << std::setprecision(2) << sigma
            << ", n = " << NUMBERS << ";" << MAGENTA
            << " D[X]_теор = " << std::setprecision(4) << sigma * sigma << RESET
            << "\n";

  double theorVar = sigma * sigma;

  std::mt19937 gen(42);
  std::normal_distribution<double> dist(mu, sigma);

  std::vector<float> data32;
  std::vector<double> data64;
  for (int i = 0; i < NUMBERS; ++i) {
    double val = dist(gen);
    data32.push_back(static_cast<float>(val));
    data64.push_back(val);
  }

  auto print_row = [&](std::string type, std::string method, double res,
                       double ref) {
    std::cout << std::left << std::setw(10) << type << " | " << std::setw(15)
              << method << " | " << std::fixed << std::setprecision(9)
              << std::setw(9) << res << (res < 0 ? " (!)" : "     ") << " | "
              << std::setprecision(6) << relative_error__(res, ref)
              << std::endl;
  };

  std::cout << std::left << std::setw(10) << "    Тип" << "    | "
            << std::setw(15) << "    Метод" << "    | " << std::setw(14)
            << "       D[X]" << "   | "
            << "Отн. ошибка (%)" << std::endl;
  std::cout << std::string(62, '-') << std::endl;

  // float32
  print_row("float32", "Быстрый      ", fast_variance(data32), theorVar);
  print_row("float32", "Двухпроходной", two_pass_variance(data32), theorVar);
  print_row("float32", "Однопроходной", one_pass_variance(data32), theorVar);

  std::cout << std::string(62, '-') << std::endl;

  // float64
  print_row("float64", "Быстрый      ", fast_variance(data64), theorVar);
  print_row("float64", "Двухпроходной", two_pass_variance(data64), theorVar);
  print_row("float64", "Однопроходной", one_pass_variance(data64), theorVar);

  std::cout << "\n\n";
}

int main() {
  std::vector<std::pair<int, float>> samples{{1, 1}, {10, 0.1}, {100, 0.01}};

  for (const auto &sample : samples)
    run_test(std::move(sample));

  // Тест, в котором наблюдается отрицательное значение дисперсии. Здесь среднее
  // большое, а дисперсия маленькая
  // run_test({10000, 0.01f});
}
