#include "perf_bench.hpp"

#include <random>

extern "C" {
float logf_scalar(float x);
void logf_avx2(const float *src, float *dst, int n);
}

int main() {
  constexpr size_t size = 5000;
  std::vector<float> data(size);
  std::mt19937 gen(42);
  std::generate(data.begin(), data.end(), gen);

  perf::print("std::exp", [](double x) { return std::exp(x); }, data);
  perf::print("std::log", [](double x) { return std::log(x); }, data);
  perf::print("std::sqrt", [](double x) { return std::sqrt(x); }, data);
  std::cout << "\n";
  perf::print("logf_scalar", logf_scalar, data);
  perf::print("logf_avx2", logf_avx2, data);
}
