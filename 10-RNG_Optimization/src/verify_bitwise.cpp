#include "minstd_vec.hpp"

#include <iostream>
#include <vector>

// Один шаг minstd_rand
static uint32_t scalar_step(uint32_t x) {
  constexpr uint64_t A = 48271;
  constexpr uint64_t M = 2147483647ULL;
  constexpr uint32_t M_HEX = 0x7FFFFFFF;

  uint64_t y = A * x;
  uint32_t z = static_cast<uint32_t>((y >> 31) + (y & M_HEX));

  return (z >= M) ? (z - M) : z;
}

int main() {
  constexpr size_t TOTAL_POINTS = 1'000'000; // 100 млн 2D-векторов
  constexpr size_t BLOCK_SIZE = 256;         // Степень двойки
  constexpr uint32_t SEED = 987654321;

  VecLCG_FP32 vec_rng(BLOCK_SIZE);
  vec_rng.init(SEED);

  std::vector<uint32_t> buffer(BLOCK_SIZE);
  uint32_t scalar_state = SEED;
  size_t processed = 0;
  bool ok = true;

  while (processed < TOTAL_POINTS && ok) {
    vec_rng.fill_uint32(buffer.data());

    for (size_t i = 0; i < BLOCK_SIZE && processed < TOTAL_POINTS;
         ++i, ++processed) {
      scalar_state = scalar_step(scalar_state);

      if (buffer[i] != scalar_state) {
        std::cerr << "Mismatch at global index " << processed
                  << ": vec=" << buffer[i] << " scalar=" << scalar_state
                  << "\n";
        ok = false;

        break;
      }
    }
  }

  if (ok) {
    std::cout << "Bitwise correctness verified for " << TOTAL_POINTS
              << " values\n";
    return 0;
  }

  std::cerr << "Verification failed\n";
  return 1;
}
