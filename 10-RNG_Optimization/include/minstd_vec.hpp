#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>


class VecLCG_FP32 {
  size_t width_;
  std::vector<uint32_t> state_;
  uint64_t ak_; // A^k mod M

  // Быстрое вычисление (a*x) mod (2^31-1)
  static uint32_t fast_mod(uint64_t y) {
    uint32_t z = static_cast<uint32_t>((y >> 31) + (y & M_HEX));
    return (z >= M) ? (z - M) : z;
  }

  // Модульная экспонента
  static uint64_t exp_mod(uint64_t base, size_t exp, uint64_t M) {
    uint64_t result = 1;
    base %= M;

    while (exp > 0) {
      if (exp & 1)
        result = fast_mod(result * base);

      base = fast_mod(base * base);
      exp >>= 1;
    }

    return result;
  }

public:
  static constexpr uint64_t A = 48271;         // A
  static constexpr uint64_t M = 2147483647ULL; // M
  static constexpr uint32_t M_HEX = 0x7FFFFFFF;

  explicit VecLCG_FP32(size_t vec_width)
      : width_(vec_width), state_(vec_width), ak_(1) {
    assert(__builtin_popcountll(vec_width) == 1 &&
           "Vector width must be power of 2");
  }

  // Инициализируем state_[i] значением x_{i+1}
  void init(uint32_t seed) {
    uint64_t cur = seed;

    for (size_t i = 0; i < width_; ++i) {
      cur = fast_mod(A * cur); // cur = x_{i+1}
      state_[i] = static_cast<uint32_t>(cur);
    }

    ak_ = exp_mod(A, width_, M);
  }

  // Инициализацируем с позиции offset
  void init_with_offset(uint32_t seed, size_t offset) {
    uint64_t a_offset = exp_mod(A, offset, M);
    uint64_t cur = fast_mod(static_cast<uint64_t>(seed) * a_offset);

    for (size_t i = 0; i < width_; ++i) {
      cur = fast_mod(A * cur);
      state_[i] = static_cast<uint32_t>(cur);
    }

    ak_ = exp_mod(A, width_, M);
  }

  // Производим нормирующее отображение из [0; M) в [0; 1)
  void fill_float(float *__restrict__ out) {
    for (size_t i = 0; i < width_; ++i)
      out[i] = static_cast<float>(state_[i]) * (1.0f / static_cast<float>(M));

    // Продвигаем состояние на width шагов вперёд
    for (size_t i = 0; i < width_; ++i) {
      uint64_t prod = static_cast<uint64_t>(ak_) * state_[i];
      state_[i] = fast_mod(prod);
    }
  }

  // Генерация блока uint32_t для побитовой проверки
  void fill_uint32(uint32_t *__restrict__ out) {
    for (size_t i = 0; i < width_; ++i)
      out[i] = state_[i];

    // Продвигаем состояние на width шагов вперёд
    for (size_t i = 0; i < width_; ++i) {
      uint64_t prod = static_cast<uint64_t>(ak_) * state_[i];
      state_[i] = fast_mod(prod);
    }
  }
};
