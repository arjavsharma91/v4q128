#pragma once

#include <immintrin.h>
#include <cstdint>

namespace v4q128 {

struct alignas(32) v4q128 {
    __m256i lo;
    __m256i hi;

    v4q128() = default;

    constexpr v4q128(__m256i low_limbs, __m256i high_limbs) noexcept
        : lo(low_limbs), hi(high_limbs) {}

    [[nodiscard]] static inline v4q128 zero() noexcept {
        __m256i z = _mm256_setzero_si256();
        return v4q128(z, z);
    }
};

} // namespace v4q128
