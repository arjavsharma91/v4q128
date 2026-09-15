#pragma once

#include <immintrin.h>
#include <cstdint>

#if defined(_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define V4Q128_INLINE inline __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

    struct alignas(32) v4q128 {
        __m256i lo; // Fractional limbs (4 x uint64_t)
        __m256i hi; // Integer limbs    (4 x int64_t)

        // Uninitialized default constructor (zero overhead when overwritten immediately)
        v4q128() = default;

        V4Q128_INLINE v4q128(__m256i low_limbs, __m256i high_limbs) noexcept
            : lo(low_limbs), hi(high_limbs) {}

        /**
         * @brief Returns a zeroed vector.
         */
        [[nodiscard]] V4Q128_INLINE static v4q128 zero() noexcept {
            __m256i z = _mm256_setzero_si256();
            return v4q128(z, z);
        }

        /**
         * @brief Broadcasts a single Q64.64 value (frac, integer) across all 4 SIMD lanes.
         */
        [[nodiscard]] V4Q128_INLINE static v4q128 set1(uint64_t frac, int64_t integer) noexcept {
            return v4q128(
                _mm256_set1_epi64x(static_cast<long long>(frac)),
                _mm256_set1_epi64x(static_cast<long long>(integer))
            );
        }

        /**
         * @brief Sets 4 individual Q64.64 lane values directly into Dual-SoA registers.
         */
        [[nodiscard]] V4Q128_INLINE static v4q128 set(
            uint64_t f0, int64_t i0,
            uint64_t f1, int64_t i1,
            uint64_t f2, int64_t i2,
            uint64_t f3, int64_t i3) noexcept
        {
            return v4q128(
                _mm256_setr_epi64x(static_cast<long long>(f0), static_cast<long long>(f1),
                                  static_cast<long long>(f2), static_cast<long long>(f3)),
                _mm256_setr_epi64x(static_cast<long long>(i0), static_cast<long long>(i1),
                                  static_cast<long long>(i2), static_cast<long long>(i3))
            );
        }
    };

} // namespace v4q128

#undef V4Q128_INLINE
