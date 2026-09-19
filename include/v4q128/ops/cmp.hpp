#pragma once

#include <immintrin.h>
#include <v4q128/core/storage.hpp>

#if defined(_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined (__GNUC__) || defined(__clang__)
    #define V4Q128_INLINE inline __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

namespace detail {
    [[nodiscard]] V4Q128_INLINE static __m256i sign_bit_mask() noexcept {
        return _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL));
    }
}

[[nodiscard]] V4Q128_INLINE __m256i equal(v4q128 a, v4q128 b) noexcept {
    __m256i hi_match = _mm256_cmpeq_epi64(a.hi, b.hi);
    __m256i lo_match = _mm256_cmpeq_epi64(a.lo, b.lo);

    return _mm256_and_si256(hi_match, lo_match);
}

[[nodiscard]] V4Q128_INLINE __m256i greater_than(v4q128 a, v4q128 b) noexcept {
    __m256i hi_gt = _mm256_cmpgt_epi64(a.hi, b.hi);
    __m256i hi_eq = _mm256_cmpeq_epi64(a.hi, b.hi);

    __m256i msb_mask = detail::sign_bit_mask();
    __m256i a_lo_masked = _mm256_xor_si256(a.lo, msb_mask);
    __m256i b_lo_masked = _mm256_xor_si256(b.lo, msb_mask);

    __m256i lo_gt = _mm256_cmpgt_epi64(a_lo_masked, b_lo_masked);

    return _mm256_or_si256(hi_gt, _mm256_and_si256(hi_eq, lo_gt));
}

[[nodiscard]] V4Q128_INLINE __m256i less_than(v4q128 a, v4q128 b) noexcept {
    return greater_than(b, a);
}

[[nodiscard]] V4Q128_INLINE __m256i less_than_or_equal(v4q128 a, v4q128 b) noexcept {
    return _mm256_andnot_si256(greater_than(a, b), _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL));
}

[[nodiscard]] V4Q128_INLINE __m256i not_equal_to(v4q128 a, v4q128 b) noexcept {
    return _mm256_andnot_si256(equal(a, b), _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL));
}

[[nodiscard]] V4Q128_INLINE __m256i greater_than_or_equal(v4q128 a, v4q128 b) noexcept {
    return _mm256_andnot_si256(less_than(a, b), _mm256_set1_epi64x(0xFFFFFFFFFFFFFFFFULL));
}

[[nodiscard]] V4Q128_INLINE __m256i operator>(v4q128 a, v4q128 b) noexcept {
    return greater_than(a, b);
}

[[nodiscard]] V4Q128_INLINE __m256i operator<(v4q128 a, v4q128 b) noexcept {
    return less_than(a, b);
}

[[nodiscard]] V4Q128_INLINE __m256i operator>=(v4q128 a, v4q128 b) noexcept {
    return greater_than_or_equal(a, b);
}

[[nodiscard]] V4Q128_INLINE __m256i operator<=(v4q128 a, v4q128 b) noexcept {
    return less_than_or_equal(a, b);
}

[[nodiscard]] V4Q128_INLINE __m256i operator==(v4q128 a, v4q128 b) noexcept {
    return equal(a, b);
}

} // namespace

#undef V4Q128_INLINE
