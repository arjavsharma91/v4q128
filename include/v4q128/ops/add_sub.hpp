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

[[nodiscard]] V4Q128_INLINE v4q128 add(v4q128 a, v4q128 b) noexcept {
    __m256i lo_sum = _mm256_add_epi64(a.lo, b.lo);

    __m256i msb_mask = detail::sign_bit_mask();
    __m256i a_lo_flipped = _mm256_xor_si256(a.lo, msb_mask);
    __m256i sum_flipped = _mm256_xor_si256(lo_sum, msb_mask);

    __m256i carry_mask = _mm256_cmpgt_epi64(a_lo_flipped, sum_flipped);
    __m256i hi_sum = _mm256_add_epi64(a.hi, b.hi);
    hi_sum = _mm256_sub_epi64(hi_sum, carry_mask);

    return v4q128(lo_sum, hi_sum);
}

[[nodiscard]] V4Q128_INLINE v4q128 sub(v4q128 a, v4q128 b) noexcept {
    __m256i lo_diff = _mm256_sub_epi64(a.lo, b.lo);

    __m256i msb_mask = detail::sign_bit_mask();
    __m256i a_lo_flipped = _mm256_xor_si256(a.lo, msb_mask);
    __m256i b_lo_flipped = _mm256_xor_si256(b.lo, msb_mask);

    __m256i borrow_mask = _mm256_cmpgt_epi64(b_lo_flipped, a_lo_flipped);
    __m256i hi_diff = _mm256_sub_epi64(a.hi, b.hi);
    hi_diff = _mm256_add_epi64(hi_diff, borrow_mask);

    return v4q128(lo_diff, hi_diff);
}

[[nodiscard]] V4Q128_INLINE v4q128 neg(v4q128 a) noexcept {
    __m256i one = _mm256_cmpeq_epi64(a.lo, a.lo);
    __m256i not_lo = _mm256_xor_si256(a.lo, one);
    __m256i not_hi = _mm256_xor_si256(a.hi, one);
    __m256i lo_neg = _mm256_add_epi64(not_lo, _mm256_set1_epi64x(1));
    __m256i carry = _mm256_cmpeq_epi64(lo_neg, _mm256_setzero_si256());
    __m256i hi_neg = _mm256_sub_epi64(not_hi, carry);
    return v4q128(lo_neg,  hi_neg)
}

[[nodiscard]] V4Q128_INLINE v4q128 operator+(v4q128 a, v4q128 b) noexcept {
    return add(a, b);
}

[[nodiscard]] V4Q128_INLINE v4q128 operator-(v4q128 a, v4q128 b) noexcept {
    return sub(a, b);
}

[[nodiscard]] V4Q128_INLINE v4q128 operator-(v4q128 a) noexcept {
    return neg(a);
}

V4Q128_INLINE v4q128& operator+=(v4q128& a, v4q128 b) noexcept {
    a = add(a, b);
    return a;
}

V4Q128_INLINE v4q128& operator-=(v4q128& a, v4q128 b) noexcept {
    a = sub(a, b);
    return a;
}

}

#undef V4Q128_INLINE
