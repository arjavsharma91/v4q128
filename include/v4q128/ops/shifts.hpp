#pragma once
#include <immintrin.h>
#include <v4q128/core/storage.hpp>
#include <cstdint>

#if defined (_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined (__GNUC__) || defined (__clang__)
    #define V4Q128_INLINE __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

namespace detail {
    template <int N>
    [[nodiscard]] V4Q128_INLINE __m256i sra_epi64_imm(__m256i v) noexcept {
        if constexpr (N == 0) return v;
        __m256i srl = _mm256_srli_epi64(v, N);
        __m256i sign_bit_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
        __m256i fill = _mm256_slli_epi64(sign_mask, 64 - N);
        return _mm256_or_si256(srl, fill);
    }

    [[nodiscard]] V4Q128_INLINE __m256i sra_epi64_var(__m256i v, __m256i count) noexcept {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i v64 = _mm256_set1_epi64x(64);
        const __m256i v63 = _mm256_set1_epi64x(63);

        __m256i srl = _mm256_srlv_epi64(v, count);
        __m256i sign_mask = _mm256_cmpgt_epi64(zero, v);

        __m256i cnt_mod64 = _mm256_and_si256(count, v63);
        __m256i valid_mask = _mm256_cmpgt_epi64(cnt_mod64, zero);
        __m256i active_sign = _mm256_and_si256(sign_mask, valid_mask);

        __m256i fill = _mm256_slli_epi64(active_sign, _mm256_sub_epi64(v64, count));
        return _mm256_or_si256(srl, fill);
    }
}

// compile time

template <int N>
[[nodiscard]] V4Q128_INLINE v4q128 shl_imm(v4q128 vec) noexcept {
    if constexpr (N == 0) {
        return vec;
    } else if constexpr (N >= 128) {
        return v4q128::zero();
    } else if constexpr (N < 64) {
        __m256i hi_shifted = _mm256_slli_epi64(vec.hi, N);
        __m256i carry = _mm256_srli_epi64(vec.lo, 64 - N);
        __m256i hi_final = _mm256_or_si256(hi_shifted, carry);
        __m256i lo_final = _mm256_slli_epi64(vec.lo, N);
        return v4q128(lo_final, hi_final);
    } else {
        __m256i hi_final = _mm256_slli_epi64(vec.lo, N-64);
        __m256i lo_final = _mm256_setzero_si256();
        return v4q128(lo_final, hi_final);
    }
}

template <int N>
[[nodiscard]] V4Q128_INLINE v4q128 srl_imm(v4q128 vec) noexcept {
    if constexpr (N == 0) {
        return vec;
    } else if constexpr (N >= 128) {
        return v4q128::zero();
    } else if constexpr (N < 64) {
        __m256i lo_shifted = _mm256_srli_epi64(vec.lo, N);
        __m256i carry = _mm256_slli_epi64(vec.hi, 64 - N);
        __m256i lo_final = _mm256_or_si256(lo_shifted, carry);
        __m256i hi_final = _mm256_srli_epi64(vec.hi, N);
        return v4q128(lo_final, hi_final);
    } else {
        __m256i lo_final = _mm256_srli_epi64(vec.hi, N-64);
        __m256i hi_final = _mm256_setzero_si256();
        return v4q128(lo_final, hi_final);
    }
}

template <int N>
[[nodiscard]] V4Q128_INLINE v4q128 sra_imm(v4q128 vec) noexcept {
    if constexpr (N == 0) {
        return vec;
    } else if constexpr (N >= 128) {
        return v4q128::zero();
    } else if constexpr (N < 64) {
        __m256i lo_shifted = _mm256_srli_epi64(vec.lo, N);
        __m256i carry = _mm256_slli_epi64(vec.hi, 64 - N);
        __m256i lo_final = _mm256_or_si256(lo_shifted, carry);
        __m256i hi_final = detail::sra_epi64_imm<N>(vec.hi);
        return v4q128(lo_final, hi_final);
    } else {
        __m256i hi_final = _mm256_cmpgt_epi64(_mm256_setzero_si256(), vec.hi);
        __m256i lo_final = detail::sra_epi64_imm<N-64>(vec.hi);
        return v4q128(lo_final, hi_final);
    }
}
    
