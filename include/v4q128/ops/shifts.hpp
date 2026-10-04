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
        __m256i fill = _mm256_slli_epi64(sign_bit_mask, 64 - N);
        return _mm256_or_si256(srl, fill);
    }

    [[nodiscard]] V4Q128_INLINE __m256i sra_epi64_var(__m256i v, __m256i count) noexcept {
        const __m256i zero = _mm256_setzero_si256();
        const __m256i v63 = _mm256_set1_epi64x(63);

        __m256i srl = _mm256_srlv_epi64(v, count);
        __m256i sign_mask = _mm256_cmpgt_epi64(zero, v);
        __m256i sign_sl1 = _mm256_slli_epi64(sign_mask, 1);
        __m256i sub63 = _mm256_sub_epi64(v63, count);
        __m256i fill = _mm256_sllv_epi64(sign_sl1);
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
        __m256i sign_fill = _mm256_cmpgt_epi64(_mm256_setzero_si256(), vec.hi);
        return v4q128(sign_fill, sign_fill);
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

// variable shifts

[[nodiscard]] V4Q128_INLINE v4q128 shl_var(v4q128 vec, __m256i count) noexcept {
    const __m256i v127 = _mm256_set1_epi64x(127);
    const __m256i v63 = _mm256_set1_epi64x(63);
    const __m256i v64 = _mm256_set1_epi64x(64);

    __m256i lo_srl = _mm256_srli_epi64(vec.lo, 1);
    __m256i sub64 = _mm256_sub_epi64(v63, count);
    __m256i carry = _mm256_srlv_epi64(lo_srl, sub64);

    __m256i hi_shifted = _mm256_sllv_epi64(vec.hi, count);
    __m256i hi_lt64 = _mm256_or_si256(hi_shifted, carry);
    __m256i lo_lt64 = _mm256_sllv_epi64(vec.lo, count);

    __m256i hi_gt64 = _mm256_sllv_epi64(vec.lo, _mm256_sub_epi64(count, v64));

    __m256i gt63 = _mm256_cmpgt_epi64(count, v63);
    __m256i gt127 = _mm256_cmpgt_epi64(count, v127);

    __m256i lo_final = _mm256_andnot_si256(gt63, lo_lt64);

    __m256i hi_blend = _mm256_blendv_epi8(hi_lt64, hi_gt64, gt63);
    __m256i final_hi = _mm256_andnot_si256(gt127, hi_blend);

    return v4q128(lo_final, final_hi);
}

    
[[nodiscard]] V4Q128_INLINE v4q128 shr_var(v4q128 vec, __m256i count) noexcept {
    const __m256i v63 = _mm256_set1_epi64x(63);
    const __m256i v64 = _mm256_set1_epi64x(64);
    const __m256i v127 = _mm256_set1_epi64x(127);

    __m256i hi_shiftone = _mm256_slli_epi64(vec.hi, 1);
    __m256i sub64 = _mm256_sub_epi64(v63, count);
    __m256i carry = _mm256_sllv_epi64(hi_shiftone, sub64);

    __m256i lo_shifted = _mm256_srlv_epi64(vec.lo, count);
    __m256i lo_lt64 = _mm256_or_si256(lo_shifted, carry);
    __m256i hi_lt64 = _mm256_srlv_epi64(vec.hi, count);

    __m256i lo_gt64 = _mm256_srlv_epi64(vec.hi, _mm256_sub_epi64(count, v64));

    __m256i gt63 = _mm256_cmpgt_epi64(count, v63);
    __m256i gt127 = _mm256_cmpgt_epi64(count, v127);

    __m256i final_hi = _mm256_andnot_si256(gt63, hi_lt64);

    __m256i lo_blend = _mm256_blendv_epi8(lo_lt64, lo_gt64, gt63);
    __m256i final_lo = _mm256_andnot_si256(gt127, lo_blend);

    return v4q128(final_lo, final_hi);
}

[[nodiscard]] V4Q128_INLINE v4q128 sra_var(v4q128 vec, __m256i count) noexcept {
    __m256i v63 = _mm256_set1_epi64x(63);
    __m256i v64 = _mm256_set1_epi64x(64);
    __m256i v127 = _mm256_set1_epi64x(127);
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_mask = _mm256_cmpgt_epi64(zero, vec.hi);

    __m256i hi_shiftone = _mm256_slli_epi64(vec.hi, 1);
    __m256i sub64 = _mm256_sub_epi64(v63, count);
    __m256i carry = _mm256_slli_epi64(hi_shiftone, sub64);
    
    __m256i lo_shifted = _mm256_srlv_epi64(vec.lo, count);
    __m256i lo_lt64 = _mm256_or_si256(lo_shifted, carry);
    __m256i hi_lt64 = detail::sra_epi64_var(vec.hi, count);

    __m256i lo_gt64 = detail::sra_epi64_var(vec.hi, _mm256_sub_epi64(count, v64));

    __m256i gt63 = _mm256_cmpgt_epi64(count, v63);
    __m256i gt127 = _mm256_cmpgt_epi64(count, v127);

    __m256i lo_blend = _mm256_blendv_epi8(lo_lt64, lo_gt64, gt63);
    __m256i final_lo = _mm256_blendv_epi8(lo_blend, sign_mask, gt127);
    __m256i final_hi = _mm256_blendv_epi8(hi_lt64, sign_mask, gt63);

    return v4q128(final_lo, final_hi);
}
}
