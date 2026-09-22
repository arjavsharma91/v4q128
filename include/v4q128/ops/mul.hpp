#pragma once

#if defined (_MSC_VER)
    #define V4Q128_INLINE __force_inline
#elif defined (__GNUC__) || defined (__clang__)
    #define V4Q128_INLINE __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

[[nodiscard]] V4Q128_INLINE v4q128 mul(v4q128 a, v4q128 b) noexcept {
    __m256i zero   = _mm256_setzero_si256();
    __m256i mask32 = _mm256_set1_epi64x(0xFFFFFFFFULL);

    __m256i sign_a   = _mm256_cmpgt_epi64(zero, a.hi);
    __m256i a_abs_lo = _mm256_sub_epi64(_mm256_xor_si256(a.lo, sign_a), sign_a);
    __m256i carry_a  = _mm256_and_si256(sign_a, _mm256_cmpeq_epi64(a.lo, zero));
    __m256i a_abs_hi = _mm256_sub_epi64(_mm256_xor_si256(a.hi, sign_a), carry_a);

    __m256i sign_b   = _mm256_cmpgt_epi64(zero, b.hi);
    __m256i b_abs_lo = _mm256_sub_epi64(_mm256_xor_si256(b.lo, sign_b), sign_b);
    __m256i carry_b  = _mm256_and_si256(sign_b, _mm256_cmpeq_epi64(b.lo, zero));
    __m256i b_abs_hi = _mm256_sub_epi64(_mm256_xor_si256(b.hi, sign_b), carry_b);

    __m256i res_sign = _mm256_xor_si256(sign_a, sign_b);

    __m256i a0 = a_abs_lo;
    __m256i a1 = _mm256_srli_epi64(a_abs_lo, 32);
    __m256i a2 = a_abs_hi;
    __m256i a3 = _mm256_srli_epi64(a_abs_hi, 32);

    __m256i b0 = b_abs_lo;
    __m256i b1 = _mm256_srli_epi64(b_abs_lo, 32);
    __m256i b2 = b_abs_hi;
    __m256i b3 = _mm256_srli_epi64(b_abs_hi, 32);

    __m256i p00 = _mm256_mul_epu32(a0, b0);
    __m256i p01 = _mm256_mul_epu32(a0, b1);
    __m256i p10 = _mm256_mul_epu32(a1, b0);

    __m256i p02 = _mm256_mul_epu32(a0, b2);
    __m256i p11 = _mm256_mul_epu32(a1, b1);
    __m256i p20 = _mm256_mul_epu32(a2, b0);

    __m256i p03 = _mm256_mul_epu32(a0, b3);
    __m256i p12 = _mm256_mul_epu32(a1, b2);
    __m256i p21 = _mm256_mul_epu32(a2, b1);
    __m256i p30 = _mm256_mul_epu32(a3, b0);

    __m256i p13 = _mm256_mul_epu32(a1, b3);
    __m256i p22 = _mm256_mul_epu32(a2, b2);
    __m256i p31 = _mm256_mul_epu32(a3, b1);

    __m256i p23 = _mm256_mul_epu32(a2, b3);
    __m256i p32 = _mm256_mul_epu32(a3, b2);

    __m256i c32_pair0 = _mm256_add_epi64(_mm256_and_si256(p01, mask32), _mm256_and_si256(p10, mask32));
    __m256i col32     = _mm256_add_epi64(_mm256_srli_epi64(p00, 32), c32_pair0);
    __m256i carry64   = _mm256_srli_epi64(col32, 32);

    __m256i c64_l0_0 = _mm256_add_epi64(carry64, _mm256_srli_epi64(p01, 32));
    __m256i c64_l0_1 = _mm256_add_epi64(_mm256_srli_epi64(p10, 32), _mm256_and_si256(p02, mask32));
    __m256i c64_l0_2 = _mm256_add_epi64(_mm256_and_si256(p11, mask32), _mm256_and_si256(p20, mask32));

    __m256i c64_l1_0 = _mm256_add_epi64(c64_l0_0, c64_l0_1);
    __m256i col64    = _mm256_add_epi64(c64_l1_0, c64_l0_2);

    __m256i res_lo_part0 = _mm256_and_si256(col64, mask32);
    __m256i carry96      = _mm256_srli_epi64(col64, 32);

    __m256i c96_l0_0 = _mm256_add_epi64(carry96,                     _mm256_srli_epi64(p02, 32));
    __m256i c96_l0_1 = _mm256_add_epi64(_mm256_srli_epi64(p11, 32), _mm256_srli_epi64(p20, 32));
    __m256i c96_l0_2 = _mm256_add_epi64(_mm256_and_si256(p03, mask32), _mm256_and_si256(p12, mask32));
    __m256i c96_l0_3 = _mm256_add_epi64(_mm256_and_si256(p21, mask32), _mm256_and_si256(p30, mask32));

    __m256i c96_l1_0 = _mm256_add_epi64(c96_l0_0, c96_l0_1);
    __m256i c96_l1_1 = _mm256_add_epi64(c96_l0_2, c96_l0_3);
    __m256i col96    = _mm256_add_epi64(c96_l1_0, c96_l1_1);

    __m256i carry128 = _mm256_srli_epi64(col96, 32);
    __m256i res_lo   = _mm256_or_si256(res_lo_part0, _mm256_slli_epi64(col96, 32));

    __m256i c128_l0_0 = _mm256_add_epi64(carry128,                    _mm256_srli_epi64(p03, 32));
    __m256i c128_l0_1 = _mm256_add_epi64(_mm256_srli_epi64(p12, 32), _mm256_srli_epi64(p21, 32));
    __m256i c128_l0_2 = _mm256_add_epi64(_mm256_srli_epi64(p30, 32), _mm256_and_si256(p13, mask32));
    __m256i c128_l0_3 = _mm256_add_epi64(_mm256_and_si256(p22, mask32), _mm256_and_si256(p31, mask32));

    __m256i c128_l1_0 = _mm256_add_epi64(c128_l0_0, c128_l0_1);
    __m256i c128_l1_1 = _mm256_add_epi64(c128_l0_2, c128_l0_3);
    __m256i col128    = _mm256_add_epi64(c128_l1_0, c128_l1_1);

    __m256i res_hi_part0 = _mm256_and_si256(col128, mask32);
    __m256i carry160     = _mm256_srli_epi64(col128, 32);

    __m256i c160_l0_0 = _mm256_add_epi64(carry160,                    _mm256_srli_epi64(p13, 32));
    __m256i c160_l0_1 = _mm256_add_epi64(_mm256_srli_epi64(p22, 32), _mm256_srli_epi64(p31, 32));
    __m256i c160_l0_2 = _mm256_add_epi64(_mm256_and_si256(p23, mask32), _mm256_and_si256(p32, mask32));

    __m256i c160_l1_0 = _mm256_add_epi64(c160_l0_0, c160_l0_1);
    __m256i col160    = _mm256_add_epi64(c160_l1_0, c160_l0_2);

    __m256i res_hi = _mm256_or_si256(res_hi_part0, _mm256_slli_epi64(col160, 32));

    __m256i res_lo_inv   = _mm256_xor_si256(res_lo, res_sign);
    __m256i res_lo_final = _mm256_sub_epi64(res_lo_inv, res_sign);

    __m256i res_hi_inv   = _mm256_xor_si256(res_hi, res_sign);
    __m256i carry_res    = _mm256_and_si256(res_sign, _mm256_cmpeq_epi64(res_lo, zero));
    __m256i res_hi_final = _mm256_sub_epi64(res_hi_inv, carry_res);

    return v4q128(res_lo_final, res_hi_final);
}
    
