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
    __m256i combined_xor = _mm256_xor_si256(a.hi, b.hi);
    __m256i sign_mask = _mm256_cmpgt_epi64(_mm256_setzero_si256(), combined_xor);

    __m256i sign_mask_a = _mm256_cmpgt_epi64(_mm256_setzero_si256(), a.hi);
    __m256i a_hi_inv = _mm256_xor_si256(a.hi, sign_mask_a);
    __m256i a_lo_inv = _mm256_xor_si256(a.lo, sign_mask_a);
    __m256i a_lo_abs = _mm256_sub_epi64(a_lo_inv, sign_mask_a);

    __m256i carrya = _mm256_and_si256(sign_mask_a, _mm256_cmpeq_epi64(a.lo, _mm256_setzero_si256()));
    __m256i a_hi_abs = _mm256_sub_epi64(a_hi_inv, carrya);

    __m256i sign_mask_b = _mm256_cmpgt_epi64(_mm256_setzero_si256(), b.hi);
    __m256i b_hi_inv = _mm256_xor_si256(b.hi, sign_mask_b);
    __m256i b_lo_inv = _mm256_xor_si256(b.lo, sign_mask_b);
    __m256i b_lo_abs = _mm256_sub_epi64(b_lo_inv, sign_mask_b);

    __m256i carryb = _mm256_and_si256(sign_mask_b, _mm256_cmpeq_epi64(b.lo, _mm256_setzero_si256()));
    __m256i b_hi_abs = _mm256_sub_epi64(b_hi_inv, carryb);

    __m256i a0 = a_lo_abs;
    __m256i a1 = _mm256_srli_epi64(a_lo_abs, 32);
    __m256i a2 = a_hi_abs;
    __m256i a3 = _mm256_srli_epi64(a_hi_abs, 32);

    __m256i b0 = b_lo_abs;
    __m256i b1 = _mm256_srli_epi64(b_lo_abs, 32);
    __m256i b2 = b_hi_abs;
    __m256i b3 = _mm256_srli_epi64(b_hi_abs, 32);

    // step 1
    __m256i a0_b0 = _mm256_mul_epu32(a0, b0);
    __m256i a0_b1 = _mm256_mul_epu32(a0, b1);
    __m256i a1_b0 = _mm256_mul_epu32(a1, b0);
   
    __m256i zero = _mm256_setzero_si256();
    __m256i a0b0_align = _mm256_srli_epi64(a0_b0, 32);
    __m256i a0_b1_32 = _mm256_blend_epi32(a0_b1, zero, 0xAA);
    __m256i a1_b0_32 = _mm256_blend_epi32(a1_b0, zero, 0xAA);
    
    __m256i sum_32 = _mm256_add_epi64(a0_b0_align, _mm256_add_epi64(a0_b1_32, a1_b0_32));
    __m256i bit_64_carry = _mm256_srli_epi64(sum_32, 32);

    // step 2
    __m256i a0_b2 = _mm256_mul_epu32(a0, b2);
    __m256i a1_b1 = _mm256_mul_epu32(a1, b1);
    __m256i a2_b0 = _mm256_mul_epu32(a2, b0);

    __m256i a0b1_align = _mm256_srli_epi64(a0_b1, 32);
    __m256i a1b0_align = _mm256_srli_epi64(a1_b0, 32);
    __m256i res_lo_base = _mm256_add_epi64(bit_64_carry, _mm256_add_epi64(a0b1_align, a1b0_align));

    __m256i sum_pair1 = _mm256_add_epi64(a0_b2, a1_b1);
    __m256i sum_pair2 = _mm256_add_epi64(a2_b0, res_lo_base);
    __m256i res_lo = _mm256_add_epi64(sum_pair1, sum_pair2);

    // step 3
    __m256i a0_b3 = _mm256_mul_epu32(a0, b3);
    __m256i a1_b2 = _mm256_mul_epu32(a1, b2);
    __m256i a2_b1 = _mm256_mul_epu32(a2, b1);
    __m256i a3_b0 = _mm256_mul_epu32(a3, b0);
    
    __m256i sum_pair_1_3 = _mm256_add_epi64(a0_b3, a1_b2);
    __m256i sum_pair_2_3 = _mm256_add_epi64(a2_b1, a3_b0);
    __m256i sum_96 = _mm256_add_epi64(sum_pair_1_3, sum_pair_2_3);

    __m256i sum_96_bl = _mm256_slli_epi64(sum_96_blended, 32);
    res_lo = _mm256_add_epi64(sum_96_bl, res_lo);
    

    
    
    
    
    

    
