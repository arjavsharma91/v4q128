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
    
    

    

