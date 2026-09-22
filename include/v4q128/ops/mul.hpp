#pragma once

#include <immintrin.h>
#include <v4q128/core/storage.hpp>

#if defined(_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define V4Q128_INLINE inline __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

[[nodiscard]] V4Q128_INLINE v4q128 mul(v4q128 a, v4q128 b) noexcept {
    // extract limbs (32 bits)
    const __m256i a0 = a.lo;
    const __m256i a1 = _mm256_srli_epi64(a.lo, 32);
    const __m256i a2 = a.hi;
    const __m256i a3 = _mm256_srli_epi64(a.hi, 32);

    const __m256i b0 = b.lo;
    const __m256i b1 = _mm256_srli_epi64(b.lo, 32);
    const __m256i b2 = b.hi;
    const __m256i b3 = _mm256_srli_epi64(b.hi, 32);

    // compute the 16 cross products (except for a3b3 because its useless)
    const __m256i p00 = _mm256_mul_epu32(a0, b0);
    const __m256i p01 = _mm256_mul_epu32(a0, b1);
    const __m256i p10 = _mm256_mul_epu32(a1, b0);
    const __m256i p02 = _mm256_mul_epu32(a0, b2);
    const __m256i p11 = _mm256_mul_epu32(a1, b1);
    const __m256i p20 = _mm256_mul_epu32(a2, b0);
    const __m256i p03 = _mm256_mul_epu32(a0, b3);
    const __m256i p12 = _mm256_mul_epu32(a1, b2);
    const __m256i p21 = _mm256_mul_epu32(a2, b1);
    const __m256i p30 = _mm256_mul_epu32(a3, b0);
    const __m256i p13 = _mm256_mul_epu32(a1, b3);
    const __m256i p22 = _mm256_mul_epu32(a2, b2);
    const __m256i p31 = _mm256_mul_epu32(a3, b1);
    const __m256i p23 = _mm256_mul_epu32(a2, b3);
    const __m256i p32 = _mm256_mul_epu32(a3, b2);

    const __m256i mask32 = _mm256_set1_epi64x(0x00000000FFFFFFFFULL);

    // accumulate in 32 bit slices to prevent 64 bit overflow

    // 2^32
    const __m256i p01_lo = _mm256_and_si256(p01, mask32);
    const __m256i p10_lo = _mm256_and_si256(p10, mask32);
    const __m256i t32 = _mm256_add_epi64(_mm256_srli_epi64(p00, 32), _mm256_add_epi64(p01_lo, p10_lo));
    const __m256i c64 = _mm256_srli_epi64(t32, 32);

    // 2^64
    const __m256i p01_hi = _mm256_srli_epi64(p01, 32);
    const __m256i p10_hi = _mm256_srli_epi64(p10, 32);
    const __m256i p02_lo = _mm256_and_si256(p02, mask32);
    const __m256i p11_lo = _mm256_and_si256(p11, mask32);
    const __m256i p20_lo = _mm256_and_si256(p20, mask32);
    const __m256i sum64_L = _mm256_add_epi64(c64, _mm256_add_epi64(_mm256_add_epi64(p01_hi, p10_hi),
                                                 _mm256_add_epi64(p02_lo, _mm256_add_epi64(p11_lo, p20_lo))));

    const __m256i bits_64_95 = _mm256_and_si256(sum64_L, mask32);
    const __m256i c96_from_L  = _mm256_srli_epi64(sum64_L, 32);

    const __m256i sum64_H = _mm256_add_epi64(_mm256_srli_epi64(p02, 32),
                             _mm256_add_epi64(_mm256_srli_epi64(p11, 32), _mm256_srli_epi64(p20, 32)));
    const __m256i c96_total = _mm256_add_epi64(c96_from_L, sum64_H);

    // 2^96
    const __m256i p03_lo = _mm256_and_si256(p03, mask32);
    const __m256i p12_lo = _mm256_and_si256(p12, mask32);
    const __m256i p21_lo = _mm256_and_si256(p21, mask32);
    const __m256i p30_lo = _mm256_and_si256(p30, mask32);
    const __m256i sum96_L = _mm256_add_epi64(c96_total,
                             _mm256_add_epi64(_mm256_add_epi64(p03_lo, p12_lo),
                                              _mm256_add_epi64(p21_lo, p30_lo)));

    const __m256i bits_96_127 = _mm256_and_si256(sum96_L, mask32);
    const __m256i c128_from_L  = _mm256_srli_epi64(sum96_L, 32);

    const __m256i sum96_H = _mm256_add_epi64(_mm256_srli_epi64(p03, 32),
                             _mm256_add_epi64(_mm256_srli_epi64(p12, 32),
                             _mm256_add_epi64(_mm256_srli_epi64(p21, 32), _mm256_srli_epi64(p30, 32))));
    const __m256i c128_total = _mm256_add_epi64(c128_from_L, sum96_H);

    // bits 64 to 127
    const __m256i res_lo = _mm256_or_si256(bits_64_95, _mm256_slli_epi64(bits_96_127, 32));

    // 2^128
    const __m256i p13_lo = _mm256_and_si256(p13, mask32);
    const __m256i p22_lo = _mm256_and_si256(p22, mask32);
    const __m256i p31_lo = _mm256_and_si256(p31, mask32);
    const __m256i sum128_L = _mm256_add_epi64(c128_total,
                              _mm256_add_epi64(p13_lo, _mm256_add_epi64(p22_lo, p31_lo)));

    const __m256i bits_128_159 = _mm256_and_si256(sum128_L, mask32);
    const __m256i c160_from_L  = _mm256_srli_epi64(sum128_L, 32);

    const __m256i sum128_H = _mm256_add_epi64(_mm256_srli_epi64(p13, 32),
                              _mm256_add_epi64(_mm256_srli_epi64(p22, 32), _mm256_srli_epi64(p31, 32)));
    const __m256i c160_total = _mm256_add_epi64(c160_from_L, sum128_H);

    // 2^160
    const __m256i p23_lo = _mm256_and_si256(p23, mask32);
    const __m256i p32_lo = _mm256_and_si256(p32, mask32);
    const __m256i sum160_L = _mm256_add_epi64(c160_total, _mm256_add_epi64(p23_lo, p32_lo));

    const __m256i bits_160_191 = _mm256_and_si256(sum160_L, mask32);

    // bits 128 to 191
    const __m256i res_hi_base = _mm256_or_si256(bits_128_159, _mm256_slli_epi64(bits_160_191, 32));

    // make sure for correct sign
    const __m256i zero = _mm256_setzero_si256();
    const __m256i sign_a_mask = _mm256_cmpgt_epi64(zero, a.hi);
    const __m256i sign_b_mask = _mm256_cmpgt_epi64(zero, b.hi);

    const __m256i corr_a = _mm256_and_si256(sign_a_mask, b.lo);
    const __m256i corr_b = _mm256_and_si256(sign_b_mask, a.lo);

    const __m256i res_hi = _mm256_sub_epi64(res_hi_base, _mm256_add_epi64(corr_a, corr_b));

    return v4q128(res_lo, res_hi);
}

[[nodiscard]] V4Q128_INLINE v4q128 operator*(v4q128 a, v4q128 b) noexcept {
    return mul(a, b);
}

V4Q128_INLINE v4q128& operator*=(v4q128& a, v4q128 b) noexcept {
    a = mul(a, b);
    return a;
}

} // namespace

#undef V4Q128_INLINE
