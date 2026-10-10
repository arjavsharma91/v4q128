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

[[nodiscard]] V4Q128_INLINE __m256d to_double(v4q128 v) noexcept {
    const __m256d scale_lo = _mm256_set1_pd(5.4210108624275221700372640043497e-20);
    const __m256d uint64_bias = _mm256_set1_pd(18446744073709551616.0);

    __m256d hi_final = _mm256_cvtepi64_pd(v.hi);

    __m256d lo_flipped = _mm256_xor_si256(v.lo, detail::sign_bit_mask());
    __m256d lo_converted = _mm256_cvepi64_pd(lo_flipped);
    __m256d lo_final = _mm256_add_pd(lo_converted, uint64_bias);

    return _mm256_fmadd_pd(lo_final, scale_lo, hi_final);
}

[[nodiscard]] V4Q128_INLINE v4q128 from_double(__m256d vec) noexcept {
    const __m256d scale_hi = _mm256_set1_pd(18446744073709551616.0);

    __m256d hi_dbl = _mm256_floor_pd(vec);
    __m256i hi_final = _mm256_cvtpd_epi64(hi_dbl);

    __m256d fraction = _mm256_sub_pd(hi_dbl, hi_final);
    __m256d scaled = _mm256_mul_pd(fraction, scale_hi);
    __m256i lo_final = _mm256_cvtpd_epi64(scaled);

    return v4q128(lo_final, hi_final);
}
} // namespace
