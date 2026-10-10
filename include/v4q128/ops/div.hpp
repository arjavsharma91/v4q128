#pragma once
#include <immintrin.h>
#include <v4q128/core/storage.hpp>
#include <v4q128/io/conversions.hpp>

#if defined(_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined (__GNUC__) || defined(__clang__)
    #define V4Q128_INLINE inline __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

[[nodiscard]] V4Q128_INLINE v4q128 div_approx(v4q128 a, v4q128 b) noexcept {
    __m256d da = to_double(a);
    __m256d db = to_double(b);
    __m256d d_double = _mm256_div_pd(da, db);
    return from_double(d_double);
}
