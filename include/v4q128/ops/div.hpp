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


[[nodiscard]] V4Q128_INLINE v4q128 div_exact(v4q128 a, v4q128 b) noexcept {
    using int128_t = __int128;
    using uint128_t = unsigned __int128;

    auto extract_lane = [](v4q128 v, int i) -> int128_t {
        uint64_t lo = 0;
        int64_t hi = 0;

        switch (i) {
            case 0:
                lo = static_case<uint64_t>(_mm256_extract_epi64(v.lo, 0));
