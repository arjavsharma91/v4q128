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
                lo = static_cast<uint64_t>(_mm256_extract_epi64(v.lo, 0));
                hi = _mm256_extract_epi64(v.hi, 0);
                break;
            case 1:
                lo = static_cast<uint64_t>(_mm256_extract_epi64(v.lo, 1));
                hi = _mm256_extract_epi64(v.hi, 1);
                break;
            case 2:
                lo = static_cast<uint64_t>(_mm256_extract_epi64(v.lo, 2));
                hi = _mm256_extract_epi64(v.hi, 2);
                break;
            case 3:
                lo = static_cast<uint64_t>(_mm256_extract_epi64(v.lo, 3));
                hi = _mm256_extract_epi64(v.hi, 3);
                break;
            default: break;
        }
        return (static_cast<int128_t>(hi) << 64) | lo;
    };

    int128_t va0 = extract_lane(a, 0);
    int128_t va1 = extract_lane(a, 1);
    int128_t va2 = extract_lane(a, 2);
    int128_t va3 = extract_lane(a, 3);
    int128_t vb0 = extract_lane(b, 0);
    int128_t vb1 = extract_lane(b, 1);
    int128_t vb2 = extract_lane(b, 2);
    int128_t vb3 = extract_lane(b, 3);

    auto compute_div = [](int128_t va, int128_t vb) -> int128_t {
        int128_t neg = (va >> 127) ^ (vb >> 127);
        uint128_t ua = va < 0 ? static_cast<uint128_t>(-va) : static_cast<uint128_t>(va);
        uint128_t ub = vb < 0 ? static_cast<uint128_t>(-vb) : static_cast<uint128_t>(vb);

        uint128_t udiv = (ua << 64) / ub;

        return static_cast<int128_t>((udiv^neg) - neg);
    };

    int128_t res0 = compute_div(va0, vb0);
    int128_t res1 = compute_div(va1, vb1);
    int128_t res2 = compute_div(va2, vb2);
    int128_t res3 = compute_div(va3, vb3);

    return v4q128(
        _mm256_set_epi64x(static_cast<int64_t>(res3), static_cast<int64_t>(res2), static_cast<int64_t>(res1), static_cast<int64_t>(res0)),
        _mm256_set_epi64x(static_cast<int64_t>(res3 >> 64), static_cast<int64_t>(res2 >> 64), static_cast<int64_t>(res1 >> 64), static_cast<int64_t>(res0 >> 64))
        );
}
} // namespace
