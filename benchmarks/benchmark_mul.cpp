#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <immintrin.h>
#include <v4q128/v4q128.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

using namespace v4q128;
using v4q128_t = v4q128::v4q128;

// =============================================================================
// Cross-Platform 128-Bit Scalar Q64.64 Multiplication (MSVC & GCC/Clang)
// =============================================================================
#if defined(__SIZEOF_INT128__)
    typedef __int128 int128_t;
    typedef unsigned __int128 uint128_t;

    inline int128_t scalar_mul_q64_64(int128_t a, int128_t b) noexcept {
        const uint64_t a_lo = static_cast<uint64_t>(a);
        const int64_t  a_hi = static_cast<int64_t>(a >> 64);
        const uint64_t b_lo = static_cast<uint64_t>(b);
        const int64_t  b_hi = static_cast<int64_t>(b >> 64);

        const uint128_t p00 = static_cast<uint128_t>(a_lo) * b_lo;
        const int128_t  p01 = static_cast<int128_t>(a_lo) * b_hi;
        const int128_t  p10 = static_cast<int128_t>(a_hi) * b_lo;
        const int128_t  p11 = static_cast<int128_t>(a_hi) * b_hi;

        return static_cast<int128_t>(p00 >> 64) + p01 + p10 + (p11 << 64);
    }

    inline uint64_t get_lo(int128_t v) noexcept { return static_cast<uint64_t>(v); }
    inline int64_t  get_hi(int128_t v) noexcept { return static_cast<int64_t>(v >> 64); }

    static int128_t generate_rand128(uint64_t& state) noexcept {
        const uint64_t lo = splitmix64_impl(state);
        const uint64_t hi = splitmix64_impl(state);
        return (static_cast<int128_t>(hi) << 64) | lo;
    }
#else
    // MSVC Native Fallback using _mul128 / _umul128 intrinsics
    struct int128_t {
        uint64_t lo;
        int64_t  hi;

        bool operator==(const int128_t& o) const noexcept {
            return lo == o.lo && hi == o.hi;
        }
    };

    inline int128_t scalar_mul_q64_64(int128_t a, int128_t b) noexcept {
        uint64_t p00_hi;
        uint64_t p00_lo = _umul128(a.lo, b.lo, &p00_hi);

        int64_t p01_hi;
        int64_t p01_lo = _mul128(static_cast<int64_t>(a.lo), b.hi, &p01_hi);

        int64_t p10_hi;
        int64_t p10_lo = _mul128(a.hi, static_cast<int64_t>(b.lo), &p10_hi);

        int64_t p11_hi;
        int64_t p11_lo = _mul128(a.hi, b.hi, &p11_hi);

        uint64_t res_lo = p00_hi + static_cast<uint64_t>(p01_lo) + static_cast<uint64_t>(p10_lo);
        int64_t  res_hi = p01_hi + p10_hi + p11_lo;

        if (res_lo < p00_hi) res_hi++; // Carry handling

        return int128_t{res_lo, res_hi};
    }

    inline uint64_t get_lo(int128_t v) noexcept { return v.lo; }
    inline int64_t  get_hi(int128_t v) noexcept { return v.hi; }

    static int128_t generate_rand128(uint64_t& state) noexcept {
        const uint64_t lo = splitmix64_impl(state);
        const int64_t  hi = static_cast<int64_t>(splitmix64_impl(state));
        return int128_t{lo, hi};
    }
#endif

static uint64_t splitmix64_impl(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main() {
    constexpr size_t N = 65536;
    static_assert(N % 4 == 0, "N must be divisible by 4 for SIMD packing");

    std::cout << "========================================================\n";
    std::cout << "   v4q128 vs Scalar 128-bit Bulk Multiplication Benchmark\n";
    std::cout << "========================================================\n\n";

    std::vector<int128_t> scalar_a(N);
    std::vector<int128_t> scalar_b(N);
    std::vector<int128_t> scalar_res(N);

    uint64_t rng_state = 0x123456789ABCDEF0ULL;
    for (size_t i = 0; i < N; ++i) {
        scalar_a[i] = generate_rand128(rng_state);
        scalar_b[i] = generate_rand128(rng_state);
    }

    std::vector<v4q128_t> simd_a(N / 4);
    std::vector<v4q128_t> simd_b(N / 4);
    std::vector<v4q128_t> simd_res(N / 4);

    // Populate SIMD structures using storage.hpp set() helper
    for (size_t i = 0; i < N / 4; ++i) {
        const size_t idx = i * 4;

        simd_a[i] = v4q128_t::set(
            get_lo(scalar_a[idx + 0]), get_hi(scalar_a[idx + 0]),
            get_lo(scalar_a[idx + 1]), get_hi(scalar_a[idx + 1]),
            get_lo(scalar_a[idx + 2]), get_hi(scalar_a[idx + 2]),
            get_lo(scalar_a[idx + 3]), get_hi(scalar_a[idx + 3])
        );

        simd_b[i] = v4q128_t::set(
            get_lo(scalar_b[idx + 0]), get_hi(scalar_b[idx + 0]),
            get_lo(scalar_b[idx + 1]), get_hi(scalar_b[idx + 1]),
            get_lo(scalar_b[idx + 2]), get_hi(scalar_b[idx + 2]),
            get_lo(scalar_b[idx + 3]), get_hi(scalar_b[idx + 3])
        );
    }

    std::cout << "Verifying equivalence between scalar and SIMD datasets... ";
    for (size_t i = 0; i < N / 4; ++i) {
        const auto res = mul(simd_a[i], simd_b[i]);

        alignas(32) uint64_t res_lo[4];
        alignas(32) int64_t  res_hi[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_lo), res.lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_hi), res.hi);

        for (int j = 0; j < 4; ++j) {
            const int128_t expected = scalar_mul_q64_64(scalar_a[i * 4 + j], scalar_b[i * 4 + j]);
            const int128_t actual   = int128_t{res_lo[j], res_hi[j]};

            assert(get_lo(expected) == get_lo(actual) && "SIMD lo limb mismatch!");
            assert(get_hi(expected) == get_hi(actual) && "SIMD hi limb mismatch!");
        }
    }
    std::cout << "PASSED!\n\n";

    // =========================================================================
    // Nanobench Configuration (Deterministic Cycles Mode for Windows/Linux)
    // =========================================================================
    ankerl::nanobench::Bench bench;
    bench.title("Bulk 128-bit Q64.64 Multiplication");
    bench.unit("128bit-mul");
    bench.warmup(100);
    bench.epochs(100);

    // Forces nanobench to directly count CPU clock cycles (RDTSC)
    bench.clockResolutionCustom([]() {
        return ankerl::nanobench::Clock::duration(1);
    });

    bench.batch(N).run("Scalar 128-bit Mul", [&] {
        for (size_t i = 0; i < N; ++i) {
            scalar_res[i] = scalar_mul_q64_64(scalar_a[i], scalar_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(scalar_res.data());
    });

    bench.batch(N).run("v4q128 AVX2 Mul (4-way SoA - 2 Wave)", [&] {
        for (size_t i = 0; i < N / 4; ++i) {
            simd_res[i] = mul(simd_a[i], simd_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(simd_res.data());
    });

    return 0;
}
