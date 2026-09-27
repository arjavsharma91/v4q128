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

static uint64_t splitmix64_impl(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}


#if defined(__SIZEOF_INT128__)
    typedef __int128 int128_t;
    typedef unsigned __int128 uint128_t;

    inline int128_t make_int128(uint64_t lo, int64_t hi) noexcept {
        return (static_cast<int128_t>(hi) << 64) | lo;
    }

    inline int128_t scalar_add(int128_t a, int128_t b) noexcept { return a + b; }
    inline int128_t scalar_sub(int128_t a, int128_t b) noexcept { return a - b; }
    inline int128_t scalar_neg(int128_t a) noexcept { return -a; }

    inline uint64_t get_lo(int128_t v) noexcept { return static_cast<uint64_t>(v); }
    inline int64_t  get_hi(int128_t v) noexcept { return static_cast<int64_t>(v >> 64); }

    static int128_t generate_rand128(uint64_t& state) noexcept {
        const uint64_t lo = splitmix64_impl(state);
        const int64_t  hi = static_cast<int64_t>(splitmix64_impl(state));
        return make_int128(lo, hi);
    }
#else
    // MSVC
    struct int128_t {
        uint64_t lo;
        int64_t  hi;

        bool operator==(const int128_t& o) const noexcept {
            return lo == o.lo && hi == o.hi;
        }
    };

    inline int128_t make_int128(uint64_t lo, int64_t hi) noexcept {
        return int128_t{lo, hi};
    }

    inline int128_t scalar_add(int128_t a, int128_t b) noexcept {
        uint64_t res_lo = a.lo + b.lo;
        int64_t  res_hi = a.hi + b.hi + (res_lo < a.lo ? 1 : 0);
        return int128_t{res_lo, res_hi};
    }

    inline int128_t scalar_sub(int128_t a, int128_t b) noexcept {
        uint64_t res_lo = a.lo - b.lo;
        int64_t  res_hi = a.hi - b.hi - (a.lo < b.lo ? 1 : 0);
        return int128_t{res_lo, res_hi};
    }

    inline int128_t scalar_neg(int128_t a) noexcept {
        return scalar_sub(int128_t{0, 0}, a);
    }

    inline uint64_t get_lo(int128_t v) noexcept { return v.lo; }
    inline int64_t  get_hi(int128_t v) noexcept { return v.hi; }

    static int128_t generate_rand128(uint64_t& state) noexcept {
        const uint64_t lo = splitmix64_impl(state);
        const int64_t  hi = static_cast<int64_t>(splitmix64_impl(state));
        return make_int128(lo, hi);
    }
#endif

int main() {
    constexpr size_t N = 65536;
    static_assert(N % 4 == 0, "N must be divisible by 4 for SIMD packing");

    std::cout << "========================================================\n";
    std::cout << "   v4q128 vs Scalar 128-bit Add / Sub / Neg Benchmark   \n";
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

    std::cout << "Verifying equivalence between scalar and SIMD datasets...\n";

    for (size_t i = 0; i < N / 4; ++i) {
        const auto res = add(simd_a[i], simd_b[i]);
        alignas(32) uint64_t res_lo[4];
        alignas(32) int64_t  res_hi[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_lo), res.lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_hi), res.hi);

        for (int j = 0; j < 4; ++j) {
            const int128_t expected = scalar_add(scalar_a[i * 4 + j], scalar_b[i * 4 + j]);
            const int128_t actual   = make_int128(res_lo[j], res_hi[j]);
            assert(get_lo(expected) == get_lo(actual) && "ADD SIMD lo limb mismatch!");
            assert(get_hi(expected) == get_hi(actual) && "ADD SIMD hi limb mismatch!");
        }
    }
    std::cout << "  - ADD: PASSED!\n";

    for (size_t i = 0; i < N / 4; ++i) {
        const auto res = sub(simd_a[i], simd_b[i]);
        alignas(32) uint64_t res_lo[4];
        alignas(32) int64_t  res_hi[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_lo), res.lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_hi), res.hi);

        for (int j = 0; j < 4; ++j) {
            const int128_t expected = scalar_sub(scalar_a[i * 4 + j], scalar_b[i * 4 + j]);
            const int128_t actual   = make_int128(res_lo[j], res_hi[j]);
            assert(get_lo(expected) == get_lo(actual) && "SUB SIMD lo limb mismatch!");
            assert(get_hi(expected) == get_hi(actual) && "SUB SIMD hi limb mismatch!");
        }
    }
    std::cout << "  - SUB: PASSED!\n";

    for (size_t i = 0; i < N / 4; ++i) {
        const auto res = neg(simd_a[i]);
        alignas(32) uint64_t res_lo[4];
        alignas(32) int64_t  res_hi[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_lo), res.lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_hi), res.hi);

        for (int j = 0; j < 4; ++j) {
            const int128_t expected = scalar_neg(scalar_a[i * 4 + j]);
            const int128_t actual   = make_int128(res_lo[j], res_hi[j]);
            assert(get_lo(expected) == get_lo(actual) && "NEG SIMD lo limb mismatch!");
            assert(get_hi(expected) == get_hi(actual) && "NEG SIMD hi limb mismatch!");
        }
    }
    std::cout << "  - NEG: PASSED!\n\n";


    ankerl::nanobench::Bench bench;
    bench.title("Bulk 128-bit Addition, Subtraction & Negation");
    bench.unit("128bit-op");
    bench.warmup(100);
    bench.epochs(100);

    // --- ADDITION ---
    bench.batch(N).run("Scalar 128-bit Add", [&] {
        for (size_t i = 0; i < N; ++i) {
            scalar_res[i] = scalar_add(scalar_a[i], scalar_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(scalar_res.data());
    });

    bench.batch(N).run("v4q128 AVX2 Add", [&] {
        for (size_t i = 0; i < N / 4; ++i) {
            simd_res[i] = add(simd_a[i], simd_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(simd_res.data());
    });

    bench.batch(N).run("Scalar 128-bit Sub", [&] {
        for (size_t i = 0; i < N; ++i) {
            scalar_res[i] = scalar_sub(scalar_a[i], scalar_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(scalar_res.data());
    });

    bench.batch(N).run("v4q128 AVX2 Sub", [&] {
        for (size_t i = 0; i < N / 4; ++i) {
            simd_res[i] = sub(simd_a[i], simd_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(simd_res.data());
    });

    bench.batch(N).run("Scalar 128-bit Neg", [&] {
        for (size_t i = 0; i < N; ++i) {
            scalar_res[i] = scalar_neg(scalar_a[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(scalar_res.data());
    });

    bench.batch(N).run("v4q128 AVX2 Neg", [&] {
        for (size_t i = 0; i < N / 4; ++i) {
            simd_res[i] = neg(simd_a[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(simd_res.data());
    });

    return 0;
}
