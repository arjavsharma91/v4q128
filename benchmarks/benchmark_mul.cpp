#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include <immintrin.h>
#include <v4q128/v4q128.hpp>

#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>

// Bring v4q128 types and functions into scope
using namespace v4q128;

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
#endif

static uint64_t splitmix64(uint64_t& state) noexcept {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static int128_t generate_rand128(uint64_t& state) noexcept {
    const uint64_t lo = splitmix64(state);
    const uint64_t hi = splitmix64(state);
    return (static_cast<int128_t>(hi) << 64) | lo;
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

    std::vector<v4q128> simd_a(N / 4);
    std::vector<v4q128> simd_b(N / 4);
    std::vector<v4q128> simd_res(N / 4);

    for (size_t i = 0; i < N / 4; ++i) {
        const size_t idx = i * 4;

        const __m256i a_lo = _mm256_set_epi64x(
            static_cast<long long>(scalar_a[idx + 3]),
            static_cast<long long>(scalar_a[idx + 2]),
            static_cast<long long>(scalar_a[idx + 1]),
            static_cast<long long>(scalar_a[idx + 0])
        );
        const __m256i a_hi = _mm256_set_epi64x(
            static_cast<long long>(scalar_a[idx + 3] >> 64),
            static_cast<long long>(scalar_a[idx + 2] >> 64),
            static_cast<long long>(scalar_a[idx + 1] >> 64),
            static_cast<long long>(scalar_a[idx + 0] >> 64)
        );
        simd_a[i] = v4q128(a_lo, a_hi);

        const __m256i b_lo = _mm256_set_epi64x(
            static_cast<long long>(scalar_b[idx + 3]),
            static_cast<long long>(scalar_b[idx + 2]),
            static_cast<long long>(scalar_b[idx + 1]),
            static_cast<long long>(scalar_b[idx + 0])
        );
        const __m256i b_hi = _mm256_set_epi64x(
            static_cast<long long>(scalar_b[idx + 3] >> 64),
            static_cast<long long>(scalar_b[idx + 2] >> 64),
            static_cast<long long>(scalar_b[idx + 1] >> 64),
            static_cast<long long>(scalar_b[idx + 0] >> 64)
        );
        simd_b[i] = v4q128(b_lo, b_hi);
    }

    std::cout << "Verifying equivalence between scalar and SIMD datasets... ";
    for (size_t i = 0; i < N / 4; ++i) {
        const v4q128 res = v4q128::mul(simd_a[i], simd_b[i]);

        alignas(32) uint64_t res_lo[4];
        alignas(32) int64_t  res_hi[4];
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_lo), res.lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(res_hi), res.hi);

        for (int j = 0; j < 4; ++j) {
            const int128_t expected = scalar_mul_q64_64(scalar_a[i * 4 + j], scalar_b[i * 4 + j]);
            const int128_t actual = (static_cast<int128_t>(res_hi[j]) << 64) | res_lo[j];
            assert(expected == actual && "SIMD output does not match scalar output!");
        }
    }
    std::cout << "PASSED!\n\n";

    ankerl::nanobench::Bench bench;
    bench.title("Bulk 128-bit Q64.64 Multiplication");
    bench.unit("128bit-mul");
    bench.warmup(20);
    bench.epochs(100);

    bench.batch(N).run("Scalar 128-bit Mul", [&] {
        for (size_t i = 0; i < N; ++i) {
            scalar_res[i] = scalar_mul_q64_64(scalar_a[i], scalar_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(scalar_res.data());
    });

    bench.batch(N).run("v4q128 AVX2 Mul (4-way)", [&] {
        for (size_t i = 0; i < N / 4; ++i) {
            simd_res[i] = v4q128::mul(simd_a[i], simd_b[i]);
        }
        ankerl::nanobench::doNotOptimizeAway(simd_res.data());
    });

    return 0;
}
