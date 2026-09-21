#include <iostream>
#include <cstdint>
#include <cassert>
#include <random>
#include <array>
#include <v4q128/v4q128.hpp>

struct Q128Scalar {
    uint64_t frac;
    int64_t integer;
};

// Scalar ground truth for Q64.64 multiplication
static Q128Scalar ref_mul(Q128Scalar a, Q128Scalar b) noexcept {
    unsigned __int128 a_lo = a.frac;
    unsigned __int128 a_hi = static_cast<uint64_t>(a.integer);
    unsigned __int128 b_lo = b.frac;
    unsigned __int128 b_hi = static_cast<uint64_t>(b.integer);

    unsigned __int128 p00 = a_lo * b_lo;
    unsigned __int128 p01 = a_lo * b_hi;
    unsigned __int128 p10 = a_hi * b_lo;
    unsigned __int128 p11 = a_hi * b_hi;

    unsigned __int128 sum_mid = (p00 >> 64) + (p01 & 0xFFFFFFFFFFFFFFFFULL) + (p10 & 0xFFFFFFFFFFFFFFFFULL);
    uint64_t res_frac = static_cast<uint64_t>(sum_mid);

    unsigned __int128 res_hi_base = (sum_mid >> 64) + (p01 >> 64) + (p10 >> 64) + p11;

    if (a.integer < 0) {
        res_hi_base -= b_lo;
    }
    if (b.integer < 0) {
        res_hi_base -= a_lo;
    }

    return { res_frac, static_cast<int64_t>(res_hi_base) };
}

static void verify_mul(Q128Scalar a[4], Q128Scalar b[4]) {
    v4q128::v4q128 va = v4q128::v4q128::set(
        a[0].frac, a[0].integer,
        a[1].frac, a[1].integer,
        a[2].frac, a[2].integer,
        a[3].frac, a[3].integer
    );

    v4q128::v4q128 vb = v4q128::v4q128::set(
        b[0].frac, b[0].integer,
        b[1].frac, b[1].integer,
        b[2].frac, b[2].integer,
        b[3].frac, b[3].integer
    );

    v4q128::v4q128 vres = v4q128::mul(va, vb);

    alignas(32) uint64_t res_frac[4];
    alignas(32) int64_t  res_int[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(res_frac), vres.lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(res_int),  vres.hi);

    for (int i = 0; i < 4; ++i) {
        Q128Scalar expected = ref_mul(a[i], b[i]);
        if (res_frac[i] != expected.frac || res_int[i] != expected.integer) {
            std::cerr << "FAIL in lane " << i << ":\n"
                      << "  A: int=" << a[i].integer << ", frac=0x" << std::hex << a[i].frac << "\n"
                      << "  B: int=" << b[i].integer << ", frac=0x" << b[i].frac << "\n"
                      << "  Got:      int=" << res_int[i] << ", frac=0x" << res_frac[i] << "\n"
                      << "  Expected: int=" << expected.integer << ", frac=0x" << expected.frac << std::dec << "\n";
            std::exit(1);
        }
    }
}

int main() {
    std::cout << "Running deterministic edge-case tests...\n";

    // Standard edge values
    constexpr uint64_t F_ZERO = 0ULL;
    constexpr uint64_t F_MAX  = 0xFFFFFFFFFFFFFFFFULL;
    constexpr uint64_t F_ONE  = 1ULL;
    constexpr uint64_t F_ALT1 = 0x5555555555555555ULL;
    constexpr uint64_t F_ALT2 = 0xAAAAAAAAAAAAAAAAULL;

    constexpr int64_t I_ZERO = 0LL;
    constexpr int64_t I_ONE  = 1LL;
    constexpr int64_t I_NEG1 = -1LL;
    constexpr int64_t I_MAX  = 0x7FFFFFFFFFFFFFFFLL;
    constexpr int64_t I_MIN  = static_cast<int64_t>(0x8000000000000000ULL);

    const std::vector<Q128Scalar> edge_cases = {
        {F_ZERO, I_ZERO}, {F_ZERO, I_ONE}, {F_ZERO, I_NEG1}, {F_ZERO, I_MAX}, {F_ZERO, I_MIN},
        {F_ONE,  I_ZERO}, {F_ONE,  I_ONE}, {F_ONE,  I_NEG1}, {F_ONE,  I_MAX}, {F_ONE,  I_MIN},
        {F_MAX,  I_ZERO}, {F_MAX,  I_ONE}, {F_MAX,  I_NEG1}, {F_MAX,  I_MAX}, {F_MAX,  I_MIN},
        {F_ALT1, I_ZERO}, {F_ALT1, I_ONE}, {F_ALT1, I_NEG1}, {F_ALT2, I_MAX}, {F_ALT2, I_MIN}
    };

    // Test matrix across edge cases
    for (const auto& a : edge_cases) {
        for (const auto& b : edge_cases) {
            Q128Scalar vec_a[4] = {a, a, a, a};
            Q128Scalar vec_b[4] = {b, b, b, b};
            verify_mul(vec_a, vec_b);
        }
    }

    std::cout << "Deterministic edge-case tests passed!\n";
    std::cout << "Running 1,000,000 randomized edge-case stress iterations...\n";

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<uint64_t> dist_u64;
    std::uniform_int_distribution<int64_t>  dist_i64;

    for (int iter = 0; iter < 1'000'000; ++iter) {
        Q128Scalar vec_a[4], vec_b[4];
        for (int i = 0; i < 4; ++i) {
            vec_a[i] = { dist_u64(rng), dist_i64(rng) };
            vec_b[i] = { dist_u64(rng), dist_i64(rng) };
        }
        verify_mul(vec_a, vec_b);
    }

    std::cout << "All 1,000,000 randomized stress tests passed successfully!\n";
    return 0;
}
