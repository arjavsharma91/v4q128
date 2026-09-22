
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>
#include <string_view>
#include <v4q128/v4q128.hpp>

struct Q128Scalar {
    uint64_t frac;
    int64_t integer;
};

// Scalar reference implementation
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

static bool verify_vector(Q128Scalar a[4], Q128Scalar b[4], bool log_errors = true) {
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

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        Q128Scalar expected = ref_mul(a[i], b[i]);
        if (res_frac[i] != expected.frac || res_int[i] != expected.integer) {
            ok = false;
            if (log_errors) {
                std::cerr << "  [FAIL] Lane " << i << ":\n"
                          << "    A: int=" << a[i].integer << ", frac=0x" << std::hex << a[i].frac << "\n"
                          << "    B: int=" << b[i].integer << ", frac=0x" << b[i].frac << "\n"
                          << "    Got:      int=" << res_int[i] << ", frac=0x" << res_frac[i] << "\n"
                          << "    Expected: int=" << expected.integer << ", frac=0x" << expected.frac << std::dec << "\n";
            }
        }
    }
    return ok;
}

static int run_deterministic_tests() {
    std::cout << "========================================\n";
    std::cout << " Running Deterministic Edge-Case Suite \n";
    std::cout << "========================================\n";

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

    uint64_t total_cases = 0;
    uint64_t passed_cases = 0;
    uint64_t failed_cases = 0;

    for (const auto& a : edge_cases) {
        for (const auto& b : edge_cases) {
            total_cases++;
            Q128Scalar vec_a[4] = {a, a, a, a};
            Q128Scalar vec_b[4] = {b, b, b, b};
            
            // Log details only for the first 10 failures to avoid log spam
            bool show_log = (failed_cases < 10);
            if (verify_vector(vec_a, vec_b, show_log)) {
                passed_cases++;
            } else {
                failed_cases++;
            }
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << " Deterministic Summary:\n";
    std::cout << "   Total Vector Tests Run: " << total_cases << "\n";
    std::cout << "   Passed:                 " << passed_cases << "\n";
    std::cout << "   Failed:                 " << failed_cases << "\n";
    std::cout << "----------------------------------------\n";

    return (failed_cases == 0) ? 0 : 1;
}

static int run_random_stress_tests() {
    std::cout << "========================================\n";
    std::cout << " Running Randomized Stress Suite (1M)  \n";
    std::cout << "========================================\n";

    std::mt19937_64 rng(1337);
    std::uniform_int_distribution<uint64_t> dist_u64;
    std::uniform_int_distribution<int64_t>  dist_i64;

    constexpr uint64_t TOTAL_ITERS = 1'000'000;
    uint64_t passed_iters = 0;
    uint64_t failed_iters = 0;

    for (uint64_t iter = 0; iter < TOTAL_ITERS; ++iter) {
        Q128Scalar vec_a[4], vec_b[4];
        for (int i = 0; i < 4; ++i) {
            vec_a[i] = { dist_u64(rng), dist_i64(rng) };
            vec_b[i] = { dist_u64(rng), dist_i64(rng) };
        }

        bool show_log = (failed_iters < 5);
        if (verify_vector(vec_a, vec_b, show_log)) {
            passed_iters++;
        } else {
            failed_iters++;
        }
    }

    std::cout << "\n----------------------------------------\n";
    std::cout << " Randomized Summary:\n";
    std::cout << "   Total Iterations Run: " << TOTAL_ITERS << "\n";
    std::cout << "   Passed:               " << passed_iters << "\n";
    std::cout << "   Failed:               " << failed_iters << "\n";
    std::cout << "----------------------------------------\n";

    return (failed_iters == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1) {
        std::string_view arg(argv[1]);
        if (arg == "--random") {
            return run_random_stress_tests();
        }
        if (arg == "--deterministic") {
            return run_deterministic_tests();
        }
    }

    int ret1 = run_deterministic_tests();
    int ret2 = run_random_stress_tests();
    return (ret1 == 0 && ret2 == 0) ? 0 : 1;
}
