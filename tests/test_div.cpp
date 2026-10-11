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

static Q128Scalar ref_div(Q128Scalar a, Q128Scalar b) noexcept {
    using int128_t  = __int128;
    using uint128_t = unsigned __int128;

    int128_t va = (static_cast<int128_t>(a.integer) << 64) | a.frac;
    int128_t vb = (static_cast<int128_t>(b.integer) << 64) | b.frac;

    int128_t mask_a = va >> 127;
    int128_t mask_b = vb >> 127;

    uint128_t ua = static_cast<uint128_t>((va ^ mask_a) - mask_a);
    uint128_t ub = static_cast<uint128_t>((vb ^ mask_b) - mask_b);

    int128_t res_mask = mask_a ^ mask_b;

    uint128_t ures = (ua << 64) / ub;

    int128_t res = static_cast<int128_t>((ures ^ res_mask) - res_mask);

    return {
        static_cast<uint64_t>(res),
        static_cast<int64_t>(res >> 64)
    };
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

    v4q128::v4q128 vres = v4q128::div_exact(va, vb);

    alignas(32) uint64_t res_frac[4];
    alignas(32) int64_t  res_int[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(res_frac), vres.lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(res_int),  vres.hi);

    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        Q128Scalar expected = ref_div(a[i], b[i]);
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
            if (b.frac == 0 && b.integer == 0) continue;

            total_cases++;
            Q128Scalar vec_a[4] = {a, a, a, a};
            Q128Scalar vec_b[4] = {b, b, b, b};
            
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
    std::cout << "    Total Vector Tests Run: " << total_cases << "\n";
    std::cout << "    Passed:                 " << passed_cases << "\n";
    std::cout << "    Failed:                 " << failed_cases << "\n";
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
            do {
                vec_b[i] = { dist_u64(rng), dist_i64(rng) };
            } while (vec_b[i].frac == 0 && vec_b[i].integer == 0); // Prevent div by zero
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
    std::cout << "    Total Iterations Run: " << TOTAL_ITERS << "\n";
    std::cout << "    Passed:               " << passed_iters << "\n";
    std::cout << "    Failed:               " << failed_iters << "\n";
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
