#include <vector>
#include <utility>
#include <random>
#include <cstdint>

struct ScalarQ128 {
    uint64_t lo;
    int64_t hi;
};

std::vector<std::pair<ScalarQ128, ScalarQ128>> generate_random_multiplication_tests(size_t num_pairs = 100000) {
    std::vector<std::pair<ScalarQ128, ScalarQ128>> random_tests;
    random_tests.reserve(num_pairs + 4);
    std::mt19937_64 rng(1337); 

    std::uniform_int_distribution<uint64_t> dist_frac(0, 0xFFFFFFFFFFFFFFFFULL);
    std::uniform_int_distribution<int64_t> dist_int_wide(-10000000000LL, 10000000000LL);
    std::uniform_int_distribution<int64_t> dist_int_extreme(-9223372036854775807LL, 9223372036854775807LL);

    for (size_t i = 0; i < num_pairs; ++i) {
        ScalarQ128 a, b;
        
        a.lo = dist_frac(rng);
        b.lo = dist_frac(rng);

        if (i % 10 == 0) {
            a.hi = dist_int_extreme(rng);
            b.hi = dist_int_extreme(rng);
        } else {
            a.hi = dist_int_wide(rng);
            b.hi = dist_int_wide(rng);
        }

        random_tests.push_back({a, b});
    }

    while (random_tests.size() % 4 != 0) {
        random_tests.push_back({ {0, 0}, {0, 0} });
    }

    return random_tests;
}
