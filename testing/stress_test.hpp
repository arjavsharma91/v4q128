#include <vector>
#include <utility>
#include <cstdint>

struct ScalarQ128 {
    uint64_t lo;
    int64_t hi;
};

std::vector<std::pair<ScalarQ128, ScalarQ128>> get_multiplication_stress_tests() {
    std::vector<std::pair<ScalarQ128, ScalarQ128>> stress_tests = {
        { {0, 0}, {0, 0} },                                             // 0 * 0
        { {0, 1}, {0, 5} },                                             // 1 * 5 = 5
        { {0, 1}, {0, -5} },                                            // 1 * -5 = -5
        { {0, -1}, {0, -5} },                                           // -1 * -5 = 5
        { {0x8000000000000000ULL, 0}, {0x8000000000000000ULL, 0} },     // 0.5 * 0.5 = 0.25 (0x4000000000000000)
        { {0x8000000000000000ULL, -2}, {0, 2} },                        // -1.5 * 2.0 = -3.0
        { {0xFFFFFFFF00000000ULL, 0}, {0x00000000FFFFFFFFULL, 0} },     // Cross-limb boundary splits
        { {0x00000000FFFFFFFFULL, 0}, {0x00000000FFFFFFFFULL, 0} },     // Lower 32-bit square
        { {0xFFFFFFFF00000000ULL, 0}, {0xFFFFFFFF00000000ULL, 0} },     // Upper 32-bit square
        { {0xFFFFFFFFFFFFFFFFULL, 0x00000000FFFFFFFFLL}, {0x00000000FFFFFFFFULL, 1} },
        { {0x00000000FFFFFFFFULL, -1}, {0xFFFFFFFFFFFFFFFFULL, -1} },
        { {0xFFFFFFFFFFFFFFFFULL, 0}, {0xFFFFFFFFFFFFFFFFULL, 0} },     // Max fraction squaring (forces multi-bit fractional carry)
        { {0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFLL}, {0, 2} },      // Forces carry out of bit 63 to rewrite high bits
        { {0xFFFFFFFFFFFFFFFFULL, 0x3FFFFFFFFFFFFFFFLL}, {0xFFFFFFFFFFFFFFFFULL, 0x3FFFFFFFFFFFFFFFLL} },
        { {0xFFFFFFFFFFFFFFFFULL, -0x3FFFFFFFFFFFFFFFLL}, {0xFFFFFFFFFFFFFFFFULL, 0x3FFFFFFFFFFFFFFFLL} },
        { {0, static_cast<int64_t>(0x8000000000000000ULL)}, {0, 1} },   // INT64_MIN * 1 (Tests strict two's complement bounds)
        { {0, static_cast<int64_t>(0x8000000000000000ULL)}, {0, -1} },  // INT64_MIN * -1
        { {0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFLL}, {0, 1} },      // INT64_MAX with full fraction * 1
        { {0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFLL}, {0, -1} },     // INT64_MAX with full fraction * -1
        { {0, 0x00000000FFFFFFFFLL}, {0, 0x00000000FFFFFFFFLL} },       // Massive integer squaring
        { {0xFFFFFFFFFFFFFFFFULL, 0}, {0, 0x00000000FFFFFFFFLL} }       // Max fraction scaled by massive integer range
    };
    while (stress_tests.size() % 4 != 0) {
        stress_tests.push_back({ {0, 0}, {0, 0} });
    }

    return stress_tests;
}
