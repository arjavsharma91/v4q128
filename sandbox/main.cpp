#include <v4q128/v4q128.hpp>
#include <iostream>

int main() {
    // AVX2 hardware test instruction
    __m256i reg = _mm256_set1_epi64x(0x123456789ABCDEF0ULL);
    __m256i shifted = _mm256_srli_epi64(reg, 4);

    alignas(32) uint64_t result[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(result), shifted);

    std::cout << "Hardware Sanity Check Passed!\n";
    std::cout << "Shifted sample lane: 0x" << std::hex << result[0] << "\n";
    return 0;
}
