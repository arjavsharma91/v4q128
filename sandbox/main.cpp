#include <iostream>
#include <cstdint>
#include <immintrin.h>
#include <include/v4q128/v4q128.hpp>

using namespace v4q128;

void print_v4q128(const char* title, v4q128 v) {
    alignas(32) uint64_t lo[4];
    alignas(32) int64_t  hi[4];

    _mm256_store_si256(reinterpret_cast<__m256i*>(lo), v.lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(hi), v.hi);

    std::cout << title << ":\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Lane " << i << ": int = " << hi[i] 
                  << ", frac = 0x" << std::hex << lo[i] << std::dec << "\n";
    }
    std::cout << "\n";
}

int main() {
    v4q128 a = v4q128::set1(0x8000000000000000ULL, 2);
    v4q128 b = v4q128::set1(0x0ULL, -3);

    print_v4q128("Vector A", a);
    print_v4q128("Vector B", b);

    v4q128 sum = a + b;
    print_v4q128("A + B", sum);

    v4q128 prod = a * b;
    print_v4q128("A * B", prod);

    return 0;
}
