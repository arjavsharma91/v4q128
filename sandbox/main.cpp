#include <iostream>
#include <cstdint>
#include <immintrin.h>
#include <v4q128/v4q128.hpp>

// Use a clean type alias to avoid namespace vs struct name ambiguity
using V4Q = v4q128::v4q128;

void print_v4q128(const char* title, V4Q v) {
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
    V4Q a = V4Q::set1(0x8000000000000000ULL, 2);
    V4Q b = V4Q::set1(0x0ULL, -3);

    print_v4q128("Vector A", a);
    print_v4q128("Vector B", b);

    V4Q sum = v4q128::add(a, b);
    print_v4q128("A + B", sum);

    V4Q prod = v4q128::mul(a, b);
    print_v4q128("A * B", prod);

    return 0;
}
