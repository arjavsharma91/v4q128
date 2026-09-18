#include <v4q128/v4q128.hpp>
#include <v4q128/core/storage.hpp>
#include <v4q128/io/transpose.hpp>
#include <iostream>
#include <csdtint>
#include <iomanip>

void print_v4q128(const char* title, v4q128 v) {
    alignas(32) uint64_t lo[4];
    alignas(32) int64_t hi[4];

    _mm256_store_si256(reinterpret_cast<__m256i*>(lo), v.lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(hi), v.hi);

    std::cout << "=== " << title << " ===\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Lane  " << i << ": hi = " << std::setw(4) << hi[i] << " | lo = 0x" << std::hex << std::setfill('0') << std::setw(16) << lo[i] << std::dec << "\n";

    cout << "\n";

int main() {
    using namespace v4q128;

    v4q128 b_carry = v4q128::set1(0x0000000000000002ULL, 5);
    v4q128 sum = a_carry + b_carry;

    print_v4q128("ADDITION WITH CARRY (Expected: hi=16, lo=0x0000000000000000)", sum);

    v4q128 a_borrow = v4q128::set1(0x0000000000000000ULL, 20);
    v4q128 b_borrow = v4q128::set1(0x0000000000000001ULL, 5);
    v4q128 diff = a_borrow - b_borrow;

    print_v4q128("SUBTRACTION WITH BORROW (Expected: hi=14, lo=0xFFFFFFFFFFFFFFFF)", diff);

    v4q128 x = v4q128::set1(0x0000000000003000ULL, 100);
    v4q128 y = v4q128::set1(0x0000000000001000ULL, 50);

    x += y;
    print_v4q128("COMPOUND ADD += (Expected: hi=150, lo=0x0000000000004000)", x);

    v4q128 neg_x = -x;
    print_v4q128("NEGATION -x (Expected: hi=-151, lo=0xFFFFFFFFFFFFC000)", neg_x);

    return 0;
}
