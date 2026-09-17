#include <v4q128/v4q128.hpp>
#include <v4q128/core/storage.hpp>
#include <v4q128/io/transpose.hpp>
#include <iostream>
#include <csdtint>
#include <iomanip>

int main() {
    using namespace v4q128;

    v4q128 a_carry = v4q128::set1(0xFFFFFFFFFFFFFFFEULL, 10);
    v4q128 b_carry = v4q128::set1(0x0000000000000002ULL, 5);
    v4q128 sum = a_carry + b_carry

    print_v4q128("ADDITION WITH CARRY, expected: hi=16, lo=0x0000000000000000)", sum);

    v4q128 a_borrow = v4q128::set1(0x0000000000000000ULL, 20);
    v4q128 b_borrow = v4q128::set1(0x0000000000000001ULL, 5);
    v4q128 diff = a_borrow - b_borrow;

    print_v4q128("SUBTRACTION WITH BORROW, expected: hi = 14 lo = 0xFFFFFFFFFFFFFFFF", diff);
