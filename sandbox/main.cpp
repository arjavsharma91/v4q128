#include <v4q128/v4q128.hpp>
#include <v4q128/core/storage.hpp>
#include <v4q128/io/transpose.hpp>
#include <iostream>

struct Q64_64 {
    uint64_t lo;
    int64_t hi;
};

int main() {
    // 4 sample Q64.64 numbers in AoS RAM format
    alignas(32) Q64_64 input_data[4] = {
        { 0x1111111111111111ULL, 100 },
        { 0x2222222222222222ULL, 200 },
        { 0x3333333333333333ULL, 300 },
        { 0x4444444444444444ULL, 400 }
    };

    // 1. Load AoS data from RAM -> Transpose to Dual-SoA Registers
    v4q128::v4q128 vec = v4q128::load_aligned(input_data);

    // 2. Store Dual-SoA Registers -> Transpose back to AoS RAM
    alignas(32) Q64_64 output_data[4];
    v4q128::store_aligned(output_data, vec);

    // 3. Assert bit-exact identity
    bool passed = true;
    for (int i = 0; i < 4; ++i) {
        if (input_data[i].lo != output_data[i].lo || input_data[i].hi != output_data[i].hi) {
            passed = false;
        }
    }

    v4q128::v4q128 vec_add_test = v4q128::load_aligned(input_data);
    

    if (passed) {
        std::cout << "Phase 1 Matrix Transpose Test Passed!\n";
        std::cout << "Lane 0: lo=0x" << std::hex << output_data[0].lo
                  << std::dec << ", hi=" << output_data[0].hi << "\n";
        std::cout << "Lane 3: lo=0x" << std::hex << output_data[3].lo
                  << std::dec << ", hi=" << output_data[3].hi << "\n";
    } else {
        std::cout << "ERROR: Transpose mismatch!\n";
    }

    return 0;
}
