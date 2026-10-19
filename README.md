# v4q128

An experimental, failed 4-lane AVX2 SIMD Q64.64 fixed-point library. Archived after a month of benchmarking proved manual vectorization is slower than compiler-optimized scalar code for 128-bit fixed-point math.

## Performance Findings

| Operation | AVX2 SIMD vs. Scalar | Result |
| :--- | :--- | :--- |
| **Addition / Subtraction** | 0.95x | Compiler auto-vectorization and scalar loops win due to register pressure. |
| **Multiplication** | 1.3x | Insufficient gain to justify dual-limb SoA complexity. |
| **Variable Shifts (`shl`, `sra`)** | Approximately 1.0x | High complexity for zero net gain. |

## Why It Failed

1. **AVX2 Lane-Crossing Overhead:** AVX2 lacks native 128-bit integer registers and cross-lane carry support. Emulating 128-bit arithmetic using split `hi`/`lo` limbs forces heavy shuffle, `vpermq`, and unpack overhead that neutralizes SIMD throughput.
2. **Compiler Superiority:** Modern LLVM/GCC auto-vectorizers crush manual intrinsic implementations on basic arithmetic by avoiding register-allocation bottlenecks.

## Salvageable Components

* **128-Bit Variable Shift Engine:** Standalone branchless carry-handling (`shl_var`, `srl_var`, `sra_var`).
* **Float/Double Conversion Pipeline:** High-precision scaling logic utilizing `2^64` bias factors and unsigned sign-bit manipulation.
