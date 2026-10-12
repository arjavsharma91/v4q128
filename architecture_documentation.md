# Architecture and Design

## 1. Memory Layout

The library uses a dual Structure-of-Arrays (SoA) layout that separates `lo` and `hi` 64-bit limbs across AVX2 registers.

* **Design Rationale:** x86_64 AVX2 lacks native 128-bit integer arithmetic instructions. Splitting 128-bit values into lower and upper 64-bit chunks enables direct use of native AVX2 64-bit integer operations (`_mm256_add_epi64`, `_mm256_sub_epi64`).

* **Cross-Lane Avoidance:** This structure prevents cross-lane permutations. Data transfers across 128-bit lane boundaries inside AVX2 registers incur high instruction latency; keeping limbs segregated within their respective lanes avoids this overhead.

## 2. Addition and Subtraction

Addition and subtraction primitives operate across the dual-limb structure using branchless bitwise logic.

* **Addition:**
  1. Compute the unsigned sum of the `lo` limbs.
  2. Apply the sign-bit mask to map unsigned sums to signed ranges.
  3. Detect overflow/carry in the `lo` limbs via wrap-around detection.
  4. If a carry occurs, adjust the `hi` limb by subtracting `1` (or adding a `-1` bitmask).

* **Subtraction:**
  1. Compute the difference of the `lo` limbs.
  2. Detect borrows by checking if the subtrahend `lo` limb exceeds the minuend `lo` limb.
  3. If a borrow occurs, adjust the `hi` limb by adding a `-1` bitmask.

## 3. Unary Operations (`neg`, `abs`, `sign`)

Unary operations are implemented branchlessly using native 64-bit AVX2 primitives:

* **`neg(v)`:** Negates a 128-bit value using two's complement arithmetic ($\sim X + 1$). The lower limb computes zero minus `lo`. The upper limb inverts `hi` (`not_hi`) and subtracts an `is_zero` mask (all ones if `lo == 0`, zero otherwise) to handle carry propagation from the lower limb without branches.

* **`abs(v)`:** Isolates the sign of `v` using `_mm256_cmpgt_epi64` against the integer limb (`hi`). It computes `neg(v)` once and uses `_mm256_blendv_epi8` to select between original and negated limbs based on the sign mask.

* **`sign(v)`:** Extracts the SIMD sign mask from the integer limb (`hi`), returning all ones for negative numbers and all zeros for non-negative numbers.

## 4. Multiplication and Instruction-Level Parallelism (ILP)

Multiplication and register allocation are optimized to maximize throughput and avoid execution stalls.

* **32-Bit Limb Splitting and Cross-Multiplication:** Multiplication breaks down 64-bit limbs into 32-bit sub-limbs to perform cross-multiplications using native instructions, ensuring precise product accumulation without widening bottlenecks.

* **Deliberate Register Spilling:** The register allocator intentionally permits spilling during complex multiplication routines.

* **Store-to-Load Forwarding Trade-Off:** Rather than stalling execution on a serial carry chain dependency, spilled data resides in the L1 store buffer. This enables store-to-load forwarding at a cost of only 5 to 7 cycles, trading minor spill overhead for high Instruction-Level Parallelism (ILP).

## 5. Division Strategy

AVX2 fixed-point division is inefficient because x86 vector registers lack native 128-bit or 64-bit integer division instructions. To bypass this hardware limitation, the library implements a dual-path division approach:

* **`div_approx` (Approximate Path):** Converts 128-bit fixed-point values into double-precision floating-point vectors (`__m256d`), performs hardware floating-point division using AVX2 units, and converts back. This trades away bit-exact accuracy for significant speed gains.
* **`div_exact` (Exact Path):** Extracts lanes and uses a branchless 128-bit scalar implementation (`__int128`) to perform exact integer division, guaranteeing bit-exact accuracy without rounding drift.

## 6. Comparison Operations

Comparison primitives return either a full bitmask (`0xFFFFFFFFFFFFFFFF`) or zero (`0`) across AVX2 lanes. All comparison operators are built as logical extensions of equality and greater-than:

* **Equality (`==`):** Compares both `lo` and `hi` limbs independently for equality, then combines the results using a bitwise `AND` instruction (`_mm256_and_si256`).
* **Greater Than (`>`):** Implemented hierarchically:
  1. Checks if the upper integer limb (`hi`) is strictly greater.
  2. Applies the sign-bit mask (MSB flip) to the lower fractional limbs (`lo`) to ensure correct signed comparison.
  3. Combines results using a bitwise `OR`: true if `hi > hi_b`, or if `hi == hi_b` and the adjusted `lo > lo_b`.
* **Other Comparisons (`<`, `<=`, `>=`, `!=`):** Implemented as logical inversions and combinations of the base `==` and `>` operations.
