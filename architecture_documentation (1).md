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