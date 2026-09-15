#pragma once

#include <immintrin.h>
#include <cstdint>
#include <v4q128/core/storage.hpp>

#if defined(_MSC_VER)
    #define V4Q128_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define V4Q128_INLINE inline __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {

/**
 * @brief Transposes two 256-bit AVX2 vectors from AoS format [loN, hiN] into Dual-SoA registers.
 */
[[nodiscard]] V4Q128_INLINE v4q128 transpose_aos_to_soa(__m256i raw0, __m256i raw1) noexcept {
    __m256i t0 = _mm256_permute2x128_si256(raw0, raw1, 0x20);
    __m256i t1 = _mm256_permute2x128_si256(raw0, raw1, 0x31);

    __m256i lo_soa = _mm256_unpacklo_epi64(t0, t1);
    __m256i hi_soa = _mm256_unpackhi_epi64(t0, t1);

    return v4q128(lo_soa, hi_soa);
}

/**
 * @brief Transposes a Dual-SoA v4q128 vector back into AoS registers for memory storage.
 */
V4Q128_INLINE void transpose_soa_to_aos(v4q128 vec, __m256i& out_raw0, __m256i& out_raw1) noexcept {
    __m256i u_lo = _mm256_unpacklo_epi64(vec.lo, vec.hi);
    __m256i u_hi = _mm256_unpackhi_epi64(vec.lo, vec.hi);

    out_raw0 = _mm256_permute2x128_si256(u_lo, u_hi, 0x20);
    out_raw1 = _mm256_permute2x128_si256(u_lo, u_hi, 0x31);
}

/**
 * @brief Load 4 Q64.64 numbers from 32-byte aligned memory into Dual-SoA registers.
 */
[[nodiscard]] V4Q128_INLINE v4q128 load_aligned(const void* ptr) noexcept {
    const auto* base = static_cast<const __m256i*>(ptr);
    __m256i raw0 = _mm256_load_si256(base);
    __m256i raw1 = _mm256_load_si256(base + 1);
    return transpose_aos_to_soa(raw0, raw1);
}

/**
 * @brief Load 4 Q64.64 numbers from unaligned memory into Dual-SoA registers.
 */
[[nodiscard]] V4Q128_INLINE v4q128 load_unaligned(const void* ptr) noexcept {
    const auto* base = static_cast<const __m256i*>(ptr);
    __m256i raw0 = _mm256_loadu_si256(base);
    __m256i raw1 = _mm256_loadu_si256(base + 1);
    return transpose_aos_to_soa(raw0, raw1);
}

/**
 * @brief Store 4 Dual-SoA Q64.64 numbers to 32-byte aligned memory in AoS layout.
 */
V4Q128_INLINE void store_aligned(void* ptr, v4q128 vec) noexcept {
    auto* base = static_cast<__m256i*>(ptr);
    __m256i raw0, raw1;
    transpose_soa_to_aos(vec, raw0, raw1);
    _mm256_store_si256(base, raw0);
    _mm256_store_si256(base + 1, raw1);
}

/**
 * @brief Store 4 Dual-SoA Q64.64 numbers to unaligned memory in AoS layout.
 */
V4Q128_INLINE void store_unaligned(void* ptr, v4q128 vec) noexcept {
    auto* base = static_cast<__m256i*>(ptr);
    __m256i raw0, raw1;
    transpose_soa_to_aos(vec, raw0, raw1);
    _mm256_storeu_si256(base, raw0);
    _mm256_storeu_si256(base + 1, raw1);
}

/**
 * @brief Non-temporal store (bypasses CPU caches directly to system RAM).
 */
V4Q128_INLINE void store_stream(void* ptr, v4q128 vec) noexcept {
    auto* base = static_cast<__m256i*>(ptr);
    __m256i raw0, raw1;
    transpose_soa_to_aos(vec, raw0, raw1);
    _mm256_stream_si256(base, raw0);
    _mm256_stream_si256(base + 1, raw1);
}

} // namespace v4q128

#undef V4Q128_INLINE
