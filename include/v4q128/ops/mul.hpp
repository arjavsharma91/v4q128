#pragma <once>

#if defined (_MSC_VER)
    #define V4Q128_INLINE __forced_inline
#elif defined (__GNUC__) || defined (__clang__)
    #define V4Q128_INLINE __attribute__((always_inline))
#else
    #define V4Q128_INLINE inline
#endif

namespace v4q128 {
