#pragma once

/**
 * Compiler optimization hints for performance-critical code paths.
 *
 * These macros provide portable abstractions for compiler-specific
 * optimization directives and branch prediction hints.
 */

// ============================================================================
// Branch Prediction Hints
// ============================================================================
// Note: C++20 [[likely]]/[[unlikely]] apply to statements, not expressions.
// For expression-based usage in if(), we use __builtin_expect universally.

#if defined(__GNUC__) || defined(__clang__)
    // GCC/Clang builtin - works in expressions like if(NPERF_LIKELY(x))
    #define NPERF_LIKELY(x)   (__builtin_expect(!!(x), 1))
    #define NPERF_UNLIKELY(x) (__builtin_expect(!!(x), 0))
#else
    // No-op fallback
    #define NPERF_LIKELY(x)   (x)
    #define NPERF_UNLIKELY(x) (x)
#endif

// ============================================================================
// Function Attributes
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
    // Force inline even at -O0
    #define NPERF_FORCE_INLINE __attribute__((always_inline)) inline
    // Hint that function is pure (no side effects, only depends on args)
    #define NPERF_PURE __attribute__((pure))
    // Hint that function is const (pure + doesn't read global memory)
    #define NPERF_CONST __attribute__((const))
    // Hot path hint - optimize for speed over size
    #define NPERF_HOT __attribute__((hot))
    // Cold path hint - optimize for size, unlikely to be called
    #define NPERF_COLD __attribute__((cold))
    // Restrict pointer aliasing
    #define NPERF_RESTRICT __restrict__
    // Prefetch memory
    #define NPERF_PREFETCH(addr, rw, locality) __builtin_prefetch(addr, rw, locality)
    // Assume condition is true for optimizer
    #define NPERF_ASSUME(cond) do { if (!(cond)) __builtin_unreachable(); } while(0)
#elif defined(_MSC_VER)
    #define NPERF_FORCE_INLINE __forceinline
    #define NPERF_PURE
    #define NPERF_CONST
    #define NPERF_HOT
    #define NPERF_COLD
    #define NPERF_RESTRICT __restrict
    #define NPERF_PREFETCH(addr, rw, locality)
    #define NPERF_ASSUME(cond) __assume(cond)
#else
    #define NPERF_FORCE_INLINE inline
    #define NPERF_PURE
    #define NPERF_CONST
    #define NPERF_HOT
    #define NPERF_COLD
    #define NPERF_RESTRICT
    #define NPERF_PREFETCH(addr, rw, locality)
    #define NPERF_ASSUME(cond)
#endif

// ============================================================================
// Loop Optimization Pragmas
// ============================================================================

// GCC/Clang loop unrolling
#if defined(__GNUC__) && !defined(__clang__)
    #define NPERF_UNROLL(n) _Pragma("GCC unroll " #n)
    #define NPERF_UNROLL_FULL _Pragma("GCC unroll 128")
#elif defined(__clang__)
    #define NPERF_UNROLL(n) _Pragma("clang loop unroll_count(" #n ")")
    #define NPERF_UNROLL_FULL _Pragma("clang loop unroll(full)")
#else
    #define NPERF_UNROLL(n)
    #define NPERF_UNROLL_FULL
#endif

// Vectorization hints
#if defined(__GNUC__) && !defined(__clang__)
    // GCC: Tell compiler loop iterations are independent (no loop-carried deps)
    #define NPERF_IVDEP _Pragma("GCC ivdep")
    // Hint that loop should be vectorized
    #define NPERF_VECTORIZE _Pragma("omp simd")
    // Vectorize with reduction
    #define NPERF_VECTORIZE_REDUCTION(op, var) _Pragma("omp simd reduction(" #op ":" #var ")")
#elif defined(__clang__)
    // Clang: Use clang-specific loop pragmas
    #define NPERF_IVDEP _Pragma("clang loop vectorize(assume_safety)")
    #define NPERF_VECTORIZE _Pragma("clang loop vectorize(enable)")
    #define NPERF_VECTORIZE_REDUCTION(op, var) _Pragma("clang loop vectorize(enable)")
#else
    #define NPERF_IVDEP
    #define NPERF_VECTORIZE
    #define NPERF_VECTORIZE_REDUCTION(op, var)
#endif

// ============================================================================
// Memory Alignment
// ============================================================================

// Aligned allocation hint (bytes)
#define NPERF_CACHE_LINE_SIZE 64

// Aligned type declaration
#define NPERF_ALIGNAS(bytes) alignas(bytes)
#define NPERF_CACHE_ALIGNED alignas(NPERF_CACHE_LINE_SIZE)

// ============================================================================
// Compile-Time Constants
// ============================================================================

namespace nperf {

// Pre-computed constants for common conversions
inline constexpr double BYTES_TO_GB = 1.0 / (1024.0 * 1024.0 * 1024.0);
inline constexpr double US_TO_SECONDS = 1e-6;
inline constexpr double SECONDS_TO_US = 1e6;

// Prefetch locality hints
enum class PrefetchLocality {
    None = 0,      // No temporal locality (use once)
    Low = 1,       // Low temporal locality
    Medium = 2,    // Medium temporal locality
    High = 3       // High temporal locality (keep in all cache levels)
};

} // namespace nperf

// ============================================================================
// Utility Macros
// ============================================================================

// Assert assumption to compiler (debug: assert, release: assume)
#ifdef NDEBUG
    #define NPERF_ASSERT_ASSUME(cond) NPERF_ASSUME(cond)
#else
    #include <cassert>
    #define NPERF_ASSERT_ASSUME(cond) assert(cond)
#endif
