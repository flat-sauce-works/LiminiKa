// name: include/liminika/gcso_config.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_CONFIG_H
#define LIMINIKA_GCSO_CONFIG_H

#if defined(__cplusplus)
    #include <cstddef>
    #include <cstdint>
#else
    #include <stddef.h>
    #include <stdint.h>
#endif

// ===================================================================
// Extern "C" Linkage Specification
// ===================================================================
#ifdef __cplusplus
    #define GCSO_EXTERN_C_BEGIN extern "C" {
    #define GCSO_EXTERN_C_END   }
#else
    #define GCSO_EXTERN_C_BEGIN
    #define GCSO_EXTERN_C_END
#endif

// ===================================================================
// API Export / Import Visibility Definitions
// ===================================================================
#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(GCSO_BUILD_DLL)
        #define GCSO_API __declspec(dllexport)
    #elif defined(GCSO_USE_DLL)
        #define GCSO_API __declspec(dllimport)
    #else
        #define GCSO_API
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_API __attribute__((visibility("default")))
#else
    #define GCSO_API
#endif

// ===================================================================
// Calling Convention Definitions
// ===================================================================
#if defined(_WIN32) && defined(_M_IX86)
    #define GCSO_CALL __stdcall
#else
    #define GCSO_CALL
#endif

// ===================================================================
// Enum Underlying Type Specification Macro for C/C++ Compatibility
// ===================================================================
#if defined(__cplusplus)
    #define GCSO_ENUM_UNDERLYING(type) : type
#else
    #define GCSO_ENUM_UNDERLYING(type)
#endif

// ===================================================================
// Compiler Optimizations & Attribute Annotations
// ===================================================================

/* Non-aliasing restrict pointer annotation */
#if defined(__cplusplus)
    #if defined(_MSC_VER)
        #define GCSO_RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define GCSO_RESTRICT __restrict__
    #else
        #define GCSO_RESTRICT
    #endif
#else
    #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
        #define GCSO_RESTRICT restrict
    #elif defined(_MSC_VER)
        #define GCSO_RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define GCSO_RESTRICT __restrict__
    #else
        #define GCSO_RESTRICT
    #endif
#endif

/* Inline execution specification */
#if defined(__cplusplus)
    #define GCSO_INLINE inline
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define GCSO_INLINE inline
#elif defined(_MSC_VER)
    #define GCSO_INLINE __inline
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_INLINE __inline__
#else
    #define GCSO_INLINE
#endif

/* Always inline annotation for critical Hot Path routines */
#if defined(_MSC_VER)
    #define GCSO_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_FORCE_INLINE __attribute__((always_inline)) GCSO_INLINE
#else
    #define GCSO_FORCE_INLINE GCSO_INLINE
#endif

/* Branch prediction likelihood annotations */
#if defined(__GNUC__) || defined(__clang__)
    #define GCSO_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define GCSO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define GCSO_LIKELY(x)   (x)
    #define GCSO_UNLIKELY(x) (x)
#endif

/* Nodiscard attribute */
#if defined(__cplusplus) && __cplusplus >= 201703L
    #define GCSO_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && _MSC_VER >= 1700
    #define GCSO_NODISCARD _Check_return_
#else
    #define GCSO_NODISCARD
#endif

/* Noexcept attribute for C++ boundary safety */
#ifdef __cplusplus
    #define GCSO_NOEXCEPT noexcept
#else
    #define GCSO_NOEXCEPT
#endif

// ===================================================================
// Explicit Struct Memory Alignment Macros
// ===================================================================
#if defined(__cplusplus)
    #define GCSO_ALIGN_PREFIX(n) alignas(n)
    #define GCSO_ALIGN_SUFFIX(n)
    #define GCSO_ALIGNAS(n) alignas(n)
    #define GCSO_ALIGNOF(type) alignof(type)
#elif defined(_MSC_VER)
    #define GCSO_ALIGN_PREFIX(n) __declspec(align(n))
    #define GCSO_ALIGN_SUFFIX(n)
    #define GCSO_ALIGNAS(n) __declspec(align(n))
    #define GCSO_ALIGNOF(type) __alignof(type)
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_ALIGN_PREFIX(n)
    #define GCSO_ALIGN_SUFFIX(n) __attribute__((aligned(n)))
    #define GCSO_ALIGNAS(n) __attribute__((aligned(n)))
    #define GCSO_ALIGNOF(type) __alignof__(type)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GCSO_ALIGN_PREFIX(n)
    #define GCSO_ALIGN_SUFFIX(n) _Alignas(n)
    #define GCSO_ALIGNAS(n) _Alignas(n)
    #define GCSO_ALIGNOF(type) _Alignof(type)
#else
    #define GCSO_ALIGN_PREFIX(n)
    #define GCSO_ALIGN_SUFFIX(n)
    #define GCSO_ALIGNAS(n)
    #define GCSO_ALIGNOF(type) sizeof(type)
#endif

// ===================================================================
// Static Assertion Compatibility Macro
// ===================================================================
#if defined(__cplusplus) && __cplusplus >= 201103L
    #define GCSO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GCSO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#elif defined(_MSC_VER)
    #define GCSO_STATIC_ASSERT(cond, msg) typedef char gcso_static_assert_##__LINE__[(cond) ? 1 : -1]
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_STATIC_ASSERT(cond, msg) typedef char gcso_static_assert_##__LINE__[(cond) ? 1 : -1] __attribute__((unused))
#else
    #define GCSO_STATIC_ASSERT(cond, msg) typedef char gcso_static_assert_##__LINE__[(cond) ? 1 : -1]
#endif

#endif /* LIMINIKA_GCSO_CONFIG_H */