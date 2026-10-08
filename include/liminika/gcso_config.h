// name: include/liminika/gcso_config.h
// SPDX-License-Identifier: MIT OR Apache-2.0

#ifndef LIMINIKA_GCSO_CONFIG_H
#define LIMINIKA_GCSO_CONFIG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
    #define GCSO_EXTERN_C extern "C"
    #define GCSO_EXTERN_C_BEGIN extern "C" {
    #define GCSO_EXTERN_C_END }
#else
    #define GCSO_EXTERN_C
    #define GCSO_EXTERN_C_BEGIN
    #define GCSO_EXTERN_C_END
#endif

/* Unified GCSO C-ABI Versioning (0.1.1) */
#define GCSO_ABI_VERSION_MAJOR 0
#define GCSO_ABI_VERSION_MINOR 1
#define GCSO_ABI_VERSION_PATCH 1
#define GCSO_ABI_VERSION_HEX   0x00000101
#define GCSO_ABI_VERSION_STRING "0.1.1"

/* Dynamic Link Library Import/Export Macros */
#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(GCSO_BUILD_DLL)
        #define GCSO_API __declspec(dllexport)
    #elif defined(GCSO_USE_DLL)
        #define GCSO_API __declspec(dllimport)
    #else
        #define GCSO_API
    #endif
    #define GCSO_CALL __cdecl
#else
    #if defined(__GNUC__) && __GNUC__ >= 4
        #define GCSO_API __attribute__((visibility("default")))
    #else
        #define GCSO_API
    #endif
    #define GCSO_CALL
#endif

/* Exception-Safety Macro for FFI Boundaries */
#ifdef __cplusplus
    #define GCSO_NOEXCEPT noexcept
#else
    #define GCSO_NOEXCEPT
#endif

/* Warn Unused Result Macro */
#if defined(__cplusplus) && __cplusplus >= 201703L
    #define GCSO_NODISCARD [[nodiscard]]
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
    #define GCSO_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER)
    #define GCSO_NODISCARD _Check_return_
#else
    #define GCSO_NODISCARD
#endif

/* Branch Prediction Optimization Hints */
#if defined(__GNUC__) || defined(__clang__)
    #define GCSO_LIKELY(x) __builtin_expect(!!(x), 1)
    #define GCSO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define GCSO_LIKELY(x) (x)
    #define GCSO_UNLIKELY(x) (x)
#endif

/* Explicit Memory Alignment & Query Macros */
#if defined(__cplusplus)
    #define GCSO_ALIGNAS(n) alignas(n)
    #define GCSO_ALIGNOF(type) alignof(type)
#elif defined(_MSC_VER)
    #define GCSO_ALIGNAS(n) __declspec(align(n))
    #define GCSO_ALIGNOF(type) __alignof(type)
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_ALIGNAS(n) __attribute__((aligned(n)))
    #define GCSO_ALIGNOF(type) __alignof__(type)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GCSO_ALIGNAS(n) _Alignas(n)
    #define GCSO_ALIGNOF(type) _Alignof(type)
#else
    #define GCSO_ALIGNAS(n)
    #define GCSO_ALIGNOF(type) sizeof(type)
#endif

/* Pointer Non-Aliasing Restrict Qualifier */
#if defined(_MSC_VER)
    #define GCSO_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_RESTRICT __restrict__
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define GCSO_RESTRICT restrict
#else
    #define GCSO_RESTRICT
#endif

/* Compile-time Static Assertion Macro */
#if defined(__cplusplus)
    #if __cplusplus >= 201103L
        #define GCSO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
    #endif
#elif defined(__STDC_VERSION__)
    #if __STDC_VERSION__ >= 202311L
        #define GCSO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
    #elif __STDC_VERSION__ >= 201112L
        #define GCSO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
    #endif
#endif

#ifndef GCSO_STATIC_ASSERT
    #define GCSO_CONCAT_IMPL(x, y) x##y
    #define GCSO_CONCAT(x, y) GCSO_CONCAT_IMPL(x, y)
    #define GCSO_STATIC_ASSERT(cond, msg) \
        typedef char GCSO_CONCAT(gcso_static_assert_typedef_, __LINE__)[(cond) ? 1 : -1]
#endif

#endif // LIMINIKA_GCSO_CONFIG_H