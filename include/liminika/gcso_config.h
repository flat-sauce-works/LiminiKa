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

#ifdef __cplusplus
    #define GCSO_NOEXCEPT noexcept
#else
    #define GCSO_NOEXCEPT
#endif

#if defined(__cplusplus) && __cplusplus >= 201703L
    #define GCSO_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_NODISCARD __attribute__((warn_unused_result))
#else
    #define GCSO_NODISCARD
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define GCSO_LIKELY(x) __builtin_expect(!!(x), 1)
    #define GCSO_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define GCSO_LIKELY(x) (x)
    #define GCSO_UNLIKELY(x) (x)
#endif

#ifdef __cplusplus
    #define GCSO_ALIGNAS(n) alignas(n)
#else
    #define GCSO_ALIGNAS(n) _Alignas(n)
#endif

#if defined(_MSC_VER)
    #define GCSO_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define GCSO_RESTRICT __restrict__
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define GCSO_RESTRICT restrict
#else
    #define GCSO_RESTRICT
#endif

#ifdef __cplusplus
    #define GCSO_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define GCSO_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#else
    #define GCSO_STATIC_ASSERT(cond, msg)
#endif

#endif // LIMINIKA_GCSO_CONFIG_H