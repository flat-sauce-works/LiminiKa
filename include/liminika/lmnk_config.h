// name: include/liminika/lmnk_config.h
// SPDX-License-Identifier: MIT OR Apache-2.0

/**
 * @file lmnk_config.h
 * @brief LiminiKa C11/C++20 ABI portability, compiler attribute, and static assertion macros.
 *
 * Designed for clean cross-platform compilation across MSVC, GCC, Clang, and embedded/WebAssembly
 * toolchains (`#![no_std]` Rust FFI counterpart: `liminika-core` v0.2.0 / `0x00000200`).
 */

#ifndef LIMINIKA_LMNK_CONFIG_H
#define LIMINIKA_LMNK_CONFIG_H

#if defined(__cplusplus)
    #include <cstddef>
    #include <cstdint>
#else
    #include <stddef.h>
    #include <stdint.h>
#endif

/* -------------------------------------------------------------------------- */
/* C++ linkage guard                                                          */
/* -------------------------------------------------------------------------- */
#ifdef __cplusplus
    #define LMNK_EXTERN_C_BEGIN extern "C" {
    #define LMNK_EXTERN_C_END }
#else
    #define LMNK_EXTERN_C_BEGIN
    #define LMNK_EXTERN_C_END
#endif

/* -------------------------------------------------------------------------- */
/* Target architecture and SIMD compile-time hints                            */
/* -------------------------------------------------------------------------- */
#if defined(_M_X64) || defined(__x86_64__)
    #define LMNK_ARCH_X86_64 1
#else
    #define LMNK_ARCH_X86_64 0
#endif

#if defined(_M_ARM64) || defined(__aarch64__)
    #define LMNK_ARCH_ARM64 1
#else
    #define LMNK_ARCH_ARM64 0
#endif

#if defined(__AVX2__)
    #define LMNK_HAS_AVX2 1
#else
    #define LMNK_HAS_AVX2 0
#endif

#if defined(__FMA__) || (LMNK_HAS_AVX2 && defined(_MSC_VER))
    #define LMNK_HAS_FMA 1
#else
    #define LMNK_HAS_FMA 0
#endif

#if defined(__AVX512F__)
    #define LMNK_HAS_AVX512 1
#else
    #define LMNK_HAS_AVX512 0
#endif

#if defined(__ARM_NEON) || LMNK_ARCH_ARM64
    #define LMNK_HAS_NEON 1
#else
    #define LMNK_HAS_NEON 0
#endif

/* -------------------------------------------------------------------------- */
/* Symbol export / visibility and calling convention                          */
/* -------------------------------------------------------------------------- */
#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(LMNK_SHARED_EXPORTS)
        #define LMNK_API __declspec(dllexport)
    #elif defined(LMNK_SHARED_IMPORTS)
        #define LMNK_API __declspec(dllimport)
    #else
        #define LMNK_API
    #endif
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_API __attribute__((visibility("default")))
#else
    #define LMNK_API
#endif

#if defined(_WIN32) && defined(_MSC_VER) && !defined(_M_X64) && !defined(_M_ARM64)
    #define LMNK_CALL __stdcall
#else
    #define LMNK_CALL
#endif

/* -------------------------------------------------------------------------- */
/* Fixed-width enum helper                                                    */
/* -------------------------------------------------------------------------- */
#ifdef __cplusplus
    #define LMNK_ENUM_UNDERLYING(type) : type
#else
    #define LMNK_ENUM_UNDERLYING(type)
#endif

/* -------------------------------------------------------------------------- */
/* Restrict, inline, branch hints, and safety attributes                      */
/* -------------------------------------------------------------------------- */
#ifdef __cplusplus
    #if defined(_MSC_VER)
        #define LMNK_RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define LMNK_RESTRICT __restrict__
    #else
        #define LMNK_RESTRICT
    #endif
#else
    #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
        #define LMNK_RESTRICT restrict
    #elif defined(_MSC_VER)
        #define LMNK_RESTRICT __restrict
    #elif defined(__GNUC__) || defined(__clang__)
        #define LMNK_RESTRICT __restrict__
    #else
        #define LMNK_RESTRICT
    #endif
#endif

#if defined(__cplusplus)
    #define LMNK_INLINE inline
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define LMNK_INLINE inline
#elif defined(_MSC_VER)
    #define LMNK_INLINE __inline
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_INLINE __inline__
#else
    #define LMNK_INLINE
#endif

#if defined(_MSC_VER)
    #define LMNK_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_FORCE_INLINE __attribute__((always_inline, unused)) LMNK_INLINE
#else
    #define LMNK_FORCE_INLINE LMNK_INLINE
#endif

#if defined(__GNUC__) || defined(__clang__)
    #define LMNK_LIKELY(x) __builtin_expect(!!(x), 1)
    #define LMNK_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    #define LMNK_LIKELY(x) (x)
    #define LMNK_UNLIKELY(x) (x)
#endif

#if defined(__cplusplus) && __cplusplus >= 201703L
    #define LMNK_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_NODISCARD __attribute__((warn_unused_result))
#elif defined(_MSC_VER) && _MSC_VER >= 1900
    #define LMNK_NODISCARD _Check_return_
#else
    #define LMNK_NODISCARD
#endif

#if defined(__cplusplus) && __cplusplus >= 201103L
    #define LMNK_NOEXCEPT noexcept
#else
    #define LMNK_NOEXCEPT
#endif

/* -------------------------------------------------------------------------- */
/* Cacheline / SIMD alignment macros (C11, C++11/20, MSVC, GCC, Clang)        */
/* -------------------------------------------------------------------------- */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #define LMNK_ALIGN_PREFIX(n) alignas(n)
    #define LMNK_ALIGN_SUFFIX(n)
    #define LMNK_ALIGNAS(n) alignas(n)
    #define LMNK_ALIGNOF(type) alignof(type)
#elif defined(_MSC_VER)
    #define LMNK_ALIGN_PREFIX(n) __declspec(align(n))
    #define LMNK_ALIGN_SUFFIX(n)
    #define LMNK_ALIGNAS(n) __declspec(align(n))
    #define LMNK_ALIGNOF(type) __alignof(type)
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_ALIGN_PREFIX(n)
    #define LMNK_ALIGN_SUFFIX(n) __attribute__((aligned(n)))
    #define LMNK_ALIGNAS(n) __attribute__((aligned(n)))
    #if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
        #define LMNK_ALIGNOF(type) _Alignof(type)
    #else
        #define LMNK_ALIGNOF(type) __alignof__(type)
    #endif
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define LMNK_ALIGN_PREFIX(n)
    #define LMNK_ALIGN_SUFFIX(n) _Alignas(n)
    #define LMNK_ALIGNAS(n) _Alignas(n)
    #define LMNK_ALIGNOF(type) _Alignof(type)
#else
    #define LMNK_ALIGN_PREFIX(n)
    #define LMNK_ALIGN_SUFFIX(n)
    #define LMNK_ALIGNAS(n)
    #define LMNK_ALIGNOF(type) sizeof(type)
#endif

/* -------------------------------------------------------------------------- */
/* Compile-time layout assertions                                             */
/* -------------------------------------------------------------------------- */
#if defined(__cplusplus) && __cplusplus >= 201103L
    #define LMNK_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
    #define LMNK_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#elif defined(__GNUC__) || defined(__clang__)
    #define LMNK_STATIC_ASSERT(cond, msg) \
        typedef char lmnk_static_assert_##__LINE__[(cond) ? 1 : -1] __attribute__((unused))
#else
    #define LMNK_STATIC_ASSERT(cond, msg) \
        typedef char lmnk_static_assert_##__LINE__[(cond) ? 1 : -1]
#endif

#endif /* LIMINIKA_LMNK_CONFIG_H */
