// name: src/core/src/foundation/simd_stats/geometry.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Constant-time $SO(2)$ slice rotation and fused Hodge obstruction / dual pairing kernels.

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use core::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
use crate::foundation::phase_ops::avx2_hsum_ps;
use crate::foundation::scalar_math::branchless_rotate_so2_pair;

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
unsafe fn avx2_hsum3_ps(
    v0: core::arch::x86_64::__m256,
    v1: core::arch::x86_64::__m256,
    v2: core::arch::x86_64::__m256,
) -> (f32, f32, f32) {
    // SAFETY: Caller guarantees `target_feature = "avx2"`.
    unsafe { (avx2_hsum_ps(v0), avx2_hsum_ps(v1), avx2_hsum_ps(v2)) }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_block32(ptr: *mut f32, v_cos: __m256, v_sin: __m256) {
    // SAFETY: Caller guarantees `ptr` is valid for 32 floats and AVX2 is enabled.
    unsafe {
        let p0 = ptr;
        let p1 = ptr.add(8);
        let v0 = _mm256_loadu_ps(p0);
        let v1 = _mm256_loadu_ps(p1);
        _mm256_storeu_ps(
            p0,
            _mm256_addsub_ps(
                _mm256_mul_ps(v0, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p1,
            _mm256_addsub_ps(
                _mm256_mul_ps(v1, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v1), v_sin),
            ),
        );
        let p2 = ptr.add(16);
        let p3 = ptr.add(24);
        let v2 = _mm256_loadu_ps(p2);
        let v3 = _mm256_loadu_ps(p3);
        _mm256_storeu_ps(
            p2,
            _mm256_addsub_ps(
                _mm256_mul_ps(v2, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v2), v_sin),
            ),
        );
        _mm256_storeu_ps(
            p3,
            _mm256_addsub_ps(
                _mm256_mul_ps(v3, v_cos),
                _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v3), v_sin),
            ),
        );
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline]
pub(crate) unsafe fn avx2_rotate_so2_dispatch(
    ptr: *mut f32,
    len: usize,
    v_cos: __m256,
    v_sin: __m256,
    sin_t: f32,
    cos_t: f32,
) {
    // SAFETY: Caller guarantees `ptr` is valid for `len` floats and AVX2 is enabled.
    // Staged 2-YMM (16-float) sub-blocks keep peak live vector register count <= 6 (`ymm0..ymm5`),
    // eliminating callee-saved `xmm6..xmm15` stack spills on Windows x64 ABI while saturating both 256-bit FMA ports.
    unsafe {
        if len == 32 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            return;
        }
        if len == 64 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            return;
        }
        if len == 128 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            return;
        }
        if len == 256 {
            avx2_rotate_so2_block32(ptr, v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(96), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(128), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(160), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(192), v_cos, v_sin);
            avx2_rotate_so2_block32(ptr.add(224), v_cos, v_sin);
            return;
        }
        let mut b = 0usize;
        if len >= 65_536 {
            // For large streaming buffers (`>= 256 KB`), step in 4KB-page chunks (`1024` floats)
            // issuing multi-page DTLB/L2 prefetches (`+8 KB`, `+16 KB`, `+32 KB`) outside a pure
            // 4-block (`128`-float, ~400B DSB-locked) inner loop with zero inner prefetch branches.
            while b + 1024 <= len {
                let page_ptr = ptr.add(b);
                if b + 2048 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(2048).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
                if b + 4096 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(4096).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                if b + 8192 < len {
                    core::arch::x86_64::_mm_prefetch(
                        page_ptr.add(8192).cast::<i8>(),
                        core::arch::x86_64::_MM_HINT_T1,
                    );
                }
                let page_end = b + 1024;
                while b < page_end {
                    let p = ptr.add(b);
                    avx2_rotate_so2_block32(p, v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
                    avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
                    b += 128;
                }
            }
        }
        while b + 128 <= len {
            let p = ptr.add(b);
            avx2_rotate_so2_block32(p, v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(32), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(64), v_cos, v_sin);
            avx2_rotate_so2_block32(p.add(96), v_cos, v_sin);
            b += 128;
        }
        while b + 32 <= len {
            avx2_rotate_so2_block32(ptr.add(b), v_cos, v_sin);
            b += 32;
        }
        if b == len {
            return;
        }
        while b + 8 <= len {
            let p0 = ptr.add(b);
            let v0 = _mm256_loadu_ps(p0);
            _mm256_storeu_ps(
                p0,
                _mm256_addsub_ps(
                    _mm256_mul_ps(v0, v_cos),
                    _mm256_mul_ps(_mm256_permute_ps::<0xB1>(v0), v_sin),
                ),
            );
            b += 8;
        }
        while b + 2 <= len {
            let (r0, r1) = branchless_rotate_so2_pair(*ptr.add(b), *ptr.add(b + 1), sin_t, cos_t);
            *ptr.add(b) = r0;
            *ptr.add(b + 1) = r1;
            b += 2;
        }
    }
}

/// Bank-conflict-free 64B cache-line-staged 3-tensor (`Q`, `K`, `V`) AVX2 $SO(2)$ 32-float block rotation.
///
/// Rotates `qp`, `kp`, and `vp` sequentially in 32-float (128-byte, 2-cache-line) blocks using shared
/// register-resident `(v_cos, v_sin)` (`<= 6` live YMM registers `ymm0..ymm5`, zero Windows x64
/// `xmm6..xmm15` callee-saved stack spills, zero L1D same-bank port conflicts, and zero 3-way 4KB
/// Load-Store Queue page-offset aliasing stalls).
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[inline(always)]
pub(crate) unsafe fn avx2_rotate_so2_triple_block32(
    qp: *mut f32,
    kp: *mut f32,
    vp: *mut f32,
    v_cos: __m256,
    v_sin: __m256,
) {
    // SAFETY: Caller guarantees `qp`, `kp`, `vp` are pairwise disjoint, valid for 32 floats, and AVX2 is enabled.
    unsafe {
        avx2_rotate_so2_block32(qp, v_cos, v_sin);
        avx2_rotate_so2_block32(kp, v_cos, v_sin);
        avx2_rotate_so2_block32(vp, v_cos, v_sin);
    }
}

/// Rotates `buf` in-place by `(sin_t, cos_t)` in strictly constant time (`0` data-dependent
/// early-return branches on `sin_t == 0.0 && cos_t == 1.0`, guaranteeing `1.00x` timing parity
/// between zero phase `0` and max saturation `+127`/`-128`).
#[inline(always)]
pub(crate) fn rotate_so2_slice_unrolled16(buf: &mut [f32], sin_t: f32, cos_t: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: `target_feature = "avx2"` is enabled and `buf` is valid for `buf.len()` floats.
        unsafe {
            avx2_rotate_so2_dispatch(
                buf.as_mut_ptr(),
                buf.len(),
                _mm256_set1_ps(cos_t),
                _mm256_set1_ps(sin_t),
                sin_t,
                cos_t,
            );
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        for pair in buf.chunks_exact_mut(2) {
            (pair[0], pair[1]) = branchless_rotate_so2_pair(pair[0], pair[1], sin_t, cos_t);
        }
    }
}

/// Computes $\sum_i (a_i - b_i)^2$ using 4-way unrolled AVX2 SIMD accumulation.
#[inline(always)]
#[must_use]
pub(crate) fn diff_norm_sq_f32_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pa = a.as_ptr();
        let pb = b.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and `i + 32 <= len` is in-bounds.
        unsafe {
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 8)),
                    _mm256_loadu_ps(pb.add(i + 8)),
                );
                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 16)),
                    _mm256_loadu_ps(pb.add(i + 16)),
                );
                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pa.add(i + 24)),
                    _mm256_loadu_ps(pb.add(i + 24)),
                );
                acc0 = _mm256_add_ps(
                    acc0,
                    _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1)),
                );
                acc1 = _mm256_add_ps(
                    acc1,
                    _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3)),
                );
                i += 32;
            }
            while i + 8 <= len {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
                acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
                i += 8;
            }
            let mut total = avx2_hsum_ps(_mm256_add_ps(acc0, acc1));
            while i < len {
                let d = *pa.add(i) - *pb.add(i);
                total += d * d;
                i += 1;
            }
            total
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s = [0.0_f32; 4];
        let mut a_chunks = a[..len].chunks_exact(4);
        let mut b_chunks = b[..len].chunks_exact(4);
        for (ac, bc) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
            for k in 0..4 {
                let d = ac[k] - bc[k];
                s[k] += d * d;
            }
        }
        let mut total = (s[0] + s[1]) + (s[2] + s[3]);
        for (&av, &bv) in a_chunks.remainder().iter().zip(b_chunks.remainder().iter()) {
            let d = av - bv;
            total += d * d;
        }
        total
    }
}

/// Computes $(\|\mathbf{r} - \mathbf{p}\|_2^2, \langle \mathbf{r} - \mathbf{p}, \boldsymbol{\gamma} \rangle, \|\boldsymbol{\gamma}\|_2^2)$
/// in a single fused AVX2 memory pass while keeping peak live YMM usage `<= 5`
/// (`0` Windows x64 `xmm6..xmm15` stack saves/restores).
#[inline]
#[must_use]
pub(crate) fn hodge_obstruction_and_dual_pairing_f32(
    r_slice: &[f32],
    p_slice: &[f32],
    g_slice: &[f32],
) -> (f32, f32, f32) {
    let n = r_slice.len().min(p_slice.len()).min(g_slice.len());
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        let pr = r_slice.as_ptr();
        let pp = p_slice.as_ptr();
        let pg = g_slice.as_ptr();
        // SAFETY: `target_feature = "avx2"` is enabled at compile time, and pointer offsets are bounds-checked against `n`.
        // Using 3 accumulator YMMs (`obs_all, dot_all, gam_all`) + 2 scratch YMMs (`d, g`) keeps peak live YMMs
        // at 5 (`ymm0..ymm4`), eliminating 8 `vmovaps` stack saves/restores of `xmm6..xmm13` on Windows x64 ABI
        // and cutting `hodge_obstruction_and_dual_homology_pairing_64` variance and latency.
        unsafe {
            let mut obs_all = _mm256_setzero_ps();
            let mut dot_all = _mm256_setzero_ps();
            let mut gam_all = _mm256_setzero_ps();
            let mut i = 0usize;
            while i + 32 <= n {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));

                let d2 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 16)),
                    _mm256_loadu_ps(pp.add(i + 16)),
                );
                let g2 = _mm256_loadu_ps(pg.add(i + 16));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d2, d2));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d2, g2));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g2, g2));

                let d3 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 24)),
                    _mm256_loadu_ps(pp.add(i + 24)),
                );
                let g3 = _mm256_loadu_ps(pg.add(i + 24));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d3, d3));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d3, g3));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g3, g3));
                i += 32;
            }
            let rem = n - i;
            if (rem & 16) != 0 {
                let d0 = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g0 = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d0, d0));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d0, g0));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g0, g0));

                let d1 = _mm256_sub_ps(
                    _mm256_loadu_ps(pr.add(i + 8)),
                    _mm256_loadu_ps(pp.add(i + 8)),
                );
                let g1 = _mm256_loadu_ps(pg.add(i + 8));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d1, d1));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d1, g1));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g1, g1));
                i += 16;
            }
            if (rem & 8) != 0 {
                let d = _mm256_sub_ps(_mm256_loadu_ps(pr.add(i)), _mm256_loadu_ps(pp.add(i)));
                let g = _mm256_loadu_ps(pg.add(i));
                obs_all = _mm256_add_ps(obs_all, _mm256_mul_ps(d, d));
                dot_all = _mm256_add_ps(dot_all, _mm256_mul_ps(d, g));
                gam_all = _mm256_add_ps(gam_all, _mm256_mul_ps(g, g));
                i += 8;
            }
            let (mut obs_sq, mut pair_dot, mut gam_sq) = avx2_hsum3_ps(obs_all, dot_all, gam_all);
            while i < n {
                let r_obs = *pr.add(i) - *pp.add(i);
                let gv = *pg.add(i);
                obs_sq += r_obs * r_obs;
                pair_dot += r_obs * gv;
                gam_sq += gv * gv;
                i += 1;
            }
            (obs_sq, pair_dot, gam_sq)
        }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        let mut s_obs = [0.0_f32; 4];
        let mut s_dot = [0.0_f32; 4];
        let mut s_gam = [0.0_f32; 4];

        let mut r_c = r_slice[..n].chunks_exact(4);
        let mut p_c = p_slice[..n].chunks_exact(4);
        let mut g_c = g_slice[..n].chunks_exact(4);

        for ((rc, pc), gc) in r_c.by_ref().zip(p_c.by_ref()).zip(g_c.by_ref()) {
            for k in 0..4 {
                let r_obs = rc[k] - pc[k];
                let gv = gc[k];
                s_obs[k] += r_obs * r_obs;
                s_dot[k] += r_obs * gv;
                s_gam[k] += gv * gv;
            }
        }
        let mut obs_sq = (s_obs[0] + s_obs[1]) + (s_obs[2] + s_obs[3]);
        let mut pair_dot = (s_dot[0] + s_dot[1]) + (s_dot[2] + s_dot[3]);
        let mut gam_sq = (s_gam[0] + s_gam[1]) + (s_gam[2] + s_gam[3]);
        for ((&rv, &pv), &gv) in r_c
            .remainder()
            .iter()
            .zip(p_c.remainder().iter())
            .zip(g_c.remainder().iter())
        {
            let r_obs = rv - pv;
            obs_sq += r_obs * r_obs;
            pair_dot += r_obs * gv;
            gam_sq += gv * gv;
        }
        (obs_sq, pair_dot, gam_sq)
    }
}
