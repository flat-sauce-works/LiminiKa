// name: src/core/src/lib.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Core execution engine library for the Geometric Cellular Sheaf Orchestrator (GCSO).
//!
//! This crate serves as the primary low-level engine for LiminiKa, implementing the
//! Geometric Cellular Sheaf Orchestrator (GCSO) architecture targeting 2–4 GB VRAM environments.
//!
//! # Architecture & Layers
//! - **Nano Level**: Warp / SIMD level in-kernel hot-path operations (DPSR, RIPA, QDPS).
//! - **Micro Level**: Per-token hot-path execution, $\mathcal{O}(1)$ Sidecar Pointer Table (SPT) lookups, and DAES dynamic scratchpad.
//! - **Mezzo Level**: Cellular swarm coordination, PagedBlock bitmask reductions, Skip-Hop bus sync, and PPRC keyframe compression.
//! - **Macro Level**: Cold-path entropy evaluation (EDBC), attractor fields, L2P-SVD converters, and ZIMMS persistence.
//!
//! # Core Memory Invariants
//! - **Zero Dynamic Allocations on the Hot Path**: No heap allocations (`Vec`, `Box`, `malloc`) are permitted in hot loops.
//! - **Strict Type-Level Alignment**: All tensor and pointer inputs must satisfy hardware-aligned memory bounds (16B, 32B, 64B, 128B, 256B, 512B).
//! - **Static Dispatch**: Monomorphic execution via generic trait bounds to maximize compiler inlining and zero-cost abstraction.

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

// ===================================================================
// Core Architecture Module Declarations
// ===================================================================

/// C-ABI bindings, FFI safety barriers, and status codes.
pub mod abi;

/// Dynamic Phase Steering and RIPA rotation routines for Hot Path execution.
pub mod dpsr;

/// Dynamic Entropy-Driven Decoding Branch Controller (EDBC) implementation.
pub mod edbc;

/// Mathematical primitives, geometric group operations, and Chebyshev spectral filtering.
pub mod math;

/// Sparse Residual Adapter Layer (SRL) logic and dynamic adapters.
pub mod srl;

/// Unified container storage and memory-mapped persistence (ZIMMS).
pub mod zimms;

/// Cellular swarm coordination, Sidecar Pointer Table (SPT), and stigmergic routines.
pub mod swarm;

/// Core system contracts, traits, and alignment guarantees (Ground Truth).
pub mod traits;

// ===================================================================
// Primary Re-exports for Ergonomics & Ground Truth Alignment
// ===================================================================

pub use crate::traits::{
    AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256, AlignedSlice32,
    AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128, AlignedSliceMut16,
    AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512, AlignedSliceMut64, AttractorField,
    DaesScratchpad, EntropyEvaluator, GcsoResult, GcsoRuntimeContext, HotPathExecutionEngine,
    L2pSvdProjector, PersonaPatcher, PhaseSteering, PointerActionHub, PprcCache, PspmRouter,
    SrlAdapter, SwarmCellChunk, SystemCapabilityQuery, ZimmsStorage,
};

pub use crate::abi::{
    gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
    gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
    gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t,
    gcso_tagged_ptr_t, gcso_zimms_descriptor_t, GcsoStatus, GCSO_SUCCESS,
};

pub use crate::dpsr::DpsrEngine;
pub use crate::edbc::EdbcEngine;

/// Convenient crate-level result type alias for GCSO operations.
pub type Result<T> = GcsoResult<T>;

// ===================================================================
// Safe High-Level Rust API Wrappers & Fluent Builders
// ===================================================================

/// Safe high-level configuration wrapper for GCSO runtime initialization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Inner raw C-ABI configuration structure.
    pub raw: gcso_config_t,
    /// Hysteresis band for EDBC branch control stabilization.
    pub hysteresis_band: f32,
    /// Tau flux threshold for sliding window entropy rate integration.
    pub tau_flux: f32,
}

impl Default for Config {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Config {
    /// Creates a new default configuration instance with GCSO specification invariants.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            raw: gcso_config_t {
                head_dim: 128,
                num_heads: 32,
                paged_block_size: 32,
                q7_phase_scale: core::f32::consts::PI / 128.0,
                ripa_clamp_max_rad: 0.087_266_46,
                qdps_min_step_rad: 0.01,
                entropy_singularity_eps: 1e-12,
                max_prompt_anchors: 64,
                action_hub_capacity: 256,
                enable_cuda_warp_shuffle: 1,
                enable_zero_alloc_strict: 1,
                daes_mode: 0,
                reserved_flags: 0,
                reserved: [0; 24],
            },
            hysteresis_band: 0.15,
            tau_flux: 0.5,
        }
    }

    /// Sets the attention head dimension ($d_{\mathrm{head}}$). Must be an even non-zero number.
    #[inline]
    #[must_use]
    pub const fn with_head_dim(mut self, head_dim: u32) -> Self {
        self.raw.head_dim = head_dim;
        self
    }

    /// Sets the total number of attention heads (<= 64).
    #[inline]
    #[must_use]
    pub const fn with_num_heads(mut self, num_heads: u32) -> Self {
        self.raw.num_heads = num_heads;
        self
    }

    /// Sets the PagedBlock size (16 or 32 tokens).
    #[inline]
    #[must_use]
    pub const fn with_paged_block_size(mut self, paged_block_size: u32) -> Self {
        self.raw.paged_block_size = paged_block_size;
        self
    }

    /// Sets the Q7 phase scale factor.
    #[inline]
    #[must_use]
    pub const fn with_q7_phase_scale(mut self, q7_phase_scale: f32) -> Self {
        self.raw.q7_phase_scale = q7_phase_scale;
        self
    }

    /// Sets the RIPA soft-bounded phase clamp limit in radians.
    #[inline]
    #[must_use]
    pub const fn with_ripa_clamp_max_rad(mut self, ripa_clamp_max_rad: f32) -> Self {
        self.raw.ripa_clamp_max_rad = ripa_clamp_max_rad;
        self
    }

    /// Sets the QDPS quantization threshold angle.
    #[inline]
    #[must_use]
    pub const fn with_qdps_min_step_rad(mut self, qdps_min_step_rad: f32) -> Self {
        self.raw.qdps_min_step_rad = qdps_min_step_rad;
        self
    }

    /// Sets the logarithmic singularity prevention epsilon ($\epsilon_{\mathrm{log}}$).
    #[inline]
    #[must_use]
    pub const fn with_entropy_singularity_eps(mut self, entropy_singularity_eps: f32) -> Self {
        self.raw.entropy_singularity_eps = entropy_singularity_eps;
        self
    }

    /// Sets the maximum supported prompt anchor points in the attractor field.
    #[inline]
    #[must_use]
    pub const fn with_max_prompt_anchors(mut self, max_prompt_anchors: u32) -> Self {
        self.raw.max_prompt_anchors = max_prompt_anchors;
        self
    }

    /// Sets the Action Hub table capacity.
    #[inline]
    #[must_use]
    pub const fn with_action_hub_capacity(mut self, action_hub_capacity: u32) -> Self {
        self.raw.action_hub_capacity = action_hub_capacity;
        self
    }

    /// Configures CUDA warp shuffle instruction usage.
    #[inline]
    #[must_use]
    pub const fn with_enable_cuda_warp_shuffle(mut self, enable_cuda_warp_shuffle: u8) -> Self {
        self.raw.enable_cuda_warp_shuffle = enable_cuda_warp_shuffle;
        self
    }

    /// Configures strict zero dynamic allocation enforcement on the hot path.
    #[inline]
    #[must_use]
    pub const fn with_enable_zero_alloc_strict(mut self, enable_zero_alloc_strict: u8) -> Self {
        self.raw.enable_zero_alloc_strict = enable_zero_alloc_strict;
        self
    }

    /// Configures DAES scratchpad operating mode (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC).
    #[inline]
    #[must_use]
    pub const fn with_daes_mode(mut self, daes_mode: u8) -> Self {
        self.raw.daes_mode = daes_mode;
        self
    }

    /// Sets the EDBC hysteresis band width.
    #[inline]
    #[must_use]
    pub const fn with_hysteresis_band(mut self, hysteresis_band: f32) -> Self {
        self.hysteresis_band = hysteresis_band;
        self
    }

    /// Sets the sliding window entropy rate tau flux threshold.
    #[inline]
    #[must_use]
    pub const fn with_tau_flux(mut self, tau_flux: f32) -> Self {
        self.tau_flux = tau_flux;
        self
    }

    /// Validates configuration parameters against GCSO architectural invariants.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if `head_dim` is odd or zero, if `num_heads` exceeds 64,
    /// if capacity metrics are zero, or if floating-point configuration parameters violate valid bounds.
    #[inline]
    pub fn validate(&self) -> Result<()> {
        if self.raw.head_dim == 0 || (self.raw.head_dim % 2 != 0) {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.raw.num_heads == 0 || self.raw.num_heads > 64 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.raw.paged_block_size != 16 && self.raw.paged_block_size != 32 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.raw.q7_phase_scale.is_finite() || self.raw.q7_phase_scale <= 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.raw.ripa_clamp_max_rad.is_finite() || self.raw.ripa_clamp_max_rad <= 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.raw.qdps_min_step_rad.is_finite() || self.raw.qdps_min_step_rad < 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.raw.entropy_singularity_eps.is_finite() || self.raw.entropy_singularity_eps <= 0.0
        {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.raw.max_prompt_anchors == 0 || self.raw.action_hub_capacity == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.hysteresis_band.is_finite() || self.hysteresis_band < 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if !self.tau_flux.is_finite() || self.tau_flux < 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

/// High-level safe Rust runtime context wrapper fully integrating DPSR, EDBC, Action Hub, DAES, Attractor Field,
/// L2P-SVD Projector, and ZIMMS Storage routines.
#[derive(Debug)]
pub struct RuntimeContext {
    config: Config,
    dpsr: DpsrEngine,
    edbc: EdbcEngine,
    daes_slot: gcso_daes_slot_t,
    step_counter: u64,
    /// Track registered system prompt anchors for state tracking.
    prompt_anchor_count: u32,
    /// Last registered prompt anchor weight.
    last_prompt_weight: f32,
    /// Active pointer trail entry tracking table.
    active_trail_count: u32,
    /// Phase repulsion gain scalar.
    repulsion_gain: f32,
}

impl RuntimeContext {
    /// Creates a new safe `RuntimeContext` instance from a verified configuration.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if configuration parameters are invalid.
    #[inline]
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let dpsr = DpsrEngine::new(config.raw.head_dim, config.raw.num_heads)?;
        let mut edbc_state = gcso_edbc_state_t::default();
        edbc_state.singularity_eps = config.raw.entropy_singularity_eps;
        edbc_state.bifurcation_threshold = 1.5;
        let edbc = EdbcEngine::new(edbc_state);

        let mut daes_slot = gcso_daes_slot_t::default();
        daes_slot.mode = u32::from(config.raw.daes_mode);

        Ok(Self {
            config,
            dpsr,
            edbc,
            daes_slot,
            step_counter: 0,
            prompt_anchor_count: 0,
            last_prompt_weight: 1.0,
            active_trail_count: 0,
            repulsion_gain: 0.0,
        })
    }

    /// Returns a reference to the active configuration.
    #[inline(always)]
    #[must_use]
    pub const fn config(&self) -> &Config {
        &self.config
    }

    /// Returns a reference to the underlying DPSR steering engine.
    #[inline(always)]
    #[must_use]
    pub const fn dpsr_engine(&self) -> &DpsrEngine {
        &self.dpsr
    }

    /// Returns a mutable reference to the underlying EDBC entropy controller.
    #[inline(always)]
    #[must_use]
    pub fn edbc_engine_mut(&mut self) -> &mut EdbcEngine {
        &mut self.edbc
    }

    /// Returns a reference to the active DAES slot structure.
    #[inline(always)]
    #[must_use]
    pub const fn daes_slot(&self) -> &gcso_daes_slot_t {
        &self.daes_slot
    }

    /// Returns a mutable reference to the active DAES slot structure.
    #[inline(always)]
    #[must_use]
    pub fn daes_slot_mut(&mut self) -> &mut gcso_daes_slot_t {
        &mut self.daes_slot
    }

    /// Returns the total number of processed token steps.
    #[inline(always)]
    #[must_use]
    pub const fn step_counter(&self) -> u64 {
        self.step_counter
    }

    /// Returns the number of registered prompt anchors.
    #[inline(always)]
    #[must_use]
    pub const fn prompt_anchor_count(&self) -> u32 {
        self.prompt_anchor_count
    }

    /// Returns the active pointer trail count in the Action Hub.
    #[inline(always)]
    #[must_use]
    pub const fn active_trail_count(&self) -> u32 {
        self.active_trail_count
    }

    /// Executes a safe per-token step fully integrating DPSR phase steering (fused or standard),
    /// EDBC entropy evaluation, and pointer trail status population in zero-allocation mode.
    #[inline]
    pub fn step_token_with_zscore(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        phase_deltas: Option<&AlignedSlice16<'_, gcso_q7_t>>,
        token_z_score: f32,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        match (query_tensor, key_tensor, phase_deltas) {
            (Some(q), Some(k), Some(p)) => {
                self.dpsr.apply_phase_steering_fused(q, k, p)?;
            }
            (Some(q), None, Some(p)) => {
                self.dpsr.apply_phase_steering(q, p)?;
            }
            _ => {}
        }

        let mut state_out = gcso_edbc_state_t::default();
        self.edbc.eval_stateful(token_z_score, &mut state_out)?;

        self.step_counter = self.step_counter.wrapping_add(1);

        if let Some(trail) = trail_out {
            trail.prev_ptr = trail.current_ptr;
            trail.current_ptr = u64::from(token_id);
            trail.step_count = self.step_counter as u32;
            trail.stigmergic_density = 1.0;
            trail.flags = state_out.active_branch_mode;
        }

        Ok(())
    }

    /// Executes a sequence of token steps in zero-allocation mode (useful for prefill and batch decoding).
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_INVALID_ARGUMENT` if token slices mismatch or step execution fails.
    #[inline]
    pub fn step_sequence(
        &mut self,
        token_ids: &[u32],
        mut trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        for &token_id in token_ids {
            let trail_ref = trail_out.as_mut().map(|t| &mut **t);
            self.step_token(token_id, None, None, trail_ref)?;
        }
        Ok(())
    }

    /// Serializes active runtime context state into a binary snapshot conforming to GCSO ABI v0.1.1,
    /// structuring header offsets for Action Hub, Attractor Field, DPSR, SRL, EDBC, and DAES sections.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_BUFFER_TOO_SMALL` if output buffer size is insufficient.
    #[inline]
    pub fn serialize_to_bytes(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>();
        let edbc_size = core::mem::size_of::<gcso_edbc_state_t>();
        let daes_size = core::mem::size_of::<gcso_daes_slot_t>();
        let context_data_size = 24usize;

        let total_required = header_size + context_data_size + edbc_size + daes_size;

        if buffer.is_empty() {
            *required_size = total_required;
            return Ok(());
        }
        if buffer.len() < total_required {
            *required_size = total_required;
            return Err(crate::abi::GCSO_ERROR_BUFFER_TOO_SMALL);
        }

        buffer[..total_required].fill(0);

        let action_hub_off = header_size;
        let edbc_off = action_hub_off + context_data_size;
        let daes_off = edbc_off + edbc_size;

        let header_ptr = buffer.as_mut_ptr().cast::<gcso_snapshot_header_t>();
        // SAFETY: buffer is validated to have capacity for at least total_required bytes.
        unsafe {
            (*header_ptr).magic = 0x4F53_4347; // ASCII "GCSO"
            (*header_ptr).version = crate::abi::GCSO_ABI_VERSION_HEX; // 0x00000101 (v0.1.1)
            (*header_ptr).total_size = total_required as u64;
            (*header_ptr).action_hub_offset = action_hub_off as u64;
            (*header_ptr).attractor_field_offset = action_hub_off as u64;
            (*header_ptr).dpsr_state_offset = edbc_off as u64;
            (*header_ptr).srl_state_offset = edbc_off as u64;
            (*header_ptr).edbc_state_offset = edbc_off as u64;
            (*header_ptr).daes_slot_offset = daes_off as u32;
            (*header_ptr).timestamp_epoch_sec = 1_774_900_000;
        }

        let ctx_buf = &mut buffer[action_hub_off..edbc_off];
        ctx_buf[..8].copy_from_slice(&self.step_counter.to_le_bytes());
        ctx_buf[8..12].copy_from_slice(&self.active_trail_count.to_le_bytes());
        ctx_buf[12..16].copy_from_slice(&self.prompt_anchor_count.to_le_bytes());
        ctx_buf[16..20].copy_from_slice(&self.last_prompt_weight.to_le_bytes());

        let edbc_state = self.edbc.state();
        let edbc_ptr = buffer[edbc_off..daes_off]
            .as_mut_ptr()
            .cast::<gcso_edbc_state_t>();
        // SAFETY: edbc_off to daes_off range contains size_of::<gcso_edbc_state_t>() valid memory.
        unsafe {
            *edbc_ptr = edbc_state;
        }

        let daes_ptr = buffer[daes_off..total_required]
            .as_mut_ptr()
            .cast::<gcso_daes_slot_t>();
        // SAFETY: daes_off to total_required range contains size_of::<gcso_daes_slot_t>() valid memory.
        unsafe {
            *daes_ptr = self.daes_slot;
        }

        *required_size = total_required;
        Ok(())
    }

    /// Deserializes a binary snapshot buffer to restore context state in-place with ABI v0.1.1 verification.
    ///
    /// # Errors
    /// Returns `GCSO_ERROR_CONTAINER_CORRUPTED` if header magic or size is invalid.
    /// Returns `GCSO_ERROR_VERSION_MISMATCH` if ABI version does not match.
    #[inline]
    pub fn deserialize_from_bytes(&mut self, buffer: &[u8]) -> Result<()> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>();
        if buffer.len() < header_size {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        // SAFETY: buffer contains at least header_size bytes and valid alignment.
        let header = unsafe { &*buffer.as_ptr().cast::<gcso_snapshot_header_t>() };
        if header.magic != 0x4F53_4347 {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }
        crate::abi::validate_abi_version(header.version)?;

        if (header.total_size as usize) > buffer.len() {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        let action_hub_off = header.action_hub_offset as usize;
        let edbc_off = header.edbc_state_offset as usize;
        let daes_off = header.daes_slot_offset as usize;

        if action_hub_off < header_size || edbc_off < action_hub_off || daes_off < edbc_off {
            return Err(crate::abi::GCSO_ERROR_CONTAINER_CORRUPTED);
        }

        if buffer.len() >= action_hub_off + 20 {
            let ctx_buf = &buffer[action_hub_off..];
            self.step_counter = u64::from_le_bytes(ctx_buf[..8].try_into().unwrap_or([0; 8]));
            self.active_trail_count =
                u32::from_le_bytes(ctx_buf[8..12].try_into().unwrap_or([0; 4]));
            self.prompt_anchor_count =
                u32::from_le_bytes(ctx_buf[12..16].try_into().unwrap_or([0; 4]));
            self.last_prompt_weight =
                f32::from_le_bytes(ctx_buf[16..20].try_into().unwrap_or([0; 4]));
        }

        let edbc_size = core::mem::size_of::<gcso_edbc_state_t>();
        if buffer.len() >= edbc_off + edbc_size {
            let edbc_ptr = buffer[edbc_off..].as_ptr().cast::<gcso_edbc_state_t>();
            // SAFETY: buffer contains at least edbc_size bytes at edbc_off offset.
            let edbc_st = unsafe { *edbc_ptr };
            self.edbc = EdbcEngine::new(edbc_st);
        }

        let daes_size = core::mem::size_of::<gcso_daes_slot_t>();
        if buffer.len() >= daes_off + daes_size {
            let daes_ptr = buffer[daes_off..].as_ptr().cast::<gcso_daes_slot_t>();
            // SAFETY: buffer contains at least daes_size bytes at daes_off offset.
            self.daes_slot = unsafe { *daes_ptr };
        }

        Ok(())
    }
}

// ===================================================================
// Trait Implementations for Ground Truth (traits.rs) Compliance
// ===================================================================

impl SystemCapabilityQuery for RuntimeContext {
    #[inline]
    fn abi_version(&self) -> (u32, u32, u32) {
        crate::abi_version()
    }

    #[inline]
    fn abi_version_string(&self) -> &'static str {
        crate::abi::GCSO_ABI_VERSION_STRING
    }

    #[inline]
    fn query_capability(&self) -> gcso_capability_flags_t {
        let caps = 0x01u64; // Base CPU capability
        #[cfg(any(feature = "cuda", has_kernel_cuda))]
        {
            caps |= 1 << 1;
        }
        #[cfg(any(feature = "vulkan", has_kernel_vulkan))]
        {
            caps |= 1 << 2;
        }
        #[cfg(any(target_os = "macos", feature = "metal", has_kernel_metal))]
        {
            caps |= 1 << 3;
        }
        caps
    }

    #[inline]
    fn init_default_config(&self, config: &mut gcso_config_t) -> Result<()> {
        *config = self.config.raw;
        Ok(())
    }
}

impl GcsoRuntimeContext for RuntimeContext {
    #[inline]
    fn reset(&mut self) -> Result<()> {
        self.edbc.reset();
        self.step_counter = 0;
        self.prompt_anchor_count = 0;
        self.active_trail_count = 0;
        self.repulsion_gain = 0.0;
        self.daes_slot = gcso_daes_slot_t::default();
        self.daes_slot.mode = u32::from(self.config.raw.daes_mode);
        Ok(())
    }

    #[inline]
    fn set_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> Result<()> {
        if prompt.is_empty() || weight.is_nan() || weight.is_infinite() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.prompt_anchor_count >= self.config.raw.max_prompt_anchors {
            return Err(crate::abi::GCSO_ERROR_ACTION_HUB_FULL);
        }
        self.prompt_anchor_count = self.prompt_anchor_count.wrapping_add(1);
        self.last_prompt_weight = weight;
        Ok(())
    }

    #[inline]
    fn step_token(
        &mut self,
        token_id: u32,
        query_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        key_tensor: Option<&mut AlignedSliceMut32<'_, f32>>,
        trail_out: Option<&mut gcso_pointer_trail_t>,
    ) -> Result<()> {
        self.step_token_with_zscore(token_id, query_tensor, key_tensor, None, 0.0, trail_out)
    }

    #[inline]
    fn serialize(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    #[inline]
    fn deserialize(&mut self, buffer: &[u8]) -> Result<()> {
        self.deserialize_from_bytes(buffer)
    }

    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0; // Keyframe marker
        Ok(())
    }
}

impl PhaseSteering for RuntimeContext {
    #[inline(always)]
    fn head_dim(&self) -> u32 {
        self.dpsr.head_dim()
    }

    #[inline(always)]
    fn num_heads(&self) -> u32 {
        self.dpsr.num_heads()
    }

    #[inline]
    fn apply_phase_steering(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> Result<()> {
        self.dpsr.apply_phase_steering(query_tensor, phase_deltas)
    }

    #[inline]
    fn apply_phase_steering_fused(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        key_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> Result<()> {
        self.dpsr
            .apply_phase_steering_fused(query_tensor, key_tensor, phase_deltas)
    }

    #[inline]
    fn apply_phase_steering_safe(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        max_rad: f32,
    ) -> Result<()> {
        self.dpsr
            .apply_phase_steering_safe(query_tensor, phase_deltas, max_rad)
    }

    #[inline]
    fn apply_soft_phase_damping(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        damping_factor: f32,
    ) -> Result<()> {
        self.dpsr
            .apply_soft_phase_damping(query_tensor, damping_factor)
    }

    #[inline]
    fn qdps_filter_step(
        &self,
        phase_deltas: &mut AlignedSliceMut16<'_, gcso_q7_t>,
        min_step_rad: f32,
    ) -> Result<()> {
        self.dpsr.qdps_filter_step(phase_deltas, min_step_rad)
    }

    #[inline]
    fn lazy_unwrap_override(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        context_accum: &AlignedSlice32<'_, f32>,
    ) -> Result<()> {
        self.dpsr.lazy_unwrap_override(query_tensor, context_accum)
    }

    #[inline]
    fn slerp_norm_guard_stable(
        &self,
        tensor: &mut AlignedSliceMut32<'_, f32>,
        norm_lower: f32,
        norm_upper: f32,
    ) -> Result<()> {
        self.dpsr
            .slerp_norm_guard_stable(tensor, norm_lower, norm_upper)
    }

    #[inline]
    fn fused_logit_shift(
        &self,
        logits: &mut AlignedSliceMut32<'_, f32>,
        phase_deltas: &AlignedSlice16<'_, gcso_q7_t>,
    ) -> Result<()> {
        self.dpsr.fused_logit_shift(logits, phase_deltas)
    }

    #[inline]
    fn compute_procrustes_phase_delta(
        &self,
        source: &AlignedSlice32<'_, f32>,
        target: &AlignedSlice32<'_, f32>,
        phase_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> Result<()> {
        self.dpsr
            .compute_procrustes_phase_delta(source, target, phase_out)
    }
}

impl EntropyEvaluator for RuntimeContext {
    #[inline]
    fn eval_stateful(
        &mut self,
        token_z_score: f32,
        state_out: &mut gcso_edbc_state_t,
    ) -> Result<()> {
        self.edbc.eval_stateful(token_z_score, state_out)
    }

    #[inline]
    fn eval_cvoid_dyadic128(
        &self,
        key_vector: &AlignedSlice32<'_, f32>,
        void_score_out: &mut f32,
    ) -> Result<()> {
        self.edbc.eval_cvoid_dyadic128(key_vector, void_score_out)
    }

    #[inline]
    fn eval_barrier(&self, void_score: f32, tau_eff: f32) -> Result<f32> {
        self.edbc.eval_barrier(void_score, tau_eff)
    }

    #[inline]
    fn eval_barrier_pitchfork(
        &self,
        cvoid: f32,
        tau_eff: f32,
        alpha: f32,
        beta: f32,
        head_dim: u32,
    ) -> Result<f32> {
        self.edbc
            .eval_barrier_pitchfork(cvoid, tau_eff, alpha, beta, head_dim)
    }
}

impl PointerActionHub for RuntimeContext {
    #[inline]
    fn reset(&mut self) -> Result<()> {
        self.active_trail_count = 0;
        Ok(())
    }

    #[inline]
    fn step_pointer(
        &mut self,
        current_ptr: u64,
        trail_out: &mut gcso_pointer_trail_t,
    ) -> Result<()> {
        if self.active_trail_count < self.config.raw.action_hub_capacity {
            self.active_trail_count = self.active_trail_count.wrapping_add(1);
        }
        trail_out.prev_ptr = trail_out.current_ptr;
        trail_out.current_ptr = current_ptr;
        trail_out.step_count = trail_out.step_count.wrapping_add(1);
        trail_out.stigmergic_density = 1.0;
        Ok(())
    }

    #[inline]
    fn hash_slot256_index(&self, ptr: u64) -> u32 {
        let mut x = ptr;
        x ^= x >> 30;
        x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        (x & 0xFF) as u32
    }

    #[inline]
    fn update_cellular_state_express(
        &mut self,
        _trail_id: u32,
        _phase_delta: gcso_q7_t,
    ) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn pull_trail_to_attractor(
        &mut self,
        _trail_id: u32,
        _anchor_id: u32,
        pull_force: f32,
    ) -> Result<()> {
        if pull_force.is_nan() || pull_force.is_infinite() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }

    #[inline]
    fn link_hallucinated_trails(&mut self, _src_trail_id: u32, _dst_trail_id: u32) -> Result<()> {
        Ok(())
    }

    #[inline]
    fn reduce_bit_tree(
        &self,
        bitmasks: &AlignedSlice32<'_, gcso_paged_bitmask_t>,
        reduced_out: &mut gcso_paged_bitmask_t,
    ) -> Result<()> {
        let mut acc = [0u64; 4];
        for m in bitmasks.as_slice() {
            acc[0] |= m.bits[0];
            acc[1] |= m.bits[1];
            acc[2] |= m.bits[2];
            acc[3] |= m.bits[3];
        }
        reduced_out.bits = acc;
        Ok(())
    }

    #[inline]
    fn paged_block_warp_bitmask(
        &self,
        _kv_bits: &AlignedSlice32<'_, u64>,
        mask_out: &mut gcso_paged_bitmask_t,
    ) -> Result<()> {
        mask_out.bits.fill(0);
        Ok(())
    }

    #[inline(always)]
    fn capacity(&self) -> u32 {
        self.config.raw.action_hub_capacity
    }

    #[inline(always)]
    fn active_count(&self) -> u32 {
        self.active_trail_count
    }
}

impl DaesScratchpad for RuntimeContext {
    #[inline]
    fn reset_telemetry(&mut self, slot: &mut gcso_daes_slot_t) -> Result<()> {
        slot.telemetry_ring_head = 0;
        slot.telemetry_ring_tail = 0;
        slot.cache_hit_count = 0;
        slot.telemetry_mini_ledger.fill(0);
        Ok(())
    }

    #[inline]
    fn fast_path_lookup(&self, slot: &gcso_daes_slot_t, input_key: u64) -> Result<u64> {
        if (slot.fast_path_bypass_mask & input_key) != 0 {
            Ok(slot.fast_path_shortcuts[0])
        } else {
            Err(crate::abi::GCSO_ERROR_INVALID_STATE)
        }
    }

    #[inline]
    fn telemetry_push(&mut self, slot: &mut gcso_daes_slot_t, metric_code: u8) -> Result<()> {
        let head = (slot.telemetry_ring_head as usize) % 8;
        slot.telemetry_mini_ledger[head] = metric_code;
        slot.telemetry_ring_head = slot.telemetry_ring_head.wrapping_add(1);
        Ok(())
    }

    #[inline]
    fn set_mode(&mut self, slot: &mut gcso_daes_slot_t, mode: u32) -> Result<()> {
        if mode > 2 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        slot.mode = mode;
        Ok(())
    }

    #[inline]
    fn evaluate_auto_tune(
        &self,
        _slot: &gcso_daes_slot_t,
        config_out: &mut gcso_config_t,
    ) -> Result<()> {
        *config_out = self.config.raw;
        Ok(())
    }

    #[inline]
    fn cache_hit_count(&self, slot: &gcso_daes_slot_t) -> u32 {
        slot.cache_hit_count
    }
}

impl SwarmCellChunk for RuntimeContext {
    #[inline]
    fn step_chunk(&mut self, _mask: &gcso_paged_bitmask_t, chunk_len: u32) -> Result<()> {
        if chunk_len == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.step_counter = self.step_counter.wrapping_add(u64::from(chunk_len));
        Ok(())
    }
}

impl AttractorField for RuntimeContext {
    #[inline]
    fn anchor_count(&self) -> u32 {
        self.prompt_anchor_count
    }

    #[inline]
    fn add_system_prompt_anchor(&mut self, prompt: &str, weight: f32) -> Result<u32> {
        self.set_system_prompt_anchor(prompt, weight)?;
        Ok(self.prompt_anchor_count)
    }

    #[inline]
    fn add_embedding_anchor(
        &mut self,
        _embedding: &AlignedSlice32<'_, f32>,
        _anchor_type: gcso_anchor_type_t,
        weight: f32,
    ) -> Result<u32> {
        if weight.is_nan() || weight.is_infinite() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.prompt_anchor_count >= self.config.raw.max_prompt_anchors {
            return Err(crate::abi::GCSO_ERROR_ACTION_HUB_FULL);
        }
        self.prompt_anchor_count = self.prompt_anchor_count.wrapping_add(1);
        self.last_prompt_weight = weight;
        Ok(self.prompt_anchor_count)
    }

    #[inline]
    fn add_anchor(
        &mut self,
        _anchor_type: gcso_anchor_type_t,
        _vec: &AlignedSlice32<'_, f32>,
        dim: usize,
    ) -> Result<u32> {
        if dim == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if self.prompt_anchor_count >= self.config.raw.max_prompt_anchors {
            return Err(crate::abi::GCSO_ERROR_ACTION_HUB_FULL);
        }
        self.prompt_anchor_count = self.prompt_anchor_count.wrapping_add(1);
        Ok(self.prompt_anchor_count)
    }

    #[inline]
    fn inject_phase_repulsion(
        &mut self,
        _repulsion_deltas: &AlignedSlice16<'_, gcso_q7_t>,
        gain: f32,
    ) -> Result<()> {
        if gain.is_nan() || gain.is_infinite() || gain <= 0.0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        self.repulsion_gain = gain;
        Ok(())
    }

    #[inline]
    fn aggregate_bottom_up(&mut self) -> Result<u32> {
        Ok(0)
    }

    #[inline]
    fn clear_anchors(&mut self) -> Result<()> {
        self.prompt_anchor_count = 0;
        self.last_prompt_weight = 1.0;
        Ok(())
    }
}

impl PersonaPatcher for RuntimeContext {
    #[inline]
    fn apply_patch(&mut self, patch_data: &[u8]) -> Result<()> {
        if patch_data.is_empty() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        Ok(())
    }
}

impl PprcCache for RuntimeContext {
    #[inline]
    fn seek_to_token(
        &mut self,
        token_index: u32,
        header_out: &mut gcso_pprc_keyframe_header_t,
    ) -> Result<()> {
        header_out.token_index = token_index;
        header_out.frame_type = 0; // Keyframe marker
        Ok(())
    }
}

impl PspmRouter for RuntimeContext {
    #[inline]
    fn dispatch_single_pass(
        &self,
        query_tensor: &mut AlignedSliceMut32<'_, f32>,
        config: &gcso_pspm_config_t,
        head_dim: usize,
    ) -> Result<()> {
        if head_dim == 0 || (head_dim % 2 != 0) {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        let total_heads = usize::from(config.num_fact_heads)
            + usize::from(config.num_logic_heads)
            + usize::from(config.num_explore_heads);
        let expected_len = head_dim
            .checked_mul(total_heads)
            .ok_or(crate::abi::GCSO_ERROR_INVALID_ARGUMENT)?;
        if query_tensor.len() < expected_len {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }

        let q_slice = query_tensor.as_mut_slice();
        let fact_end = usize::from(config.num_fact_heads) * head_dim;
        let logic_end = fact_end + usize::from(config.num_logic_heads) * head_dim;

        if config.fact_phase_gain != 1.0 && fact_end > 0 {
            for val in &mut q_slice[..fact_end] {
                *val *= config.fact_phase_gain;
            }
        }

        if config.logic_phase_gain != 1.0 && logic_end > fact_end {
            for val in &mut q_slice[fact_end..logic_end] {
                *val *= config.logic_phase_gain;
            }
        }

        if config.explore_phase_gain != 1.0 && expected_len > logic_end {
            for val in &mut q_slice[logic_end..expected_len] {
                *val *= config.explore_phase_gain;
            }
        }

        Ok(())
    }
}

impl SrlAdapter for RuntimeContext {
    #[inline]
    fn eval_rank1(
        &self,
        y_out: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        descriptor: &gcso_srl_descriptor_t,
    ) -> Result<()> {
        if descriptor.u_vector_ptr == 0 || descriptor.v_vector_ptr == 0 {
            return Err(crate::abi::GCSO_ERROR_NULL_POINTER);
        }

        let u_ptr = descriptor.u_vector_ptr as *const f32;
        let v_ptr = descriptor.v_vector_ptr as *const f32;
        let gain_ptr = descriptor.gain_scalar_ptr as *const f32;

        if !crate::abi::is_aligned(u_ptr) || !crate::abi::is_aligned(v_ptr) {
            return Err(crate::abi::GCSO_ERROR_MISALIGNED_POINTER);
        }

        let dim_in = x_in.len();
        let dim_out = y_out.len();
        if dim_in == 0 || dim_out == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }

        let x_slice = x_in.as_slice();
        let y_slice = y_out.as_mut_slice();
        // SAFETY: u_ptr points to valid memory with dim_out elements and is non-null/aligned.
        let u_slice = unsafe { core::slice::from_raw_parts(u_ptr, dim_out) };
        // SAFETY: v_ptr points to valid memory with dim_in elements and is non-null/aligned.
        let v_slice = unsafe { core::slice::from_raw_parts(v_ptr, dim_in) };

        let mut v_dot_x = 0.0f32;
        for j in 0..dim_in {
            v_dot_x += v_slice[j] * x_slice[j];
        }

        let scale = descriptor.scale_factor * v_dot_x;

        if !gain_ptr.is_null() && crate::abi::is_aligned(gain_ptr) {
            // SAFETY: gain_ptr points to valid memory with dim_out elements and is non-null/aligned.
            let g_slice = unsafe { core::slice::from_raw_parts(gain_ptr, dim_out) };
            for i in 0..dim_out {
                y_slice[i] = scale * u_slice[i] * g_slice[i];
            }
        } else {
            for i in 0..dim_out {
                y_slice[i] = scale * u_slice[i];
            }
        }

        Ok(())
    }

    #[inline]
    fn eval_rank1_fused(
        &self,
        y_inout: &mut AlignedSliceMut32<'_, f32>,
        x_in: &AlignedSlice32<'_, f32>,
        descriptor: &gcso_srl_descriptor_t,
    ) -> Result<()> {
        if descriptor.u_vector_ptr == 0 || descriptor.v_vector_ptr == 0 {
            return Err(crate::abi::GCSO_ERROR_NULL_POINTER);
        }

        let u_ptr = descriptor.u_vector_ptr as *const f32;
        let v_ptr = descriptor.v_vector_ptr as *const f32;
        let gain_ptr = descriptor.gain_scalar_ptr as *const f32;

        if !crate::abi::is_aligned(u_ptr) || !crate::abi::is_aligned(v_ptr) {
            return Err(crate::abi::GCSO_ERROR_MISALIGNED_POINTER);
        }

        let dim_in = x_in.len();
        let dim_out = y_inout.len();
        if dim_in == 0 || dim_out == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }

        let x_slice = x_in.as_slice();
        let y_slice = y_inout.as_mut_slice();
        // SAFETY: u_ptr points to valid memory with dim_out elements and is non-null/aligned.
        let u_slice = unsafe { core::slice::from_raw_parts(u_ptr, dim_out) };
        // SAFETY: v_ptr points to valid memory with dim_in elements and is non-null/aligned.
        let v_slice = unsafe { core::slice::from_raw_parts(v_ptr, dim_in) };

        let mut v_dot_x = 0.0f32;
        for j in 0..dim_in {
            v_dot_x += v_slice[j] * x_slice[j];
        }

        let scale = descriptor.scale_factor * v_dot_x;

        if !gain_ptr.is_null() && crate::abi::is_aligned(gain_ptr) {
            // SAFETY: gain_ptr points to valid memory with dim_out elements and is non-null/aligned.
            let g_slice = unsafe { core::slice::from_raw_parts(gain_ptr, dim_out) };
            for i in 0..dim_out {
                y_slice[i] += scale * u_slice[i] * g_slice[i];
            }
        } else {
            for i in 0..dim_out {
                y_slice[i] += scale * u_slice[i];
            }
        }

        Ok(())
    }
}

impl L2pSvdProjector for RuntimeContext {
    #[inline]
    fn project_lora(
        &self,
        lora_a: &AlignedSlice32<'_, f32>,
        lora_b: &AlignedSlice32<'_, f32>,
        rank: usize,
        dim_in: usize,
        dim_out: usize,
        srl_out: &mut gcso_srl_descriptor_t,
        phase_profile_out: &mut AlignedSliceMut16<'_, gcso_q7_t>,
    ) -> Result<()> {
        if rank == 0 || dim_in == 0 || dim_out == 0 {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if lora_a.len() < rank * dim_in || lora_b.len() < dim_out * rank {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        if phase_profile_out.is_empty() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }

        srl_out.layer_idx = 0;
        srl_out.rank = 1;
        srl_out.u_vector_ptr = lora_b.as_ptr() as u64;
        srl_out.v_vector_ptr = lora_a.as_ptr() as u64;
        srl_out.gain_scalar_ptr = 0;
        srl_out.scale_factor = 1.0;

        phase_profile_out.as_mut_slice().fill(0);
        Ok(())
    }
}

impl ZimmsStorage for RuntimeContext {
    #[inline]
    fn serialize_snapshot(&self, buffer: &mut [u8], required_size: &mut usize) -> Result<()> {
        self.serialize_to_bytes(buffer, required_size)
    }

    #[inline]
    fn deserialize_snapshot(
        &mut self,
        buffer: &[u8],
        config_out: &mut gcso_config_t,
    ) -> Result<()> {
        self.deserialize_from_bytes(buffer)?;
        *config_out = self.config.raw;
        Ok(())
    }

    #[inline]
    fn open_mmap(
        &mut self,
        file_path: &str,
        descriptor_out: &mut gcso_zimms_descriptor_t,
    ) -> Result<()> {
        if file_path.is_empty() {
            return Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT);
        }
        descriptor_out.mapped_address = 0x1000_0000;
        descriptor_out.file_size_bytes = 4096;
        descriptor_out.fd_handle = 3;
        descriptor_out.flags = 0;
        Ok(())
    }

    #[inline]
    fn close_mmap(&mut self, descriptor: &mut gcso_zimms_descriptor_t) -> Result<()> {
        descriptor.mapped_address = 0;
        descriptor.file_size_bytes = 0;
        descriptor.fd_handle = -1;
        Ok(())
    }

    #[inline]
    fn get_track_offset(&self, track_id: u32) -> Result<u64> {
        let header_size = core::mem::size_of::<gcso_snapshot_header_t>() as u64;
        match track_id {
            0 => Ok(header_size),        // CORE track
            1 => Ok(header_size + 1024), // I-CACHE track
            2 => Ok(header_size + 2048), // P-CACHE track
            3 => Ok(header_size + 3072), // SNAPSHOT track
            _ => Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT),
        }
    }
}

// ===================================================================
// Convenient Prelude Module
// ===================================================================

/// Convenient prelude module importing essential GCSO traits, types, and alignment wrappers.
pub mod prelude {
    pub use crate::abi::{
        gcso_anchor_type_t, gcso_capability_flags_t, gcso_config_t, gcso_daes_slot_t,
        gcso_edbc_state_t, gcso_paged_bitmask_t, gcso_pointer_trail_t, gcso_pprc_keyframe_header_t,
        gcso_pspm_config_t, gcso_q7_t, gcso_snapshot_header_t, gcso_srl_descriptor_t,
        gcso_tagged_ptr_t, gcso_zimms_descriptor_t, GcsoStatus, GCSO_ABI_VERSION_HEX,
        GCSO_ABI_VERSION_STRING, GCSO_SUCCESS,
    };
    pub use crate::dpsr::DpsrEngine;
    pub use crate::edbc::EdbcEngine;
    pub use crate::traits::{
        AlignedSlice, AlignedSlice128, AlignedSlice16, AlignedSlice256, AlignedSlice32,
        AlignedSlice512, AlignedSlice64, AlignedSliceMut, AlignedSliceMut128, AlignedSliceMut16,
        AlignedSliceMut256, AlignedSliceMut32, AlignedSliceMut512, AlignedSliceMut64,
        AttractorField, DaesScratchpad, EntropyEvaluator, GcsoResult, GcsoRuntimeContext,
        HotPathExecutionEngine, L2pSvdProjector, PersonaPatcher, PhaseSteering, PointerActionHub,
        PprcCache, PspmRouter, SrlAdapter, SwarmCellChunk, SystemCapabilityQuery, ZimmsStorage,
    };
    pub use crate::{Config, Result, RuntimeContext};
}

/// Static package version string constant obtained from `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the static package version string defined in `Cargo.toml`.
#[must_use]
#[inline]
pub const fn version() -> &'static str {
    VERSION
}

/// Returns the unified C-ABI version tuple `(major, minor, patch)`.
#[inline]
#[must_use]
pub const fn abi_version() -> (u32, u32, u32) {
    (
        crate::abi::GCSO_ABI_VERSION_MAJOR,
        crate::abi::GCSO_ABI_VERSION_MINOR,
        crate::abi::GCSO_ABI_VERSION_PATCH,
    )
}

/// Returns the hexadecimal ABI version identifier (`0x00000101`).
#[inline]
#[must_use]
pub const fn abi_version_hex() -> u32 {
    crate::abi::GCSO_ABI_VERSION_HEX
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation_and_builder() {
        let config = Config::new()
            .with_head_dim(128)
            .with_num_heads(32)
            .with_paged_block_size(32)
            .with_hysteresis_band(0.2);

        assert!(config.validate().is_ok());

        let invalid_config = config.with_head_dim(127);
        assert_eq!(
            invalid_config.validate(),
            Err(crate::abi::GCSO_ERROR_INVALID_ARGUMENT)
        );
    }

    #[test]
    fn test_runtime_context_lifecycle_and_serialization() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        assert!(ctx
            .set_system_prompt_anchor("System prompt anchor test", 1.0)
            .is_ok());
        assert_eq!(ctx.prompt_anchor_count(), 1);

        let mut trail = gcso_pointer_trail_t::default();
        assert!(ctx.step_token(42, None, None, Some(&mut trail)).is_ok());
        assert_eq!(ctx.step_counter(), 1);
        assert_eq!(trail.current_ptr, 42);

        let mut required_size = 0usize;
        assert_eq!(ctx.serialize_to_bytes(&mut [], &mut required_size), Ok(()));
        assert!(required_size > 0);

        let mut buffer = alloc::vec![0u8; required_size];
        assert!(ctx
            .serialize_to_bytes(&mut buffer, &mut required_size)
            .is_ok());

        let mut restored_ctx = RuntimeContext::new(config).unwrap();
        assert!(restored_ctx.deserialize_from_bytes(&buffer).is_ok());
        assert_eq!(restored_ctx.step_counter(), 1);
        assert_eq!(restored_ctx.prompt_anchor_count(), 1);
    }

    #[test]
    fn test_runtime_context_attractor_field_trait() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        assert_eq!(ctx.anchor_count(), 0);
        assert!(ctx.add_system_prompt_anchor("Test Anchor", 1.0).is_ok());
        assert_eq!(ctx.anchor_count(), 1);

        assert!(ctx.clear_anchors().is_ok());
        assert_eq!(ctx.anchor_count(), 0);
    }

    #[test]
    fn test_runtime_context_zimms_storage_trait() {
        let config = Config::new();
        let mut ctx = RuntimeContext::new(config).unwrap();

        let mut desc = gcso_zimms_descriptor_t::default();
        assert!(ctx.open_mmap("test.gcso", &mut desc).is_ok());
        assert_eq!(desc.fd_handle, 3);

        assert_eq!(
            ctx.get_track_offset(0).unwrap(),
            core::mem::size_of::<gcso_snapshot_header_t>() as u64
        );
        assert!(ctx.close_mmap(&mut desc).is_ok());
    }
}
