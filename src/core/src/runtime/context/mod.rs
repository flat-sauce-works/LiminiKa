// name: src/core/src/runtime/context/mod.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! High-level configuration builder ([`Config`]), [`CoreStateStack`] (128B `Pod`),
//! [`OrchestratorStatusSnapshot`], unified [`RuntimeContext`], and capability query helpers.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]

use crate::drivers::{
    AnonymousComputePool, DistortionTension, EgoDifferentialNet, EntropyController, EntropyState,
    LiminalEgoState, SystemSpineState,
};
use crate::foundation::{
    branchless_clamp_f32, branchless_is_finite_f32, Pod, DEFAULT_PHASE_CLAMP_MAX_RAD,
    DEFAULT_PHASE_STEP_MIN_RAD,
};
use crate::liminal::{
    branchless_select_f32, branchless_select_u32, branchless_select_u8, LacunaPartition8,
    LiminalThresholdState, LmnkResult, RuntimeConfig, DEFAULT_ENTROPY_EPS, LMNK_ABI_VERSION_MAJOR,
    LMNK_ABI_VERSION_MINOR, LMNK_ABI_VERSION_PATCH, LMNK_ABI_VERSION_STRING,
    LMNK_CAP_INPLACE_CONTEXT_INIT, LMNK_CAP_SIMD_AVX2, LMNK_CAP_SIMD_NEON, LMNK_CAP_SIMD_WASM128,
    LMNK_CAP_SWAR_BASELINE, LMNK_CAP_ZERO_ALLOC_CORE, LMNK_ERR_INVALID_ARGUMENT,
    MAX_ATTENTION_HEADS, MAX_ATTRACTOR_DOMAINS, Q7_PHASE_SCALE_RAD,
};
use crate::runtime::boundary_delta::LiminalBoundaryDelta;
use crate::runtime::gateway::{
    CausalProvenanceSignal, GatewayFrame, GatewayPhaseSteeringExt, ManifoldGatewaySignal,
    ManifoldProbeBaseline, SubstrateGatewayExt, SystemConstraintSignal,
};
use crate::runtime::horizon::{EpistemicHorizonReading, HorizonDwellState};
use crate::runtime::telemetry::{CoreTelemetryBank, TelemetryCapturePolicy};
use crate::stigmergy::{
    CognitiveTrajectoryLedger, CorePrimitiveState, DissipativeOverflowReport,
    OverlapDiscoveryReport, PhaseSteeringEngine, SandboxLifecycleEvent, StigmergicGranuleState,
    StigmergicSubstrate, StigmergicSubstrateOps, DEFAULT_SUBSTRATE_CAPACITY, MAX_PHASE_ANCHORS,
};

mod accessors;
mod config;
mod orchestration;
mod snapshot;
mod traits;

pub use config::Config;

/// Default number of tokens per PagedAttention KV cache block (`32`).
pub const DEFAULT_PAGED_BLOCK_SIZE: u32 = 32;

/// Returns the active semantic ABI version tuple `(major, minor, patch)`.
#[inline(always)]
#[must_use]
pub const fn abi_version() -> (u32, u32, u32) {
    (
        LMNK_ABI_VERSION_MAJOR,
        LMNK_ABI_VERSION_MINOR,
        LMNK_ABI_VERSION_PATCH,
    )
}

/// Returns the static ABI version string ([`LMNK_ABI_VERSION_STRING`]).
#[inline(always)]
#[must_use]
pub const fn abi_version_string() -> &'static str {
    LMNK_ABI_VERSION_STRING
}

/// Queries active compute capability bitmask flags without branching (`LMNK_CAP_*`).
#[inline(always)]
#[must_use]
pub const fn query_capability() -> u64 {
    LMNK_CAP_SWAR_BASELINE
        | ((cfg!(target_feature = "avx2") as u64) * LMNK_CAP_SIMD_AVX2)
        | ((cfg!(target_feature = "neon") as u64) * LMNK_CAP_SIMD_NEON)
        | ((cfg!(target_feature = "simd128") as u64) * LMNK_CAP_SIMD_WASM128)
        | LMNK_CAP_ZERO_ALLOC_CORE
        | LMNK_CAP_INPLACE_CONTEXT_INIT
}

/// Populates default runtime configuration parameters conforming to LiminiKa architectural invariants.
///
/// # Errors
/// Never returns an error in the default implementation.
#[inline(always)]
pub fn init_default_config(config: &mut RuntimeConfig) -> LmnkResult<()> {
    *config = RuntimeConfig {
        head_dim: 128,
        num_heads: 32,
        paged_block_size: DEFAULT_PAGED_BLOCK_SIZE,
        q7_phase_scale: Q7_PHASE_SCALE_RAD,
        phase_clamp_max_rad: DEFAULT_PHASE_CLAMP_MAX_RAD,
        phase_step_min_rad: DEFAULT_PHASE_STEP_MIN_RAD,
        entropy_singularity_eps: DEFAULT_ENTROPY_EPS,
        max_prompt_anchors: MAX_PHASE_ANCHORS as u32,
        substrate_capacity: DEFAULT_SUBSTRATE_CAPACITY,
        execution_flags: 0,
        enable_zero_alloc_strict: 1,
        ledger_mode: 0,
        rotary_dim: 0,
        rope_layout: 0,
        guard_hold_tokens: 0,
        max_observe_stride: 1,
        reserved: [0; 18],
    };
    Ok(())
}

/// Branchlessly validates a [`RuntimeConfig`] descriptor against LiminiKa architectural invariants.
///
/// # Errors
/// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if any dimension, scale, or capacity invariant is violated.
#[inline(always)]
pub fn validate_config_invariants(config: &RuntimeConfig) -> LmnkResult<()> {
    let valid_dims = (config.head_dim != 0)
        & ((config.head_dim & 1) == 0)
        & (config.num_heads != 0)
        & ((config.num_heads as usize) <= MAX_ATTENTION_HEADS)
        & (config.rotary_dim <= config.head_dim)
        & ((config.rotary_dim & 1) == 0)
        & (config.rope_layout <= 1)
        & ((config.paged_block_size == 16) | (config.paged_block_size == 32));
    let valid_scales = branchless_is_finite_f32(config.q7_phase_scale)
        & (config.q7_phase_scale > 0.0)
        & branchless_is_finite_f32(config.phase_clamp_max_rad)
        & (config.phase_clamp_max_rad > 0.0)
        & branchless_is_finite_f32(config.phase_step_min_rad)
        & (config.phase_step_min_rad >= 0.0)
        & branchless_is_finite_f32(config.entropy_singularity_eps)
        & (config.entropy_singularity_eps > 0.0);
    let valid_caps = (config.max_prompt_anchors != 0)
        & (config.substrate_capacity != 0)
        & (config.ledger_mode <= 2);

    if valid_dims & valid_scales & valid_caps {
        Ok(())
    } else {
        Err(LMNK_ERR_INVALID_ARGUMENT)
    }
}

// ===================================================================
// 7. Unified 3-Slot Autonomous Core Stack (128 B = 2 Cache Lines)
// ===================================================================

/// Unified 128-byte (2 cache lines, `#[repr(C, align(64))]`) 3-slot autonomous core stack pairing:
/// - **Slot 0 ([`LiminalThresholdState`], 32 B)**: Epistemic Horizon & Trait Cognitive Apertures
/// - **Slot 1 ([`StigmergicGranuleState`], 32 B)**: Stigmergic Granule Approximation, Overlap & Discovery
/// - **Slot 2 ([`LiminalEgoState`], 64 B)**: Top-Down Subjective Ego Boundary & Phase Intervention
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CoreStateStack {
    /// Core Stack Slot 0: Epistemic Horizon & Trait Cognitive Aperture Controller (`offset 0..32`).
    pub liminal: LiminalThresholdState,
    /// Core Stack Slot 1: Stigmergic Granule Approximation & Discovery Controller (`offset 32..64`).
    pub stigmergy: StigmergicGranuleState,
    /// Core Stack Slot 2: Wide-Aperture Subjective Ego Controller (`offset 64..128`).
    pub ego: LiminalEgoState,
}

impl CoreStateStack {
    /// Constructs a neutral 128-byte [`CoreStateStack`] with all 3 core slots initialized.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            liminal: LiminalThresholdState::new(),
            stigmergy: StigmergicGranuleState::new(),
            ego: LiminalEgoState::new(),
        }
    }

    /// Returns a 64-byte [`CorePrimitiveState`] view uniting the two foundational primitives
    /// (`LiminalThresholdState` + `StigmergicGranuleState`).
    #[inline(always)]
    #[must_use]
    pub const fn core_primitive_state(&self) -> CorePrimitiveState {
        CorePrimitiveState {
            liminal: self.liminal,
            stigmergy: self.stigmergy,
        }
    }

    /// Replaces the `Liminal` + `Stigmergy` halves from `primitives`.
    #[inline(always)]
    pub const fn set_core_primitive_state(&mut self, primitives: CorePrimitiveState) {
        self.liminal = primitives.liminal;
        self.stigmergy = primitives.stigmergy;
    }

    /// Branchlessly co-adapts Slot 0 (`LiminalThresholdState`) and Slot 1 (`StigmergicGranuleState`)
    /// with cross-slot feedback (exploratory discovery in Slot 1 widens Slot 0's horizon).
    #[inline]
    pub fn adapt_layers_branchless(
        &mut self,
        ledger: &CognitiveTrajectoryLedger,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
        overflow_opt: Option<&DissipativeOverflowReport>,
    ) {
        self.liminal
            .adapt_from_circuit_and_overflow_branchless(ledger, horizon, overflow_opt);
        let discovered = overlap_report.discovery_triggered != 0;
        self.liminal.threshold_floor_q7[1] = branchless_select_u8(
            discovered,
            self.liminal.threshold_floor_q7[1].saturating_sub(4).max(8),
            self.liminal.threshold_floor_q7[1],
        );
        self.liminal.threshold_floor_q7[6] = branchless_select_u8(
            discovered,
            self.liminal.threshold_floor_q7[6].saturating_sub(4).max(8),
            self.liminal.threshold_floor_q7[6],
        );
        self.liminal.aperture_gain_q7[6] = branchless_select_u8(
            discovered,
            self.liminal.aperture_gain_q7[6].saturating_add(6).min(127),
            self.liminal.aperture_gain_q7[6],
        );
        self.liminal.active_aperture_mask |=
            branchless_select_u32(discovered, (1 << 1) | (1 << 6), 0);

        self.stigmergy
            .adapt_from_horizon_and_overlap_branchless(horizon, overlap_report, ledger);
    }

    /// Branchlessly modulates `frame` in-place across Core Stack Slot 0 (`Liminal`) and Slot 1 (`Stigmergic`).
    #[inline(always)]
    pub fn modulate_gateway_frame_branchless(
        &self,
        frame: &mut GatewayFrame,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
    ) {
        self.liminal
            .modulate_gateway_frame_branchless(frame, horizon);
        let core_cnt = overlap_report.overlap_core_mask.count_ones() as f32;
        let has_overlap =
            (overlap_report.overlap_core_mask | overlap_report.overlap_fringe_mask) != 0;
        let coop_boost = branchless_clamp_f32(1.0 + 0.05 * core_cnt, 1.0, 1.40);
        let liminal_gated = horizon.is_liminal()
            | (horizon.total_memory_support < self.liminal.channel_threshold_floor_f32(1));
        frame.effective_energy = branchless_select_f32(
            !liminal_gated & (overlap_report.overlap_core_mask != 0),
            frame.effective_energy * coop_boost,
            frame.effective_energy,
        );
        let active_granules = overlap_report.overlap_core_mask
            | overlap_report.overlap_fringe_mask
            | overlap_report.discovery_excursion_mask;
        frame.active_swarm_cell_mask =
            branchless_select_u8(has_overlap, active_granules, frame.active_swarm_cell_mask);
    }

    /// Branchlessly modulates `frame` in-place across Core Stack Slot 0 (`Liminal`), Slot 1
    /// (`Stigmergic`), and Slot 2 (`Ego`) **without mutating** sub-slot state or
    /// `intervention_epochs` (for read-only per-layer Hot-Path RoPE modulation across all
    /// transformer layers).
    #[inline]
    pub fn modulate_gateway_frame_with_ego_readonly(
        &self,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
        frame: &mut GatewayFrame,
    ) {
        self.modulate_gateway_frame_branchless(frame, horizon, overlap_report);
        if self.ego.is_active() {
            let _ = self.ego.modulate_frame_pure(
                &self.liminal,
                &self.stigmergy,
                delta,
                horizon,
                overlap_report,
                frame,
            );
        }
    }

    /// Branchlessly modulates `frame` in-place across Core Stack Slot 0 (`Liminal`) and Slot 1
    /// (`Stigmergic`) and, when Core Stack Slot 2 (`ego`) is active, performs in-place Ego
    /// intervention across all three slots and `frame`.
    #[inline]
    pub fn intervene_ego_and_modulate_frame_branchless(
        &mut self,
        delta: &LiminalBoundaryDelta,
        horizon: &EpistemicHorizonReading,
        overlap_report: &OverlapDiscoveryReport,
        frame: &mut GatewayFrame,
    ) {
        self.modulate_gateway_frame_branchless(frame, horizon, overlap_report);
        if self.ego.is_active() {
            self.ego.intervene_on_sublayers_and_frame_branchless(
                &mut self.liminal,
                &mut self.stigmergy,
                delta,
                horizon,
                overlap_report,
                frame,
            );
        }
    }
}

/// Convenient crate-level result type alias for LiminiKa runtime operations.
pub type Result<T> = LmnkResult<T>;

/// Default `EntropyController` hysteresis band width ($\delta_{\mathrm{hyst}}$).
pub const DEFAULT_ENTROPY_GATE_HYSTERESIS_BAND: f32 = 0.15;

/// Default sliding-window entropy rate flux threshold ($\tau_{\mathrm{flux}}$).
pub const DEFAULT_ENTROPY_GATE_TAU_FLUX: f32 = 0.50;

/// Default `EntropyController` pitchfork bifurcation Z-score threshold ($\tau_{\mathrm{bifurcation}}$).
pub const DEFAULT_ENTROPY_GATE_BIFURCATION_THRESHOLD: f32 = 1.50;

/// Default Contrastive Phase Bloom Filter collision count threshold triggering anti-phase repulsion.
pub const DEFAULT_BLOOM_REPULSION_THRESHOLD: u8 = 2;

/// Default Q7 unit scaling factor (`128` = $1.0\times$) for phase-conjugate repulsion injection.
pub const DEFAULT_BLOOM_REPULSION_SCALE_U7: u8 = 128;

/// Total Q7 phase elements across all 8 Attractor Working Lanes (`8 * 64 = 512` bytes).
pub const SLOT_PHASE_BANK_LEN: usize = MAX_ATTRACTOR_DOMAINS * MAX_ATTENTION_HEADS;

/// Default initial weights for the 8 anonymous free-by-default Workbench Lanes (`0..=7`).
pub const DEFAULT_SLOT_WEIGHTS: [f32; MAX_ATTRACTOR_DOMAINS] = [0.0; MAX_ATTRACTOR_DOMAINS];

// ===================================================================
// High-Level Safe Rust Runtime Context Engine
// ===================================================================

/// Single-glance status snapshot summarizing the active concrete core engines, the 3-layer
/// autonomous core stack (`Liminal` + `Stigmergic` + `Ego`), and dynamically slid-in horizon slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OrchestratorStatusSnapshot {
    /// Total tokens/steps processed by `StigmergicSubstrate`.
    pub step_counter: u64,
    /// Number of active trails registered in `StigmergicSubstrate`.
    pub active_trail_count: u32,
    /// Active `EntropyController` branch mode (`0` = normal, `1` = bifurcation, `2` = repulsion).
    pub entropy_gate_mode: u32,
    /// Active enabled working-lane count in [`StigmergicSubstrate`].
    pub attractor_anchor_count: u32,
    /// Bitmask of enabled working lanes (`0..=7`) in `StigmergicSubstrate`.
    pub slot_enabled_mask: u8,
    /// Bitmask of explicitly pinned working lanes (`0..=7`) in `StigmergicSubstrate`.
    pub slot_pinned_mask: u8,
    /// Bitmask of anti-phase repulsion working lanes (`0..=7`) in `StigmergicSubstrate`.
    pub slot_repulsion_mask: u8,
    /// Bitmask of isolated exterior-only sandbox working lanes (`0..=7`) in `StigmergicSubstrate`.
    pub slot_sandbox_mask: u8,
    /// Active Core Stack Slot 0 `Liminal` channel aperture mask (`8` channels).
    pub liminal_threshold_mask: u32,
    /// Active Core Stack Slot 1 `Stigmergic` overlap-core granule mask.
    pub stigmergic_overlap_core_mask: u8,
    /// Whether Core Stack Slot 2 `LiminalEgoState` intervention is active.
    pub ego_active: bool,
    /// Monotonic count of Core Stack Slot 2 Ego interventions applied.
    pub ego_intervention_epochs: u32,
    /// Whether Core Stack Slot 0 boundaries are quiescent (`stationary_shortcut_ready != 0`).
    pub stationary_shortcut_ready: bool,
    /// Bitmask of external [`crate::runtime::horizon::LatentHorizonSlot`] traits currently slid into the cognitive window.
    pub active_horizon_slots_mask: u32,
    /// Consecutive destabilized steps observed by the orchestrator.
    pub unstable_dwell_steps: u16,
    /// Consecutive quiescent steps observed by the orchestrator.
    pub quiescent_dwell_steps: u16,
}

/// Allocation provenance tag for [`RuntimeContext::allocation_mode`]: stack / uninitialized.
pub(crate) const CTX_ALLOC_UNINIT: u8 = 0;
/// Allocation provenance tag for [`RuntimeContext::allocation_mode`]: heap `Box<RuntimeContext>`.
pub(crate) const CTX_ALLOC_HEAP_BOXED: u8 = 1;
/// Allocation provenance tag for [`RuntimeContext::allocation_mode`]: caller-owned `in_place` memory.
pub(crate) const CTX_ALLOC_IN_PLACE: u8 = 2;

/// Exact payload byte size (`384` bytes) of a serialized `RuntimeSnapshot` section.
pub const LMNK_RUNTIME_SNAPSHOT_PAYLOAD_BYTES: usize = 384;

/// Exact total byte size (`512` bytes) of a serialized single-section `RuntimeContext` container.
pub const LMNK_RUNTIME_SNAPSHOT_TOTAL_BYTES: usize = 512;

/// Deterministically scrubs `size_of::<T>()` bytes at `dst` to zero using volatile stores
/// followed by a `SeqCst` compiler fence to prevent dead-store elimination.
///
/// # Safety
/// Caller must ensure `dst` is non-null, writable for `size_of::<T>()` bytes, and not aliased.
#[inline(never)]
pub(crate) unsafe fn zeroize_volatile<T>(dst: *mut T) {
    let byte_len = core::mem::size_of::<T>();
    if dst.is_null() || byte_len == 0 {
        return;
    }
    let word_bytes = core::mem::size_of::<usize>();
    let mut ptr_u8 = dst.cast::<u8>();
    let mut rem = byte_len;

    if ((ptr_u8 as usize) & (core::mem::align_of::<usize>() - 1)) == 0 {
        let word_count = rem / word_bytes;
        let mut ptr_word = ptr_u8.cast::<usize>();
        let mut w = 0usize;
        while w < word_count {
            // SAFETY: `ptr_u8` is naturally `usize`-aligned and within `byte_len`.
            unsafe {
                core::ptr::write_volatile(ptr_word, 0usize);
                ptr_word = ptr_word.add(1);
            }
            w += 1;
        }
        ptr_u8 = ptr_word.cast::<u8>();
        rem %= word_bytes;
    }

    let mut b = 0usize;
    while b < rem {
        // SAFETY: `ptr_u8.add(b)` stays within the writable `size_of::<T>()` allocation.
        unsafe {
            core::ptr::write_volatile(ptr_u8.add(b), 0u8);
        }
        b += 1;
    }
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// High-level safe Rust single orchestrator facade integrating concrete core engines
/// ([`StigmergicSubstrate`], [`PhaseSteeringEngine`], [`EntropyController`]),
/// the 128-byte 3-slot core stack ([`CoreStateStack`]: `Liminal` + `Stigmergic` + `Ego`),
/// and dynamic slide-in/slide-out [`crate::runtime::horizon::LatentHorizonSlot`] boundaries.
///
/// Field ordering follows Hot -> Warm -> Cold cache locality under `#[repr(C, align(64))]`,
/// guaranteeing `size_of % 64 == 0` and zero [`core::mem::needs_drop`] glue for caller-managed
/// zero-heap `in_place` initialization.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct RuntimeContext {
    // --- Hot (per-token step working set) ---
    config: Config,
    core_layers: CoreStateStack,
    entropy_controller: EntropyController,
    system_spine: SystemSpineState,
    last_horizon: EpistemicHorizonReading,
    last_overlap_report: OverlapDiscoveryReport,
    distortion_tension: DistortionTension,
    phase_steering: PhaseSteeringEngine,
    last_lacuna_partition: LacunaPartition8,
    last_boundary_delta: LiminalBoundaryDelta,
    probe_baseline: ManifoldProbeBaseline,
    horizon_dwell: HorizonDwellState,
    active_horizon_slots_mask: u32,
    sandbox_settlements: u32,
    escalated_workbench_lane: Option<u8>,
    sandbox_workbench_lane: Option<u8>,
    last_sandbox_event: SandboxLifecycleEvent,
    allocation_mode: u8,
    // --- Warm (substrate & differential network) ---
    compute_pool: AnonymousComputePool,
    ego_differential_net: EgoDifferentialNet,
    substrate: StigmergicSubstrate,
    // --- Cold (telemetry counters & breakdown) ---
    telemetry: CoreTelemetryBank,
}

const _: () = {
    assert!(core::mem::size_of::<RuntimeContext>() == 5120);
    assert!(core::mem::align_of::<RuntimeContext>() == 64);
    assert!(core::mem::offset_of!(RuntimeContext, system_spine) == 384);
    assert!(core::mem::offset_of!(RuntimeContext, telemetry) == 4544);
    assert!(!core::mem::needs_drop::<RuntimeContext>());
};

impl RuntimeContext {
    /// Creates a new verified [`RuntimeContext`] instance from [`Config`].
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if configuration parameters violate
    /// LiminiKa architectural invariants.
    #[inline]
    pub fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let phase_steering = PhaseSteeringEngine::new(config.raw.head_dim, config.raw.num_heads)?;
        let mut substrate = StigmergicSubstrate::from_config(&config.raw)?;
        substrate.repulsion_threshold = config.repulsion_threshold;
        substrate.repulsion_scale_u7 = config.repulsion_scale_u7;
        substrate.coupling_matrix_8x8 = config.slot_coupling_matrix;
        let entropy_state = EntropyState {
            singularity_eps: config.raw.entropy_singularity_eps,
            bifurcation_threshold: config.bifurcation_threshold,
            ..Default::default()
        };
        let mut entropy_ctrl = EntropyController::new(entropy_state);
        entropy_ctrl.set_hysteresis_band(config.hysteresis_band);
        entropy_ctrl.set_tau_flux(config.tau_flux);

        let mut config = config;
        if config.raw.max_observe_stride == 0 {
            config.raw.max_observe_stride = 1;
        }
        let system_spine = SystemSpineState::with_policy(
            config.raw.guard_hold_tokens,
            config.raw.max_observe_stride,
        );

        Ok(Self {
            config,
            core_layers: CoreStateStack::new(),
            entropy_controller: entropy_ctrl,
            probe_baseline: ManifoldProbeBaseline::new(),
            system_spine,
            last_horizon: EpistemicHorizonReading::default(),
            last_overlap_report: OverlapDiscoveryReport::default(),
            distortion_tension: DistortionTension::default(),
            phase_steering,
            last_lacuna_partition: LacunaPartition8::default(),
            last_boundary_delta: LiminalBoundaryDelta::default(),
            horizon_dwell: HorizonDwellState::default(),
            active_horizon_slots_mask: 0,
            sandbox_settlements: 0,
            escalated_workbench_lane: None,
            sandbox_workbench_lane: None,
            last_sandbox_event: SandboxLifecycleEvent::Idle,
            allocation_mode: CTX_ALLOC_UNINIT,
            compute_pool: AnonymousComputePool::default(),
            ego_differential_net: EgoDifferentialNet::default(),
            substrate,
            telemetry: CoreTelemetryBank::default(),
        })
    }

    /// Initializes a caller-allocated `RuntimeContext` buffer in-place with zero heap
    /// allocation and without staging the full ~5 KB struct on the stack.
    ///
    /// # Safety
    /// Caller must ensure `dst` is non-null, properly aligned (`align_of::<RuntimeContext>() == 64`),
    /// writable for `size_of::<RuntimeContext>()` bytes, and not currently aliased.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] before modifying `dst` if `config` violates
    /// architectural invariants or dimension limits.
    pub unsafe fn write_in_place(dst: *mut Self, config: Config) -> Result<()> {
        config.validate()?;
        if config.raw.substrate_capacity == 0
            || config.raw.head_dim > u16::MAX as u32
            || config.raw.num_heads > u16::MAX as u32
        {
            return Err(LMNK_ERR_INVALID_ARGUMENT);
        }
        let phase_steering = PhaseSteeringEngine::new(config.raw.head_dim, config.raw.num_heads)?;
        let entropy_state = EntropyState {
            singularity_eps: config.raw.entropy_singularity_eps,
            bifurcation_threshold: config.bifurcation_threshold,
            ..Default::default()
        };
        let mut entropy_ctrl = EntropyController::new(entropy_state);
        entropy_ctrl.set_hysteresis_band(config.hysteresis_band);
        entropy_ctrl.set_tau_flux(config.tau_flux);

        let mut config = config;
        if config.raw.max_observe_stride == 0 {
            config.raw.max_observe_stride = 1;
        }
        let raw_cfg = config.raw;
        let rep_thresh = config.repulsion_threshold;
        let rep_scale_u7 = config.repulsion_scale_u7;
        let coupling = config.slot_coupling_matrix;

        // SAFETY: Caller guarantees `dst` is valid, writable, non-aliased, and 64-byte aligned.
        unsafe {
            core::ptr::write_bytes(dst, 0u8, 1);
            core::ptr::write(core::ptr::addr_of_mut!((*dst).config), config);
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).core_layers),
                CoreStateStack::new(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).entropy_controller),
                entropy_ctrl,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).probe_baseline),
                ManifoldProbeBaseline::new(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).system_spine),
                SystemSpineState::with_policy(
                    raw_cfg.guard_hold_tokens,
                    raw_cfg.max_observe_stride,
                ),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_horizon),
                EpistemicHorizonReading::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_overlap_report),
                OverlapDiscoveryReport::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).distortion_tension),
                DistortionTension::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).phase_steering),
                phase_steering,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_lacuna_partition),
                LacunaPartition8::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_boundary_delta),
                LiminalBoundaryDelta::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).horizon_dwell),
                HorizonDwellState::default(),
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).escalated_workbench_lane),
                None,
            );
            core::ptr::write(core::ptr::addr_of_mut!((*dst).sandbox_workbench_lane), None);
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).last_sandbox_event),
                SandboxLifecycleEvent::Idle,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).allocation_mode),
                CTX_ALLOC_IN_PLACE,
            );
            core::ptr::write(
                core::ptr::addr_of_mut!((*dst).compute_pool),
                AnonymousComputePool::default(),
            );
            EgoDifferentialNet::init_in_place_on_zeroed(
                core::ptr::addr_of_mut!((*dst).ego_differential_net),
                1,
            );
            StigmergicSubstrate::init_in_place_on_zeroed(
                core::ptr::addr_of_mut!((*dst).substrate),
                &raw_cfg,
                rep_thresh,
                rep_scale_u7,
                coupling,
            );
            CoreTelemetryBank::init_in_place_on_zeroed(
                core::ptr::addr_of_mut!((*dst).telemetry),
                TelemetryCapturePolicy::default(),
            );
        }
        Ok(())
    }

    /// Allocates and initializes a [`RuntimeContext`] directly on the heap via `Box::new_uninit`
    /// without staging the 5,120-byte structure on the caller's stack.
    ///
    /// # Errors
    /// Returns [`LMNK_ERR_INVALID_ARGUMENT`] if `config` violates architectural invariants or
    /// dimension limits.
    pub fn new_boxed(config: Config) -> Result<alloc::boxed::Box<Self>> {
        let mut uninit = alloc::boxed::Box::<Self>::new_uninit();
        // SAFETY: `uninit.as_mut_ptr()` points to a valid, 64B-aligned, writable, non-aliased
        // `RuntimeContext` heap allocation; `write_in_place` validates `config` before writing
        // and initializes every field upon returning `Ok(())`.
        unsafe {
            Self::write_in_place(uninit.as_mut_ptr(), config)?;
            let mut boxed = uninit.assume_init();
            boxed.allocation_mode = CTX_ALLOC_HEAP_BOXED;
            Ok(boxed)
        }
    }

    /// Scrubs all 5,120 bytes of a raw [`RuntimeContext`] pointer via volatile zeroization without
    /// creating any intermediate Rust reference (preserving Stacked Borrows provenance for FFI teardown).
    ///
    /// # Safety
    /// Caller must ensure `ptr` is non-null, properly aligned (`align_of::<RuntimeContext>() == 64`),
    /// writable for `size_of::<RuntimeContext>()` bytes, not aliased, and never subsequently read as
    /// a live Rust [`RuntimeContext`] without re-initializing non-zero-guaranteed enum fields.
    #[inline]
    pub(crate) unsafe fn zeroize_raw(ptr: *mut Self) {
        // SAFETY: Caller upholds non-nullness, 64B alignment, exclusive writability, and `!needs_drop`.
        unsafe {
            zeroize_volatile(ptr);
        }
    }

    /// Reads [`RuntimeContext::allocation_mode`] directly via raw pointer projection without forming
    /// an intermediate `&RuntimeContext` reference that could invalidate teardown write provenance.
    ///
    /// # Safety
    /// Caller must ensure `ptr` is non-null, naturally aligned, and points to an initialized
    /// [`RuntimeContext`] allocation.
    #[inline]
    #[must_use]
    pub(crate) unsafe fn read_allocation_mode_raw(ptr: *const Self) -> u8 {
        // SAFETY: `allocation_mode` is a `u8` field inside a valid initialized `RuntimeContext`.
        unsafe { core::ptr::addr_of!((*ptr).allocation_mode).read() }
    }

    /// Scrubs all 5,120 bytes of this [`RuntimeContext`] via volatile zeroization and restores valid
    /// empty enum discriminants so the reference remains structurally valid without drop glue.
    pub fn zeroize_in_place(&mut self) {
        let raw = core::ptr::from_mut(self);
        // SAFETY: `raw` derives from a uniquely borrowed 5,120-byte `RuntimeContext` with `!needs_drop`
        // and is never re-borrowed through `self` after `zeroize_raw`. Recursive field audit confirms
        // `EntropyController.is_active: bool` (`false`), `CoreTelemetryBank.ema_initialized: bool`
        // (`false`), `ComputePressureRegime::Quiescent = 0`, `TelemetryVerbosityMode::Coarse = 0`,
        // `TelemetryCapturePolicy::Pinned(Coarse)` (discriminant `0`), and `SandboxLifecycleEvent::Idle = 0`
        // are all zero-valid; `Option<u8>` and `SandboxLifecycleEvent` are explicitly re-written via `raw`.
        unsafe {
            Self::zeroize_raw(raw);
            core::ptr::write(
                core::ptr::addr_of_mut!((*raw).escalated_workbench_lane),
                None,
            );
            core::ptr::write(core::ptr::addr_of_mut!((*raw).sandbox_workbench_lane), None);
            core::ptr::write(
                core::ptr::addr_of_mut!((*raw).last_sandbox_event),
                SandboxLifecycleEvent::Idle,
            );
        }
    }

    /// Updates the internal allocation provenance tag (`CTX_ALLOC_*`).
    #[inline(always)]
    pub(crate) fn set_allocation_mode(&mut self, mode: u8) {
        self.allocation_mode = mode;
    }
}

// Compile-time verification that `RuntimeContext` satisfies `StigmergicSubstrateOps` and
// Gateway / Coordinator trait bounds.
const _: () = {
    const fn assert_full_stack_bounds<
        T: StigmergicSubstrateOps
            + SubstrateGatewayExt
            + GatewayPhaseSteeringExt
            + CausalProvenanceSignal
            + SystemConstraintSignal
            + ManifoldGatewaySignal,
    >() {
    }
    assert_full_stack_bounds::<RuntimeContext>();
};

// SAFETY: `CoreStateStack` is `#[repr(C, align(64))]` of 128 bytes with zero padding traps.
unsafe impl Pod for CoreStateStack {}

#[cfg(test)]
mod tests;
