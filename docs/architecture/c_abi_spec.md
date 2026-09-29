# C-ABI Specification: Geometric Cellular Sheaf Orchestrator (GCSO)

## 1. Architectural Worldview & Terminology Mapping (Alias Index)

### 1.1 Intuitive Alias & Terminology Mapping Index

| GCSO Theoretical Concept | Practical Engineering Alias | Primary System Role & Functional Description |
| --- | --- | --- |
| **Cellular Sheaf Cohomology Obstruction $H^1(K; \mathcal{F})$** | **Context Hallucination Lock-in Detector** | Detects topological loop disconnects and non-factual generation lock-in states from token activation residuals. |
| **Phase-Conjugate Attractor Repulsion** | **Anti-Hallucination Vector Repulser** | Flips false local minima energy valleys into repulsive potential peaks using anti-phase markers ($-\boldsymbol{\Delta\theta}$). |
| **Stigmergic Swarm-Attractor Duality** | **Pointer-Trace Dynamic Memory Convergence** | Replaces heavy online matrix differential solves with $\mathcal{O}(1)$ micro bitwise/pointer updates that converge to global attractors. |
| **Dynamic Phase-Shifted RoPE (DPSR)** | **Zero-Weight Context Phase Modulator** | Applies dynamic relative phase rotation to Query/Key attention registers without modifying base weight tensors in VRAM. |
| **Quantization-Discretized Phase Steering (QDPS)** | **Discrete Phase Rotation Filter** | Cuts off phase rotation steps falling below the minimum discretization step ($\Delta\theta_{\mathrm{min\_step}}$) on quantized grids. |
| **Sidecar Pointer Table (SPT) / Action Hub** | **$\mathcal{O}(1)$ Hot-Path Pointer Dispatcher** | Executes constant-time tagged pointer transitions, creating persistent memory trails for intelligence routing. |
| **Entropy-Driven Decoding Branch Controller (EDBC)** | **Dynamic Decoding Switchboard** | Monitors entropy flux ($\tilde{H}$) across layers to trigger potential-driven sampling or phase-conjugate repulsion at critical points. |
| **Dynamic Adaptive Extension Scratchpad (DAES)** | **Multi-Layer Dynamic Adaptive Scratchpad** | Reuses 64B cacheline memory as an in-place telemetry ledger, $\mathcal{O}(1)$ bypass shortcut table, and dynamic parameter auto-tuner. |
| **Predictive Phase-Motion & Residual Compensation (PPRC)** | **Instant-Replay Keyframe KV Cache** | Structures KV caches into I-Frames (Anchors) and P-Frames (Phase Motion), enabling zero-forward latency history seek. |
| **Sparse Residual Adapter Layer (SRL)** | **Dynamic Rank-1 Outer Product Adapter** | Provides lightweight FP8 outer product corrections ($\mathbf{u}\mathbf{v}^T$) to supplement non-linear model capacities. |
| **LoRA-to-Phase SVD Converter (L2P-SVD)** | **Training-Free Adapter Phase Projector** | Projects fine-tuned LoRA matrices via SVD into phase profiles and Rank-1 SRL vectors without gradient training. |
| **Zero-Overhead In-Memory Mapped Storage (ZIMMS)** | **Zero-Copy Memory-Mapped Engine** | Provides zero-copy memory-mapped file access and direct DMA stream buffers for `.gcso` unified binary containers. |
| **Phase-Steered Parallel Multi-head (PSPM) Router** | **Sub-Head Phase Group Allocator** | Dynamically routes attention heads into Fact, Logic, and Explore sub-groups in a single forward pass. |

---

### 1.2 Multi-Layer Execution Topology & Dynamic Layer-Adaptive Hierarchy

```text
+-----------------------------------------------------------------------------------+
|               High-Level Runtime (Rust Core / LiminiKa DSL / CLI)                 |
+-----------------------------------------------------------------------------------+
                                         │
                   [C11 / C++20 C-ABI Boundary Barrier Layer]
                                         │
+-----------------------------------------------------------------------------------+
| [Macro Level] Cold Path / Attractor Field & EDBC Control                          |
|   - Primary Endpoint: System Prompt (Natural Language) as Anchor Attractor        |
|   - Moving Z-Score Entropy (H~) & Pitchfork Bifurcation Evaluation                |
|   - Dynamic Persona Patch Application & Dyadic Ultrametric Barrier                |
|   - ZIMMS Memory-Mapped Storage (.gcso Container Serialization / Restoration)     |
|   - Cohomological Obstruction Class H^1(K; F) & Phase-Conjugate Repulsion         |
|   - Bottom-Up Attractor Crystallization from Aggregated Pointer Trails            |
|   - Layer-Adaptive DAES Role: Autonomous Self-Tuning Parameter Optimizer          |
|   - Layer-Adaptive EDBC Role: Pitchfork Bifurcation & Energy-Guided Decoding      |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ (Cascading Lower Aggregates & Attractor Pulls)
+-----------------------------------------------------------------------------------+
| [Mezzo Level] Inter-Block Coordination (Cellular Layer-Hub / Swarm)               |
|   - PagedBlock (16-32 Tokens) Aggregate Processing & Skip-Hop Synchronization     |
|   - Buffered Bit-Tree Reduction & Interlinking Hallucinated Trails                |
|   - Memory Pushout Alignment & Stigmergic Offloading to Disk                      |
|   - PPRC Keyframe Seek & Variable GOP Frame Indexing                              |
|   - Layer-Adaptive DAES Role: Block-Level Hit-Rate Aggregator & Cache Indexer     |
|   - Layer-Adaptive EDBC Role: Sliding-Window Entropy Rate Integrator (Phi_M)      |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ (Cascading Pointer Trails & Stigmergic Traces)
+-----------------------------------------------------------------------------------+
| [Micro Level] Hot Path / Action Hub (Sidecar Pointer Table - SPT)                 |
|   - Intelligence as Pointer Chains: O(1) Tagged Pointer Transitions               |
|   - Bottom-Up Stigmergy: Connect Cellular Hallucinations & Pull Trails to Anchor  |
|   - Extensible Payload Mapping (user_data / cluster_id / target_anchor_id)        |
|   - Lazy Phase Unwrapping & Query-Only Relative Phase Shift                       |
|   - Hash Slot 256 Indexing & Local Swarm Cell Step Execution                      |
|   - PSPM Sub-Head Group Routing & SRL Dynamic Rank-1 Adapter Integration          |
|   - Layer-Adaptive DAES Role: O(1) Fast-Path Shortcut Lookup & Bypass Table       |
|   - Layer-Adaptive EDBC Role: Token Z-Score Entropy Calculation                   |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ (Warp / SIMD Bitmask & Register Cascade)
+-----------------------------------------------------------------------------------+
| [Nano Level] In-Kernel Hot Path (C++20 / CUDA / Metal / Vulkan Kernels)           |
|   - Dynamic Phase-Shifted RoPE (DPSR) & RIPA tanh Angle Clamping                  |
|   - QDPS Quantization-Discretized Phase Steering & Step Thresholding              |
|   - Slerp Norm-Guarded Stabilization & Fused Logit Phase Shift                    |
|   - Isotropic Block-Diagonal Scaling & Sparse Residual Adapter Layer (SRL)        |
|   - Branchless SIMD Bitmask Operations & Warp Register Shuffle                    |
|   - Speculative Bit-Level Phase Prefetching                                       |
|   - Layer-Adaptive DAES Role: Lock-Free In-Register Ring Ledger Push              |
|   - Layer-Adaptive EDBC Role: Warp-Level 1st-Order Entropy Surge Detection        |
+-----------------------------------------------------------------------------------+
                                         │
        [ Dynamic Adaptive Extension Scratchpad (DAES) Multi-Layer Dynamic Barrier ]
                                         │
+-----------------------------------------------------------------------------------+
| [DAES Multi-Layer Dynamic Engine] In-Memory Scratchpad & Layer Adaptation         |
|   - Dynamic Data Structure: In-Place Ring Buffer, Fast-Path Shortcuts & Counters  |
|   - Dynamic Mode 0: Local Fast-Path Shortcut & Multi-Layer Telemetry Scratchpad   |
|   - Dynamic Mode 1: External Distributed Network Plugin Slot (GCSO-DNP)           |
|   - Dynamic Mode 2: Multi-GPU / Inter-Process Shared Memory Buffer Slot           |
+-----------------------------------------------------------------------------------+

```

---

### 1.3 Execution Responsibility & Dynamic Data Cascade Matrix

| Layer Level | Compute Granularity | Local Execution (Micro Computation) | Local Control (Macro Evaluation) | Data Cascade Output | DAES Multi-Layer Behavior | EDBC Multi-Layer Behavior |
| --- | --- | --- | --- | --- | --- | --- |
| **Macro Level** | Session / Attractor Field | Asynchronous AOT phase base updating, L2P-SVD projection & ZIMMS file mapping | Moving Z-Score Entropy ($\tilde{H}$), Pitchfork Bifurcation tracking & Cohomology $H^1(K; \mathcal{F})$ | Atomic global phase parameters, System Prompt Anchors & Macro Attractor Clusters | Evaluates telemetry ring to auto-tune PSPM ratios, RIPA clamps & EDBC thresholds | Drives Pitchfork Bifurcation & Energy-Guided Decoding transitions |
| **Mezzo Level** | Chunk / PagedBlock (16–32 Tokens) | Skip-Hop Bus inter-layer topology updates & PPRC I/P-Frame Seek | Buffered Bit-Tree Reduction, Interlinking Hallucinated Trails & Gershgorin Upper Bounds | Aggregated block phase status & Keyframe Indices | Aggregates block-level cache hit-rate metrics across token ranges | Tracks Sliding-Window Entropy Rate Integrator ($\Phi_M(t)$) |
| **Micro Level** | Per-Token / Action Hub (SPT) | $\mathcal{O}(1)$ Tagged Pointer lookup, PSPM Sub-Head Routing & Dynamic Shortcut Lookup | Bitmask reduction, SRL Rank-1 gating & Trail pulling toward Anchor Attractors | Pointer Trail, Q7 Phase delta & Adapter Residuals | Operates $\mathcal{O}(1)$ Fast-Path Bypass Table to skip redundant evaluations | Computes Token Z-Score Entropy ($\tilde{H}$) from activation distance |
| **Nano Level** | Thread-Warp / SIMD Lane | Branchless bitwise ops, Q7 addition in register shuffle & QDPS grid filtering | In-register RIPA $\tanh$ clamping, Speculative Phase Prefetch & immediate steering | Warp bitmask & Fused RoPE angles | Pushes lock-free profiling telemetry into scratchpad ring buffer | Detects 1st-order local entropy surge via Warp Shuffle |

---

### 1.4 Core Architectural Pillars & Layer-Adaptive Behaviors

| Pillar Name | Target Layer | Primary Memory & Execution Guarantee |
| --- | --- | --- |
| **Intelligence as Pointer Chains** | Micro / Hot Path | $\mathcal{O}(1)$ cacheline-friendly tagged pointer layout. Zero dynamic allocation in Hot Path. Pointer transitions represent the fundamental primitive of intelligence. |
| **Bottom-Up Stigmergy & Attractor Crystallization** | Micro $\to$ Macro | Pointer processing trails bridge cellular hallucinations, accumulating stigmergic density to macro-crystallize into latent Anchor Attractors or pull trails toward existing anchors. |
| **Phase-Conjugate Repulsion via Pointer Traces** | Micro $\to$ Macro | When cellular hallucinations hit false local minima, anti-phase markers ($-\boldsymbol{\Delta\theta}$) are stamped onto pointer trails, converting energy valleys into repulsive potential peaks. |
| **Quantization-Discretized Phase Steering (QDPS)** | Nano / Micro | Discrete steering filter eliminating phase updates below discretization threshold ($\Delta\theta_{\mathrm{min\_step}}$) to prevent grid oscillation under 1.5–3.5 bit weights. |
| **Multi-Layer Self-Adaptive System Components** | All Layers (Nano–Macro) | High-level components (DAES, EDBC) dynamically change their internal execution role depending on whether they are accessed at Nano, Micro, Mezzo, or Macro levels. |
| **System Prompt Anchor Endpoint** | Macro / Cold Path | Natural language system prompts projected into phase space as primary baseline Anchor Attractors. |
| **Descriptor-Based ABI & Facade** | Boundary | Unified Facade pattern via context handles. APIs accept descriptors with `struct_size` and `abi_version` validation. |
| **Unified Binary Persistence (ZIMMS)** | Boundary / Storage | Action Hub pointer chains, Attractor Field states, PPRC keyframes, SRL weights, and DAES state unified into `.gcso` container via zero-copy mmap. |
| **Polyfilled Compatibility Baseline** | Boundary | Standard C11 / C++20 compilation baseline with fallback polyfill macros (`GCSO_RESTRICT`, `GCSO_NOEXCEPT`, `GCSO_ALIGNAS`). |
| **Zero-Alloc Exception Isolation** | Boundary / Hot Path | Boundary exception protection (`try-catch`) and `catch_unwind` on Rust FFI. Non-aliased (`GCSO_RESTRICT`) aligned memory. |
| **PSPM Single-Pass Head Steering** | Nano / Micro | Single-pass head-group phase steering supporting Fact, Logic, and Explore sub-head routing. |
| **PPRC Keyframe KV Compression** | Mezzo / Storage | Temporal Key-Frame KV cache compression via Variable GOP Structures, decoupling Fact Anchors (I-Cache) and Phase Motion (P-Cache). |
| **Dynamic Adaptive Extension Scratchpad (DAES)** | DAES Layer | Reuses 64B extension memory as a dynamic in-place scratchpad data structure (Telemetry Ledger / Fast-Path Bypass Table / DNP Slot) for zero-alloc hot-path acceleration and self-optimization. |

---

## 2. Intelligence as Pointer Chains & Bottom-Up Attractor Dynamics

### 2.1 Pointer-Chain Intelligence & Action Hub (SPT) Conceptual Flow

Micro-level pointer transitions build a persistent Stigmergic Trail Map in the Action Hub (SPT). Micro pointer trails (`linked_trail_id`) continuously accumulate stigmergic density ( $\rho_{\mathrm{stigmergy}}$ ). When density passes the pull threshold, an attractor pull force ( $F_{\mathrm{pull}}$ ) emerges to steer active pointer chains toward macro anchor attractors. High-density trails autonomously crystallize in phase space as new dynamic latent attractors without updating base model weights.

```text
[ Micro Cellular Hallucinations ] ──(Link Pointer Trails)──> [ Action Hub (SPT) Pointer Trail Map ]
                                                                      │
                                                                (Stigmergic Density
                                                                 Accumulation & Pull)
                                                                      │
                                                                      ▼
[ Bottom-Up Macro Attractor ] <───(Phase Crystallization)───────[ Attractor Field ]
  (Self-Organized Phase Field)                                        ▲
                                                                      │ (Attractor Pull Force F_pull)
[ System Prompt Anchor Attractor ] ───────────────────────────────────┘
  (Primary Baseline Endpoint)

```

---

### 2.2 Pointer Trail to Attractor Life-Cycle Matrix

| Stage Name | Execution Layer | Trigger Condition | Memory & Topological Operation |
| --- | --- | --- | --- |
| **Pointer Step** | Micro / Action Hub | Token transition execution | Writes 64-bit Tagged Pointer transition and updates Q7 phase delta in `gcso_pointer_trail_t`. |
| **Stigmergic Trace** | Micro / Mezzo | Repeated trail traversal | Increases `stigmergic_density` and links adjacent trails via `linked_trail_id`. |
| **QDPS Grid Filter** | Nano / Micro | Phase angle update evaluation | Evaluates $\vert\Delta\theta\vert \ge \Delta\theta_{\mathrm{min\_step}}$. If below threshold, rounds to 0 to prevent quantization noise. |
| **Attractor Pull** | Micro $\leftarrow$ Macro | `stigmergic_density > tau_pull` | Applies `attractor_pull_force` $F_{\mathrm{pull}}$ to steer active pointer chain toward `target_anchor_id`. |
| **Phase-Conjugate Repulsion** | Micro $\to$ Macro | Cohomological obstruction $H^1(K; \mathcal{F})$ | Marks pointer trail with anti-phase delta ($-\boldsymbol{\Delta\theta}$), flipping local minimum into a repulsive peak. |
| **Macro Crystallization** | Macro / Attractor Field | `stigmergic_density > tau_crystal` | Instantiates a new dynamic Anchor Attractor in phase space, converting high-density pointer trails into a persistent macro attractor. |

---

### 2.3 QDPS (Quantization-Discretized Phase Steering) Resolution Grid Topology

```text
[ Continuous Phase Delta Input (dTheta) ]
                   │
                   ▼
  ┌─────────────────────────────────┐
  │  Magnitude Check: |dTheta|      │
  └────────────────┬────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
  [ < dTheta_min_step ]  [ >= dTheta_min_step ]
         │                   │
         ▼                   ▼
  ┌─────────────┐     ┌───────────────────────────────────┐
  │ Cutoff to 0 │     │ Quantized Step Rounding           │
  │ (No-Op)     │     │ dTheta_q = round(dTheta / Step)   │
  └─────────────┘     └─────────────────┬─────────────────┘
                                        │
                                        ▼
                      ┌───────────────────────────────────┐
                      │ Apply to DPSR Register Shuffle    │
                      └───────────────────────────────────┘

```

---

## 3. Subsystem Architectural Dynamics & State Machines

### 3.1 EDBC Entropy-Driven Decoding & Pitchfork Bifurcation State Machine

The Entropy-Driven Decoding Branch Controller (EDBC) monitors activation entropy ( $\tilde{H}$ ) across execution layers, switching decoding paths without dynamic allocation.

```text
               ┌──────────────────────────────────────────────────┐
               │         NORMAL_ENTROPY_MODE (Default)            │
               │   - Standard Nucleus / Dynamic Sampling          │
               │   - Low Phase Distortions                        │
               └──────────────────────┬───────────────────────────┘
                                      │
            [ Z-Score Entropy H~ > Tau_Bifurcation ]
                                      │
                                      ▼
               ┌──────────────────────────────────────────────────┐
               │          PITCHFORK_BIFURCATION_MODE              │
               │   - Split Exploration & Logic Sub-Heads          │
               │   - Apply Dual Phase Rotation Angle (+/- dTheta) │
               └──────────────────────┬───────────────────────────┘
                                      │
     ┌────────────────────────────────┴────────────────────────────────┐
     │                                                                 │
[ Cohomological Obstruction Class ]                [ Entropy Normalizes H~ <= Tau_Bifurcation ]
     │                                                                 │
     ▼                                                                 ▼
┌──────────────────────────────────────────┐      ┌──────────────────────────────────────────┐
│      PHASE_CONJUGATE_REPULSION_MODE      │      │          NORMAL_ENTROPY_MODE             │
│  - Inject Anti-Phase Shift (-dTheta)     │      │  - Merge Sub-Heads & Recalibrate Phase   │
│  - Convert Minimum to Potential Peak     │      └──────────────────────────────────────────┘
└────────────────────┬─────────────────────┘
                     │
         [ Energy Valley Overcome ]
                     │
                     ▼
┌──────────────────────────────────────────┐
│             RECOVERY_COMPLETE            │
│  - Resume Normal Trajectory Steering     │
└──────────────────────────────────────────┘

```

---

### 3.2 PPRC Keyframe KV Cache & Temporal Motion Delta Lifecycle

PPRC decouples Key-Value cache storage into sparse Fact Anchors (I-Frames) and continuous Phase Motion Deltas (P-Frames), enabling zero-forward latency context seeks.

```text
[ Context Input Stream ] ───> [ Token Entropy Evaluation ]
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
           [ Keyframe Trigger (I-Frame) ]    [ Delta Token Step (P-Frame) ]
                     │                                 │
                     ▼                                 ▼
         ┌───────────────────────┐         ┌───────────────────────┐
         │  I-Frame KV Cache     │         │  P-Frame Motion Cache │
         │  - Full Key Vector    │         │  - Phase Delta dTheta │
         │  - High Precision FP16│         │  - Q7 Quantized Delta │
         └───────────┬───────────┘         └───────────┬───────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │  Zero-Forward Latency     │
                        │  Keyframe Reconstruction  │
                        └───────────────────────────┘

```

---

### 3.3 DAES Polymorphic Dynamic Scratchpad & Mode Switch State Machine

DAES reuses a single 64-byte cacheline block as an active in-memory scratchpad in Mode 0, as an FFI boundary extension handle for distributed networks in Mode 1, or as a shared IPC memory slot in Mode 2.

```text
+-----------------------------------------------------------------------------------+
|                        Core GCSO Runtime Execution Engine                         |
|   (Context Facade / Action Hub / Attractor Field / DPSR / SRL / EDBC / Storage)   |
+-----------------------------------------------------------------------------------+
                                         │
                      [ DAES Polymorphic Execution Barrier ]
                                         │
     ┌───────────────────────────────────┼───────────────────────────────────┐
     │ (Mode 0: Default)                 │ (Mode 1: Network Plugin)          │ (Mode 2: Shared Buffer)
     ▼                                   ▼                                   ▼
+-------------------------+   +-------------------------+   +-------------------------+
| [DAES Scratchpad Engine]|   | [GCSO-DNP Plugin Engine]|   | [Shared IPC Memory Slot]|
| - Fast-Path Bypass Table|   | - Remote Node Sync      |   | - Multi-GPU Shared Ring |
| - Telemetry Ring Ledger |   | - Dynamic Network Frame |   | - Process Interop Queue |
| - Auto-Tuner (PSPM/RIPA)|   | - Repulsion Callbacks   |   | - Lock-Free IPC Sync    |
+-------------------------+   +-------------------------+   +-------------------------+

```

---

### 3.4 L2P-SVD & SRL Dynamic Rank-1 Adapter Lifecycle & Mapping Flow

```text
[ Fine-Tuned LoRA Weights (W_A, W_B) ] ───(L2P-SVD Projection)───> [ First-Order SVD: U * Sigma * V^T ]
                                                                                   │
                                                                                   ▼
[ SRL Outer Product Vectors (u, v) & Gain (s) ] <───(Extract Principal Component)──┘
                      │
                      ▼
[ Hot Path In-Kernel Execution: y = W_base * x + s (*) (u * (v^T * x)) ]

```

---

## 4. Header Topology & Polyfill Specification

### 4.1 Header Module Inclusion Network

```text
  +-----------------------------------+
  |   include/liminika/gcso_config.h  |  <-- Versioning, Export Visibility, C++20/C11 Polyfills
  +-----------------------------------+
                    ▲
                    │ #include
  +-----------------------------------+
  |   include/liminika/gcso_types.h   |  <-- POD Struct Layouts, Enums, Handles, Alignment
  +-----------------------------------+
                    ▲
                    │ #include
  +-----------------------------------+
  |    include/liminika/gcso_abi.h    |  <-- Exported C-ABI Function Contracts & Facade API
  +-----------------------------------+

```

---

### 4.2 Compatibility & Polyfill Feature Mapping

| Feature / Macro Keyword | Target C++ Standard | Target C Standard | ABI Binary Guarantee |
| --- | --- | --- | --- |
| `GCSO_EXTERN_C` | `extern "C"` Linkage Block | Standard C Linkage | Prevents C++ name mangling across FFI boundaries. |
| `GCSO_API` | Symbol Export Attribute | Symbol Export Attribute | Ensures explicit symbol export visibility in shared libraries. |
| `GCSO_CALL` | Standard C Calling Convention | Standard C Calling Convention | Enforces standard C calling convention across language boundaries. |
| `GCSO_NOEXCEPT` | `noexcept` Keyword | Exception-Free Attribute | Guarantees exception non-propagation across FFI layer. |
| `GCSO_NODISCARD` | `[[nodiscard]]` Attribute | Warn Unused Attribute | Enforces status code checks at function call sites. |
| `GCSO_LIKELY / UNLIKELY` | `[[likely]] / [[unlikely]]` | `__builtin_expect` / Empty | Optimizes branch prediction for hot/cold path routing. |
| `GCSO_ALIGNAS(n)` | `alignas(n)` Specifier | Alignment Attribute | Enforces strict memory boundary alignment ($n \in \{4, 16, 32, 64, 128\}$). |
| `GCSO_RESTRICT` | `__restrict` | `restrict` Qualifier | Asserts non-aliasing pointers for SIMD vectorization. |
| `GCSO_STATIC_ASSERT` | `static_assert(c, m)` | `_Static_assert(c, m)` | Enforces compile-time layout and size verification. |

---

### 4.3 Hardware Execution Granularity Matrix

| Hardware Target Backend | Hardware Register Primitive | SIMD / Subgroup Granularity | In-Kernel Hot Path Optimization Strategy |
| --- | --- | --- | --- |
| **NVIDIA CUDA** | `__shfl_xor_sync`, `__popc` | 32 Threads (Warp) | Register Shuffle Q7 Phase Addition, QDPS thresholding, Warp PagedBitmask reduction & DAES Ring Ledger push. |
| **Apple Metal** | `simd_shuffle_xor`, `simd_sum` | 32 Threads (Simdgroup) | Metal Shading Language threadgroup memory zero-copy phase rotation & Fast-Path index lookup. |
| **Vulkan Compute** | `subgroupShuffleXor`, `subgroupBallot` | 16 / 32 / 64 Threads | Subgroup-wide bitwise phase prefetching, barrier-free reduction & DAES Scratchpad update. |
| **CPU AVX-512 / AVX2** | `_mm512_shuffle_epi8`, `_mm256_add_epi8` | 16 / 32 / 64 Bytes | Branchless vector Q7 phase offset addition, Slerp norm guard & Ring Ledger logging. |

---

## 5. Data Layout & Memory Topology Specification

### 5.1 Primitive Types & Opaque Handles Matrix

| Type Name | Underlying Primitive | Binary Role & Ownership |
| --- | --- | --- |
| `gcso_q7_t` | Signed 8-bit Integer | Fixed-point Q7 phase representation ($\beta_{\mathrm{Q7}} = 1/128$, range $-1.0 \sim +0.9921875$). |
| `gcso_status_t` | Signed 32-bit Integer | Standardized 32-bit status code for 1:1 cross-language FFI mapping. |
| `gcso_capability_flags_t` | Unsigned 64-bit Integer | Bitmask matrix defining active feature caps, DAES mode, and extension slots. |
| `gcso_context_handle_t` | Opaque Pointer | Opaque handle for Unified Runtime Context Facade. |
| `gcso_action_hub_handle_t` | Opaque Pointer | Opaque handle for Hot-Path Tagged Pointer Table (Action Hub / SPT). |
| `gcso_attractor_field_handle_t` | Opaque Pointer | Opaque handle for Macro Target Attractor Field. |
| `gcso_dpsr_kernel_handle_t` | Opaque Pointer | Opaque handle for DPSR Phase Steering Engine. |
| `gcso_srl_adapter_handle_t` | Opaque Pointer | Opaque handle for Sparse Residual Adapter Layer. |
| `gcso_edbc_controller_handle_t` | Opaque Pointer | Opaque handle for Dynamic Entropy Decoding Branch Controller. |
| `gcso_container_handle_t` | Opaque Pointer | Opaque handle for `.gcso` Memory-Mapped Storage. |
| `gcso_pspm_router_handle_t` | Opaque Pointer | Opaque handle for Phase-Steered Parallel Multi-head Router. |
| `gcso_daes_slot_handle_t` | Opaque Pointer | Opaque handle for registered DAES modules or plugin extensions. |

---

### 5.2 Status Classification Matrix (`gcso_status_t`)

| Status Identifier | Numeric Value | Classification | Operational Meaning & Recovery Strategy |
| --- | --- | --- | --- |
| `GCSO_SUCCESS` | `0` | Success | Operation completed without errors. |
| `GCSO_ERROR_INVALID_ARGUMENT` | `-1` | Parameter Error | Out-of-bounds parameter or invalid flag passed. |
| `GCSO_ERROR_OUT_OF_MEMORY` | `-2` | Memory Allocation | Allocation failed during Cold Path creation. |
| `GCSO_ERROR_BUFFER_TOO_SMALL` | `-3` | Buffer Constraints | Output buffer capacity insufficient for query. |
| `GCSO_ERROR_PANIC_CAUGHT` | `-4` | FFI Safety Barrier | Intercepted internal exception or panic at FFI boundary. |
| `GCSO_ERROR_NULL_POINTER` | `-5` | Pointer Validation | Mandatory pointer argument was NULL. |
| `GCSO_ERROR_INVALID_STATE` | `-6` | Lifecycle State | Operation requested on uninitialized or closed state. |
| `GCSO_ERROR_VERSION_MISMATCH` | `-7` | ABI Verification | Descriptor `struct_size` or ABI version incompatible. |
| `GCSO_ERROR_MISALIGNED_POINTER` | `-8` | Memory Alignment | Buffer violates required SIMD or cacheline alignment. |
| `GCSO_ERROR_IO_FAILURE` | `-9` | Storage Persistence | System I/O or memory mapping failed. |
| `GCSO_ERROR_ACTION_HUB_FULL` | `-10` | Resource Capacity | Action Hub slot table capacity exceeded. |
| `GCSO_ERROR_NOT_IMPLEMENTED` | `-11` | PoC Boundary | Feature or kernel backend not implemented in current build. |
| `GCSO_ERROR_ATTRACTOR_NOT_FOUND` | `-20` | Attractor Search | Requested anchor ID missing in attractor field. |
| `GCSO_ERROR_DPSR_PHASE_OVERFLOW` | `-30` | Kernel Safety | Phase angle accumulation exceeded safety bound. |
| `GCSO_ERROR_QDPS_UNDERFLOW` | `-31` | QDPS Steering | Phase update step fell below minimum step threshold $\Delta\theta_{\mathrm{min\_step}}$. |
| `GCSO_ERROR_EDBC_SINGULARITY` | `-40` | Dynamic Math | Logarithmic singularity encountered in entropy calculation. |
| `GCSO_ERROR_CONTAINER_CORRUPTED` | `-50` | Storage Checksum | `.gcso` file header checksum or structure invalid. |
| `GCSO_ERROR_ZIMMS_MAPPING_FAILED` | `-51` | ZIMMS Engine | Zero-copy mmap or Direct DMA memory buffer mapping failed. |
| `GCSO_ERROR_PSPM_ROUTING_FAILED` | `-60` | PSPM Router | Head group phase allocation failed or head index invalid. |
| `GCSO_ERROR_PPRC_SEEK_FAILED` | `-70` | Keyframe Indexing | Keyframe indexing corrupt or token out of bounds. |
| `GCSO_ERROR_OBSTRUCTION_UNRESOLVED` | `-80` | Sheaf Cohomology | Cohomological obstruction class cannot be reduced by scalar potential. |
| `GCSO_ERROR_EXTENSION_NOT_LOADED` | `-90` | DAES Layer | Requested extension plugin or acceleration mode is inactive. |
| `GCSO_ERROR_DAES_SCRATCHPAD_FULL` | `-91` | DAES Layer | Fast-Path shortcut table or Telemetry Ring Buffer capacity reached. |

---

### 5.3 Anchor Classification Matrix (`gcso_anchor_type_t`)

| Enumerator Value | Numeric Code | Phase Trajectory Steering Role |
| --- | --- | --- |
| `GCSO_ANCHOR_TYPE_SYSTEM_PROMPT` | `0` | Natural language prompt anchor projected into phase space (Primary Endpoint). |
| `GCSO_ANCHOR_TYPE_EMBEDDING` | `1` | Dense feature vector anchor in latent space. |
| `GCSO_ANCHOR_TYPE_PHASE_REPULSE` | `2` | Anti-phase repulsion vector for hallucination suppression. |
| `GCSO_ANCHOR_TYPE_TOPOLOGICAL` | `3` | Manifold topological anchor for hierarchical tree structures. |
| `GCSO_ANCHOR_TYPE_CRYSTALLIZED` | `4` | Self-organized macro anchor crystallized bottom-up from high-density pointer trails. |

---

### 5.4 Binary Memory Layout Specifications

#### A. Tagged Pointer Layout (64-bit Unsigned Integer)

```text
 63          56 55          48 47                                              0
+--------------+--------------+-------------------------------------------------+
| Phase State  |  Tag Bits    |               Raw Address Pointer               |
|   (8 bits)   |   (8 bits)   |                    (48 bits)                    |
+--------------+--------------+-------------------------------------------------+

```

| Field Identifier | Bit Range | Mask Bit Pattern | Encoding / Masking Rules |
| --- | --- | --- | --- |
| `Raw Address` | 0–47 | `0x0000FFFFFFFFFFFF` | Canonical 48-bit virtual address pointer. Unpack via bitwise AND. |
| `Tag Bits` | 48–55 | `0x00FF000000000000` | Domain or AST node classification identifier. Shift right by 48. |
| `Phase State` | 56–63 | `0xFF00000000000000` | Quantized phase state index. Shift right by 56. |

---

#### B. Paged Block Bitmask Layout (`gcso_paged_bitmask_t`)

* **Size**: 32 Bytes | **Alignment**: 32 Bytes (SIMD AVX-256 / Warp Aligned)

```text
 0                      64                     128                    192                   255
+----------------------+----------------------+----------------------+----------------------+
|       bits[0]        |       bits[1]        |       bits[2]        |       bits[3]        |
|  (64-bit Unsigned)   |  (64-bit Unsigned)   |  (64-bit Unsigned)   |  (64-bit Unsigned)   |
+----------------------+----------------------+----------------------+----------------------+

```

---

#### C. Stigmergic Pointer Trail Layout (`gcso_pointer_trail_t`)

* **Size**: 128 Bytes (2 Cache Lines) | **Alignment**: 128 Bytes

```text
 Offset (Bytes)
  0  +-----------------------------------------------------------------+
     | current_ptr (64-bit Unsigned Tagged Pointer)                    |
  8  +-----------------------------------------------------------------+
     | previous_ptr (64-bit Unsigned Tagged Pointer)                   |
 16  +-----------------------------------------------------------------+
     | user_data (Extensible Payload: 64-bit Unsigned)                 |
 24  +-------------------------------+---------------------------------+
     | transition_cost (Signed 32)   | step_count (Unsigned 32)        |
 32  +-------------------------------+---------------------------------+
     | stigmergic_density (Float 32) | target_anchor_id (Unsigned 32)  |
 40  +-------------------------------+---------------------------------+
     | cluster_id (Unsigned 32)      | linked_trail_id (Unsigned 32)   |
 48  +-------------------------------+---------------------------------+
     | attractor_pull_force (F32)    | flags (Unsigned 32)             |
 56  +-----------------------------------------------------------------+
     | accumulated_phase_delta[64] (gcso_q7_t Array: 64 Bytes)         |
120  +-----------------------------------------------------------------+
     | reserved_padding[8] (Unsigned 8-bit Array: Padding to 128 B)    |
128  +-----------------------------------------------------------------+

```

---

#### D. Base Configuration Layout (`gcso_config_t`)

* **Size**: 64 Bytes | **Alignment**: 4 Bytes

| Offset | Field Identifier | Type Category | Operational Role / Invariant State |
| --- | --- | --- | --- |
| `0` | `head_dim` | Unsigned 32 | Attention head dimension (Must be even, e.g., 128). |
| `4` | `num_heads` | Unsigned 32 | Total number of attention heads. |
| `8` | `paged_block_size` | Unsigned 32 | Tokens per paged block (16 or 32). |
| `12` | `q7_phase_scale` | Float 32 | Q7 scale factor ($\beta_{\mathrm{Q7}} = 1.0 / 128.0$). |
| `16` | `ripa_clamp_max_rad` | Float 32 | RIPA soft-bounded phase limit in radians (Default: $5^\circ \approx 0.087$). |
| `20` | `qdps_min_step_rad` | Float 32 | QDPS quantization threshold angle below which phase steering is cut off. |
| `24` | `entropy_singularity_eps` | Float 32 | Singularity prevention clamp ($\epsilon_{\mathrm{log}} = 10^{-12}$). |
| `28` | `max_prompt_anchors` | Unsigned 32 | Maximum allowed system prompt anchors. |
| `32` | `action_hub_capacity` | Unsigned 32 | Slot capacity of Action Hub (SPT) tagged pointer table. |
| `36` | `enable_cuda_warp_shuffle` | Unsigned 8 | Enable GPU Warp Shuffle inline evaluation. |
| `37` | `enable_zero_alloc_strict` | Unsigned 8 | Enforce Hot-Path zero dynamic allocation checks. |
| `38` | `daes_mode` | Unsigned 8 | DAES Slot Mode (`0` = Dynamic Scratchpad, `1` = Dynamic Plugin, `2` = Shared Buffer). |
| `39` | `reserved_flags` | Unsigned 8 | Alignment padding. |
| `40` | `reserved[24]` | Unsigned 8 Array | Reserved for future ABI core extensions. |

---

#### E. Container Snapshot Header Layout (`gcso_snapshot_header_t`)

* **Size**: 128 Bytes | **Alignment**: 64 Bytes

| Offset | Field Identifier | Type Category | Binary Purpose & Validation Role |
| --- | --- | --- | --- |
| `0` | `magic` | Unsigned 32 | Magic Constant (`0x4F534347` = ASCII `"GCSO"`). |
| `4` | `version` | Unsigned 32 | ABI Version Identifier (`0x00020000` = v2.0.0). |
| `8` | `total_size` | Unsigned 64 | Total byte size of container payload. |
| `16` | `action_hub_offset` | Unsigned 64 | Byte offset to Action Hub binary payload. |
| `24` | `attractor_field_offset` | Unsigned 64 | Byte offset to Attractor Field payload. |
| `32` | `dpsr_state_offset` | Unsigned 64 | Byte offset to DPSR kernel state payload. |
| `40` | `srl_state_offset` | Unsigned 64 | Byte offset to SRL adapter payload. |
| `48` | `edbc_state_offset` | Unsigned 64 | Byte offset to EDBC controller payload. |
| `56` | `checksum_crc32` | Unsigned 32 | CRC32 integrity checksum over payload. |
| `60` | `daes_slot_offset` | Unsigned 32 | Byte offset to DAES binary state/telemetry section. |
| `64` | `timestamp_epoch_sec` | Unsigned 64 | Container creation timestamp (Epoch seconds). |
| `72` | `reserved_padding[56]` | Unsigned 8 Array | Extension padding to 128 bytes. |

---

#### F. Dynamic Adaptive Extension Scratchpad (DAES) Memory Layout (`gcso_daes_slot_t`)

* **Size**: 64 Bytes (1 Cacheline) | **Alignment**: 64 Bytes

```text
 Mode 0: Telemetry & Fast-Path Dynamic Scratchpad Layout (Default)
 Offset (Bytes)
  0  +-------------------------------+----------------------------------+
     | mode = 0 (Unsigned 32)        | telemetry_ring_head (Unsigned 16)|
  4  +-------------------------------+----------------------------------+
     | telemetry_ring_tail (U16)     | cache_hit_count (Unsigned 32)    |
  8  +-------------------------------+----------------------------------+
     | fast_path_bypass_mask (U64)   | auto_tune_flags (Unsigned 32)    |
 20  +------------------------------------------------------------------+
     | fast_path_shortcuts[4] (Four 64-bit Tagged Pointer Shortcuts)    |
 52  +------------------------------------------------------------------+
     | telemetry_mini_ledger[12] (Fixed-size byte ring for profiling)   |
 64  +------------------------------------------------------------------+

 Mode 1: External Plugin / GCSO-DNP Slot Layout
 Offset (Bytes)
  0  +-------------------------------+---------------------------------+
     | mode = 1 (Unsigned 32)        | plugin_version (Unsigned 32)    |
  8  +-----------------------------------------------------------------+
     | capability_mask (Unsigned 64)                                   |
 16  +-----------------------------------------------------------------+
     | reserved_payload_offset (Unsigned 64)                           |
 24  +-----------------------------------------------------------------+
     | plugin_context_ptr (Unsigned 64)                                |
 32  +-----------------------------------------------------------------+
     | reserved_padding[32] (Unsigned 8-bit Array)                     |
 64  +-----------------------------------------------------------------+

 Mode 2: Shared IPC Memory Slot Layout
 Offset (Bytes)
  0  +-------------------------------+---------------------------------+
     | mode = 2 (Unsigned 32)        | shared_memory_key (Unsigned 32) |
  8  +-----------------------------------------------------------------+
     | ring_buffer_ipc_offset (Unsigned 64)                            |
 16  +-----------------------------------------------------------------+
     | lockfree_sync_atomic_counter (Unsigned 64)                      |
 24  +-----------------------------------------------------------------+
     | reserved_padding[40] (Unsigned 8-bit Array)                     |
 64  +-----------------------------------------------------------------+

```

---

#### G. PPRC Keyframe Header, PSPM, EDBC, SRL & ZIMMS Layouts

##### PPRC Keyframe Header Layout (`gcso_pprc_keyframe_header_t`)

* **Size**: 64 Bytes | **Alignment**: 32 Bytes

| Offset | Field Identifier | Type Category | Keyframe Storage Role |
| --- | --- | --- | --- |
| `0` | `frame_type` | Unsigned 32 | Frame Classification (`0` = I-Frame Anchor, `1` = P-Frame Motion Delta). |
| `4` | `token_index` | Unsigned 32 | Absolute token position sequence index. |
| `8` | `gop_length` | Unsigned 32 | Dynamic entropy-driven GOP length. |
| `12` | `composite_vector_length` | Float 32 | Resultant vector length $\bar{R}_c$ in circular statistics. |
| `16` | `von_mises_kappa` | Float 32 | Concentration parameter $\kappa_c$ of phase residual distribution. |
| `20` | `sparse_scalar_residual` | Float 32 | Compression residual scalar gain $s_t$. |
| `24` | `icache_payload_offset` | Unsigned 64 | Offset to Fact Anchor Key Cache payload. |
| `32` | `pcache_payload_offset` | Unsigned 64 | Offset to Phase Motion Delta payload. |
| `40` | `reserved[24]` | Unsigned 8 Array | Alignment padding to 64 bytes. |

---

##### PSPM Router Configuration Layout (`gcso_pspm_config_t`)

* **Size**: 32 Bytes | **Alignment**: 16 Bytes

| Offset | Field Identifier | Type Category | Sub-Head Group Allocation |
| --- | --- | --- | --- |
| `0` | `num_fact_heads` | Unsigned 16 | Number of heads allocated to Fact sub-group. |
| `2` | `num_logic_heads` | Unsigned 16 | Number of heads allocated to Logic sub-group. |
| `4` | `num_explore_heads` | Unsigned 16 | Number of heads allocated to Explore sub-group. |
| `6` | `flags` | Unsigned 16 | Routing flags & single-pass execution mode. |
| `8` | `fact_phase_gain` | Float 32 | Phase scale factor for Fact sub-heads. |
| `12` | `logic_phase_gain` | Float 32 | Phase scale factor for Logic sub-heads. |
| `16` | `explore_phase_gain` | Float 32 | Phase scale factor for Explore sub-heads. |
| `20` | `reserved[12]` | Unsigned 8 Array | Reserved for future router parameters. |

---

##### EDBC Dynamic Controller State Layout (`gcso_edbc_state_t`)

* **Size**: 64 Bytes | **Alignment**: 32 Bytes

| Offset | Field Identifier | Type Category | State Tracking Role |
| --- | --- | --- | --- |
| `0` | `moving_z_entropy` | Float 32 | Moving average normalized Z-score entropy $\tilde{H}$. |
| `4` | `bifurcation_threshold` | Float 32 | Z-score threshold $\tau_{\mathrm{bifurcation}}$ triggering branch split. |
| `8` | `singularity_eps` | Float 32 | Logarithmic singularity safety guard $\epsilon_{\mathrm{log}}$. |
| `12` | `sliding_entropy_rate` | Float 32 | Sliding-window entropy rate integrator $\Phi_M(t)$. |
| `16` | `repulsion_gain` | Float 32 | Scaling gain for anti-phase repulsion pulse ($-\boldsymbol{\Delta\theta}$). |
| `20` | `sample_temperature` | Float 32 | Energy-guided dynamic temperature value. |
| `24` | `active_branch_mode` | Unsigned 32 | Enum code (`0` = Normal, `1` = Bifurcation, `2` = Repulsion). |
| `28` | `reserved[36]` | Unsigned 8 Array | Extension padding to 64 bytes. |

---

##### SRL Dynamic Rank-1 Descriptor Layout (`gcso_srl_descriptor_t`)

* **Size**: 64 Bytes | **Alignment**: 32 Bytes

| Offset | Field Identifier | Type Category | Operational Purpose |
| --- | --- | --- | --- |
| `0` | `layer_idx` | Unsigned 32 | Target Transformer layer index for Rank-1 insertion. |
| `4` | `rank` | Unsigned 32 | Rank parameter (Fixed to `1` for SRL). |
| `8` | `u_vector_ptr` | Unsigned 64 | Pointer to FP8/FP16 column vector $\mathbf{u}$. |
| `16` | `v_vector_ptr` | Unsigned 64 | Pointer to FP8/FP16 row vector $\mathbf{v}$. |
| `24` | `gain_scalar_ptr` | Unsigned 64 | Pointer to FP32 diagonal gain vector $\mathbf{s}$. |
| `32` | `scale_factor` | Float 32 | Global scaling multiplier for Rank-1 product. |
| `36` | `flags` | Unsigned 32 | Flags defining FP8 format and dynamic activation state. |
| `40` | `reserved[24]` | Unsigned 8 Array | Extension padding to 64 bytes. |

---

##### ZIMMS Memory Mapping Descriptor Layout (`gcso_zimms_descriptor_t`)

* **Size**: 64 Bytes | **Alignment**: 32 Bytes

| Offset | Field Identifier | Type Category | Memory Mapping Role |
| --- | --- | --- | --- |
| `0` | `mapped_address` | Unsigned 64 | Virtual memory address returned by zero-copy `mmap`. |
| `8` | `file_size_bytes` | Unsigned 64 | Total byte length of mapped `.gcso` file. |
| `16` | `dma_buffer_handle` | Unsigned 64 | Direct DMA ring-buffer handle for async page streaming. |
| `24` | `flags` | Unsigned 32 | Protection & mapping flags (`PROT_READ`, `MAP_SHARED`, etc.). |
| `28` | `fd_handle` | Signed 32 | Operating system file descriptor integer. |
| `32` | `reserved[32]` | Unsigned 8 Array | Extension padding to 64 bytes. |

---

### 5.5 Memory Lifetime & Allocation Topology

```text
[ Caller / Host Layer ]
       │
       │ Allocates Descriptor on Stack / Heap
       ▼
[ ABI Facade Boundary ] ────── Create Function Call ───────┐
                                                           ▼
                                         [ Internal Managed Heap / ZIMMS mmap ]
                                         ├── Context Facade Object
                                         ├── Action Hub Slot Table (SPT)
                                         ├── Attractor Field Anchors
                                         ├── DPSR Kernel State
                                         ├── SRL Adapter Tables (Rank-1 Vectors)
                                         ├── EDBC History Buffers
                                         ├── PSPM Router Table
                                         └── DAES Dynamic Scratchpad Buffer
       │                                                   │
       │ Reads/Writes Hot-Path Data (Zero Alloc)           │
       ▼                                                   ▼
[ Output Buffers / Pointers ] ◄── Destroy Function Call ───┘
 (Caller Allocated & Passed)     (Deallocates All Internal Structures / Unmaps ZIMMS)

```

---

## 6. Functional C-ABI Interface Specifications

### 6.1 System & Capability Query Interface Matrix

| API Identifier | Operational Target | Compute Complexity | Memory Constraint | Operational Summary |
| --- | --- | --- | --- | --- |
| `gcso_abi_get_version` | System Query | $\mathcal{O}(1)$ | **Zero Allocation** | Retrieves major, minor, patch ABI versions. |
| `gcso_abi_get_version_string` | System Query | $\mathcal{O}(1)$ | **Zero Allocation** | Returns static version string pointer. |
| `gcso_status_to_string` | System Query | $\mathcal{O}(1)$ | **Zero Allocation** | Converts status code to static descriptive string. |
| `gcso_abi_query_capability` | System Query | $\mathcal{O}(1)$ | **Zero Allocation** | Queries active hardware/kernel capability flags. |
| `gcso_config_init_default` | Core Config | $\mathcal{O}(1)$ | **Zero Allocation** | Initializes configuration descriptor with default parameters. |

---

### 6.2 High-Level Runtime Context Facade Interface Matrix

| API Identifier | Execution Path | Parameter Ownership & Constraints | Operational Summary |
| --- | --- | --- | --- |
| `gcso_context_create` | Cold Path | Input descriptor, output context handle pointer | Allocates runtime context handle and initializes sub-components. |
| `gcso_context_reset` | Cold Path | Context handle | Resets phase accumulators and trails without freeing allocated tables. |
| `gcso_context_set_system_prompt_anchor` | Cold Path | Context handle, UTF-8 text string, weight float | **Primary Endpoint**: Registers natural language prompt as Anchor Attractor. |
| `gcso_context_step_token` | **Hot Path** | Context handle, non-aliased tensors, trail output | Executed per token. Applies inline DPSR, QDPS step checks, SRL evaluation, EDBC tracking, and DAES Telemetry in a zero-alloc loop. |
| `gcso_context_serialize` | Cold Path | Context handle, output buffer pointer, capacity pointer | Two-pass serialization of Action Hub, Attractors, and DAES state into `.gcso` binary. |
| `gcso_context_deserialize` | Cold Path | Input buffer pointer, buffer size, output context handle pointer | Restores full runtime state snapshot from `.gcso` binary container payload. |
| `gcso_context_destroy` | Cold Path | Context handle (Allows `NULL` as no-op) | Safely deallocates context and all child components. |

---

### 6.3 Action Hub, Stigmergic Pointer Trail & DAES Acceleration Matrix

| API Identifier | Target Layer | Compute Complexity | Memory Allocation Rule | Operational Summary |
| --- | --- | --- | --- | --- |
| `gcso_action_hub_step_pointer` | Micro | $\mathcal{O}(1)$ | **Zero Allocation** | Executes $\mathcal{O}(1)$ Tagged Pointer transitions in Action Hub sidecar table (SPT). |
| `gcso_action_hub_hash_slot256_index` | Micro | $\mathcal{O}(1)$ | **Zero Allocation** | Computes direct 256-slot hash index for pointer trail caching. |
| `gcso_swarm_cell_chunk_step` | Micro | $\mathcal{O}(1)$ | **Zero Allocation** | Updates local cellular swarm cell state across token chunk boundaries. |
| `gcso_action_hub_pull_trail_to_attractor` | Micro | $\mathcal{O}(1)$ | **Zero Allocation** | Applies attractor pull force $F_{\mathrm{pull}}$ to pointer trails bottom-up toward anchor. |
| `gcso_action_hub_link_hallucinated_trails` | Micro | $\mathcal{O}(1)$ | **Zero Allocation** | Links cellular hallucinated trails into contiguous stigmergic trace graphs. |
| `gcso_action_hub_reduce_bit_tree` | Mezzo | $\mathcal{O}(1)$ | **Zero Allocation** | Reduces block-level bit-tree structures across multi-head PagedBlocks. |
| `gcso_action_hub_paged_block_warp_bitmask` | Mezzo | $\mathcal{O}(1)$ | **Zero Allocation** | Evaluates SIMD/Warp bitmask reduction over PagedBlock KV caches. |
| `gcso_daes_fast_path_lookup` | Micro Hot Path | $\mathcal{O}(1)$ | **Zero Allocation** | Executes $\mathcal{O}(1)$ fast-path bypass lookup in DAES dynamic scratchpad. |
| `gcso_daes_telemetry_push` | Nano Hot Path | $\mathcal{O}(1)$ | **Zero Allocation** | Pushes profiling metrics to DAES telemetry ring ledger in zero-alloc mode. |
| `gcso_daes_set_mode` | DAES Layer | $\mathcal{O}(1)$ | **Zero Allocation** | Configures DAES slot operating mode (`0` = Scratchpad, `1` = Plugin, `2` = Shared IPC Buffer). |
| `gcso_daes_evaluate_auto_tune` | DAES Layer | $\mathcal{O}(1)$ | **Zero Allocation** | Processes Telemetry Ring Ledger to derive optimized PSPM routing ratios, RIPA bounds & EDBC thresholds. |

---

### 6.4 DPSR Kernel Steering, QDPS, PSPM & SRL Adapter Matrix

| API Identifier | Operational Target | Invariants & Mathematical Constraints | Memory Allocation Rule |
| --- | --- | --- | --- |
| `gcso_dpsr_apply_phase_steering` | Nano Hot Path | Inline Query-Only phase rotation ($\theta_{m,i} \to \theta_{m,i} + \Delta\theta_i$). Non-overlapping Q/K tensors. | **Zero Allocation** |
| `gcso_dpsr_apply_phase_steering_safe` | Nano Hot Path | Applies RIPA soft-bounded $\tanh$ clamping ( $\Delta\theta_{\mathrm{safe}} = \theta_{\max} \tanh(\Delta\theta/\theta_{\max})$ ) on low-frequency channels. | **Zero Allocation** |
| `gcso_qdps_filter_step` | Nano Hot Path | Evaluates minimum step threshold $\Delta\theta_{\mathrm{min\_step}}$. Cuts off updates below threshold to prevent quantization jitter. | **Zero Allocation** |
| `gcso_dpsr_lazy_unwrap_override` | Micro Hot Path | Applies relative phase difference against context accumulator on Query tensor. | **Zero Allocation** |
| `gcso_dpsr_slerp_norm_guard_stable` | Nano Hot Path | Executes norm-guarded Slerp phase stabilization on state vectors. | **Zero Allocation** |
| `gcso_dpsr_fused_logit_shift` | Nano Hot Path | Fused inline logit phase shift prior to Softmax layer. | **Zero Allocation** |
| `gcso_pspm_dispatch_single_pass` | Nano Hot Path | Dispatches PSPM head-group phase profiles in a single forward pass. | **Zero Allocation** |
| `gcso_srl_eval_rank1` | Micro Hot Path | Evaluates SRL Dynamic Rank-1 outer product ( $\mathbf{y} = W\mathbf{x} + \mathbf{s} \odot (\mathbf{u}(\mathbf{v}^T \mathbf{x}))$ ). | **Zero Allocation** |
| `gcso_l2p_svd_project_lora` | Cold Path | Computes SVD on input LoRA matrices ($W_A, W_B$) to output Rank-1 SRL vectors ($\mathbf{u}, \mathbf{v}$) and phase profiles. | Cold Path Alloc Allowed |

---

### 6.5 Attractor Field, Stigmergic Aggregation & ZIMMS Storage Matrix

| API Identifier | Target Layer | Phase Modulation & Attractor Steering Role |
| --- | --- | --- |
| `gcso_attractor_field_add_anchor` | Macro | Registers topological anchor point in attractor field. |
| `gcso_attractor_field_add_system_prompt_anchor` | Macro | Maps natural language system prompt text as primary Anchor Attractor. |
| `gcso_attractor_field_add_embedding_anchor` | Macro | Maps dense feature embedding vector as continuous attractor anchor. |
| `gcso_attractor_field_inject_phase_repulsion` | Macro | Injects phase-conjugate repulsion vector ($-\boldsymbol{\Delta\theta}$) to flip spurious local minima into repulsive potential peaks. |
| `gcso_attractor_field_aggregate_bottom_up` | Macro | Aggregates high-density pointer trails bottom-up to macro-crystallize new dynamic anchors. |
| `gcso_persona_apply_patch` | Macro | Dynamic application of persona phase modulation patches without altering base weights. |
| `gcso_zimms_open_mmap` | Storage / ZIMMS | Maps `.gcso` container payload into memory space using zero-copy `mmap`. |
| `gcso_zimms_close_mmap` | Storage / ZIMMS | Unmaps zero-copy ZIMMS memory handle and releases Direct DMA resources. |

---

## 7. Architectural Execution Sequence Diagrams

### 7.1 Hot-Path Token Step Execution & DAES Fast-Path Bypass

```text
User / CLI           Rust Core / FFI               C-ABI Boundary                Hot-Path Kernel
   │                       │                               │                             │
   │─── step_token() ─────>│                               │                             │
   │                       │─── gcso_context_step_token ──>│                             │
   │                       │    (GCSO_RESTRICT Pointers)   │─── DAES Fast-Path Lookup ──>│ (O(1) Shortcut)
   │                       │                               │─── QDPS Step Filter ───────>│ (Threshold Check)
   │                       │                               │─── Inline DPSR Phase ──────>│ (Register Shuffle)
   │                       │                               │    (RIPA tanh Clamping)     │
   │                       │                               │─── PSPM Single-Pass Router> │ (Sub-Head Routing)
   │                       │                               │─── SRL Rank-1 Eval ────────>│ (Outer Product)
   │                       │                               │─── EDBC Entropy Eval ──────>│ (Z-Score H~)
   │                       │                               │─── DAES Telemetry Push ────>│ (Ring Ledger)
   │                       │                               │                             │
   │                       │<── GCSO_SUCCESS ──────────────│<── Trail & Status Output ───│
   │<── Token Result ──────│                               │                             │

```

---

### 7.2 Pointer Chain Stigmergy to Bottom-Up Attractor Crystallization & Pull Loop

```text
Action Hub (SPT)           Stigmergic Density Evaluator    Macro Attractor Field       EDBC / Persona Controller
       │                                │                            │                           │
       │── Step Tagged Pointers ───────>│                            │                           │
       │   (Update Pointer Trails)      │                            │                           │
       │                                │── Accumulate Stigmergy ───>│                           │
       │                                │ (rho_stigmergy > threshold)│                           │
       │                                │                            │── Pull Trails to Anchor ─>│ (Attractor Pull F_pull)
       │                                │                            │                           │
       │                                │                            │── Macro-Crystallize ─────>│ (Self-Organize New
       │                                │                            │   Bifurcated Attractor    │  Dynamic Anchor)
       │                                │                            │                           │
       │<── Dynamic Shortcut in DAES ───┼----------------------------┼---------------------------┘

```

---

### 7.3 Dynamic Self-Optimization & Multi-Layer Auto-Tuning Loop

```text
Hot-Path Kernel (Nano)       DAES Scratchpad Ledger      EDBC / Macro Controller     PSPM / DPSR Kernel
       │                           │                              │                         │
       │── Record Telemetry ──────>│                              │                         │
       │   (Hit/Miss, Phase Surge) │                              │                         │
       │                           │                              │                         │
       │                           │── Evaluate Auto-Tune ───────>│                         │
       │                           │   (Read Ring Ledger)         │                         │
       │                           │                              │── Update Sub-Head Ratio>│ (Fact/Logic/Explore)
       │                           │── Adjust RIPA Clamps ───────>│ (Dynamic Angle Max)     │
       │                           │                              │                         │
       │<── Fast-Path Shortcut ────┼------------------------------┴-------------------------┘
       │    Updated (O(1) Bypass)  │

```

---

### 7.4 Cohomological Repulsion Pulse Injection & L2P-SVD Hot-Swap Flow

```text
Hot-Path Kernel             EDBC Controller           Sheaf Cohomology Evaluator    Attractor Field / SRL Engine
       │                           │                              │                        │
       │── Token Step Result ─────>│                              │                        │
       │                           │── Evaluate Entropy H~ ──────>│                        │
       │                           │   (Check Singularity & H~)   │                        │
       │                           │                              │── Compute Residual ───>│
       │                           │                              │   r_obs in im(delta0)┴ │
       │                           │                              │                        │
       │                           │<── Cohomological Obstruction ┼────────────────────────┤
       │                           │    Class H^1(K; F) Detected  │                        │
       │                           │                                                       │
       │                           │── Inject Phase Repulsion Pulse (-delta theta) ───────>│ (Flip Local Minima
       │                           │                                                       │  to Repulsive Peak)
       │                           │── Hot-Swap L2P-SVD Rank-1 Vectors (u, v) ────────────>│ (Attach SRL Adapter)

```

---

## 8. ABI Verification & Boundary Safety Constraints

### 8.1 Memory Layout Alignment & Size Verification Matrix

| Structure Identifier | Exact Size (Bytes) | Alignment Invariant | Mandatory Offset Constraints |
| --- | --- | --- | --- |
| `gcso_paged_bitmask_t` | 32 | 32 Bytes (SIMD) | `bits`: Offset 0 |
| `gcso_pointer_trail_t` | 128 | 128 Bytes (Cacheline) | `current_ptr`: Offset 0, `user_data`: Offset 16, `accumulated_phase_delta`: Offset 56 |
| `gcso_config_t` | 64 | 4 Bytes | `head_dim`: Offset 0, `qdps_min_step_rad`: Offset 20, `daes_mode`: Offset 38 |
| `gcso_snapshot_header_t` | 128 | 64 Bytes | `magic`: Offset 0, `checksum_crc32`: Offset 56, `timestamp_epoch_sec`: Offset 64 |
| `gcso_pprc_keyframe_header_t` | 64 | 32 Bytes | `frame_type`: Offset 0, `icache_payload_offset`: Offset 24, `pcache_payload_offset`: Offset 32 |
| `gcso_pspm_config_t` | 32 | 16 Bytes | `num_fact_heads`: Offset 0, `fact_phase_gain`: Offset 8 |
| `gcso_edbc_state_t` | 64 | 32 Bytes | `moving_z_entropy`: Offset 0, `active_branch_mode`: Offset 24 |
| `gcso_srl_descriptor_t` | 64 | 32 Bytes | `layer_idx`: Offset 0, `u_vector_ptr`: Offset 8, `v_vector_ptr`: Offset 16 |
| `gcso_zimms_descriptor_t` | 64 | 32 Bytes | `mapped_address`: Offset 0, `file_size_bytes`: Offset 8 |
| `gcso_descriptor_header_t` | 8 | 4 Bytes | `struct_size`: Offset 0, `abi_version`: Offset 4 |
| `gcso_daes_slot_t` | 64 | 64 Bytes (Cacheline) | `mode`: Offset 0, `fast_path_shortcuts`: Offset 20 |

---

### 8.2 FFI Memory Isolation & Exception Boundary

```text
  [ Rust DSL / Core Runtime ]
               │
  (FFI Boundary: std::panic::catch_unwind)
               │  Intercepts Rust panics -> returns GCSO_ERROR_PANIC_CAUGHT
               ▼
   [ C11 / C++20 C-ABI Entry ]  <-- Exception-Free Barrier & Standard Calling Convention
               │
  (C++ Boundary: try-catch Exception Interception -> returns GCSO_ERROR_PANIC_CAUGHT)
               ▼
  [ C++20 / CUDA / Metal Core Kernels ]

```

1. **Opaque Pointer Encapsulation & Facade Pattern**: Internal runtime structures are strictly shielded behind opaque handles and exposed via Facade design patterns. Direct internal struct access across FFI is prohibited.
2. **C++ Exception Boundary**: All ABI entry points isolate C++ exceptions via exception interception blocks (`try-catch`), returning `GCSO_ERROR_PANIC_CAUGHT`.
3. **Rust Panic Unwinding Boundary (`catch_unwind`)**: All Rust callbacks or FFI exports wrap execution in `std::panic::catch_unwind` to prevent panics from unwinding across C-ABI boundaries.
4. **Zero-Allocation Hot Path**: Hot Path entry points execute zero dynamic allocations. Pointers must adhere to strict alignment constraints and non-aliasing rules (`GCSO_RESTRICT`).
5. **Safe Memory Deallocation**: All deallocation functions safely handle `NULL` pointers as no-ops, returning `GCSO_SUCCESS`.

---

### 8.3 Safety Assertions & Invariant Matrix

| Invariant Category | Verification Mechanism | Runtime Action / Status Output |
| --- | --- | --- |
| **Zero Allocation Enforcement** | Custom allocator tracking in unit tests | Hot-path allocations trigger immediate runtime status error `GCSO_ERROR_INVALID_STATE`. |
| **Memory Alignment Assertions** | FFI boundary memory pointer check | Misaligned input tensors return `GCSO_ERROR_MISALIGNED_POINTER`. |
| **RIPA Angle Bounds** | In-register $\tanh$ soft clamping | Phase overflow soft-clamped to `ripa_clamp_max_rad` bounds in register shuffle. |
| **QDPS Resolution Threshold** | Magnitude evaluation against $\Delta\theta_{\mathrm{min\_step}}$ | Sub-threshold updates rounded to zero or return `GCSO_ERROR_QDPS_UNDERFLOW`. |
| **Descriptor Compatibility** | `struct_size` and `abi_version` header check | Mismatched descriptor layouts trigger `GCSO_ERROR_VERSION_MISMATCH`. |
| **ZIMMS Zero-Copy Mapping** | Page boundary alignment and file descriptor check | Unaligned offset or failed mmap triggers status `GCSO_ERROR_ZIMMS_MAPPING_FAILED`. |
| **DAES Dynamic Bounds** | In-place ring head/tail mask modulo checks | Out-of-bounds telemetry writes trigger status `GCSO_ERROR_DAES_SCRATCHPAD_FULL`. |
