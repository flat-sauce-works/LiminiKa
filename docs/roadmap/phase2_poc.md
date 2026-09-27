# Phase 2: PoC ( Proof of Concept ) Development Roadmap

## 1. Objectives & Scope

The objective of Phase 2 is to implement and verify a "Vertical Slice" that proves the mathematical model defined in the GCSO ( Geometric Cellular Sheaf Orchestrator ) whitepaper and the core concepts of the LiminiKa DSL actually operate at high speed, with a lightweight footprint, and with zero dynamic memory allocation ( Zero-Allocation ) under the constraints of a 2–4 GB VRAM environment.

As a fundamental design principle of this project, direct online computational costs from the macro level—such as heavy online matrix inversion or global topology decoding—are completely eliminated. The entire system is explicitly decoupled into two poles: a **"Bottom-up Execution & Parsing Path ( Hot Path )"** and a **"Macro Control & Evaluation Mechanism ( Macro Control Rule / Cold Path )"** that aggregates and evaluates at specific lines.

Even within the Hot Path, a self-similar ( fractal ) evaluation and steering architecture is maintained throughout: "executing minimal local computations ( micro ), locally evaluating them at a specific line directly above ( local macro ), and cascading the result as a single micro unit viewed from higher layers." Similarly, in DSL parsing, a bottom-up extraction approach starting from lowest-level literals / key-value pairs is adopted to transmit parameters to the C-ABI as macro steering rules ( `GCSOConfig` ).

---

### Fractal Bottom-Up Cascade Architecture

This system possesses a self-similar structure across every computational granularity ( Nano / Micro / Mezzo / Macro ), where "minimal computation ( micro )" and "local evaluation ( macro )" are paired, and lower-level evaluation results cascade bottom-up into upper-level micro elements. By thoroughly eliminating direct online calculation costs from the macro level, global steering is completed through the aggregation of local operations.

```text
+-----------------------------------------------------------------------------------------+
| [Macro Level] Session / Attractor Level ( Cold Path / EDBC & Attractor Field )          |
|   - Macro Control   : Observation of aggregated lower states, Moving Z-score            |
|                       entropy ( H~ ) evaluation                                         |
|   - Macro Steering  : AOT phase-based atomic updates, phase transition &                |
|                       attractor evaluation                                              |
+-----------------------------------------------------------------------------------------+
                                         ▲
                                         │ ( Cascading aggregation of lower block values )
+-----------------------------------------------------------------------------------------+
| [Mezzo Level] Chunk / Segment Level ( Cellular Layer-Hub )                              |
|   - Mezzo Execution : PagedBlock aggregate processing ( 16-32 tokens ) via Micro inputs |
|   - Mezzo Control   : Batch phase sync via Skip-Hop Bus & Bit-Tree Reduction            |
+-----------------------------------------------------------------------------------------+
                                         ▲
                                         │ ( Cascading aggregation of token evaluation results )
+-----------------------------------------------------------------------------------------+
| [Micro Level] Token / PagedBlock Level ( Hot Path Core )                                |
|   - Micro Execution : O( 1 ) SPT lookup, Fused DPSR phase addition ( Δθ ), Lazy Unwrap  |
|   - Micro Control   : Bitmask reduction, 1st-order local entropy surge evaluation       |
+-----------------------------------------------------------------------------------------+
                                         ▲
                                         │ ( Cascading aggregation of Warp/SIMD local values )
+-----------------------------------------------------------------------------------------+
| [Nano Level] Warp / SIMD Level ( In-Kernel Hot Path )                                   |
|   - Nano Execution  : Branchless bitwise ops & Q7 fixed-point addition in 32 threads    |
|   - Nano Control    : In-register RIPA clamping & immediate steering via Warp Shuffle   |
+-----------------------------------------------------------------------------------------+

```

1. **Nano Level ( Warp / SIMD Level / In-Kernel Hot Path )**:
* **Nano Execution ( Micro Computation )**: Completes branchless bitwise operations and Q7 fixed-point addition within minimal compute regions such as 32 threads ( Warp ) or PagedBlocks.
* **Nano Control ( Local Macro Evaluation )**: Instantly evaluates and determines local clamp checks ( RIPA ) and steering states at the Warp level via Warp register shuffles and bitmask conjunctions ( `AND`/`OR` ) without accessing VRAM or L2 cache.


2. **Micro Level ( Token / PagedBlock Level / Hot Path Core )**:
* **Micro Execution ( Micro Computation )**: Treats Nano-level aggregated results as single micro units, executing $O( 1 )$ index lookups on the Sidecar Pointer Table ( SPT ) and Fused RoPE phase addition ( $\Delta\theta$ ) during every token's inference loop.
* **Micro Control ( Local Macro Evaluation )**: Executes PagedBlock-unit bitmask reductions and primary evaluations of local entropy surges, evaluating and finalizing steering directions for subsequent tokens on-the-fly.


3. **Mezzo Level ( Chunk / Segment Level / Cellular Layer-Hub )**:
* **Mezzo Execution ( Micro Computation )**: Receives Micro-level outputs as block aggregate units, performing intermediate execution of topological updates across 16–32 token ranges.
* **Mezzo Control ( Local Macro Evaluation )**: Performs inter-layer phase synchronization and block-level evaluations via the Skip-Hop Bus or Buffered Bit-Tree Reduction, supplying compact state values to the upper Macro layer.


4. **Macro Level ( Session / Attractor Level / Cold Path )**:
* **Macro Control ( Observation & Global Control )**: Asynchronously and non-blockingly evaluates moving Z-score entropy ( $\tilde{H}$ ) and phase transition flux ( $\Phi_M( t )$ ) aggregated bottom-up from lower levels, atomically swapping and supplying global steering parameters ( e.g., AOT phase-based ).



---

### IP Compliance & Independent Implementation Policy

This project is an open-source project released under dual **MIT / Apache-2.0** licensing ( for source code ) and **CC BY 4.0** ( for documentation ). Clean intellectual property compliance is thoroughly maintained throughout the entire development process.

* **Complete Elimination of Copyleft Code & Independent Implementation Policy**:
No code subject to Copyleft licenses ( such as GPL, LGPL, AGPL, SSPL ) or non-commercial restrictions ( such as CC BY-NC ) will be copied, ported, modified, or adapted. The core control routines and kernel layers ( DPSR, SPT, EDBC, SRL, etc. ) do not copy or adapt source code from existing inference engines ( such as llama.cpp, vLLM, Hugging Face transformers, etc. ), enforcing a completely independent implementation based strictly on the GCSO whitepaper and C-ABI specification ( `docs/architecture/c_abi_spec.md` ).
* **Legitimate Use of Permissive External Ecosystems & Attribution Management**:
For file format handling ( `safetensors` ), basic build infrastructure, standard C-ABI bindings, parsing helpers ( `winnow`, etc. ), and test generation environments ( Python scripts, etc. ), only crates/libraries with Permissive licenses explicitly allowed in `deny.toml` ( MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, CC0-1.0, 0BSD, Unlicense, etc. ) will be introduced as valid dependencies. Copyright attribution and NOTICE requirements for all used dependencies will be strictly observed.
* **License Markings & Comment-Based SPDX Header Policy**:
* **Source Code & Scripts**: Considering the possibility of code snippets being extracted, used, or redistributed individually, the header comment block of every source file ( `.rs`, `.cpp`, `.cu`, `.metal`, `.h`, `.hpp`, `.py` ), C-ABI header, and build definition file ( `build.rs`, `CMakeLists.txt` ) must explicitly state `SPDX-License-Identifier: MIT OR Apache-2.0`.
* **Documentation & Specifications**: To enhance maintainability and maintain consistency with existing documentation, individual license headers will not be added into Markdown documentation files; instead, they will collectively adhere to `LICENSE-CC-BY-4.0` and `LICENSE.md` at the repository root.


* **Base Model Selection & Model-Agnostic Governance Policy**:
Base models used for evaluation, verification, and prototyping are strictly limited to major open-source / open-weight models with clear terms of use and redistribution terms ( SmolLM2, Llama 3.2, Gemma 2, Phi-3.5-mini, etc. ), strictly complying with their respective terms of use ( Meta Llama 3.2 Community License, Gemma Terms of Use, etc. ) without bypassing terms or redistributing weights automatically. The LiminiKa runtime itself is designed as a model-agnostic engine that does not depend on specific model weights or proprietary license terms. Large model weight binaries must never be committed to the Git repository and are managed under local acquisition assumptions ( enforced via `.gitignore` ).
* **Automated Dependency License Auditing**:
* **Rust Domain**: Allowed licenses ( MIT, Apache-2.0, BSD, ISC, CC0, etc. ) are strictly managed in `deny.toml`, making automated auditing via `cargo-deny` mandatory in CI ( `license-safety-check` in `ci.yml` ) and local environments.
* **C/C++ Domain**: Unauthorized inclusion of third-party external libraries is prevented, building strictly with self-authored headers ( `include/liminika/` ) and C++17 compliant source code.



---

### Model Sourcing & Fixture Strategy

Large weight binaries of pre-trained models are not stored directly in the Git repository; they are handled via local environments specified in `.gitignore` and test fixture generators.

* **CI & Unit Testing via Synthetic Dummy Weights**:
Using `tests/fixtures/generate_dummy_model.py` ( a self-authored tool adhering to Permissive licensing ), lightweight, randomly initialized `safetensors` structures ( a few MB to tens of MB ) are generated on-demand. These are used to verify pipeline connectivity, data integrity across C-ABI / FFI boundaries, and zero dynamic allocation ( managed under `tests/fixtures/`, with `SPDX-License-Identifier: MIT OR Apache-2.0` explicitly stated in the file header comment ).
* **Selection of Open-Source / Open-Weight Models for On-Device Inference & Qualitative Evaluation**:
For experiments on DPSR phase steering and long-context retention in actual 2–4 GB VRAM environments, the following lightweight models are adopted ( assuming local downloads outside the repository and user consent to respective licenses ):
* **SmolLM2 ( 135M / 360M / 1.7B )**: Model for ultra-fast prototyping during kernel development and C-ABI boundary testing ( Apache-2.0 License ).
* **Llama 3.2 ( 1B / 3B )**: Primary validation model featuring standard RoPE structure ( Meta Llama 3.2 Community License ).
* **Gemma 2 ( 2B )**: Small evaluation model with high reasoning capability ( Gemma Terms of Use ).
* **Phi-3.5-mini ( 3.8B )**: Model for long-context ( 128k ) phase retention experiments ( MIT License, assuming 4-bit/3-bit quantization ).



---

### Out of Scope ( Intentionally Excluded in PoC Phase )

* **Direct Online Execution of Heavy Macro Computations**: Strict online Hodge decomposition and real-time matrix inversion of 0-Laplacian / 1-Laplacian matrices ( restricted to hooking/referencing pre-generated AOT Eigen-Phase Bases and bottom-up aggregation ).
* **Complex Multi-threaded Synchronization**: Asynchronous lock-free structures ( assuming single-threaded / basic mutual exclusion, with non-blocking / atomic pointer swapping on the read side only ).
* **Complex Compiler Optimizations in DSL**: Construction of top-down giant ASTs, type inference, and complex optimization passes ( restricted for PoC to constructing a minimal bottom-up parser/AST framework that extracts primary key-value pairs of `persona` / `session` and directly transfers parameters to FFI / C-ABI ).
* **Advanced Binary Compression**: Advanced GOP compression and motion prediction algorithms in `.gcso` format ( restricted to serialization of headers and fixed-length P-Cache/I-Cache structures ).

---

## 2. Milestones & Tasks

```text
[M2.1: C-ABI & Types] ( Boundary Layer )
   │
   ├──> [M2.1.5: Early E2E Mock Pipeline] ( Boundary Layer ) ────────────────────────────┐
   │                                                                                     │
   ├──> [M2.2: DPSR Kernel] ( Micro / Hot Path & Nano Hot Path ) ─┐                      │
   │                                                              ├─> [M2.4.5: Core Integration] ──> [M2.5: DSL Parser & FFI Skeleton]
   ├──> [M2.3: SPT, Swarm & Storage] ( Micro / Hot Path ) ────────┤          ( Boundary )                  ( Boundary Layer )
   │                                                              │                                   │
   └──> [M2.4: EDBC & Math] ( Macro / Cold Path ) ────────────────┘                                   └──> [M2.6: Full E2E Demo]
                                                                                                           ( E2E Verification )

```

### Milestone 2.1: C-ABI Specification & Type Foundation [Boundary Layer]

* **Objective**: Establish the C-ABI data structures, error-handling conventions, license header notations, license checking, and build pipeline that serve as the boundary between the C++ Core/Kernel layer and the Rust Core / DSL.
* **Target Directories/Files**: `include/liminika/`, `src/core/src/abi/`, `src/dsl/src/ffi/`, `tests/ffi/`, `deny.toml`, `CITATION.cff`
* **Primary Tasks**:
* [ ] `include/liminika/gcso_types.h`, `gcso_config.h`: Define Q7 fixed-point ( $\beta_{\mathrm{Q7}} = 1/128$ ), PagedBlock, Bitmask, and phase vector structures, constant macros, and error code enums ( `LiminiKaStatus` ). Explicitly include error states such as `GCSO_ERROR_PANIC_CAUGHT` for panic handling and `GCSO_ERROR_OUT_OF_MEMORY` for memory exhaustion. Explicitly state `SPDX-License-Identifier: MIT OR Apache-2.0` in all header comment blocks. Write structure size and alignment verification code using `static_assert` on the C++17 side.
* [ ] `include/liminika/gcso_abi.h`: Define C-ABI function signatures and declare memory deallocation functions ( `gcso_free_*` ) based on GCSO Whitepaper Table 0.1 and `docs/architecture/c_abi_spec.md`. Enforce `noexcept` on all C++ side ABI entry point functions, catching internal exceptions via `try-catch` and converting them into error codes. Explicitly state the SPDX identifier in the header comment.
* [ ] `src/core/build.rs` & `src/dsl/build.rs`: Configure automatic C++ kernel builds and automatic FFI binding generation via `cmake-rs` and `bindgen`, as well as target Cargo feature branching settings. Explicitly state the SPDX identifier in file header comments.
* [ ] `tests/ffi/`: Automate C-ABI structure alignment verification tests on the Rust side ( `std::mem::size_of`, `align_of`, `offset_of!` ), and test handling of exception/panic catching ( `std::panic::catch_unwind` ) and error codes across FFI boundaries.
* [ ] `deny.toml` and `CITATION.cff` Verification: Run `cargo deny check licenses` to verify that all added dependencies satisfy permissive licenses and that repository metadata definitions are normal.



---

### Milestone 2.1.5: Early E2E Mock Pipeline & Test Fixture Foundation [Boundary Layer]

* **Objective**: Pass through the execution chain from Rust CLI → Rust DSL / Core → C-ABI → C++ early using dummy C++ cores and synthetic test weights/data fixtures, preventing interface design and data passing regressions in advance.
* **Target Directories/Files**: `src/cli/`, `src/core/src/abi/`, `src/dsl/src/ffi/`, `src/kernels/`, `examples/c_api/`, `tests/fixtures/`, `CMakeLists.txt`, `src/core/build.rs`
* **Primary Tasks**:
* [ ] Implement empty mock functions ( stubs like `gcso_dpsr_apply_phase_steering` ) on the C++ side and add target definitions inside `src/kernels/CMakeLists.txt`. Integrate build targets by adding `add_subdirectory( src/kernels )` in the root `CMakeLists.txt`. Explicitly state `SPDX-License-Identifier: MIT OR Apache-2.0` in each file header comment.
* [ ] Cross-platform support for `cmake-rs` invocations in `src/core/build.rs`: Flexibly handle compiler flags, C++17 specifications, and GPU kernel ( CUDA / Metal / Vulkan ) build branches across Windows ( MSVC ), macOS ( Clang ), and Linux ( GCC ).
* [ ] `tests/fixtures/generate_dummy_model.py`: Create an automated generation script for synthetic random weights ( `safetensors` / lightweight model layouts ) and dummy phase vectors for testing and CI ( placed under `tests/fixtures/`, with `SPDX-License-Identifier: MIT OR Apache-2.0` explicitly stated in the comment block ).
* [ ] `tests/fixtures/`: Set up binary placement and script output configurations for test pseudo-logit data and dummy phase vectors.
* [ ] Execute dummy C-ABI functions from CLI via Rust FFI to verify that return values and steering logs are correctly returned.



---

### Milestone 2.2: DPSR Kernel Implementation & Verification [Micro Layer / Hot Path & Nano Hot Path]

* **Objective**: Verify micro-computation behavior of fast phase steering using Query-Only Relative Phase Shift ( Lazy Phase Unwrapping ) and RIPA ( Restricted Inline Phase Alignment ). Use the CPU reference implementation as Ground Truth for correctness, and verify branchless design within Warp / PagedBlock units and complete avoidance of GPU divergence via Warp Shuffle local macro evaluation in prototype GPU kernels ( CUDA/Metal/Vulkan ).
* **Target Directories**: `src/core/src/dpsr/`, `src/kernels/common/`, `src/kernels/cpu/`, `src/kernels/cuda/`, `src/kernels/metal/`, `src/kernels/vulkan/`, `examples/c_api/`, `benches/`, `tests/core/`
* **Primary Tasks**:
* [ ] CPU Reference Implementation ( Ground Truth ): Implement reference algorithms such as `gcso_dpsr_apply_phase_steering` in `src/kernels/cpu/` and `src/core/src/dpsr/`. Explicitly state `SPDX-License-Identifier: MIT OR Apache-2.0` in all source header comments.
* [ ] C-API Unit Verification: Create a minimal example directly driving the CPU reference implementation from C in `examples/c_api/`.
* [ ] RIPA Angle Clamping & Phase Cycling Control: Implement $\tanh$ clamping control ( $\Delta\theta_{\mathrm{safe}} = \theta_{\max} \tanh( \Delta\theta/\theta_{\max} )$ ) for low-frequency channels ( $d_{\mathrm{head}}/4$ ) and perform numerical stability tests for Q7 overflow / phase wrap-around prevention ( branchless implementation eliminating conditional branching ).
* [ ] Prototype GPU / SIMD Kernels ( `src/kernels/cuda/`, `src/kernels/metal/`, `src/kernels/vulkan/` ): Build inline fusion prototypes for register-level phase addition ( $\theta_{m,i} \to \theta_{m,i} + \Delta\theta_i$ ) and verify Nano Hot Path local macro evaluation using Warp Shuffle ( phased connection based on environment priority ).
* [ ] Automated Zero-Allocation Verification Test in Hot Path: Integrate a custom allocator in `tests/core/` to verify via unit tests that zero dynamic memory allocations ( `Vec`, `Box`, etc. ) occur during per-token phase application and inference loops.
* [ ] `benches/`: Measure micro-benchmarks comparing standard RoPE vs DPSR-fused RoPE across C++ / GPU / Rust environments ( Target: compute time overhead within +5%, register pressure invariance, Zero-Allocation in Hot Path confirmed ).



---

### Milestone 2.3: Sidecar Pointer Table ( SPT ), Swarm Space, and Storage Implementation [Micro Layer / Hot Path]

* **Objective**: Implement $O( 1 )$ array index pointer management and Bitmask operations, and create a prototype for the minimal `.gcso` binary format.
* **Target Directories**: `src/core/src/swarm/`, `src/core/src/storage/`, `tests/core/`
* **Primary Tasks**:
* [ ] Implement $O( 1 )$ index lookup and update logic ( `gcso_hash_slot256_index`, etc. ) for the Sidecar Pointer Table ( SPT ) in `src/core/src/swarm/`. Explicitly state SPDX identifiers in all source header comments.
* [ ] Prototype bitwise operation routines for PagedBlock Bitmasks ( 16–32 token units ) and block-level local evaluation logic ( `gcso_paged_block_warp_bitmask` ).
* [ ] `src/core/src/storage/`: Implement minimal `.gcso` binary container routines ( Header + uncompressed I-Cache / P-Cache Frame basic Data Dump/Read ) and ownership drop logic.



---

### Milestone 2.4: EDBC & Dynamic Entropy Engine Implementation [Macro Evaluation Layer / Cold Path]

* **Objective**: Verify macro evaluation rules that hook and attach pre-defined phase vectors based on bottom-up entropy surge detection. Supply steering information in a non-blocking manner to the Hot Path via atomic parameter swapping.
* **Target Directories**: `src/core/src/edbc/`, `src/core/src/math/`, `src/core/src/srl/`, `examples/c_api/`, `tests/core/`
* **Primary Tasks**:
* [ ] `src/core/src/math/`: Implement maximum eigenvalue upper-bound estimation based on Gershgorin Disc Theorem ( $\hat{\lambda}_{\max} = 2 \max_i D_{ii}$ ) and basic functions for the Langevin dynamics Pitchfork Bifurcation model.
* [ ] Implement calculation logic for Moving Z-Score Normalized Attention Entropy ( $\tilde{H}$ ) ( including logarithmic singularity prevention clamp $\epsilon_{\mathrm{log}} = 10^{-12}$ and range $[0, \ln 2]$ verification tests ).
* [ ] Implement hysteresis evaluation using the Sliding-Window Entropy Rate Integrator ( $\Phi_M( t )$ ).
* [ ] `src/core/src/srl/`: Build minimal evaluation logic for Sparse Residual Adapter Layer ( SRL ) Dynamic Rank-1 structures ( $\mathbf{y} _ {\mathrm{srl}} = W_{\mathrm{base}} \mathbf{x} + \mathbf{s} \odot ( \mathbf{u} ( \mathbf{v}^T \mathbf{x} ) )$ ).
* [ ] Create unit tests for EDBC branch evaluation mocks using a Mock Logit Generator to hook and attach specified phase vectors/pulsations upon entropy surge detection.
* [ ] **EDBC-DPSR Dynamic Coupling Verification ( C-API Level )**: Implement C API unit integration tests in `examples/c_api/` to verify whether the DPSR kernel's phase vector $\Delta\theta$ is dynamically modulated and steered upon entropy surge detection.



---

### Milestone 2.4.5: Core Module Integration [Boundary Layer]

* **Objective**: Interconnect submodules ( `dpsr`, `swarm`, `edbc`, `math`, `srl`, `storage`, `abi` ) under `src/core/src/` in `lib.rs`, establishing an integrated internal Rust pipeline combining micro and macro processing.
* **Target Directories**: `src/core/src/lib.rs`, `tests/integration/`, `tests/core/`
* **Primary Tasks**:
* [ ] In `src/core/src/lib.rs`, build high-level APIs that integrate entropy evaluation ( `edbc`/`math` ) → phase generation → SPT index updates ( `swarm` ) → DPSR invocation ( `dpsr`/`abi` ) → storage reading/writing ( `storage` ) ( ensuring asynchronous, non-blocking steering parameter update structures for the Hot Path ).
* [ ] Add automated Zero-Allocation integration tests: Apply custom allocators to the integrated environment to automatically ensure that no illegal dynamic memory allocations occur on the Hot Path during Core pipeline execution.
* [ ] Add unit and integration tests at the integrated level. Explicitly state SPDX identifiers in header comments of all test files.



---

### Milestone 2.5: LiminiKa DSL Parser & AST Compiler Minimal Skeleton ( PoC ) [Boundary Layer / Bottom-Up Parsing]

* **Objective**: Parse core elements of declarative DSL syntax ( `.lmnk` ) via a bottom-up approach that constructs from atomic key-value pairs, establishing an execution pipeline that directly transmits parameters to FFI / C-ABI structures ( `GCSOConfig` ) via AST skeletons.
* **Target Directories**: `src/dsl/src/parser/`, `src/dsl/src/ast/`, `src/dsl/src/compiler/`, `src/dsl/src/ffi/`, `src/cli/`, `tests/dsl/`
* **Primary Tasks**:
* [ ] `src/dsl/src/parser/`: Using a lightweight parser combinator ( `winnow`, etc., permitted permissive crates in `deny.toml` ), implement an ultra-lightweight bottom-up parser skeleton built from lowest-level tokens and literals. For PoC scope, construct accurate extraction of key-value pairs and safe forward recovery for priority blocks: `persona` ( phase/repel settings inside `attractors` and style/memory path extraction ) and `session` ( `boost_phase` / `repel_phase` control rules inside `events` ).
* [ ] `src/dsl/src/ast/`: Define minimal AST structures ( `PersonaDecl`, `SessionDecl`, `EventRule`, etc. ) to store parsing results, and implement bottom-up data mapping into internal representations.
* [ ] `src/dsl/src/compiler/`: Implement a conversion layer ( skeleton ) that directly maps AST structures to C-compatible GCSO execution configuration structures ( `GCSOConfig` ).
* [ ] `src/dsl/src/ffi/`: Build FFI bindings and memory layout transformation logic that safely transmit converted parameters from Rust to the C++ GCSO C-ABI ( `include/liminika/gcso_abi.h` ).
* [ ] `src/cli/`: Create entry point ( `liminika run <file.lmnk>` ) and perform end-to-end execution testing of the FFI pipeline using C++ mock cores.



---

### Milestone 2.6: End-to-End PoC Integration Demo & Verification [E2E Verification Layer]

* **Objective**: Drive the GCSO Core and open-weight models ( SmolLM2 / Llama 3.2 1B / Gemma 2 2B, etc. ) from DSL scripts ( `.lmnk` ) via CLI, running an E2E demo that outputs phase steering logs.
* **Target Directories**: `examples/c_api/`, `examples/dsl/`, `tests/integration/`, `src/cli/`
* **Primary Tasks**:
* [ ] `examples/c_api/demo_dpsr.c`: Create a C sample directly invoking the C API to execute phase shifts and entropy measurements ( explicitly state `SPDX-License-Identifier: MIT OR Apache-2.0` in the file header comment ).
* [ ] `examples/dsl/monika_poc.lmnk`: Create a DSL script for PoC verification ( an integrated validation script including `persona` and `session` ).
* [ ] Implement real-time log displaying step-by-step entropy variations $\tilde{H}$ and phase shifts $\Delta\theta$ during CLI execution.
* [ ] `tests/integration/`: Automate integration tests across all layers: CLI → DSL Parser → AST → FFI → GCSO Core.



---

## 3. Definition of Done & Acceptance Criteria

> 💡 **Development Flexibility vs. Strict Milestone Completion Policy**
> Based on the low-pressure development policy defined in `docs/dev/guidelines.md`, temporary CI failures ( red status ) and work-in-progress states are fully permitted during individual development tasks and Draft PR stages. However, upon official milestone completion for Phase 2 ( formal release into `main` / `develop` ), all of the following acceptance criteria must be met, and the CI pipeline must pass completely.

1. **IP Compliance & Legal Adherence**:
* `cargo deny check licenses` passes successfully ( automatically verified by the `license-safety-check` job in `deny.toml` and `ci.yml` ).
* Zero inclusion of GPL/Copyleft code, no direct copying or adaptation from existing inference engine codebases ( guaranteeing self-authored independent implementation ), SPDX license identifiers correctly attached to comment blocks of all source code, scripts, and build files, and verification conducted strictly with open-source/open-weight models having explicit public licenses/terms of use ( enforcing non-tracking protection of weight binaries via `.gitignore` ).


2. **C-ABI / FFI Consistency & Safety**:
* `cargo test --workspace` and `ctest` pass all tests.
* No discrepancies exist in structure layouts ( `sizeof`, `align`, `offset_of` ) between C++ and Rust, ensuring zero memory leaks and no undefined behavior across FFI boundaries ( such as unhandled panics or leaked C++ exceptions ) verified via `catch_unwind` capture and C++ side `noexcept`/`try-catch` conversion to `LiminiKaStatus`. Verify that `LiminiKaStatus` contains error codes for panic and memory exhaustion.


3. **Resources & Overhead ( 2–4 GB VRAM Environment & Fractal Execution )**:
* Under the constraints of 2–4 GB VRAM environments, no unnecessary memory allocations occur or expand, automatically verifying via custom allocator tests that Zero Dynamic Allocation ( Zero-Allocation ) is maintained within the per-token Hot Path.
* DPSR phase processing operates within target bounds ( compute overhead within +5% ) relative to standard RoPE processing ( verified on CPU reference and GPU kernels ).
* Unnecessary conditional branching ( Branch Divergence ) and register spills do not occur at the SIMD/Warp level, ensuring local evaluations at Nano/Micro/Mezzo/Macro levels function correctly bottom-up.


4. **Steering Operation ( E2E Pipeline )**:
* When running `liminika run` with parameters defined in a DSL script ( `.lmnk` ) ( e.g., `boost_phase`, `repel_phase` ), the DPSR kernel and EDBC entropy evaluation are invoked as intended via C-ABI, outputting logs that show real-time phase vector $\Delta\theta$ steering and modulation.


5. **Code Quality & CI Automation**:
* `cargo fmt --all -- --check` and `cargo clippy` pass without warnings, and all build/test jobs in `ci.yml` ( `cpp-build-format`, `rust-check`, `license-safety-check`, `meta-citation-check` ) pass completely.


6. **Full Synchronization of Documentation & Headers**:
* Any changes to C-ABI specifications or DSL syntax arising during development are fully synchronized with specifications in `docs/architecture/` ( `c_abi_spec.md`, `container_spec.md`, `dsl_spec.md`, `overview.md` ) and C header files in `include/liminika/`.
