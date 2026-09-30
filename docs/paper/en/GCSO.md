# Geometric Cellular Sheaf Orchestrator (GCSO): Design Concept of a RoPE Phase-Modulated Sheaf-Theoretic Memory Control Runtime for Local LLMs Based on Swarm-Attractor Stigmergic Architecture

> Note: This document is an architecture proposal/whitepaper presenting a system architecture and mathematical formulation.

---

## Abstract

To address the issues of local context inconsistency, attention saturation and context lock-in (**Attention Saturation / Context Lock-in**) in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems, physical constraints of VRAM/DRAM bandwidth and capacity during long-context processing (especially resource-constrained environments with 2–4 GB VRAM), as well as the reduction of non-factual generation (Off-manifold Latent Attractors / Spurious Local Minima) while maintaining model expressiveness, this paper proposes **GCSO (Geometric Cellular Sheaf Orchestrator: Swarm-Attractor Stigmergic Architecture)**, an inference and memory runtime architecture that integrates graph spectral theory, higher-order activation Taylor dynamics, differential geometric attention control, and low-layer memory management.

Unless otherwise specified, the dimension $d$ denoted in this paper refers to the Attention Head dimension $d _ {\mathrm{head}}$ (an even number), which is strictly distinguished from the overall hidden layer dimension $d _ {\mathrm{model}}$ and the FFN intermediate dimension $d _ {\mathrm{ffn}}$. Therefore, the phase rotation group is formulated as $SO(2)^{d _ {\mathrm{head}}/2}$.

The structural rationale for why this framework exhibits high processing efficiency in a 2–4 GB VRAM environment lies in the fact that advanced continuous geometric structures (Hodge decomposition, cohomological obstruction classes $H^1(K; \mathcal{F})$, and Eyring-Kramers potential fields) are not computed by directly solving them online, but are defined as a macro Target Attractor Field that the system should reach. The actual computation (Hot Path) is reduced to micro-level stigmergic (environment-mediated) interactions—such as rewriting Bitmasks by Attention Heads (swarm cells), Tagged Pointers in the Sidecar Pointer Table (SPT), and Q7 phase differentials $\boldsymbol{\Delta\theta}$ (**Two-Layer Complementary Swarm-Attractor Architecture / Stigmergic Swarm-Attractor Duality**). The processing for each token is completed via $\mathcal{O}(1)$ bit/pointer operations, autonomously converging to the global attractor as a mean-field limit.

While keeping a single weight model (equivalent to 2 GB) resident in VRAM, Asymmetric Layer-wise Quantization is applied. It combines inline fusion within the RoPE kernel for $SO(2)^{d _ {\mathrm{head}}/2}$ register-level phase rotation in Query and Key tensors (Dynamic Phase-Shifted RoPE / Context-Conditioned Rotary Offsets: DPSR), 2D block-isotropic scaling (amplitude modulation via Isotropic Block-Diagonal Scaling), and limited intermediate-layer Sparse Residual Adapter Layers (SRL, residual adapters with Dynamic Rank-1 structure). Furthermore, by applying the **Phase-Gated Head-Wise Subspace Ensemble (Parameter-Efficient Sub-Head MoE)** structure—which multiplexes phase offsets held by pointers and performs expert allocation to Attention Head subgroups—it realizes the **Phase-Steered Parallel Multi-head Ensemble (PSPM, Single-Pass Head-Wise Ensemble)** architecture, which concurrently drives multiple specialized processing units in a single pass.

In this framework, the cellular layer (micro) and the attractor field (macro) are interpreted as representations at different resolutions on the same phase space $\mathcal{M}$. Non-factual generation (hallucination) is not merely treated as external noise, but is geometrically identified as inappropriate local minima (**Spurious Local Minima / Off-manifold Latent Attractors**) existing on the complementary space $\mathcal{M} \setminus \mathcal{X}$ of the same phase space. Precisely because they exist within the same phase field, simply recording the phase pattern of erroneous local solutions onto the stigmergic environmental field as an anti-phase dynamically flips its energy "valley" into a "repulsive potential peak" (**Phase-Conjugate Attractor Repulsion**), autonomously guiding trajectories toward legitimate attractors without modifying model weights.

For the 1-cochain residual $\mathbf{r} \in C^1(K; \mathcal{F})$ arising from uncertainty, a canonical orthogonal decomposition based on the Regularized Green Operator $\mathbf{G} _ {\epsilon} = (\Delta _ 0 + \epsilon \mathbf{I})^{-1}: C^0(K; \mathcal{F}) \to C^0(K; \mathcal{F})$ with Tikhonov regularization is introduced:

$$\begin{aligned}
\mathbf{r} &= \delta _ 0 \boldsymbol{\phi} + \mathbf{r} _ {\mathrm{obs}} \\
\boldsymbol{\phi} &= \mathbf{G} _ {\epsilon} \delta _ 0^* \mathbf{r} \in C^0(K; \mathcal{F}) \\
\mathbf{r} _ {\mathrm{obs}} &= \delta _ 1^* \boldsymbol{\psi} + \mathbf{r} _ {\mathrm{harm}} = (\mathbf{I} _ {C^1} - \mathbf{P} _ {\mathrm{im}(\delta _ 0)}) \mathbf{r} \in \mathrm{im}(\delta _ 0)^\perp
\end{aligned}$$

(where $\boldsymbol{\phi} \in C^0(K; \mathcal{F}), \boldsymbol{\psi} \in C^2(K; \mathcal{F})$, and $\mathbf{P} _ {\mathrm{im}(\delta _ 0)} = \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^* : C^1(K; \mathcal{F}) \to C^1(K; \mathcal{F})$ is a Tikhonov-regularized quasi-orthogonal projection operator). This separately evaluates the solvable component $\mathrm{im}(\delta _ 0)$ derived by the smoothed scalar potential $\boldsymbol{\phi} \in C^0(K; \mathcal{F})$, and the structural obstruction residual $\mathbf{r} _ {\mathrm{obs}} \in \mathrm{im}(\delta _ 0)^\perp$ that includes the solenoidal component $\delta _ 1^* \boldsymbol{\psi}$ and the harmonic component $\mathbf{r} _ {\mathrm{harm}} \in \mathrm{ker}(\Delta _ 1)$ (the subscript $\mathrm{obs}$ refers to the cohomological **obstruction** class; global 1-cohomology obstruction space $H^1(K; \mathcal{F}) \cong \mathrm{ker}(\Delta _ 1)$). Based on the algebraic confrontation between the Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ on the cell complex $K$ and the Cellular Sheaf Homology $H _ 1(K; \mathcal{F})$ on the Dual Cell Complex (dual pairing $\langle \cdot, \cdot \rangle: C^1(K; \mathcal{F}) \times C _ 1(K; \mathcal{F}) \to \mathbb{R}$ and dual diagonal invariant evaluation based on discrete Morse theory), Persistent Homology is evaluated as dual cycles on the 1-homology space $H _ 1(K; \mathcal{F})$.

To reduce numerical oscillations (Gibbs phenomenon), CPU/GPU execution delays, and VRAM bandwidth contention associated with 0-Laplacian pseudo-inverse computations, this framework integrates **Graph Laplacian Spectral Filtering**—which incorporates the construction of a **Block-Level Coarse Graph (supernodes)** during the Prefill stage and dynamic upper-bound evaluation of maximum eigenvalues via the Gershgorin Disc Theorem (upper bound $\hat{\lambda} _ {\max} = 2 \cdot \max _ {i} D _ {ii}$ based on the unnormalized Graph Laplacian $\Delta _ 0 = D - A$ under non-negative weighted undirected graph conditions $A _ {ij} \ge 0, A _ {ij} = A _ {ji}$)—alongside an **Entropy-Gated Phase Lookup Table (EG-LUT, static lookup)** and **Entropy-Gated Dynamic Phase Lookup (EGDPL, dynamic steering/prefetching)** referencing pre-extracted phase bases (Ahead-Of-Time Eigen-Phase Base: AOT Eigen-Phase Base). By constructing the Graph Laplacian $\Delta _ 0$ in advance using PagedAttention block units (16–32 tokens) as supernodes and processing them as linear combinations of representative phase base vectors, the computational complexity is compressed to linear $\mathcal{O}(K _ {\mathrm{base}} \cdot d _ {\mathrm{head}})$ ($K _ {\mathrm{base}}$ is the number of bases, $d _ {\mathrm{head}}$ is the Head dimension), suppressing the computational load on the Cold Path.

In per-token processing (Hot Path), computation is reduced strictly to reading and adding Sidecar Pointer Table (SPT) values ($\mathcal{O}(1)$ index lookups). To avoid $\mathcal{O}(N)$ re-rotation operations across the entire past Key cache during dynamic phase transitions, phase-independent standard Keys are preserved within the KV cache, and **Query-Only Relative Phase Shift / Lazy Phase Unwrapping** is introduced to apply the relative phase difference against the context accumulated phase value in a single batch on the Query side inside the attention kernel. This unifies the structure inside existing RoPE kernels into one that adds angle arrays ($\theta _ {m,i} \to \theta _ {m,i} + \Delta\theta _ i$), suppressing VRAM reads/writes and keeping operations to constant-time $\mathcal{O}(1)$ additions. Furthermore, register-level **Bank-Conflict-Free Register Shuffle**, **Warp-Cooperative Paged Block Bitmasking**, and 2D phase rotations in constant time $\mathcal{O}(d _ {\mathrm{head}})$ are executed, deploying **Speculative Bit-Level Phase Prefetching** to predictively load the next phase from the quantized KV bit pattern of the immediately preceding token.

For maintaining facts across long contexts, the framework introduces a **Decoupled Fact-Semantic Memory Architecture**, **Entropy-Gated Key Anchor Preservation**, **Sparse Cross-Attention Correction**, and **Phase Residual Aggregated KV Pruning**. By cumulatively saving the directional influence that discarded KV caches exerted on attention space into phase residuals as continuous approximation transitions of composite vector length $\bar{R} _ c$ and von Mises concentration parameter $\kappa _ c$ in circular statistics, it suppresses phase dissipation (Phase Cancellation) and maintains topological shadows of long-range context while reducing VRAM consumption.

When applying to pre-trained models, **Asymmetric Layer-wise Quantization** is adopted, protecting shallow layers and the LM Head layer with FP8/INT4 while adjusting intermediate layers to 1.5–3.5 bit quantization. Furthermore, applying **Restricted Inline Phase Alignment (RIPA, Soft-Bounded Phase Clamping)** (which clamps angles restricted to low-frequency channels, i.e., upper $d _ {\mathrm{head}}/4$ dimensions), **Dynamic Anchor Head** selection (focusing on intermediate-layer guidance and concept extraction heads), and **Softmax Temperature dynamic correction** achieves Perplexity stabilization while suppressing attention drift and context structure distortion.

To address the discretization grid (Quantization Floor) of weight space in ultra-low quantization environments, **Quantization-Discretized Phase Steering (QDPS)** is introduced to evaluate and cut off rotations below the minimum effective phase rotation angle $\Delta\theta _ {\mathrm{min\ _ step}}$. Additionally, for limited layers such as intermediate FFN Down-Projection ($W _ {\mathrm{down}} \in \mathbb{R}^{d _ {\mathrm{model}} \times d _ {\mathrm{ffn}}}$) and Attention Out-Projection, the **Sparse Residual Adapter Layer (SRL)** mechanism is deployed to add FP8 outer product vectors ($\mathbf{u} \in \mathbb{R}^{d _ {\mathrm{model}}}, \mathbf{v} \in \mathbb{R}^{d _ {\mathrm{ffn}}}$, Rank-1) and diagonal gain scalars $\mathbf{s} \in \mathbb{R}^{d _ {\mathrm{model}}}$, algebraically supplementing information transformation capabilities equivalently to dynamic LoRA (at a size of a few KB to a few MB) beyond mere attention score modulation. A **Training-free SVD LoRA-to-Phase Converter (LoRA-to-Phase SVD Projection: L2P-SVD)** is placed to directly convert existing fine-tuned LoRA weights into phase profiles and SRL vectors without requiring additional gradient training or optimization.

Unobserved knowledge regions are interpreted as closed cycles on the 1-homology space $H _ 1(K; \mathcal{F})$, deploying **Moving Z-Score Normalized Attention Entropy ($\tilde{H}$)** that incorporates moving-standardized cosine distances of SRL-corrected activations and a logarithmic singularity prevention clamp $\epsilon _ {\mathrm{log}}$. Via local normalization of Top-2 probabilities, its value range is bounded to $[0, \ln 2]$, ensuring numerical stability. Furthermore, a **Sliding-Window Entropy Rate Integrator** executing moving-time differential integration over $M$ consecutive tokens is deployed to identify sustained high-entropy states strictly as phase transition critical points.

Based on this, an **Entropy-Driven Decoding Branch Controller (EDBC)** is deployed to identify phase transitions. In exploration mode, it achieves **Potential-Driven Sampling / Energy-Guided Decoding**, which controllably induces phase perturbations (**Metastable Transition Phase Perturbation**) that trigger transitions out of metastable states. **Lemma 3.1 (Pitchfork Bifurcation Model in Decoding Dynamics based on Langevin Equation)**, formulated using a scalar order parameter $x \in \mathbb{R}$ on the primary bifurcation axis, is introduced.

Upon detecting erroneous local solutions (Spurious Local Minima), **Phase-Conjugate Attractor Repulsion** is triggered to apply anti-phase pulses to the corresponding Head's phase, inverting energy valleys into repulsive potential fields (peaks). The phase pattern of the corresponding Head is stored as an anti-phase $-\boldsymbol{\Delta\theta} _ {\mathrm{hallucination}}$ into **Contrastive Phase Memory (Dynamic Negative Constraint Memory)**, functioning as a repulsive potential field in subsequent search spaces. For highly coherent exploratory thought trajectories, **Bifurcated Latent Attractors** protect them, storing them into a **Dynamic Phase Modulation Profile** through **Persistent Topological Mode Quantization**, thereby achieving commutative personality switching without rewriting model weights.

For memory management, the **Predictive Phase-Motion & Residual Compensation (PPRC / Temporal Key-Frame KV Cache Compression)** scheme is presented, defining **`.gcso` (GCSO Unified Container Format)** as the physical binary container for unified data management. Conversation context is structured into I-Cache (Key Anchor) and P-Cache (Phase Motion Vector $\boldsymbol{\Delta\theta} _ t$ and Sparse Scalar Residual $s _ t, \mathbf{e} _ t$), applying a Variable Group of Pictures (GOP) Structure at entropy surge points to achieve **Zero-Forward Latency Seek (Instantaneous Replay)** without forward pass calculations. Long-context phase errors are reduced and absorbed into a **Multi-Head Virtual Position Shift ($\Delta m^{(h)}$)**.

For projecting external natural language prompts and RAG contexts, a **Product of Lie Group Rotations** is introduced, structuring text to apply Lie group rotations $\mathbf{R} _ l(\boldsymbol{\Delta\theta} _ l)$ to sub-channels of each hierarchical node $l$. Utilizing system prompts or external knowledge structures not as rewritten weight parameters in VRAM, but as an **"External Attractor Field"** that deforms the latent phase field, offloads context and knowledge storage costs from VRAM to high-speed NVMe SSD / DRAM (`.gcso` containers) (**Storage Cost Offloading**), expanding expressiveness under physical constraints of 2–4 GB VRAM.

Furthermore, it specifies modified formulas for **Pseudo-ultrametric Spaces ($d _ {\mathcal{V}}$) in Hierarchical Tree Structures**, **Quantized Depth Pointer Tables (QDPT)**, **Cascade Pointer Tables (CPT)**, **Visual Attention Steering** for image/video tokens, and specifications for the common container format `.gcso` and its deployment format `.gcsopack`, ensuring compatibility with existing inference engines.

---

## Notation Table

| Symbol | Definition and Geometric / Engineering Meaning |
| --- | --- |
| $d _ {\mathrm{head}}$ | Dimension of Attention Head (even number). Phase rotation in this architecture is based on $d = d _ {\mathrm{head}}$. |
| $SO(2)^{d _ {\mathrm{head}}/2}$ | Direct sum structure of $d _ {\mathrm{head}}/2$ copies of the 2D phase rotation group $SO(2)$. |
| $K$, $\mathcal{F}$ | Simplicial/Cell Complex $K$ and Coefficient Sheaf $\mathcal{F}$ assigned upon it. |
| $C^k(K; \mathcal{F})$, $\delta _ k$ | $k$-Cochain space and coboundary operator $\delta _ k: C^k \to C^{k+1}$. |
| $H^1(K; \mathcal{F})$ | 1st-degree Cellular Sheaf Cohomology space (uncertainty / global obstruction space). |
| $H _ 1(K; \mathcal{F})$ | 1st-degree Cellular Sheaf Homology space (cycle space on the dual complex). |
| $\Delta _ 0$, $\Delta _ 1$ | 0-Laplacian operator ($\delta _ 0^* \delta _ 0$) and 1-Laplacian operator ($\delta _ 1^* \delta _ 1 + \delta _ 0 \delta _ 0^*$). |
| $\mathbf{G} _ {\epsilon}$ | Tikhonov-regularized Green operator $(\Delta _ 0 + \epsilon \mathbf{I})^{-1} : C^0 \to C^0 \quad (\epsilon > 0)$. |
| $\mathbf{P} _ {\mathrm{im}(\delta _ 0)}$ | Tikhonov-regularized quasi-orthogonal projection operator $\delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^* : C^1 \to C^1$ onto the solvable subspace. |
| $\boldsymbol{\Delta\theta}$ | DPSR phase modulation offset vector (Q7 fixed-point representation). |
| $\boldsymbol{\theta} _ {\mathrm{base}}$ | Base standard positional phase angle vector of RoPE. |
| $\mathbf{p} _ {\mathrm{SPT}}$ | Q7-quantized 1-cochain offset vector stored in Sidecar Pointer Table (SPT). |
| $\beta _ {\mathrm{Q7}}$ | Q7 fixed-point quantization scale factor ($\beta _ {\mathrm{Q7}} = 1/128$). |
| $d _ {\mathcal{V}}$ | Hierarchical tree pseudo-ultrametric defined on Dyadic tree structures. |
| $\tilde{H}$ | Moving Z-Score Normalized Attention Entropy (dynamic attention entropy metric, range $[0, \ln 2]$). |
| $\mathcal{C} _ {\mathrm{void}}$ | Coherent Vector Alignment Metric for Out-of-Distribution Latent Space (latent manifold alignment score / coherence metric). |
| $\mathbf{r} _ {\mathrm{obs}}$, $\mathbf{r} _ {\mathrm{harm}}$ | Structural obstruction residual component ($\mathrm{obs}$ indicates obstruction) and harmonic component in 1-cochain residual $\mathbf{r}$. |
| $\mathbf{e} _ t$, $\mathbf{r} _ t^{\mathrm{residual}}$ | Compression error/residual scalar and vector within P-Cache (distinguished from 1-cochain residual $\mathbf{r}$). |
| $\bar{R} _ c$, $\kappa _ c$ | Resultant vector length and von Mises distribution concentration parameter in circular statistics. |
| $\tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$ | Continuous target attractor field on latent manifold parameterized by phase parameters $\boldsymbol{\theta} = \boldsymbol{\theta} _ {\mathrm{base}} + \boldsymbol{\Delta\theta}$. |

---

## Acronym Index

* **DPSR (Dynamic Phase-Shifted RoPE / Context-Conditioned Rotary Offsets):** Mechanism to inline-execute dynamic phase modulation inside the RoPE kernel.
* **RIPA (Restricted Inline Phase Alignment / Soft-Bounded Phase Clamping):** Phase alignment with angle clamping restricted to low-frequency channels.
* **QDPS (Quantization-Discretized Phase Steering):** Discrete steering that cuts off phase rotations below the minimum step on a quantization grid.
* **PSPM (Phase-Steered Parallel Multi-head Ensemble / Single-Pass Head-Wise Ensemble):** Ensemble structure driving Head groups with different phase profiles in parallel in a single pass.
* **EG-LUT (Entropy-Gated Phase Lookup Table):** Static lookup structure referencing pre-extracted AOT phase bases.
* **EGDPL (Entropy-Gated Dynamic Phase Lookup):** Dynamic phase prefetching and in-kernel synthesis structure according to entropy metrics.
* **EDBC (Entropy-Driven Decoding Branch Controller):** Decoding branch controller executing exploration/convergence routing control at entropy surge points.
* **PPRC (Predictive Phase-Motion & Residual Compensation / Temporal Key-Frame KV Cache Compression):** Temporal Key-Frame KV cache compression scheme via Variable GOP Structures.
* **SPT (Sidecar Pointer Table):** Sidecar pointer table holding and referencing phase offsets and topology information in $\mathcal{O}(1)$.
* **QDPT (Quantized Depth Pointer Table) / CPT (Cascade Pointer Table):** Pointer structures supporting resolution depth and cascade referencing in hierarchical tree structures.
* **ZIMMS (Zero-Overhead In-Memory Mapped Storage):** Zero-overhead storage deployment mechanism based on memory-mapped access.
* **SRL (Sparse Residual Adapter Layer / Dynamic Rank-1 Residual Adapter):** Residual adapter with Dynamic Rank-1 structure supplementing intermediate-layer FFN/Attention projections at low cost.
* **L2P-SVD (LoRA-to-Phase SVD Projection):** Converter projecting trained LoRA weights via first-order SVD decomposition into phase profiles and SRL vectors without additional training.
* **GCSO-DNP / DNP (Dynamic Node Protocol):** Communication protocol for phase and topology control information between dynamic distributed nodes.

---

## Chapter 0: Correspondence Table of Geometric Mathematical Concepts, Engineering Data Structures, and Processing Mechanisms

### 0.1 Two-Layer Complementary Dynamics and Swarm-Attractor Geometric Approximation Hybrid Model

The GCSO framework is fundamentally rooted in a **Two-Layer Complementary Swarm-Attractor Architecture (hybrid approximation model)** consisting of a continuous macro attractor field governed by differential geometry and algebraic topology, alongside a micro discrete space of bitwise and pointer operations acting as executable instructions in the inference Hot Path.

1. **Macro Target Field (Macro Target Attractor Field / Continuous Target Space):**

A continuous space formulated by Hodge decomposition, cohomological obstruction classes $H^1(K; \mathcal{F})$, Eyring-Kramers potential fields $\tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$, etc. These are not heavy differential equations solved directly during inference, but function as a **Target Attractor Field** that the micro system as a whole should autonomously track and reach.

2. **Micro Execution Layer (Micro Stigmergic Swarm Layer / Discrete Execution Space):**

A discrete space where individual Attention Heads (cellular swarm agents) and Thread-Warps inside Multi-Head Attention locally update Bitmasks on PagedBlocks, Tagged Pointers in the Sidecar Pointer Table (SPT), and Q7 fixed-point phase differentials $\boldsymbol{\Delta\theta}$. By relying solely on asynchronous updates (stigmergy) of bit patterns and pointers left in the shared memory field (environment), Hot Path processing per token is completed in $\mathcal{O}(1)$, accumulating micro-operations to autonomously converge to the macro target attractor.

3. **Synergy of Local Minima Interpretation and Stigmergy in the Same Phase Space:**

The micro cellular layer and macro attractor field are not separate spaces, but **the same phase space $\mathcal{M}$ viewed at different micro/macro resolutions**. Non-factual generation (hallucination) is interpreted as local solutions (another attractor / Spurious Local Minima) on the complementary space of the observable manifold within this same field. Precisely because they exist in the same field, writing the phase of non-factual local solutions back into the stigmergic field as an anti-phase enables inverting the potential field into a "repulsive peak" (Phase-Conjugate Attractor Repulsion), exerting a synergistic effect that guides global trajectories toward legitimate solutions through minimal computation.

---

### Table 0.1: Geometric Mathematical Concepts ⇔ Practical Engineering Data Structures / Algorithms ⇔ C-ABI Interface Correspondence Table

| Geometric Mathematical Concept | Practical Engineering Data Structure / Algorithm | C-ABI Interface Specification |
| --- | --- | --- |
| Pseudo-ultrametric space in hierarchical tree structures, boundary control | Dynamic resolution depth ($D _ {\mathcal{V},\mathrm{continuous}}^*, \bar{D} _ {\mathcal{V},\mathrm{quantized}}$), Quantized Depth Pointer Table (QDPT), Cascade Pointer Table (CPT), hysteresis quantization ($\delta _ {\mathrm{hyst}}$), modified pseudo-ultrametric definition formula, Dyadic regularization weight ($w _ {\mathrm{key}}$), Dyadic Key bit-width ($B _ {\mathrm{dyadic}}=128$), Entropy-Driven Decoding Branch Controller (EDBC, Moving Z-Score Normalized Attention Entropy $\tilde{H}$, Sliding-Window Entropy Rate Integrator $\Phi _ M(t)$), Potential-Driven Sampling, Metastable Transition Phase Perturbation, Eyring-Kramers potential barrier deformation model, pseudo-ultrametric ($d _ {\mathcal{V}}$), cohomological obstruction classes in latent spaces ($H^1(K; \mathcal{F})$), Persistent Topological Mode Quantization | `gcso_edbc_init`<br>`gcso_edbc_eval_stateful`<br>`gcso_edbc_free`<br>`gcso_cvoid_eval_dyadic128`<br>`gcso_cvoid_eval_barrier` |
| Dynamic Phase-Shifted RoPE (DPSR), Restricted Inline Phase Alignment (RIPA), Quantization-Discretized Phase Steering (QDPS), Phase-Steered Parallel Multi-head Ensemble (PSPM), Phase-Gated Head-Wise Subspace Ensemble | Phased deployment engine (Tier-0~3), Givens/Householder coupled Rank-$k$ geometric modulation, soft-bounded clamp ($\boldsymbol{\Delta\theta} _ {\mathrm{safe}} = \theta _ {\max} \tanh(\boldsymbol{\Delta\theta}/\theta _ {\max})$), Query-Only Relative Phase Shift (Lazy Phase Unwrapping), Dynamic Anchor Head selection (intermediate-layer top 10–20% Head allocation), Single-Pass Head Group Router, Dynamic Head Subspace Routing, Entropy-Gated Phase Lookup Table (EG-LUT / EGDPL), Speculative Bit-Level Phase Prefetching, Sparse Residual Adapter Layer (SRL) with Dynamic Rank-1 structure, Training-free SVD LoRA-to-Phase Converter (L2P-SVD), $SO(2)^{d _ {\mathrm{head}}/2}$ commutativity ($[\mathbf{R} _ {\mathrm{RoPE}}, \mathbf{R} _ {\mathrm{DPSR}}] = \mathbf{0}$), attention entropy control | `gcso_dpsr_init`<br>`gcso_dpsr_apply_phase_steering`<br>`gcso_dpsr_apply_phase_steering_safe`<br>`gcso_dpsr_lazy_unwrap_override`<br>`gcso_dpsr_apply_soft_phase_damping`<br>`gcso_srl_dynamic_mlp_gate_eval`<br>`gcso_dpsr_slerp_norm_guard_stable`<br>`gcso_dpsr_fused_logit_shift`<br>`gcso_compute_procrustes_phase_delta` |
| Decomposed Spatial Activation Chunks, regularized Hodge decomposition and indicator-field guided asynchronous control, stigmergic bit updates | Hierarchical Local Topology Coordinator (Cellular Layer-Hub), Tikhonov-regularized Graph Laplacian Spectral Filtering ($T _ k(\tilde{\mathbf{L}})$) on coarse supernodes, Gershgorin Disc upper-bound evaluation, AOT phase base clustering, lock-free async queues for heterogeneous CPU-GPU boundaries, Windowed-Tree Sheaf canonical residual projection ($\mathbf{P} _ {\mathrm{im}(\delta _ 0)} = \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^*$), Hierarchical Bit-Tree Reduction, Sidecar Pointer Table (SPT), dual Tagged Pointer index, Warp Bitmasking | `gcso_swarm_cell_chunk_step`<br>`gcso_hub_reduce_bit_tree`<br>`gcso_cellular_state_update_express`<br>`gcso_paged_block_warp_bitmask`<br>`gcso_hash_slot256_index` |
| Memory engine and resource-constrained heterogeneous memory structures | `.gcso` (GCSO Unified Container Format), `.gcsocore` (Base Model Core), Cascade Pointer Table (CPT), `.gcsopack` (Unified Deployment Package) and Zero-Conversion Sidecar Mode, Zero-Overhead In-Memory Mapped Storage (ZIMMS), Direct-DMA Async Ring-Buffer (`io_uring` SQPOLL / DirectStorage API), 2-Level Ring Buffer Prefetching, resident Asymmetric Layer-wise Quantization backbone (shallow/LM Head: FP8/INT4, intermediate: 1.5-3.5bit), Decoupled Fact-Semantic Memory Architecture (Fact Anchor Track + Semantic Phase Track), Phase Residual Aggregated KV Pruning & von Mises concentration $\kappa _ c$, Predictive Phase-Motion & Residual Compensation (.gcso / PPRC) & Entropy-Gated Key Anchor Preservation & Instantaneous Replay, Product of Lie Group Rotations ($\prod _ l \mathbf{R} _ l(\boldsymbol{\Delta\theta} _ l)$), External Natural Language Attractor & Storage Cost Offloading, Contrastive Phase Memory (Dynamic Negative Constraint Memory), Phase-Conjugate Attractor Repulsion, Dynamic Node Protocol (GCSO-DNP), visual attention spatial geometric steering | `gcso_container_open_mmap`<br>`gcso_container_get_track`<br>`gcso_pprc_seek_to_token`<br>`gcso_container_close`<br>`gcso_mem_pushout_align`<br>`gcso_mem_stigmergic_offload`<br>`gcso_persona_apply_patch`<br>`gcso_dnp_dispatch_packet` |

---

## Chapter 1: Theoretical Background and Geometric/Algebraic Formulations

### 1.1 Cohomological Obstruction Classes and Regularized Hodge Decomposition in Out-of-Distribution Latent Space

Let the learned phase space in a large language model be defined as an observable manifold $\mathcal{X} \subset \mathcal{M}$. Finite contexts directly referenced/generated by the model are described as geometric structures on $\mathcal{X}$, whereas non-factual generation (Off-manifold Latent Attractors) and context inconsistency phenomena occur on the complementary space $\mathcal{M} \setminus \mathcal{X}$ of the observable domain across the full phase space $\mathcal{M}$—that is, as local minima (**Spurious Local Minima / Off-manifold Latent Attractors**) in the **Out-of-Distribution Latent Space / Complementary Subspace**. In this framework, non-factual generation is formulated not as mere numerical noise, but as inappropriate local solutions (attractors) existing within the same phase field.

For a coefficient sheaf $\mathcal{F}$ assigned on a cell complex $K$, let the $k$-cochain space be $C^k(K; \mathcal{F})$, and the coboundary operator be $\delta _ k: C^k(K; \mathcal{F}) \to C^{k+1}(K; \mathcal{F})$. An uncertainty residual $\mathbf{r} \in C^1(K; \mathcal{F})$ generated during context generation satisfies the following canonical orthogonal decomposition by the Hodge Decomposition Theorem in differential geometry:

$$\begin{aligned}
\mathbf{r} &= \delta _ 0 \boldsymbol{\phi} + \mathbf{r} _ {\mathrm{obs}} \\
\boldsymbol{\phi} &= \mathbf{G} _ {\epsilon} \delta _ 0^* \mathbf{r} \in C^0(K; \mathcal{F}) \\
\mathbf{r} _ {\mathrm{obs}} &= \delta _ 1^* \boldsymbol{\psi} + \mathbf{r} _ {\mathrm{harm}} = (\mathbf{I} _ {C^1} - \mathbf{P} _ {\mathrm{im}(\delta _ 0)}) \mathbf{r} \in \mathrm{im}(\delta _ 0)^\perp
\end{aligned}$$

where $\boldsymbol{\phi} \in C^0(K; \mathcal{F}), \boldsymbol{\psi} \in C^2(K; \mathcal{F})$, the coboundary dual operator of $\delta _ 0: C^0 \to C^1$ is $\delta _ 0^*: C^1 \to C^0$, $\delta _ 1^*: C^2 \to C^1$, and the 0-Laplacian operator is $\Delta _ 0 = \delta _ 0^* \delta _ 0 : C^0(K; \mathcal{F}) \to C^0(K; \mathcal{F})$.

The Regularized Green Operator $\mathbf{G} _ {\epsilon}$ incorporating Tikhonov regularization is defined as an operator on $C^0(K; \mathcal{F}) \to C^0(K; \mathcal{F})$ as follows:

$$\mathbf{G} _ {\epsilon} = (\Delta _ 0 + \epsilon \mathbf{I})^{-1}: C^0(K; \mathcal{F}) \to C^0(K; \mathcal{F}) \quad (\epsilon > 0)$$

The spectral response of this operator follows $g _ \epsilon(\lambda) = \frac{1}{\lambda + \epsilon}$, playing the role of deriving a smoothed scalar potential $\boldsymbol{\phi} = \mathbf{G} _ {\epsilon} \delta _ 0^* \mathbf{r} \in C^0(K; \mathcal{F})$ from the residual $\mathbf{r} \in C^1(K; \mathcal{F})$. The Tikhonov-regularized quasi-orthogonal projection operator $\mathbf{P} _ {\mathrm{im}(\delta _ 0)}$ onto the solvable space is a mapping $C^1(K; \mathcal{F}) \to C^1(K; \mathcal{F})$, formulated as:

$$\mathbf{P} _ {\mathrm{im}(\delta _ 0)} = \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^* : C^1(K; \mathcal{F}) \to C^1(K; \mathcal{F})$$

In this decomposition, $\boldsymbol{\phi} \in C^0(K; \mathcal{F})$ is a scalar potential providing a smoothable solvable component (exact cochain $\mathrm{im}(\delta _ 0)$), while the complementary component $\mathbf{r} _ {\mathrm{obs}} \in \mathrm{im}(\delta _ 0)^\perp$ (the subscript $\mathrm{obs}$ indicates **obstruction**) is formulated under regularization parameter $\epsilon > 0$ as a quasi-orthogonal component (a completely orthogonal component in the limit $\epsilon \to 0^+$). It is collectively decomposed into a solenoidal component $\delta _ 1^* \boldsymbol{\psi} \in \mathrm{im}(\delta _ 1^*)$ (local rotational noise) and a harmonic component $\mathbf{r} _ {\mathrm{harm}} \in \mathrm{ker}(\Delta _ 1)$ (global obstruction) (the 1-Laplacian is $\Delta _ 1 = \delta _ 1^* \delta _ 1 + \delta _ 0 \delta _ 0^* : C^1(K; \mathcal{F}) \to C^1(K; \mathcal{F})$).

In particular, a residual component that satisfies the local closedness condition ($\delta _ 1 \mathbf{r} \approx \mathbf{0}$) yet cannot be reduced to a solvable component is uniquely identified as the harmonic component $\mathbf{r} _ {\mathrm{harm}}$, which is an element of the 1-cohomology obstruction space:

$$H^1(K; \mathcal{F}) = \frac{\mathrm{ker}(\delta _ 1)}{\mathrm{im}(\delta _ 0)} \cong \mathrm{ker}(\Delta _ 1)$$

That is, non-coboundary closed cycles remaining beyond exact cochains constitute **Cohomological Obstructions in Latent Sheaves**, geometrically identified as the structural cause of non-factual generation (attraction into false attractors).

Based on the algebraic confrontation between the Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ (representation via cochain complex $C^*$) on cell complex $K$ and the Cellular Sheaf Homology $H _ 1(K; \mathcal{F})$ (representation via chain complex $C _ *$) on the Dual Cell Complex (dual pairing $\langle \cdot, \cdot \rangle: C^1(K; \mathcal{F}) \times C _ 1(K; \mathcal{F}) \to \mathbb{R}$ and dual diagonal invariant evaluation based on discrete Morse theory), Persistent Homology is evaluated as dual cycles on the 1-homology space $H _ 1(K; \mathcal{F})$.

---

### 1.2 Graph Laplacian Spectral Filtering, Tikhonov Regularization, Gershgorin Disc Upper-Bound Evaluation, and AOT Phase Bases

To prevent CPU/GPU load from online real-time full graph construction and Laplacian pseudo-inverse computations, this framework builds a **Block-Level Coarse Graph (supernodes)** during the Prefill stage, using PagedAttention blocks (16–32 tokens) as nodes. The Graph Laplacian $\Delta _ 0$ is calculated from the cosine similarity matrix between text chunks, constraining edge count $\vert E \vert$. Furthermore, it unifies **Graph Laplacian Spectral Filtering** with an **Entropy-Gated Phase Lookup Table (EG-LUT, static lookup)** and **Entropy-Gated Dynamic Phase Lookup (EGDPL, dynamic steering)**.

To accurately and rapidly estimate the maximum eigenvalue $\lambda _ {\max}$ of the Graph Laplacian $\Delta _ 0$ during normalization to the Chebyshev expansion domain $[-1, 1]$, a non-negative ($A _ {ij} \ge 0$) and symmetric ($A _ {ij} = A _ {ji}$) undirected graph is constructed as a prerequisite for adjacency matrix $A _ {ij}$ using Shifted Cosine or $\max(0, \cos(\cdot))$. In the unnormalized Graph Laplacian ($\Delta _ 0 = D - A$), the $i$-th row sum is $\sum _ j (\Delta _ 0) _ {ij} = 0$, disc center $a _ {ii} = D _ {ii}$, and radius $R _ i = \sum _ {j \neq i} \vert a _ {ij} \vert = \sum _ {j \neq i} A _ {ij} = D _ {ii}$. Therefore, by the **Gershgorin Disc Theorem**, the eigenvalue spectrum is contained within $\bigcup _ i [D _ {ii} - R _ i, D _ {ii} + R _ i] = [0, 2 D _ {ii}]$, and the upper bound $\hat{\lambda} _ {\max}$ of the maximum eigenvalue is derived via a single-pass calculation ($\mathcal{O}(\vert V \vert)$ complexity) from node degrees:

$$\hat{\lambda} _ {\max} = \max _ {i} \left( D _ {ii} + \sum _ {j \neq i} \vert a _ {ij} \vert \right) = 2 \cdot \max _ {i} D _ {ii}$$

Using the derived $\hat{\lambda} _ {\max}$ ($\lambda _ {\min} = 0$), a uniform affine linear transformation is executed:

$$\tilde{\mathbf{L}} = \frac{2}{\hat{\lambda} _ {\max}} \Delta _ 0 - \mathbf{I}$$

This guarantees that the eigenvalue spectrum is normalized into the $[-1, 1]$ interval, and for the Tikhonov-regularized filter function $g _ \epsilon(\lambda) = \frac{1}{\lambda + \epsilon}$, using a function with re-defined variable dependencies following affine transformation:

$$g _ \epsilon(\tilde{\lambda}) = \frac{1}{\frac{\hat{\lambda} _ {\max}}{2}(\tilde{\lambda} + 1) + \epsilon}$$

the Chebyshev expansion expanded with order $K _ {\mathrm{cheb}}$ (typically $K _ {\mathrm{cheb}} = 3 \sim 5$) first-kind Chebyshev polynomials $T _ k(\tilde{\lambda})$:

$$c _ k = \frac{2-\delta _ {k0}}{\pi} \int _ {-1}^{1} \frac{g _ \epsilon(\tilde{\lambda}) T _ k(\tilde{\lambda})}{\sqrt{1-\tilde{\lambda}^2}} d\tilde{\lambda}$$

is guaranteed to converge.

Through Ahead-Of-Time (AOT) analysis, representative phase base vectors $\mathbf{v} _ 1, \mathbf{v} _ 2, \dots, \mathbf{v} _ {K _ {\mathrm{base}}}$ ($K _ {\mathrm{base}} \ll \vert E \vert$) corresponding to domains or thought patterns are pre-extracted and statically placed in VRAM. Based on the division of roles between static lookup EG-LUT and dynamic prefetching/routing EGDPL, in the Prefill stage, initial coefficients are calculated using a lightweight 1-layer router MLP that directly outputs combination coefficients $\alpha _ k$ of AOT phase bases from text feature vectors of the prompt. Inside the Hot Path kernel, a scale factor $\gamma(\tilde{H})$ is calculated immediately from Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) described later, synthesizing phase modulation angles in-kernel:

$$\boldsymbol{\Delta\theta} = \gamma(\tilde{H}) \cdot \sum _ {k=1}^{K _ {\mathrm{base}}} \alpha _ k \mathbf{v} _ k$$

This formulation suppresses Cold Path computational load and CPU-GPU communication, compressing computational complexity to linear $\mathcal{O}(K _ {\mathrm{base}} \cdot d _ {\mathrm{head}})$ with respect to base count $K _ {\mathrm{base}}$ and Head dimension $d _ {\mathrm{head}}$. Derived geometric potentials are meshed and stored into the Sidecar Pointer Table (SPT) as quantized phase indices, accessed from the Hot Path as constant-time $\mathcal{O}(1)$ lookup table (LUT) references.

---

### 1.3 Definition and Proof of Pseudo-ultrametric Conditions in Hierarchical Tree Pseudo-ultrametric Spaces

To control conceptual distance and structural resolution in out-of-distribution latent space, a **Hierarchical Tree Pseudo-ultrametric Space** is formulated.

In a Dyadic tree structure, let the depth (distance from root) of the Lowest Common Ancestor (LCA) of two state points $x, y \in \mathcal{M}$ be $l _ {\mathrm{lca}}(x, y) \in \{0, 1, \dots, l _ {\max}\}$. State sets assigned to the same deepest subtree (or same PagedBlock structure) at maximum depth $l _ {\max}$ of the Dyadic tree are defined as equivalence class $x \sim y$, where $l _ {\mathrm{lca}}(x, y) = l _ {\max}$ is assumed. As equivalence classes are assigned to deeper internal nodes, $l _ {\mathrm{lca}}(x, y)$ becomes larger, and $l _ {\max} - l _ {\mathrm{lca}}(x, y)$ becomes smaller.

Using a bounded, monotonically non-decreasing function $f: \mathbb{R} _ {\ge 0} \to [0, 1)$ passing through the origin ($f(0) = 0$), the distance $d _ {\mathcal{V}}(x, y)$ is defined as follows:

$$d _ {\mathcal{V}}(x, y) = f\left( w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(x, y)) \right)$$

where $w _ {\mathrm{key}} > 0$ is a Dyadic regularization weight, adopting monotonic non-decreasing functions such as $f(u) = \tanh(u)$ or $f(u) = 1 - e^{-u}$.

#### Lemma 1.1 (Axiomatic Satisfaction of Pseudo-ultrametric Space)

##### [Claim]

The distance $d _ {\mathcal{V}}(x, y)$ defined above satisfies the axioms of a pseudo-ultrametric space (identity of indiscernibles/zero distance, non-negativity, symmetry, strong triangle inequality) for any $x, y, z \in \mathcal{M}$. Furthermore, for state points belonging to the same equivalence class $x \sim y$ (including $x=y$) where $l _ {\mathrm{lca}}(x, y) = l _ {\max}$ is determined, $d _ {\mathcal{V}}(x, y) = 0$, naturally corresponding to an ultrametric space on quotient space $\mathcal{M} / \sim$.

##### [Proof]

1. **Zero Distance Condition and Non-negativity:**

* For state points belonging to the same equivalence class $x \sim y$ (including $x = y$), the lowest common ancestor depth is $l _ {\mathrm{lca}}(x, y) = l _ {\max}$ by definition of the deepest subtree. Thus, the difference is $l _ {\max} - l _ {\mathrm{lca}}(x, y) = 0$, and since $f(0) = 0$, $d _ {\mathcal{V}}(x, y) = 0$ holds.
* For $x, y$ belonging to different equivalence classes, $l _ {\mathrm{lca}}(x, y) < l _ {\max}$ yields $w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(x, y)) > 0$. By the monotonic non-decreasing property of $f$ and $f(0) = 0$, $d _ {\mathcal{V}}(x, y) \ge 0$, assigning a positive distance to $x \not\sim y$.

2. **Symmetry:** From the symmetry of LCA depth definition $l _ {\mathrm{lca}}(x, y) = l _ {\mathrm{lca}}(y, x)$, $d _ {\mathcal{V}}(x, y) = d _ {\mathcal{V}}(y, x)$ holds.
3. **Strong Triangle Inequality:** From the geometric properties of Dyadic tree structures, LCA depth for any 3 points $x, y, z$ satisfies:

$$l _ {\mathrm{lca}}(x, z) \ge \min(l _ {\mathrm{lca}}(x, y), l _ {\mathrm{lca}}(y, z))$$

Multiplying both sides by $-1$ and adding $l _ {\max}$ yields:

$$(l _ {\max} - l _ {\mathrm{lca}}(x, z)) \le \max(l _ {\max} - l _ {\mathrm{lca}}(x, y), \, l _ {\max} - l _ {\mathrm{lca}}(y, z))$$

Here, since $f$ is a monotonically non-decreasing function, monotonic commutativity with respect to $\max$ operations holds for any $a, b \ge 0$: $f(\max(a, b)) = \max(f(a), f(b))$. Therefore, applying the monotonically non-decreasing function $f$ to both sides transforms the inequality as follows:

$$\begin{aligned}
f(w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(x, z))) &\le f(\max(w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(x, y)), w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(y, z)))) \\
&= \max(f(w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(x, y))), f(w _ {\mathrm{key}} \cdot (l _ {\max} - l _ {\mathrm{lca}}(y, z))))
\end{aligned}$$

Thus:

$$d _ {\mathcal{V}}(x, z) \le \max(d _ {\mathcal{V}}(x, y), d _ {\mathcal{V}}(y, z))$$

Because it satisfies the strong triangle inequality, the standard triangle inequality $d _ {\mathcal{V}}(x, z) \le d _ {\mathcal{V}}(x, y) + d _ {\mathcal{V}}(y, z)$ is automatically satisfied as well. $\blacksquare$

Dynamic resolution depth $D^* _ {\mathcal{V},\mathrm{continuous}}$ based on pseudo-ultrametric $d _ {\mathcal{V}}$ is converted by a hysteresis quantizer into discrete index $\bar{D} _ {\mathcal{V},\mathrm{quantized}}$ and held in the Quantized Depth Pointer Table (QDPT). QDPT immediately retrieves phase offsets corresponding to topological levels of each PagedBlock. Cross-referencing multiple pointers across hierarchical tree structures uses the Cascade Pointer Table (CPT), supplying topology modulation parameters from parent nodes to child nodes into the Hot Path in $\mathcal{O}(1)$ cascade lookups.

---

### 1.4 Stigmergic Local Rules and Global Attractor Convergence (Two-Layer Complementary Swarm-Attractor Structure)

As theoretical proof that this architecture avoids solving heavy matrix differential operations or numerical differential equations online during inference, we demonstrate that the ensemble dynamics (swarm) of local bitwise operations and pointer additions autonomously converge to global geometric attractor fields (Swarm-Attractor Duality).

The total phase angle vector is composed as the sum $\boldsymbol{\theta} = \boldsymbol{\theta} _ {\mathrm{base}} + \boldsymbol{\Delta\theta}$ of base position phase $\boldsymbol{\theta} _ {\mathrm{base}}$ and phase modulation offset $\boldsymbol{\Delta\theta}$, and potential field gradients are evaluated with respect to modulation offset $\boldsymbol{\Delta\theta}$.

#### Lemma 1.2 (Global Attractor Convergence of Local Update Rules)

##### [Claim]

The expected value of the average rate of change (drift vector) per 1 step (time step $\Delta t$) of local bit operations (Bitwise AND/OR, Atomic Pointer Increment) executed by each Attention Head (cellular swarm agent) $h \in \{1, \dots, H\}$ upon shared Sidecar Pointer Table (SPT) and Bitmask memory fields converges, under the mean-field limit ($H \to \infty$ swarm count and $B \to \infty$ bit-width), to a gradient flow accompanied by the global Tikhonov-regularized quasi-orthogonal projection operator $\mathbf{P} _ {\mathrm{im}(\delta _ 0)} = \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^*$:

$$\begin{aligned}
\lim _ {B \to \infty, H \to \infty} \frac{1}{\Delta t} \mathbb{E}\left[ \mathbf{p} _ {\mathrm{SPT}}^{(t + \Delta t)} - \mathbf{p} _ {\mathrm{SPT}}^{(t)} \right] &= - \mu _ {\mathrm{step}} \cdot \mathbf{P} _ {\mathrm{im}(\delta _ 0)} \nabla _ {\boldsymbol{\Delta\theta}} \tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta})) + \mathcal{O}\left( \frac{1}{\sqrt{B}} \right) \\
&= - \mu _ {\mathrm{step}} \cdot \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^* \nabla _ {\boldsymbol{\Delta\theta}} \tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta})) + \mathcal{O}\left( \frac{1}{\sqrt{B}} \right)
\end{aligned}$$

(where $\boldsymbol{\theta} = \boldsymbol{\theta} _ {\mathrm{base}} + \boldsymbol{\Delta\theta}$)

where $\mathbf{p} _ {\mathrm{SPT}}$ is the 1-cochain offset vector of SPT pointer values, isomorphic to discrete geometric elements on $C^1(K; \mathcal{F})$ via the scale conversion relationship with Q7-quantized phase offset vector $\boldsymbol{\Delta\theta}$ and Q7 fixed-point scale factor $\beta _ {\mathrm{Q7}} = 1/128$:

$$\boldsymbol{\Delta\theta} = \beta _ {\mathrm{Q7}} \cdot \mathbf{p} _ {\mathrm{SPT}}$$

The effective step gain $\mu _ {\mathrm{step}} > 0$ on the right-hand side represents an effective gain incorporating this Q7 scale factor $\beta _ {\mathrm{Q7}}$. $\tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$ is a continuous target attractor field on latent manifold parameterized by phase parameters $\boldsymbol{\theta}$, and $B$ is the bitmask width inside a PagedBlock.

##### [Proof]

Each cellular swarm agent $h$ writes to the shared memory field (Stigmergic Medium) via local Bitmask $\mathbf{M} _ h$ in VRAM and 1st-difference Q7 phase differential $\boldsymbol{\Delta\theta} _ h$. Define the average state of the swarm as $\bar{\mathbf{p}} = \frac{1}{H} \sum _ {h=1}^H \mathbf{p} _ h$.

Selective pheromone updating via local bitmask logical AND $\mathbf{M} _ h \text{ AND } \mathbf{M} _ {h'}$ is equivalent to evaluating the adjacency relation $A _ {ij} = \mathbf{M} _ i^T \mathbf{M} _ j$ of Graph Laplacian $\Delta _ 0$ as discrete logical operations. Under the mean-field limit of large swarm count $H \to \infty$ and block size $B \to \infty$, the variance of individual discrete bit noise shrinks into micro-fluctuations of $\mathcal{O}(1/\sqrt{B})$ via the Central Limit Theorem.

Directly corresponding to agent-based particle approximations of McKean-Vlasov processes / Markov processes in statistical mechanics, as discrete operations accumulated as stochastic micro-displacements of individual particles pile up through shared memory fields, the progression of the macroscopic mean-field follows Fokker-Planck equations, converging to steep descent trajectories of target continuous potential field $\tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$. Thus, expected drift velocity $\frac{1}{\Delta t} \mathbb{E}[\Delta \mathbf{p}]$ of asynchronous local pointers of agents on 1-cochain space matches the 1-cochain smoothed vector field obtained by applying regularized quasi-orthogonal projection operator $\mathbf{P} _ {\mathrm{im}(\delta _ 0)} = \delta _ 0 \mathbf{G} _ {\epsilon} \delta _ 0^*$ to steepest descent direction $-\nabla _ {\boldsymbol{\Delta\theta}} \tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$ of attractor field $\tilde{V}(\mathbf{\Omega}(\boldsymbol{\theta}))$. $\blacksquare$

##### [Engineering Significance]

By this lemma, there is no need to solve heavy pseudo-inverse calculations or tensor decompositions directly inside the Hot Path. Executing only $\mathcal{O}(1)$ Bitwise AND and pointer addition operations (stigmergic local rules) per token causes solution trajectories to autonomously converge to global target attractors (geometric optimal solutions) within an error bound of $\mathcal{O}(1/\sqrt{B})$. This constitutes the engineering foundation of the "Swarm-Attractor Geometric Approximation Hybrid Model via Local Swarm Operations."

---

### 1.5 Local State Connection and Galerkin Projection in Stalk Spaces

To suppress numerical fluctuations and noise amplification of higher-order differences under ultra-low quantization (1.5–2.5 bit) environments, local state connections in Stalk space are restricted to 1st-order inter-layer activation differences ($\mathbf{h} _ {\mathrm{deep}} - \mathbf{h} _ {\mathrm{mid}}$) and moving standardization (Z-score) of smoothed cosine similarities. This suppresses embedding discretization noise while reducing Galerkin projection error upper bounds from continuous spaces to discrete cell complexes, curbing attention division noise stemming from phase discontinuities.

---

### 1.6 Restricted Inline Phase Alignment (RIPA) and Proof of $SO(2)^{d _ {\mathrm{head}}/2}$ Commutativity

To extend the domain of phase modulation operators beyond attention scores into attention sensitivity (Attention Temperature), a composite modulation operator $\mathbf{M} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta}, \mathbf{D}) = \mathbf{D} \mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta})$ is introduced, combining phase rotation operator $\mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta}) \in SO(2)^{d _ {\mathrm{head}}/2}$ with diagonal scale (amplitude modulation) matrix $\mathbf{D} = \mathrm{diag}(s _ 1 \mathbf{I} _ 2, \dots, s _ {d _ {\mathrm{head}}/2} \mathbf{I} _ 2)$. Here, $\mathbf{D}$ is required to be **Isotropic Block-Diagonal Scaling**, giving identical scale within 2D sub-blocks.

To suppress distortion of high-frequency channels (dimensions maintaining fundamental context structure) in ultra-low quantization environments, the application region of phase rotation is restricted strictly to low-frequency channels of RoPE (upper $d _ {\mathrm{head}}/4$ dimensions), formulating **Restricted Inline Phase Alignment (RIPA, Soft-Bounded Phase Clamping)** incorporating $\tanh$ clamping:

$$\boldsymbol{\Delta\theta} _ {\mathrm{safe}} = \theta _ {\max} \cdot \tanh\left( \frac{\boldsymbol{\Delta\theta}}{\theta _ {\max}} \right) \quad (\text{Standard: } \theta _ {\max} = 5^\circ \approx 0.087 \text{ rad})$$

#### Theorem 1.1 (Commutativity in $SO(2)^{d _ {\mathrm{head}}/2}$)

##### [Claim]

Under 2D diagonal sub-block decomposition in attention space of Head dimension $d _ {\mathrm{head}}$ (even number), the phase rotation operator $\mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta} _ {\mathrm{safe}}) \in SO(2)^{d _ {\mathrm{head}}/2}$ by DPSR and the Rotational Position Embedding (RoPE) operator $\mathbf{R} _ {\mathrm{RoPE}}(m \boldsymbol{\Theta}) \in SO(2)^{d _ {\mathrm{head}}/2}$ are algebraically commutative (abelian group structure) for identical channel pairs. Furthermore, isotropic diagonal scale matrix $\mathbf{D} _ i = s _ i \mathbf{I} _ 2$ (Isotropic Block-Diagonal Scaling) within each 2D block is also commutative with this rotation operator.

##### [Proof]

$\mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta} _ {\mathrm{safe}})$ and $\mathbf{R} _ {\mathrm{RoPE}}(m \boldsymbol{\Theta})$ are both block-diagonalized as the direct sum of $d _ {\mathrm{head}}/2$ copies of 2D rotation matrix $\mathbf{R} _ 2(\theta _ i)$:

$$\mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta} _ {\mathrm{safe}}) = \bigoplus _ {i=1}^{d _ {\mathrm{head}}/2} \begin{pmatrix} \cos \Delta\theta _ {\mathrm{safe},i} & -\sin \Delta\theta _ {\mathrm{safe},i} \\ \sin \Delta\theta _ {\mathrm{safe},i} & \cos \Delta\theta _ {\mathrm{safe},i} \end{pmatrix}$$

$$\mathbf{R} _ {\mathrm{RoPE}}(m \boldsymbol{\Theta}) = \bigoplus _ {i=1}^{d _ {\mathrm{head}}/2} \begin{pmatrix} \cos(m \Theta _ i) & -\sin(m \Theta _ i) \\ \sin(m \Theta _ i) & \cos(m \Theta _ i) \end{pmatrix}$$

For any $i$, because the 2D rotation group $SO(2)$ is a commutative (abelian) group, the following holds:

$$\mathbf{R} _ 2(\Delta\theta _ {\mathrm{safe},i}) \mathbf{R} _ 2(m \Theta _ i) = \mathbf{R} _ 2(\Delta\theta _ {\mathrm{safe},i} + m \Theta _ i) = \mathbf{R} _ 2(m \Theta _ i) \mathbf{R} _ 2(\Delta\theta _ {\mathrm{safe},i})$$

Within each 2D block, isotropic scale matrix $\mathbf{D} _ i = s _ i \mathbf{I} _ 2$ is a scalar multiple, making it commutative with any 2D rotation matrix $\mathbf{R} _ 2$:

$$\mathbf{D} _ i \mathbf{R} _ 2 = (s _ i \mathbf{I} _ 2) \mathbf{R} _ 2 = s _ i \mathbf{R} _ 2 = \mathbf{R} _ 2 (s _ i \mathbf{I} _ 2) = \mathbf{R} _ 2 \mathbf{D} _ i$$

Since commutativity holds in each block of the direct sum structure, in full space:

$$\mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta} _ {\mathrm{safe}}) \mathbf{R} _ {\mathrm{RoPE}}(m \boldsymbol{\Theta}) = \mathbf{R} _ {\mathrm{RoPE}}(m \boldsymbol{\Theta}) \mathbf{R} _ {\mathrm{DPSR}}(\boldsymbol{\Delta\theta} _ {\mathrm{safe}})$$

That is, the commutator $[\mathbf{R} _ {\mathrm{RoPE}}, \mathbf{R} _ {\mathrm{DPSR}}] = \mathbf{0}$ holds. $\blacksquare$

This commutativity allows additive application of DPSR phase modulation and amplitude scaling in arbitrary order without damaging positional encoding representations already applied by RoPE.

---

## Chapter 2: Runtime Architecture and In-Kernel Low-Layer Control System

### 2.1 Definition of Swarm Cells and Stigmergic Environmental Fields

To engineer swarm intelligence and spatial algebraic control at low layers, components are defined as follows:

* **Swarm Agents:** Individual Attention Heads inside Multi-Head Attention, or Thread-Warp units processing PagedBlocks (16–32 tokens), are defined as autonomous swarm cells.
* **Stigmergic Medium:** Sidecar Pointer Table (SPT), Cascade Pointer Table (CPT), and Paged Block Bitmask placed in VRAM/DRAM are defined as the "environmental field."
* **Stigmergic Interaction:** Agents (Heads) do not communicate directly with each other for synchronization; instead, they leave phase differences (Q7 fixed integers) or Bitmasks in shared pointers, which subsequent agents immediately reference ($\mathcal{O}(1)$) to apply phase rotation.

---

### 2.2 Per-Token Decoding Processing (Hot Path) and Dynamic Phase-Shifted RoPE (DPSR) Inline Fusion

To resolve the problem where future dynamic phase transitions (EDBC) trigger rewriting the entire past KV cache ($\mathcal{O}(N)$ VRAM stalls) during Key vector writes at decode time, this framework introduces **Lazy Phase Unwrapping (Query-Only Relative Phase Shift)**.

Instead of attaching DPSR as an external standalone tensor transformation kernel, this approach adopts a **Fused DPSR Kernel** structure embedded directly into rotation angle calculation registers inside existing RoPE kernels:

$$\theta _ {m, i}' = m \cdot \Theta _ i + \Delta\theta _ i$$

By directly adding $\Delta\theta _ i$ on registers against rotation angle $\theta _ {m,i}$ inside CUDA/Metal kernels such as FlashAttention or PagedAttention, computation is completed via a single addition instruction $\mathcal{O}(1)$ without incurring extra VRAM read/write operations.

#### Table 2.1: Hot Path Inline Execution Sequence Specification

| Step | Processing Summary | Computational Complexity | Applied Engineering Element | Interface Function |
| --- | --- | --- | --- | --- |
| 1. Phase Lookup | Read phase offset $\boldsymbol{\Delta\theta}$ from Sidecar Pointer Table (SPT) | $\mathcal{O}(1)$ | Tagged Pointer Lookup | Phase lookup initialization |
| 2. RIPA Clamp | Restriction to low-frequency channels ($d _ {\mathrm{head}}/4$) and $\tanh$ angle control | $\mathcal{O}(d _ {\mathrm{head}})$ | Single-Instruction Math Clamp | Soft-bounded phase steering |
| 3. Inline Fusion | Addition onto RoPE rotation angle registers ($\theta _ {m,i} + \Delta\theta _ i$) & Lazy Phase Unwrapping | $\mathcal{O}(d _ {\mathrm{head}})$ | Fused DPSR Kernel, Register Shuffle | Relative phase shift override |
| 4. KV Cache | Write and reference phase-independent standard Key | $\mathcal{O}(1)$ | Standard Paged KV Storage | Norm-guarded Slerp stabilization |
| 5. Attention | MMA (Matrix Multiply-Accumulate) integrated kernel processing & Softmax evaluation | $\mathcal{O}(d _ {\mathrm{head}})$ | Warp-Cooperative Block Bitmasking | Fused Logit phase shift |

---

### 2.3 Lazy Phase Unwrapping (Query-Only Relative Phase Shift) and VRAM Bandwidth Optimization

To avoid $\mathcal{O}(N)$ memory stalls that rewrite all KV caches in VRAM when context changes or phase transitions occur, phase-independent standard Key vectors (states with standard RoPE applied) are retained in the KV cache.

In attention kernels during per-token decoding, an effective composite phase $\boldsymbol{\Delta\theta} _ {\mathrm{query\ _ eff}}$ containing cumulative context phase differences is batch-multiplied and evaluated in dot products strictly on the Query side:

$$\boldsymbol{\Delta\theta} _ {\mathrm{query\ _ eff}} = \boldsymbol{\Delta\theta} _ {\mathrm{query}} - \boldsymbol{\theta} _ {\mathrm{context}}$$

where $\boldsymbol{\theta} _ {\mathrm{context}}$ is the global context phase accumulator updated based on circular statistics described later. By the relativity of rotations in inner products ($\langle \mathbf{R}(\boldsymbol{\theta} _ q) \mathbf{q}, \mathbf{R}(\boldsymbol{\theta} _ k) \mathbf{k} \rangle = \langle \mathbf{R}(\boldsymbol{\theta} _ q - \boldsymbol{\theta} _ k) \mathbf{q}, \mathbf{k} \rangle$), computational cost is concentrated into a single rotation instruction on the Query side without re-rotating Key caches.

---

### 2.4 Phase-Steered Parallel Multi-head Ensemble (PSPM) and Phase-Gated Head-Wise Subspace Ensemble

Unlike traditional MoE structures that dynamically swap multi-GB parameters, this framework formulates a **Phase-Gated Head-Wise Subspace Ensemble (Parameter-Efficient Sub-Head MoE)** structure, keeping a single model (2 GB) resident in VRAM while assigning unique "micro phase profiles (a few KB)" and SRL vectors to each Attention Head (or Head subgroup) as independent experts.

From token activation $\mathbf{x}$, a lightweight router dynamically controls the phase and gain of each Head or Sub-Head:

$$g _ e(\mathbf{x}) = \mathrm{SoftmaxTopK}\left( W _ r \mathbf{x} \right) _ e$$

Each phase expert $e$ consists of a unique phase modulation vector $\boldsymbol{\Delta\theta} _ e$, continuous attention gain $s _ h \in [0, 1]$, and SRL outer product vectors $(\mathbf{u} _ e, \mathbf{v} _ e)$.

Furthermore, it introduces the **Phase-Steered Parallel Multi-head Ensemble (PSPM, Single-Pass Head-Wise Ensemble)** execution architecture, switching attention Head activation circuits based on token entropy and context attributes. Within a single Forward pass, Attention Head groups are split and applied into functional sub-networks:

* **Group A (Fact Sub-Heads):** Apply knowledge-matching phase profile $\boldsymbol{\Delta\theta} _ {\mathrm{fact}}$ + strict angle clamping (RIPA)
* **Group B (Logic Sub-Heads):** Apply logical-reasoning phase profile $\boldsymbol{\Delta\theta} _ {\mathrm{logic}}$ + standard phases
* **Group C (Explore Sub-Heads):** Apply exploration/hypothesis-generation phase profile $\boldsymbol{\Delta\theta} _ {\mathrm{explore}}$ + stochastic pulses

At the Logit generation stage immediately preceding the LM Head, these output representations are ensemble-synthesized (Single-Pass Logit Ensembling). While remaining a single base model, it transparently switches diverse specialized circuits, boosting expressiveness without adding VRAM.

---

### 2.5 Entropy-Gated Phase Lookup (EG-LUT / EGDPL) and Speculative Phase Prefetching

To avoid synchronous stalls from Sheaf/Hodge computations in the Cold Path, static lookup **Entropy-Gated Phase Lookup Table (EG-LUT)** and dynamic prefetching/routing **Entropy-Gated Dynamic Phase Lookup (EGDPL)** are explicitly distinguished to optimize execution:

* **EG-LUT (Static Lookup):** Responsible for directly referencing pre-defined AOT phase base tables.
* **EGDPL (Dynamic Steering):** Responsible for in-kernel dynamic phase synthesis and pre-loading phase values for subsequent tokens.

#### Table 2.2: EG-LUT / EGDPL Control Processing Flow Specification

| Phase | Execution Processing | Applied Component / Output |
| --- | --- | --- |
| **Phase 1: Evaluation** | Evaluate entropy flow from activation state at decode time | Softmax Attention Layer / $\tilde{H}$ metric |
| **Phase 2: Lookup** | Directly lookup pre-defined phase base tables (AOT LUT) (EG-LUT) | EG-LUT Controller / Scale factor $\gamma(\tilde{H})$ |
| **Phase 3: Synthesis** | Dynamic phase prefetching/routing with RIPA angle clamp (EGDPL) | EGDPL Module / $\boldsymbol{\Delta\theta} = \gamma(\tilde{H}) \cdot \mathbf{v} _ {\mathrm{active}}$ |
| **Phase 4: Steering** | Direct steering to RoPE kernel registers | Hot Path Attention Kernel |

Against active phase bases $\mathbf{v} _ {\mathrm{active}}$ resident in VRAM, scale factor $\gamma(\tilde{H})$ computed in-kernel immediately after Softmax is applied via lookup. This reduces CPU-GPU communication overhead.

Furthermore, to hide EG-LUT lookup latency, **Speculative Bit-Level Phase Prefetching** is used concurrently, using the bit pattern (Bitmask) of quantized KV caches of the recent $k$ tokens as a hash key to speculatively pre-load the next phase change $\boldsymbol{\Delta\theta} _ {\mathrm{next}}$ onto GPU L2 cache 1 step in advance.

---

### 2.6 Dynamic Layer-Skipping with Phase Compensation

To further increase computational efficiency during lightweight inputs such as boilerplate text or low-entropy states, **Dynamic Layer-Skipping** is introduced to bypass forward pass computations of ultra-low quantized intermediate layers (e.g., layers 8–20).

Non-linear expressive capacity lost to skipping is approximated and substituted solely via Query phase rotation $\boldsymbol{\Delta\theta} _ {\mathrm{comp}}$ and SRL (Dynamic Rank-1 correction) of subsequent layers. This configuration boosts inference efficiency while reducing memory bandwidth consumption.

---

### 2.7 Hierarchical Local Topology Coordinator

To bind resource constraints across Heterogeneous Memory Architectures, a **Hierarchical Local Topology Coordinator (Cellular Layer-Hub)** is deployed.

#### Table 2.3: Hierarchical Topology Layer Specification

| Topology Hierarchy | Control Unit | Primary Responsibilities & Control Method |
| --- | --- | --- |
| **Macro Attractor Level** | Global Concept / Persona | Holds long-term context anchors and global phase profiles (`.gcso` SNAPSHOT track) |
| **Mezzo Cellular Layer-Hub** | Transformer Block Groups | Inter-layer topology coordination and batch phase synchronization via Skip-Hop Bus |
| **Micro Activation Chunks** | PagedBlock Units | Distributed local geometric modulation and fast in-kernel processing every 16–32 tokens |

Using thread-local accumulators aligned to 128-byte boundaries reduces Compare-And-Swap (CAS) atomic instruction contention during multi-thread batch reductions, maintaining a batch commit protocol (**Buffered Bit-Tree Reduction**) every 16–32 tokens.

---

## Chapter 3: Phase Transition Detection, Phase Perturbation Processing, and Geometric Autonomous Control

### 3.1 Moving Z-Score Normalized Attention Entropy ($\tilde{H}$) and Singularity Avoidance/Stabilization

In 1.5–2.5 bit ultra-low quantization environments, directly computing activation differences $\Vert \mathbf{h} _ {\mathrm{mid}} - \mathbf{h} _ {\mathrm{deep}} \Vert _ 2$ between intermediate and deep layers causes rounding errors from quantization to superimpose onto moving components, which can cause entropy metrics to remain constantly high. Furthermore, logarithmic singularity risks ($\ln(0)$) in surrogate entropy computations of attention must be avoided.

This framework introduces a logarithmic singularity prevention clamp $\epsilon _ {\mathrm{log}} = 10^{-12}$ and uses a 2-state distribution representation locally normalized by $P _ {\max} + P _ {\mathrm{2nd}}$:

$$\tilde{P} _ {\max} = \frac{P _ {\max}}{P _ {\max} + P _ {\mathrm{2nd}}}, \quad \tilde{P} _ {\mathrm{2nd}} = \frac{P _ {\mathrm{2nd}}}{P _ {\max} + P _ {\mathrm{2nd}}}$$

to define surrogate entropy as follows:

$$H _ {\mathrm{sparse-surrogate}} = - \tilde{P} _ {\max} \ln(\tilde{P} _ {\max} + \epsilon _ {\mathrm{log}}) - \tilde{P} _ {\mathrm{2nd}} \ln(\tilde{P} _ {\mathrm{2nd}} + \epsilon _ {\mathrm{log}})$$

This formula $H _ {\mathrm{sparse-surrogate}}$ is a normalized surrogate metric detecting boundaries between certainty and uncertainty in $\mathcal{O}(1)$, bounded to value range $[0, \ln 2]$ (maximum value $\ln 2 \approx 0.693$).

Furthermore, using LayerNorm-applied same-dimension activations $\mathbf{h} _ {\mathrm{mid,srl}}, \mathbf{h} _ {\mathrm{deep,srl}}$ (immediately after SRL correction), a division-by-zero clamp $\sigma _ {\min}$ and smooth bounded function $\tanh$ are introduced to formulate **Moving Z-Score Normalized Attention Entropy ($\tilde{H}$)** as follows:

$$\begin{aligned}
Z _ {\mathrm{dist}} &= \frac{d _ {\mathrm{cosine}}(\mathbf{h} _ {\mathrm{mid,srl}}, \mathbf{h} _ {\mathrm{deep,srl}}) - \mu _ {\mathrm{dist}}}{\max(\sigma _ {\mathrm{dist}}, \sigma _ {\min})} \\
\hat{d} _ {\mathrm{cosine}} &= \tanh\left( \alpha _ z \cdot \mathrm{ReLU}(Z _ {\mathrm{dist}}) \right) \\
\tilde{H} &= H _ {\mathrm{sparse-surrogate}} \cdot \left( 1 + \gamma _ {\mathrm{mod}} \cdot \hat{d} _ {\mathrm{cosine}} \right)
\end{aligned}$$

$$d _ {\mathrm{cosine}}(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a}^T \mathbf{b}}{\Vert \mathbf{a} \Vert _ 2 \Vert \mathbf{b} \Vert _ 2}$$

where $\mu _ {\mathrm{dist}}, \sigma _ {\mathrm{dist}}$ are moving mean and moving standard deviation of cosine distance across the recent $W _ {\mathrm{norm}}$ tokens, $\sigma _ {\min} > 0$ is a lower-bound threshold for numerical stabilization in flat context, and $\gamma _ {\mathrm{mod}} > 0$ is a weighted modulation coefficient for geometric variance. Combining the correlated elevation $\hat{d} _ {\mathrm{cosine}}$ of inter-layer distance as a weighted multiplicative term $(1 + \gamma _ {\mathrm{mod}} \cdot \hat{d} _ {\mathrm{cosine}})$ against base attention diffusion $H _ {\mathrm{sparse-surrogate}}$ prevents basic information quantity of entropy from collapsing to 0 during normal states ($Z _ {\mathrm{dist}} \le 0$), while autonomously amplifying sensitivity evaluation strictly during geometric variance.

---

### 3.2 Sliding-Window Entropy Rate Integrator

To suppress chattering phenomena caused by overreaction to transient single-token high-entropy states, a **Sliding-Window Entropy Rate Integrator** is deployed.

Moving-window entropy flux $\Phi _ M(t)$ corresponding to continuous-time differential integration over token step differences $\tilde{H}(t) - \tilde{H}(t-M)$ across recent $M$ tokens (standard $M = 3 \sim 5$) is defined and evaluated as follows:

$$\Phi _ M(t) = \int _ {t-M}^{t} \frac{d \tilde{H}}{d\tau} d\tau = \tilde{H}(t) - \tilde{H}(t-M)$$

Evaluation applies a state-holding structure incorporating a hysteresis band $\delta _ {\mathrm{hyst}} > 0$, recognizing sustained high-entropy surges ($\Phi _ M(t) > \tau _ {\mathrm{flux}} + \delta _ {\mathrm{hyst}}$) strictly as phase transition critical points to reduce false triggers from transient noise.

---

### 3.3 Phase Perturbation Triggering Metastable Transitions and Phase-Conjugate Attractor Repulsion (Local Minima Synergy in the Same Field)

In regions identified as phase transition critical points, the **Entropy-Driven Decoding Branch Controller (EDBC)** executes dual-branch routing control based on latent manifold alignment score $\mathcal{C} _ {\mathrm{void}}$ (Coherent Vector Alignment Metric for Out-of-Distribution Latent Space) and layer drop $\Delta H _ {\mathrm{layer}}$.

In this framework, the cellular layer (micro) and attractor field (macro) are interpreted as **views at different resolutions within the same phase space $\mathcal{M}$**. Thus, non-factual generation (hallucination) is formulated not as external noise to be suppressed from the system, but as "inappropriate local minima (Spurious Local Minima / Off-manifold Attractors) on the same phase field" existing in the complementary space $\mathcal{M} \setminus \mathcal{X}$ of the observable domain.

#### Dual-Branch Control Procedure and Local Solution Repulsion Synergy

1. **Entropy Flux Signal Evaluation ( $\Phi _ M(t)$ ):**

EDBC evaluates phase transition critical points.

2. **Exploration Path ($\mathcal{C} _ {\mathrm{void}} \ge \tau _ {\mathrm{eff}}$):**

Triggers **Potential-Driven Sampling / Energy-Guided Decoding**. When detecting highly aligned exploratory tendencies, it injects phase perturbations (**Metastable Transition Phase Perturbation**) $\boldsymbol{\Delta\theta} _ {\mathrm{tunnelling}}$ orthogonal to Eyring-Kramers potential barriers to induce transitions out of metastable states. This shifts the model out of local solutions, transitioning to new conceptual solutions (bifurcated latent attractors) on high-dimensional phase space.

3. **Dissipation and Erroneous Local Solution Control Path ($\mathcal{C} _ {\mathrm{void}} < \tau _ {\mathrm{eff}}$):**

Evaluates local closedness ($\delta _ 1 \mathbf{r} \approx \mathbf{0}$) and global non-boundaries ($\mathbf{r} \notin \mathrm{im}(\delta _ 0)$) of 1-cochain residual $\mathbf{r}$. Upon detecting cohomological obstruction class $[\mathbf{r}] \in H^1(K; \mathcal{F}) \neq \{\mathbf{0}\}$, it determines that local hypothesis branches explored by the micro layer have been pulled into inappropriate local solutions (Spurious Local Minima) in complementary space. At this point, rather than simply discarding the branch, the system triggers **Phase-Conjugate Attractor Repulsion**.
It applies anti-phase pulses $-\boldsymbol{\Delta\theta} _ {\mathrm{hallucination}}$ to the corresponding Head's phase, writing that phase pattern into a Counting Bloom Filter (**Contrastive Phase Memory / Dynamic Negative Constraint Memory**) (stigmergic environmental rewriting). Precisely because it is a local solution on the same phase field, writing repulsive phases into the environment flips the attractor's energy "valley" into a "peak (repulsive potential field)," allowing the model to autonomously and smoothly correct trajectories to true target attractors without altering weights (synergy with stigmergy).

To mitigate output collapse risks from excessive anti-phase rotation, an **Entropy-Regularized Attention Steering** mechanism is applied, accompanied by amplitude scale $\mathbf{D}$ attenuation and Softmax Temperature correction ($\tau _ {\mathrm{attn}} \uparrow$) on corresponding Heads. Highly coherent exploratory thought trajectories are protected by **Bifurcated Latent Attractors**, accumulating into a **Dynamic Phase Modulation Profile** via **Persistent Topological Mode Quantization**, thereby achieving commutative personality switching without rewriting model weights.

---

### 3.4 Pitchfork Bifurcation Model and Scale Invariance in Langevin Dynamics

To mathematically describe state transition dynamics in decoding branch control, overdamped Langevin dynamics (Overdamped Langevin Dynamics) against a scalar order parameter $x \in \mathbb{R}$ (state displacement) on the primary bifurcation axis are introduced.

#### Lemma 3.1 (Pitchfork Bifurcation Model in Decoding Dynamics Based on Langevin Equation)

##### [Claim]

The time evolution of order parameter $x \in \mathbb{R}$ is described by the following overdamped Langevin equation:

$$dx = \left( \mu x - \beta x^3 \right) dt + \sqrt{2D} \, dW _ t$$

where $\beta > 0$ is a stabilizing non-linear saturation constant, and $dW _ t$ is a standard Wiener process (Gaussian white noise). Phase transition bifurcation parameter $\mu$ and effective noise intensity $D$ are normalized and formulated based on latent manifold alignment score $\mathcal{C} _ {\mathrm{void}}$, threshold $\tau _ {\mathrm{eff}}$, and Attention Head dimension $d _ {\mathrm{head}}$ as follows:

$$\mu = \frac{\alpha (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})}{\sqrt{d _ {\mathrm{head}}}}, \quad D = \frac{D _ 0}{d _ {\mathrm{head}}}$$

Equivalent to $\frac{1}{\sqrt{d _ {\mathrm{head}}}}$ scaling in Transformer Scaled Dot-Product Attention, this normalized formulation keeps bifurcation thresholds and stochastic phase perturbation sensitivity model-scale invariant across variations in Head dimension.

1. **Convergence Mode ($\mu < 0 \iff \mathcal{C} _ {\mathrm{void}} < \tau _ {\mathrm{eff}}$):**

Exhibits a supercritical pitchfork bifurcation phase where the origin $x^* = 0$ is the sole strongly stable fixed point (potential valley), autonomously correcting the system to a unique deterministic optimal solution.

2. **Exploration Mode ($\mu > 0 \iff \mathcal{C} _ {\mathrm{void}} > \tau _ {\mathrm{eff}}$):**

The origin $x^* = 0$ destabilizes, yielding two symmetric stable fixed points $x^* _ {\pm} = \pm \sqrt{\mu / \beta}$ (pitchfork bifurcation). Against potential barrier $\Delta V = \frac{\mu^2}{4\beta} = \frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4\beta d _ {\mathrm{head}}}$, tunneling probability follows the Eyring-Kramers Mean Exit Time Formula:

$$\tau _ {\mathrm{escape}} \propto \exp\left( \frac{\Delta V}{D} \right) = \exp\left( \frac{\frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4\beta d _ {\mathrm{head}}}}{\frac{D _ 0}{d _ {\mathrm{head}}}} \right) = \exp\left( \frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4 \beta D _ 0} \right)$$

driving escape from metastable states to achieve smooth phase transitions into bifurcated high-dimensional conceptual solutions (thought branches).

##### [Proof]

The potential function $V(x)$ corresponding to order parameter $x \in \mathbb{R}$ is given by integrating the drift term $-\frac{\partial V}{\partial x} = \mu x - \beta x^3$ of the overdamped Langevin equation as follows:

$$V(x) = -\frac{\mu}{2} x^2 + \frac{\beta}{4} x^4$$

1. When $\mu < 0$, extremum condition $\frac{\partial V}{\partial x} = 0 \implies x( -\mu + \beta x^2 ) = 0$ yields $x^* = 0$ as the sole real solution. Since $\frac{\partial^2 V}{\partial x^2}(0) = -\mu > 0$, the origin constitutes the sole strongly stable minimum (single attractor).
2. When $\mu > 0$, origin $x^* = 0$ destabilizes as $\frac{\partial^2 V}{\partial x^2}(0) = -\mu < 0$. Instead, two stable minimum points $x^* _ {\pm} = \pm \sqrt{\frac{\mu}{\beta}}$ arise, with potential valley depth $V(x^* _ {\pm}) = -\frac{\mu^2}{4\beta}$. The potential barrier height between the origin and stable minima is $\Delta V = V(0) - V(x^* _ {\pm}) = \frac{\mu^2}{4\beta}$.
3. Substituting parameter $\mu = \frac{\alpha (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})}{\sqrt{d _ {\mathrm{head}}}}$ into potential barrier height $\Delta V$ yields:

$$\Delta V = \frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4 \beta d _ {\mathrm{head}}}$$

Calculating dimensionless ratio $\frac{\Delta V}{D}$ in the Eyring-Kramers formula:

$$\frac{\Delta V}{D} = \frac{\frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4 \beta d _ {\mathrm{head}}}}{\frac{D _ 0}{d _ {\mathrm{head}}}} = \frac{\alpha^2 (\mathcal{C} _ {\mathrm{void}} - \tau _ {\mathrm{eff}})^2}{4 \beta D _ 0}$$

The denominator and numerator $d _ {\mathrm{head}}$ cancel out, making the ratio $\frac{\Delta V}{D}$ a constant independent of Attention Head dimension $d _ {\mathrm{head}}$. Consequently, escape velocity and stochastic tunneling probability of phase transitions remain invariant regardless of model Head scale. $\blacksquare$

---

## Chapter 4: Long-Context Preservation and Phase Residual KV Pruning Structure

### 4.1 Decoupled Fact-Semantic Memory Architecture

To tackle attention fixation/saturation and capacity pressure during long-context processing, this framework deploys a **Decoupled Fact-Semantic Memory Architecture** that splits memory retention into two tracks:

* **Fact Anchor Track ($K _ {\mathrm{fact}}$):** Holds quantitative information (anchors) such as proper nouns, numerical values, and code syntax. Via Entropy-Gated Key Anchor Preservation, it protects tokens with high attention concentration and low entropy, or Key regions with high norm/importance.
* **Semantic Phase Track ($\boldsymbol{\theta} _ {\mathrm{semantic}}$):** Holds topic shifts and qualitative semantic structures of context as geometric phase transitions. Via phase residual aggregated KV pruning described below, instead of discarding individual token KV caches, it cumulatively preserves their phase residuals geometrically.

---

### 4.2 Phase Residual Aggregated KV Pruning, Circular Statistics, and von Mises Concentration $\kappa _ c$

To prevent VRAM depletion, when pruning low-contribution KV cache items, simple deletion causes topological corrective forces previously exerted by past context upon attention space to dissipate.

This framework applies circular statistics to phase vectors of pruned KV token groups, calculating and compensating their composite direction and dissipation concentration.

#### Calculation of Circular Statistics and von Mises Concentration

Let the pruned KV token set be $J$, weight of each token be $w _ j$, and phase angle be $\theta _ j$. Composite orthogonal components $\bar{C}, \bar{S}$, mean resultant vector length $\bar{R} _ c \in [0, 1]$, and mean resultant phase angle $\bar{\theta} _ c$ are computed as follows:

$$\begin{aligned} \bar{C} &= \sum _ {j \in J} w _ j \cos \theta _ j, \quad \bar{S} = \sum _ {j \in J} w _ j \sin \theta _ j \\ \bar{R} _ c &= \frac{\sqrt{\bar{C}^2 + \bar{S}^2}}{\sum _ {j \in J} w _ j}, \quad \bar{\theta} _ c = \mathrm{atan2}(\bar{S}, \bar{C}) \end{aligned}$$

von Mises concentration parameter $\kappa _ c$ on circular distributions is evaluated according to mean resultant vector length $\bar{R} _ c$ via piecewise approximations (based on Best & Fisher, 1979 / Banerjee et al., 2005):

$$\kappa _ c \approx \begin{cases} 2 \bar{R} _ c + \bar{R} _ c^3 + \frac{5}{6} \bar{R} _ c^5 & (\bar{R} _ c < 0.53) \\ \frac{1}{1.28(1 - \bar{R} _ c^2)} & (0.53 \le \bar{R} _ c < 0.85) \\ \frac{1}{\bar{R} _ c (1 - \bar{R} _ c)(3 - \bar{R} _ c)} & (\bar{R} _ c \ge 0.85) \end{cases}$$

To suppress chattering phenomena arising from gradient discontinuities (step jumps) near boundary points $\bar{R} _ c = 0.53$ and $\bar{R} _ c = 0.85$ during numerical computation, smooth sigmoidal blending is applied near boundaries:

$$\sigma _ {\mathrm{blend}}(x; x _ 0, \delta) = \frac{1}{1 + e^{-(x - x _ 0)/\delta}}$$

(where $x _ 0 \in \{0.53, 0.85\}$, smoothing parameter $\delta = 0.01 \sim 0.05$), ensuring continuous, smooth concentration evaluation.

Topological corrective forces previously exerted by discarded KV caches are added and updated into the global context phase accumulator $\boldsymbol{\theta} _ {\mathrm{context}}$ as phase residuals multiplied by composite phase angle $\bar{\theta} _ c$ and concentration $\kappa _ c$. This maintains topological shadows of past context to preserve long-range tracking capabilities while significantly compressing physical KV cache sizes.

---

### 4.3 Predictive Phase-Motion & Residual Compensation (PPRC) and `.gcso` Format Structure

To achieve persistence and instantaneous seeking of conversation history and long-range KV caches, the **Predictive Phase-Motion & Residual Compensation (PPRC / Temporal Key-Frame KV Cache Compression)** scheme is formulated.

Extending concepts from Video Codec Architecture Group of Pictures (GOP) structures, KV cache sequences are structured into Variable GOP Structures:

* **I-Cache (Intra Key-Frame Cache):** Uncompressed KV cache anchors placed at entropy surge points or conceptual transition points.
* **P-Cache (Predictive Motion Cache):** Per-token phase motion vectors $\boldsymbol{\Delta\theta} _ t$, scalar gains $s _ t$, and micro residual vectors $\mathbf{e} _ t$ between I-Caches (denoted as $\mathbf{e} _ t$ to distinguish from 1-cochain residual $\mathbf{r}$).

#### Table 4.1: `.gcso` Unified Binary Container Format (`.gcso` Unified Container Format) Specification

| Track Name | Data Structure and Format Specification | Functions and Purpose |
| --- | --- | --- |
| `CORE` | Asymmetric Layer-wise Quantized Model Weights | Fixed parameter group of resident base model |
| `I-CACHE` | FP8/INT4 Retained Anchor Key/Value | Reference baseline Key cache at fixed keyframes |
| `P-CACHE` | Q7 $\boldsymbol{\Delta\theta} _ t$ + INT4 Scalar Gain $s _ t$ + Residual $\mathbf{e} _ t$ | Inter-token phase motion vectors and micro-difference corrections |
| `SNAPSHOT` | Global Phase Accumulator $\boldsymbol{\theta} _ {\mathrm{context}}$ & Persistent Phase Profiles | Context phases and persona configurations |

During restoration, without forward pass model computations, adding P-Cache phase motion vectors onto I-Cache standard Keys achieves **Zero-Forward Latency Seek (Instantaneous Replay)** to arbitrary token positions. Accumulated phase errors are reduced and absorbed into Multi-Head Virtual Position Shifts ($\Delta m^{(h)}$).

---

## Chapter 5: High-Dimensional Context Projection and External Attractor Field Structure

### 5.1 Product of Lie Group Rotations in Hierarchical Topic Phase Trees

To project external context supplied from natural language prompts or RAG systems into latent space, a **Product of Lie Group Rotations** is formulated, sequentially applying Lie group $SO(2)^{d _ {\mathrm{head}}/2}$ rotation operators $\mathbf{R} _ l(\boldsymbol{\Delta\theta} _ l)$ to each node $l$ of a text's hierarchical topic tree:

$$\mathbf{R} _ {\mathrm{total}} = \prod _ {l \in \mathrm{Path}} \mathbf{R} _ l(\boldsymbol{\Delta\theta} _ l)$$

Batch-synthesizing phase offsets corresponding to each level of the hierarchical structure (domain, section, paragraph) as product operations achieves phase steering while algebraically preserving context inclusion relationships.

---

### 5.2 Dynamic Rank-1 Structure Sparse Residual Adapter Layer (SRL) and LoRA-to-Phase SVD Converter (L2P-SVD)

To dynamically compensate inter-layer information transformation capabilities that are difficult via attention score modulation alone, a **Sparse Residual Adapter Layer (SRL)** operating with minimal parameters is deployed for intermediate FFN and Attention projection layers. This adapter features a dynamic Dynamic Rank-1 structure.

In the SRL correction formula:

$$\mathbf{y} _ {\mathrm{srl}} = W _ {\mathrm{base}} \mathbf{x} + \mathbf{s} \odot \left( \mathbf{u} (\mathbf{v}^T \mathbf{x}) \right)$$

dimensional formulations matching target FFN / Attention projection layer transformation directions are defined as follows:

1. **For FFN Down-Projection ($W _ {\mathrm{down}} \in \mathbb{R}^{d _ {\mathrm{model}} \times d _ {\mathrm{ffn}}}$) or Attention Out-Projection ($W _ {\mathrm{out}} \in \mathbb{R}^{d _ {\mathrm{model}} \times d _ {\mathrm{model}}}$):**

Input activation is $\mathbf{x} \in \mathbb{R}^{d _ {\mathrm{ffn}}}$ ($\mathbb{R}^{d _ {\mathrm{model}}}$ for Out-Projection), and vector dimensions are $\mathbf{v} \in \mathbb{R}^{d _ {\mathrm{ffn}}}, \mathbf{u} \in \mathbb{R}^{d _ {\mathrm{model}}}, \mathbf{s} \in \mathbb{R}^{d _ {\mathrm{model}}}$.

2. **For FFN Up-Projection / Gate-Projection ($W _ {\mathrm{up}}, W _ {\mathrm{gate}} \in \mathbb{R}^{d _ {\mathrm{ffn}} \times d _ {\mathrm{model}}}$):**

Input activation is $\mathbf{x} \in \mathbb{R}^{d _ {\mathrm{model}}}$, and vector dimensions flip to $\mathbf{v} \in \mathbb{R}^{d _ {\mathrm{model}}}, \mathbf{u} \in \mathbb{R}^{d _ {\mathrm{ffn}}}, \mathbf{s} \in \mathbb{R}^{d _ {\mathrm{ffn}}}$.

When pre-trained LoRA adapter weights $W _ A \in \mathbb{R}^{r \times d _ {\mathrm{in}}}, W _ B \in \mathbb{R}^{d _ {\mathrm{out}} \times r}$ (rank $r$) exist, a **Training-free SVD LoRA-to-Phase Converter (LoRA-to-Phase SVD Projection: L2P-SVD)** projects them into phase profiles and SRL vectors via first-order SVD decomposition without additional training:

$$W _ B W _ A \approx U \Sigma V^T \implies \mathbf{u} = \sqrt{\sigma _ 1} \mathbf{u} _ 1, \quad \mathbf{v} = \sqrt{\sigma _ 1} \mathbf{v} _ 1$$

Despite data sizes of a few KB to a few MB, it achieves expressiveness corrections algebraically equivalent to dynamic LoRA.

---

### 5.3 Storage Cost Offloading / External Attractor Fields

External natural language prompts, system instructions, and RAG retrieval contexts are defined as an **"External Attractor Field"** that deforms latent phase fields without rewriting model weights (VRAM consumption).

**Storage Cost Offloading** is executed, shifting context retention and knowledge structure storage locations from VRAM to high-speed NVMe SSDs or DRAM `.gcso` containers. This allows transparent referencing and driving of external knowledge attractor fields even under physical VRAM limits of 2–4 GB.

---

### 5.4 Multimodal Extension: Visual Attention Steering

The phase modulation mechanism of this architecture applies not only to language tokens, but also to visual patch tokens of images and videos.

Against visual patch encodings holding 2D/3D spatial position information, **Visual Attention Steering** is deployed to dynamically steer spatial attention focus areas via product structures of 2D rotation group $SO(2)$. This controls selective attention to image regions during multimodal inference.

---

### 5.5 Deployment Package Specification (`.gcsopack`) and Zero-Conversion Sidecar Mode

To ensure interoperability with existing inference ecosystems (llama.cpp, vLLM, TensorRT-LLM, etc.), **`.gcsopack` (Unified Deployment Package)** is specified as the standard distribution/deployment format.

#### Table 5.1: Package Composition and Compatible Operation Mode Specifications

| Operation Mode | Component Composition | Operational Specifications |
| --- | --- | --- |
| **Integrated Native Mode** | `.gcsopack` (Core Model + Sidecar Track Integrated) | Execution via dedicated GCSO runtime |
| **Zero-Conversion Sidecar Mode** | Standard GGUF / Safetensors + External `.gcso` Sidecar File | Externally applies phase/SRL corrections without changing existing base model weights |

Without modifying base models in existing environments, attaching sidecar files alone extends geometric and phase memory control capabilities.

---

## Chapter 6: Conclusion

The **GCSO (Geometric Cellular Sheaf Orchestrator)** proposed in this paper is an inference and memory runtime architecture integrating Cellular Sheaf Cohomology from algebraic topology, overdamped Langevin dynamics / pitchfork bifurcation models from dynamical systems, stigmergy from swarm intelligence, circular statistics, and low-layer GPU register-level memory management.

Through the **Two-Layer Complementary Swarm-Attractor Structure (Swarm-Attractor Duality)** bridging constant-time $\mathcal{O}(1)$ bit/pointer operations of micro swarm cells (Attention Heads) with autonomous convergence to macro continuous target attractor fields, it formulates the theoretical and engineering foundation for achieving processing efficiency gains, long-context preservation, autonomous correction of non-factual generation, and dynamic personality control without rewriting model weights in resource-constrained 2–4 GB VRAM environments.

---

## Declaration of Generative AI Use

During the preparation of this work, the author(s) utilized Google's Gemini (LLM) as an auxiliary tool for proofreading, reviewing mathematical and architectural formulations, refining prose, and assisting with language translation. All outputs and suggestions provided by the AI were thoroughly reviewed, edited, and validated by the author(s), who take full responsibility for the final content of this document.

## License & Copyright

### Documentation

The text and diagrams of this document (architecture proposal / whitepaper) are provided under [CC BY 4.0 (Creative Commons Attribution 4.0 International)](https://creativecommons.org/licenses/by/4.0/).

© 2026 flat-sauce-works

### Source Code

Source code and scripts contained within this project belong to the copyright holder below and are available/redistributable under either of the following licenses at the user's choice (dual-licensed):

© 2026 flat-sauce-works

* [MIT License](https://opensource.org/licenses/MIT) (See: `LICENSE-MIT`)
* [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (See: `LICENSE-APACHE`)
