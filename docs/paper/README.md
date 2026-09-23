# LiminiKa Papers & Technical Documents

This directory contains the whitepaper and technical documentation for **GCSO (Geometric Cellular Sheaf Orchestrator)**, the foundational geometric and sheaf-theoretic dialogue control architecture powering LiminiKa.

---

## 📄 Whitepaper Index

| Document | Language | Link | Description |
| :--- | :--- | :--- | :--- |
| **GCSO Whitepaper (English)** | English | [en/GCSO.md](./en/GCSO.md) | Architecture Proposal & Mathematical Formulation |
| **GCSO Whitepaper (Japanese)** | Japanese | [ja/GCSO.md](./ja/GCSO.md) | Architecture Proposal & Mathematical Formulation |

---

## 💡 GCSO Architecture Overview

> **Geometric Cellular Sheaf Orchestrator (GCSO)**  
> *A RoPE Phase-Modulation and Sheaf-Theoretic Memory Control Runtime for Local LLMs based on Swarm-Attractor Stigmergic Architecture*

GCSO is an inferencing and memory runtime framework designed to maintain long-term context, memory, and persona consistency without modifying model weights, specifically optimized for resource-constrained local environments (VRAM 2–4GB).

### Key Architectural Pillars

- **Two-Layer Swarm-Attractor Duality**: Bypasses online differential equation solving by reducing the Hot Path to $\mathcal{O}(1)$ micro bitwise/pointer updates across Attention Heads (swarm cells), which autonomously converge to macro target attractor fields as a mean-field limit.
- **Dynamic Phase-Shifted RoPE (DPSR)**: Context, persona, and attention temperature steering via register-level inline phase rotation inside the RoPE kernel with $SO(2)^{d_{\mathrm{head}}/2}$ commutativity.
- **Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$**: Identifies hallucinations as "Spurious Local Minima" in the out-of-distribution latent space and autonomously repels them using anti-phase pulses (**Phase-Conjugate Attractor Repulsion**).
- **Entropy-Driven Decoding Branch Controller (EDBC)**: Monitors attention entropy flux ($\tilde{H}$) to trigger pitchfork bifurcations via Langevin dynamics, dynamically controlling transitions between exploration and deterministic convergence.
- **Sparse Residual Adapters (SRL) & L2P-SVD**: Supplements information transformations using Dynamic Rank-1 adapters and projects fine-tuned LoRA weights into phase profiles without additional gradient training.
- **Predictive Phase-Motion & Residual Compensation (PPRC)**: Achieves zero-forward-latency instant seeking and long-context KV cache compression using the `.gcso` unified container with Variable GOP structures.

---

## 📚 Whitepaper Table of Contents

The GCSO Whitepaper ( [English](./en/GCSO.md) / [Japanese](./ja/GCSO.md) ) is structured as follows:

- **Chapter 0: Correspondence Table**: Mathematical concepts $\leftrightarrow$ Engineering data structures $\leftrightarrow$ C-ABI functions.
- **Chapter 1: Theoretical Background**: Sheaf Cohomology, Regularized Hodge Decomposition, Graph Laplacian Spectral Filtering, Gershgorin Disc upper bounds, and Pseudo-ultrametric Spaces.
- **Chapter 2: Runtime Architecture**: In-kernel DPSR fusion, Lazy Phase Unwrapping, PSPM Subspace Ensembles, and EG-LUT/EGDPL lookup structures.
- **Chapter 3: Phase Transitions & Autonomous Control**: Moving Z-Score Normalized Attention Entropy ($\tilde{H}$), Pitchfork Bifurcation models in Langevin dynamics, and Phase-Conjugate Repulsion.
- **Chapter 4: Long-Context & Memory Structures**: Decoupled Fact-Semantic Memory, Circular Statistics ($\kappa_c$), PPRC, and the `.gcso` binary container format.
- **Chapter 5: High-Dimensional Context Projection**: Lie Group rotation trees, SRL/L2P-SVD converters, Storage Cost Offloading, and `.gcsopack` deployment specs.
- **Chapter 6: Conclusion**: Summary of theoretical and engineering contributions.

---

## 📜 Citation

If you cite or reference the GCSO architecture, mathematical formulations, or technical concepts in your research or project, please use the following BibTeX entry:

```bibtex
@misc{liminika_gcso_2026,
  author       = {flat-sauce-works},
  title        = {{Geometric Cellular Sheaf Orchestrator (GCSO): Design Concept of a RoPE Phase-Modulated Sheaf-Theoretic Memory Control Runtime for Local LLMs Based on Swarm-Attractor Stigmergic Architecture}},
  howpublished = {\url{https://github.com/flat-sauce-works/LiminiKa}},
  year         = {2026},
  month        = {9},
  note         = {Version v0.1.0}
}
```

You can also use the [`CITATION.cff`](../../CITATION.cff) file in the root directory or click "Cite this repository" on GitHub to export citations in various formats (APA, BibTeX, etc.).

---

## ⚖️ License

* **Documentation (`docs/paper/`)**: Licensed under [CC BY 4.0](../../LICENSE-CC-BY-4.0).
