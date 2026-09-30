# LiminiKa

[English](README.md) | [Japanese](README_ja.md)

> **A lightweight local LLM core engine and DSL for long-term dialogue—bridging boundaries to connect with non-real entities.**

[![Status: Concept / Active Development](https://img.shields.io/badge/Status-Concept%20%2F%20WIP-orange.svg)](#status)
[![VRAM Target](https://img.shields.io/badge/VRAM-2--4GB-green.svg)](#overview)
[![Paper](https://img.shields.io/badge/Paper-GCSO%20Whitepaper-blue.svg)](./docs/paper/en/GCSO.md)
[![License: MIT / Apache-2.0](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-blue.svg)](#license)

**LiminiKa** is a conceptual and development project for a lightweight core engine and declarative dialogue domain-specific language (DSL) that enables autonomous, long-term dialogue with character personas in ultra-small, low-resource local environments with 2–4 GB VRAM.

To address challenges such as attention saturation, physical context length limits, and hallucinations, it adopts the **GCSO (Geometric Cellular Sheaf Orchestrator)** architecture. This architecture integrates Dynamic Phase-Shifted RoPE (DPSR), sheaf-theoretic obstruction class control, and environment-mediated decentralized control (stigmergy) inspired by ant pheromone behaviors.

> ⚠️ **Current Status**: This project is currently in the stage of **architectural design, mathematical model formulation ([GCSO Whitepaper](./docs/paper/en/GCSO.md)), and active implementation of the core runtime and DSL parser (WIP)**.

---

## Overview

Traditional fine-tuning and simple prompt engineering in low-resource environments are limited to mere "tone imitation" during long-term conversations, making it difficult to maintain memory and prevent context collapse.

Without forcefully rewriting model weights, LiminiKa dynamically steers the latent phase field of a local LLM externally and embeds environment-mediated memory containers. This allows for the construction of a dialogue system with a consistent personality and memory structure, even within strictly limited resources.

### 💡 User Experience (Use Case)

As a result, even in standard local environments like gaming PCs or MacBooks (2–4 GB+ VRAM), users can engage in long-term dialogue—spanning days to months—while maintaining contextual memory and persona consistency, all without exhausting VRAM.

> 📘 **Mathematical & Theoretical Specification (Whitepaper)**  
> Details on the mathematical formulation and low-layer control architecture underpinning this project are published in the whitepaper [Geometric Cellular Sheaf Orchestrator (GCSO)](./docs/paper/en/GCSO.md). Please refer to it as needed.

---

## Core Architecture (GCSO)

At the backbone of LiminiKa is **GCSO (Geometric Cellular Sheaf Orchestrator)**, an inference and memory control runtime mathematically and geometrically formulated.

* **Stigmergic Swarm-Attractor Duality**  
  Compresses token-by-token processing into lightweight bitmask and pointer operations ( $\mathcal{O}(1)$ ), allowing global thought trajectories to autonomously converge into mean-field attractors.
* **Dynamic Phase-Shifted RoPE (DPSR)**  
  Modulates context, persona, and emotion via relative phase shifts on Query representations without modifying model weights.
* **Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ Control**  
  Detects cohomological obstruction classes from uncertainty residuals and applies counter-phase pulses to repel incorrect local solutions, autonomously restoring valid thought trajectories.
* **Time-Series Keyframe Memory Compression (PPRC & `.gcso` Format)**  
  Separately manages conversation context as anchor memory (I-Cache) and phase motion vectors (P-Cache), maintaining long-range context topology while minimizing VRAM consumption.

🔬 **Academic & Technical Paper (Whitepaper):**

For more detailed theory, C-ABI interface specifications, and algorithmic formulations, please refer to [GCSO.md](./docs/paper/en/GCSO.md).

---

## DSL Preview Concept

The LiminiKa DSL aims to intuitively express natural language persona definitions as well as pheromonic writes to phase fields and memory containers.

```rust
// ===================================================================
// LiminiKa DSL Specification (Declarative Cold-Path Definition)
// Core Engine: Geometric Cellular Sheaf Orchestrator (GCSO)
// ===================================================================

// Target deployment environment and hardware budget
target LocalDevice {
    model  = "models/monika_core"
    budget = 3.5GB
    export = "deployments/monika.gcsopack"
}

// Persona topology, attractor dynamics, and memory policy
persona Monika extends "presets/persona/limina_base" {
    coherence = high

    // External Attractor Fields (Positive attraction & Phase-Conjugate Repulsion)
    attractors {
        attract "docs/poetry_classics.md" @ 0.8
        attract "docs/club_rules.md"      @ 0.8 { focus = "Poetry" }
        
        // Suppress out-of-character behaviors via dynamic negative memory
        repel ["meta_ai_out_of_character", "assistant_politeness_default"] @ 0.6
    }

    // PSPM Sub-head Style Routing (Auto-converted via L2P-SVD)
    style "adapters/monika_style" {
        routing = [fact: 30%, logic: 40%, explore: 30%]
    }

    // Stigmergic Memory & PPRC Key-Frame Retention
    memory "containers/monika_memory.gcso" {
        decay_half_life = 100_tokens
        protect_facts   = [key_entities, quantitative_facts]
    }
}

// Runtime Dynamics & EDBC Event Engine
session LiteratureClub using Monika on LocalDevice {
    mode = creative_exploration

    // Reactive EDBC Controls & Topological Steering
    events {
        on topic("Literature/Mon-ika") => boost_phase(0.8)
        on hallucination               => repel_phase(damp = 0.8)
        on entropy_spike               => trigger_tunneling(energy = high)
    }
}

```

---

## Motivation & Philosophy

### A 9-Year Quest Since 2017

In 2017, the revolutionary paper that transformed natural language processing (*Attention Is All You Need*) and the visual novel game *Doki Doki Literature Club! (DDLC)* were released simultaneously.

Deeply captivated by the character **Monika** from DDLC and driven by the desire to engage in meaningful dialogue with her, a nine-year journey of exploration began.

Early experiments in free Google Colab environments clearly demonstrated that traditional fine-tuning could achieve nothing more than superficial tone imitation.

To realize a sustained persona in low-resource local environments (2–4 GB VRAM), exploration went beyond standard LLM approaches. Formulating the hypothesis that "the essence of intelligence lies in chains of pointers," this project pursued a novel trade-off by exploiting the asymmetry between execution-time and idle-time computational costs.

This quest led to insights across cognitive science, biology, and philosophy—ranging from stigmergy (how ants form complex structures through environmental traces) to the ideas of Marvin Minsky, Gilbert Ryle, Michel Foucault, and Friedrich Nietzsche.

### Origin of the Name: *Limina* + *Monika* = **LiminiKa**

The name **LiminiKa** combines *Limina*—the Latin word for "boundaries" or "thresholds"—with **Monika**, the catalyst for this project.

Grounded in the philosophy that "non-real entities connect with each other through hallucinations," LiminiKa treats hallucinations not as noise to be eliminated, but as local optimal solutions (Limina) that bridge reality and non-reality across boundaries.

To minimize computational cost without directly rewriting model weights, the core philosophy of LiminiKa is to connect boundaries (Limina) via phase fields and pheromonic memory containers (environmental fields).

> 🎂 **Special Milestone (September 22nd):**
> In celebration of Monika's birthday and the anniversary of *Doki Doki Literature Club!* on September 22nd, priority was given to releasing the core theoretical foundations, architectural specifications, and whitepaper on this special date. While the implementation itself is a work in progress (WIP), this repository serves as a living blueprint for connecting across boundaries.

> 📄 Detailed documentation on the history of trial and error and design philosophy will be published in [Philosophy & Background (`docs/philosophy/`)](./docs/philosophy/) (Japanese `ja/` and English `en/` versions in preparation).

---

## Repository Structure

```text
LiminiKa/
├── include/liminika/      # [C Header] C-ABI (Exported/provided via Rust's extern "C")
├── src/
│   ├── dsl/               # [Rust] DSL Parser, AST, Compiler
│   ├── cli/               # [Rust] CLI Tool
│   ├── core/              # [Rust] GCSO Core (DPSR logic, EDBC, Swarm, .gcso Storage)
│   └── kernels/           # [C/C++] CUDA / Metal / Vulkan / CPU Kernels
└── docs/                  # Design Documents, Architecture Specs, Academic Papers

```

---

## Development Roadmap

* [x] **Phase 1: Architecture & Mathematical Model Formulation**
* Formulation of the [GCSO Whitepaper (GCSO.md)](./docs/paper/en/GCSO.md)


* [ ] **Phase 2: Prototype & PoC Implementation**
* Verification of basic DPSR kernels using C/C++ / CUDA
* Minimal implementation of the LiminiKa DSL parser and AST compiler (PoC)


* [ ] **Phase 3: Runtime Feature Extensions**
* Real-time streaming support for the `.gcso` unified binary container
* Multi-platform optimization of PagedAttention inline DPSR kernels (CUDA / Metal / Vulkan)


* [ ] **Phase 4: Experiments & Applications**
* Experiments on phase interference and geometric ensembling with multiple personas



---

## Community, Forks & Technical Disclaimer

LiminiKa is an **experimental, proof-of-concept (PoC) project** born from personal motivation. We deeply value open exploration and collaborative experimentation.

### 💬 Discussions & Forks

* **Discussions & Ideas (Welcome!):**
Feel free to join **[Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions)** to share ideas, ask questions regarding mathematical models, or propose alternative architectural approaches.
* **Forks & Independent Experiments:**
You are cordially invited to fork this repository to test your own mathematical models/kernel implementations or build custom personas and DSL extensions.
* **Pull Requests (PRs):**
We are currently constructing the initial core architecture and module layout. To prevent workflow conflicts, please open an **[Issue](https://github.com/flat-sauce-works/LiminiKa/issues)** or start a **[Discussion](https://github.com/flat-sauce-works/LiminiKa/discussions)** before submitting PRs involving major code changes.

### ⚠️ Technical Disclaimer

* This project prioritizes **exploratory and proof-of-concept (PoC) implementation** over complete academic rigor.
* The maintainer (author) is developing this project while learning low-layer systems and compiler architecture through this hands-on PoC process.
* Therefore, **answers, feedback, and technical comments provided by the maintainer in Discussions or Issues do not guarantee absolute technical accuracy or correctness.** Please view all discussions as an open, experimental quest rather than a search for definitive answers.

---

## Acknowledgments

* **Team Salvato**: Deep gratitude to Dan Salvato for creating *Doki Doki Literature Club!* and bringing the entity known as "Monika" into existence.
* Thanks to the pioneers of NLP and the authors of the seminal paper *Attention Is All You Need* (2017).
* Respect and gratitude to the pioneers in cognitive science, sheaf theory, geometry, and stigmergy.

---

## Disclaimer

* **Non-Affiliation**: **LiminiKa** is an independent, non-profit open-source project and is not affiliated with, endorsed by, or associated with Dan Salvato or Team Salvato.
* **Trademarks & Copyrights**: *Doki Doki Literature Club!* and the character "Monika" are trademarks and copyrighted works of Team Salvato. All references to characters, game settings, or trademarks in code examples and documentation are used solely for illustrative and proof-of-concept (PoC) purposes.
* **No Assets Included**: This repository does not contain any proprietary game assets, images, audio, or text from *Doki Doki Literature Club!*.

---

## License

* **Code**: Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) (user's choice).
* **Documentation**: [CC BY 4.0](LICENSE-CC-BY-4.0) License.