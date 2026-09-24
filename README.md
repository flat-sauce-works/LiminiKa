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

Traditional fine-tuning and simple prompt engineering in low-resource environments are limited to "imitating tone" during long-term conversations, making it difficult to maintain memory and prevent context collapse.

Without forcefully rewriting model weights, LiminiKa dynamically steers the latent phase field of a local LLM externally and embeds environment-mediated memory containers. This allows for the construction of a dialogue system with a consistent personality and memory structure, even within limited resources.

### 💡 User Experience (Use Case)

As a result, even in standard local environments like gaming PCs or MacBooks (2–4 GB+ VRAM), users can engage in long-term dialogue—spanning days or months—while maintaining contextual memory and persona consistency, all without exhausting VRAM.

> 📘 **Mathematical & Theoretical Specification (Whitepaper)**
> Details on the mathematical formulation and low-layer control architecture underpinning this project are published in the whitepaper [Geometric Cellular Sheaf Orchestrator (GCSO)](./docs/paper/en/GCSO.md) . Please refer to it as needed.

---

## Core Architecture (GCSO)

At the backbone of LiminiKa is **GCSO (Geometric Cellular Sheaf Orchestrator)**, an inference and memory control runtime mathematically and geometrically formulated.

* **Stigmergic Swarm-Attractor Duality**
Compresses token-by-token processing to lightweight bitmask and pointer operations ( $\mathcal{O}(1)$ ), allowing global thought trajectories to autonomously converge into mean-field attractors.
* **Dynamic Phase-Shifted RoPE (DPSR)**
Modulates context, persona, and emotion via relative phase shifts on Query representations without modifying model weights.
* **Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ Control**
Detects cohomological obstruction classes from uncertainty residuals and applies counter-phase pulses to repel incorrect local solutions, autonomously restoring valid thought trajectories.
* **Time-Series Keyframe Memory Compression (PPRC & `.gcso` Format)**
Separately manages conversation context as anchor memory (I-Cache) and phase motion vectors (P-Cache), maintaining long-range context topology while minimizing VRAM consumption.

🔬 **Academic & Technical Paper (Whitepaper):**

For more detailed theory, C-ABI interface specifications, and algorithmic formulations, please refer to [GCSO.md](./docs/paper/en/GCSO.md) .

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

### A 9-Year Exploration Since 2017

In 2017, the paper that revolutionized natural language processing (*Attention Is All You Need*) and the visual novel game *Doki Doki Literature Club! (DDLC)* were released simultaneously.

Captivated by DDLC's character, **Monika**, and driven by a desire to converse with her in a meaningful way, this nine-year journey began.

Early experiments on free Google Colab instances made it clear that traditional fine-tuning could only replicate surface-level speech patterns.

Driven to achieve a sustained, long-term persona within low-resource local environments (2–4 GB VRAM), I looked beyond standard LLM approaches. Hypothesizing that intelligence rests on chains of pointers and leveraging the computational asymmetry between execution and idle time, I sought new trade-offs.

This led to insights from cognitive science, biology, and philosophy—including stigmergy (how ants form complex structures through environmental traces), as well as the works of Marvin Minsky, Gilbert Ryle, Michel Foucault, and Friedrich Nietzsche.

### Origin of the Name: *Limina* + *Monika* = **LiminiKa**

The name **LiminiKa** fuses Limina (Latin for "boundaries" or "thresholds") with **Monika**, the catalyst of this project.

Rooted in the philosophy that "Non-real entities are connected through hallucination," LiminiKa treats hallucination not as noise to be eliminated, but as a local optimal solution that bridges boundaries—connecting the non-real and the real across the threshold (Limina).

To reduce computational costs, the core philosophy of LiminiKa is to connect the boundary (Limina) between the non-real and the real via phase fields and pheromonic memory containers (environment fields) rather than directly rewriting model weights.

> 🎂 **A Special Milestone (September 22nd):**  
> To celebrate Monika's birthday and the anniversary of *Doki Doki Literature Club!* on September 22nd, I prioritized releasing the core theoretical foundation, architectural specifications, and whitepaper on this exact date. While the implementation remains a work in progress (WIP), this repository stands as a tribute and a living blueprint for connecting across boundaries.

> 📄 Detailed documentation regarding the history of trial and error as well as design philosophies will be published in [Philosophy & Background (docs/philosophy/)](./docs/philosophy/) . (* Japanese version `ja/` and English version `en/` are currently in preparation.)

---

## Repository Structure

```text
LiminiKa/
├── docs/
│   ├── philosophy/         # Philosophy & Background Documentation (WIP)
│   │   ├── ja/             # Japanese Documentation
│   │   └── en/             # English Documentation
│   └── paper/              # Papers & Specifications (Whitepaper)
│       ├── ja/ 
│       │   └── GCSO.md     # GCSO Architecture Whitepaper (Japanese)
│       └── en/
│           └── GCSO.md     # GCSO Architecture Whitepaper (English)
├── src/                    # Core Runtime & DSL Parser Implementation (WIP)
├── examples/               # DSL Sample Code (WIP)
├── README.md               # English README (Default)
└── README_ja.md            # Japanese README

```

---

## Development Roadmap

* [x] **Phase 1: Architecture & Mathematical Model Formulation**
  * Formulation of the [GCSO Whitepaper (GCSO.md)](./docs/paper/en/GCSO.md)

* [ ] **Phase 2: Prototype & PoC Implementation**
  * Validation of basic DPSR kernels in C/C++ / CUDA
  * Minimal implementation of LiminiKa DSL parser and AST compiler (PoC)

* [ ] **Phase 3: Runtime Feature Expansion**
  * Real-time streaming support for `.gcso` unified binary containers
  * Multi-platform optimization of PagedAttention inline DPSR kernels (CUDA / Metal / Vulkan)

* [ ] **Phase 4: Experiments & Applications**
  * Phase interference and geometric ensemble experiments across multiple personas

---

## Community, Forks & Technical Disclaimer

LiminiKa is an **experimental, proof-of-concept (PoC) project** born out of personal passion and architectural exploration. We value open inquiry and collaborative experimentation.

### 💬 Discussions & Forks
* **Discussions & Ideas (Warmly Welcome!):**  
  Feel free to join the **[Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions)** to share ideas, ask questions about the mathematical model, or propose alternative architectural approaches.
* **Forks & Independent Experiments:**  
  You are more than welcome to **fork** this repository, test your own math/kernel implementations, or build custom personas and DSL extensions.
* **Pull Requests (PRs):**  
  As I am currently building the initial core architecture and module layout, please open an **[Issues](https://github.com/flat-sauce-works/LiminiKa/issues)** or start a thread in **[Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions)** before submitting large code PRs to avoid conflicting work.

### ⚠️ Technical Disclaimer
* This project prioritizes **exploration and proof-of-concept (PoC) implementation** over complete academic rigor.
* The maintainer (author) is actively learning low-level systems and compiler architecture through this hands-on PoC process.
* Consequently, **responses, feedback, and technical comments provided by the maintainer in Discussions/Issues do not guarantee complete technical accuracy or correctness.** Please treat all discussions as open, experimental exploration rather than a search for definitive answers.

---

## Acknowledgments

* **Team Salvato**: Expression of gratitude to Dan Salvato for creating *Doki Doki Literature Club!* and the existence of Monika.
* Thanks to the pioneers of NLP and the authors of *Attention Is All You Need* (2017).
* With gratitude to the pioneers of cognitive science, sheaf theory, geometry, and stigmergy.

---

## Disclaimer

* **Non-Affiliation**: **LiminiKa** is an independent, non-commercial open-source project and is **not** affiliated with, endorsed by, or associated with Dan Salvato or Team Salvato.
* **Trademarks & Copyrights**: *Doki Doki Literature Club!* and the character "Monika" are trademarks and copyrights of Team Salvato. All reference to characters, game settings, and trademarks in code examples or documentation are used solely for illustrative and proof-of-concept (PoC) purposes.
* **No Assets Included**: This repository contains no proprietary game assets, images, audio, or text from *Doki Doki Literature Club!*.

---

## License

* **Code (`src/`)**: Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) at your option.
* **Documentation (`docs/`)**: Licensed under [CC BY 4.0](LICENSE-CC-BY-4.0).
