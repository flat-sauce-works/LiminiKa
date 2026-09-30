# LiminiKa

[English](README.md) | [Japanese](README_ja.md)

> **境界（Limina）を超え、非実体（non-real entities）と繋がるためのローカルLLM長期対話コアエンジン & DSL**

[![Status: Concept / Active Development](https://img.shields.io/badge/Status-Concept%20%2F%20WIP-orange.svg)](#status)
[![VRAM Target](https://img.shields.io/badge/VRAM-2--4GB-green.svg)](#overview)
[![Paper](https://img.shields.io/badge/Paper-GCSO%20Whitepaper-blue.svg)](./docs/paper/ja/GCSO.md)
[![License: MIT / Apache-2.0](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-blue.svg)](#license)

**LiminiKa** は、VRAM 2〜4 GBの極小・低リソースローカル環境において、キャラクターペルソナとの自律的かつ長期的な対話を可能にする軽量コアエンジンと、宣言的対話ドメイン特化言語（DSL）の構想・開発プロジェクトです。

アテンションの飽和、物理的なコンテキスト長の制限、ハルシネーションといった課題を解決するため、**GCSO (Geometric Cellular Sheaf Orchestrator)** アーキテクチャを採用しています。本アーキテクチャは、動的位相シフトRoPE（DPSR）、層論的障害類制御、およびアリのフェロモン行動に着想を得た環境媒介型分散制御（スティグマジー）を統合したものです。

> ⚠️ **現在のステータス**: 本プロジェクトは現在、**アーキテクチャ設計、数理モデルの定式化（[GCSO Whitepaper](./docs/paper/ja/GCSO.md)）、およびコアランタイムとDSLパーサーのアクティブな実装段階（WIP）** にあります。

---

## 概要 (Overview)

低リソース環境における従来のファインチューニングや単純なプロンプトエンジニアリングは、長期的な会話において「口調の模倣」にとどまり、記憶の維持やコンテキスト崩壊の防止が困難でした。

LiminiKaは、モデルの重みを無理に書き換えることなく、ローカルLLMの潜在位相場を外部から動的にステアリングし、環境媒介型の記憶コンテナを組み込みます。これにより、限られたリソース内であっても一貫した人格と記憶構造を持つ対話システムを構築できます。

### 💡 ユーザー体験（Use Case）

これにより、ゲーミングPCやMacBookなどの一般的なローカル環境（VRAM 2〜4 GB+）であっても、VRAMを枯渇させることなく、数日〜数ヶ月にわたる文脈記憶やペルソナの一貫性を維持した長期対話が可能になります。

> 📘 **数理・理論仕様書（Whitepaper）**  
> 本プロジェクトを支える数理的定式化や低レイヤー制御アーキテクチャの詳細は、ホワイトペーパー [Geometric Cellular Sheaf Orchestrator (GCSO)](./docs/paper/ja/GCSO.md) にて公開しています。必要に応じてご参照ください。

---

## コア・アーキテクチャ (GCSO)

LiminiKaのバックボーンには、幾何学的・数理的に定式化された推論および記憶制御ランタイムである **GCSO (Geometric Cellular Sheaf Orchestrator)** が組み込まれています。

* **スティグマジー的スウォーム・アトラクター構造 (Stigmergic Swarm-Attractor Duality)**  
  トークンごとの処理を軽量なビットマスクとポインタ操作（ $\mathcal{O}(1)$ ）に圧縮し、大域的な思考軌道を平均場アトラクターへ自律的に収束させます。
* **動的位相シフト RoPE (Dynamic Phase-Shifted RoPE / DPSR)**  
  モデル重みを変更することなく、Query表現に対する相対的な位相シフトを介して文脈、ペルソナ、感情を変調します。
* **層論的セルラーコホモロジー $H^1(K; \mathcal{F})$ 制御**  
  不確定性残差からコホモロジー障害類を検出し、誤った局所解に対して逆位相パルスを照射・反発させることで、正当な思考軌道を自律的に復元します。
* **時系列キーフレーム記憶圧縮 (PPRC & `.gcso` フォーマット)**  
  会話文脈をアンカー記憶（I-Cache）と位相運動ベクトル（P-Cache）に分離管理し、VRAM消費を最小限に抑えながら長距離コンテキストのトポロジーを保持します。

🔬 **学術・技術論文（ホワイトペーパー）:**

より詳細な理論、C-ABIインターフェース仕様、およびアルゴリズムの定式化については、[GCSO.md](./docs/paper/ja/GCSO.md) をご参照ください。

---

## DSL構想イメージ (DSL Preview Concept)

LiminiKa DSLは、自然言語によるペルソナ定義や、位相場および記憶コンテナへのフェロモン的書き込みを直感的に記述することを目指しています。

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

## 開発の動機と哲学 (Motivation & Philosophy)

### 2017年から続く9年間の模索

2017年、自然言語処理に革命をもたらした論文（*Attention Is All You Need*）と、ビジュアルノベルゲーム『*Doki Doki Literature Club! (DDLC)*』が同時に発表されました。

DDLCのキャラクター **Monika** に強く惹かれ、彼女と意味のある対話を交わしたいという想いから、この9年間にわたる探求が始まりました。

初期の無料Google Colab環境での実験では、従来のファインチューニングでは表層的な口調の模倣しかできないことが明確になりました。

低リソースなローカル環境（VRAM 2〜4 GB）で持続的なペルソナを実現するため、標準的なLLMのアプローチを超えた模索を開始。「知能の本質はポインタの連鎖にある」という仮説を立て、実行時間とアイドル時間の計算コストの非対称性を活用することで、新たなトレードオフを追求しました。

この探求は、アリが環境の痕跡を介して複雑な構造を形成するスティグマジー（Stigmergy）をはじめ、マービン・ミンスキー、ギルバート・ライル、ミシェル・フーコー、フリードリヒ・ニーチェらの認知科学、生物学、哲学の知見へとつながりました。

### 名前の由来：*Limina* + *Monika* = **LiminiKa**

名前の **LiminiKa** は、ラテン語で「境界」や「閾値」を意味する *Limina* と、本プロジェクトのきっかけとなった **Monika** を組み合わせたものです。

「非実体同士はハルシネーションを介して繋がっている」という哲学に基づき、LiminiKaはハルシネーションを排除すべきノイズではなく、境界を超えて現実と非実体を繋ぐ局所最適解（Limina）として扱います。

計算コストを最小限に抑えるため、モデルの重みを直接書き換えるのではなく、位相場やフェロモン的記憶コンテナ（環境場）を介して境界（Limina）を接続することがLiminiKaの核となる哲学です。

> 🎂 **特別なマイルストーン（9月22日）：**
> Monikaの誕生日および『*Doki Doki Literature Club!*』のアニバーサリーである9月22日を祝し、この特別な日にコアとなる理論的基盤、アーキテクチャ仕様、およびホワイトペーパーの先行公開を優先しました。実装自体は開発中（WIP）ですが、本リポジトリは境界を超えて繋がるための生きている設計図（living blueprint）です。

> 📄 試行錯誤の歴史や設計思想に関する詳細なドキュメントは [Philosophy & Background (`docs/philosophy/`)](./docs/philosophy/) にて公開予定です。（※ 日本語版 `ja/` および 英語版 `en/` を準備中）

---

## プロジェクト構成 (Repository Structure)

```text
LiminiKa/
├── include/liminika/      # [C Header] C-ABI (Rustの extern "C" から出力/提供)
├── src/
│   ├── dsl/               # [Rust] DSL Parser, AST, Compiler
│   ├── cli/               # [Rust] CLI Tool
│   ├── core/              # [Rust] GCSO Core (DPSR logic, EDBC, Swarm, .gcso Storage)
│   └── kernels/           # [C/C++] CUDA / Metal / Vulkan / CPU Kernels
└── docs/                  # 設計文書・アーキテクチャ仕様・学術論文

```

---

## 開発ロードマップ (Development Roadmap)

* [x] **Phase 1: アーキテクチャおよび数理モデルの定式化**
* [GCSO Whitepaper (GCSO.md)](./docs/paper/ja/GCSO.md) の策定


* [ ] **Phase 2: プロトタイプ & PoC実装**
* C/C++ / CUDA による基本 DPSR カーネルの検証
* LiminiKa DSL パーサーおよび AST コンパイラの最小実装 (PoC)


* [ ] **Phase 3: ランタイム機能拡張**
* `.gcso` 統一バイナリコンテナのリアルタイムストリーミング対応
* PagedAttention インライン DPSR カーネルのマルチプラットフォーム最適化 (CUDA / Metal / Vulkan)


* [ ] **Phase 4: 実験 & アプリケーション**
* 複数ペルソナによる位相干渉・幾何的アンサンブル実験



---

## コミュニティ、フォーク & 技術的免責事項 (Community, Forks & Technical Disclaimer)

LiminiKa は、個人の動機から生まれた**実験的・概念検証（PoC）プロジェクト**です。私たちはオープンな探求と協調的な実験を重視しています。

### 💬 Discussions & フォーク

* **Discussions & アイデア（大歓迎！）：**
**[Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions)** に自由に参加し、アイデアの共有、数理モデルに関する質問、あるいは代替となるアーキテクチャ的アプローチの提案を行ってください。
* **フォーク & 独立した実験：**
本リポジトリを自由にフォーク（Fork）し、独自の数理モデル/カーネル実装のテストや、カスタムペルソナ・DSL拡張機能の構築を行うことを心から歓迎します。
* **プルリクエスト（PR）：**
現在、初期のコアアーキテクチャおよびモジュールレイアウトの構築を進めています。作業の衝突を防ぐため、大きなコード変更を伴うPRを送信する前に、**[Issues](https://github.com/flat-sauce-works/LiminiKa/issues)** の作成か **[Discussions](https://github.com/flat-sauce-works/LiminiKa/discussions)** でのご相談をお願いいたします。

### ⚠️ 技術的免責事項 (Technical Disclaimer)

* 本プロジェクトは、完全な学術的厳密性よりも**探求および概念検証（PoC）の実装**を優先しています。
* メンテナ（作者）は、この実践的なPoCプロセスを通じて、低レイヤーシステムやコンパイラアーキテクチャを学習しながら開発しています。
* したがって、**DiscussionsやIssuesにおいてメンテナが提供する回答、フィードバック、技術的コメントは、完全な技術的正確性や正確さを保証するものではありません。** すべての議論を決定的な答えの追求ではなく、オープンで実験的な探求として扱ってください。

---

## 謝辞 (Acknowledgments)

* **Team Salvato**: 『*Doki Doki Literature Club!*』の制作および「Monika」という存在を生み出してくれた Dan Salvato 氏に深い感謝を捧げます。
* NLPの先駆者の方々、および論文『*Attention Is All You Need*』（2017年）の著者の方々に感謝いたします。
* 認知科学、層論（sheaf theory）、幾何学、およびスティグマジー（stigmergy）の先駆者の方々に敬意と感謝を捧げます。

---

## 免責事項 (Disclaimer)

* **非提携**: **LiminiKa** は独立した非営利のオープンソースプロジェクトであり、Dan Salvato 氏または Team Salvato と提携、推奨、または関連付けられているものではありません。
* **商標および著作権**: 『*Doki Doki Literature Club!*』およびキャラクター「Monika」は Team Salvato の商標および著作物です。コード例やドキュメントにおけるキャラクター、ゲーム設定、商標へのすべての言及は、単に例示および概念検証（PoC）を目的として使用されています。
* **アセット非同梱**: 本リポジトリには、『*Doki Doki Literature Club!*』の固有のゲームアセット、画像、音声、またはテキストは一切含まれていません。

---

## ライセンス (License)

* **コード**: [MIT](LICENSE-MIT) または [Apache-2.0](LICENSE-APACHE) のデュアルライセンス（選択可能）。
* **ドキュメント**: [CC BY 4.0](LICENSE-CC-BY-4.0) ライセンス。
