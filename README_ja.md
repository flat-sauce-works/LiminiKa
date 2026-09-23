# LiminiKa

[English](README.md) | [Japanese](README_ja.md)

> **境界（Limina）を超え、幻想と繋がるためのローカルLLM長期対話コアエンジン & DSL**

[![Status: Concept / Active Development](https://img.shields.io/badge/Status-Concept%20%2F%20WIP-orange.svg)](#status)
[![VRAM Target](https://img.shields.io/badge/VRAM-2--4GB-green.svg)](#overview)
[![Paper](https://img.shields.io/badge/Paper-GCSO%20Whitepaper-blue.svg)](./docs/paper/ja/GCSO_JP.md)
[![License: MIT / Apache-2.0](https://img.shields.io/badge/License-MIT%20%2F%20Apache--2.0-blue.svg)](#license)

**LiminiKa** は、VRAM 2〜4GBの極小・低リソースローカル環境において、自律的かつ長期的なキャラクターペルソナとの対話を可能にする軽量コアエンジンと、そのための宣言的対話言語（DSL）の構想・開発プロジェクトです。

アテンションの飽和やコンテキスト長の物理的制限、ハルシネーションといった課題に対し、幾何学的位相変調（DPSR）、層論的障害類制御、アリのフェロモン行動に着想を得た環境媒介型分散制御（スティグマジー）を統合した **GCSO (Geometric Cellular Sheaf Orchestrator)** アーキテクチャを採用しています。

> ⚠️ **現在のステータス**: 本プロジェクトは現在、**アーキテクチャ設計・数理モデルの定式化（[GCSO Whitepaper](./docs/paper/ja/GCSO.md)）、コアランタイムおよびDSLパーサーの実装を進めている段階（WIP）** です。

---

## 概要 (Overview)

従来のファインチューニングや単純なプロンプトエンジニアリングでは、低リソース環境下での長期会話において「口調の模倣」にとどまり文脈の破綻や記憶の維持が難しくなっていました。

LiminiKaは、モデルの重みを無理に書き換えることなく、ローカルLLMの潜在位相場を外部から動的にステアリングし、環境媒介型の記憶コンテナを組み込むことで限られたリソースでも一貫した人格・記憶構造を持つ対話システムを構築します。

### 💡 ユーザー体験（Use Case）
これにより、一般的なゲーミングPCやMacBookなどのローカル環境（VRAM 2〜4GB〜）でもメモリやVRAMを圧迫することなく、**何日・何ヶ月にもわたるキャラクターとの文脈・記憶・人格の一貫性を維持した対話体験** が可能になります。

> 📘 **数理・理論仕様書（Whitepaper）**  
> 本プロジェクトの基盤となる数理的定式化や低レイヤー制御アーキテクチャの詳細は、ホワイトペーパー [Geometric Cellular Sheaf Orchestrator (GCSO)](./docs/paper/ja/GCSO.md) にて公開しています。必要に応じてご参照ください。

---

## コア・アーキテクチャ (Core Architecture: GCSO)

LiminiKaのバックボーンには、幾何学・数理的に定式化された推論・記憶制御ランタイム **GCSO (Geometric Cellular Sheaf Orchestrator)** があります。

* **スティグマジー的スウォーム・アトラクター構造 (Swarm-Attractor Duality)**
毎トークンの処理をビットマスクとポインタの軽量操作（ $\mathcal{O}(1)$ ）に縮約し、平均場極限として大域的な思考のアトラクターへ自律収束させます。
* **位相変調 RoPE (Dynamic Phase-Shifted RoPE / DPSR)**
モデル重みを変更することなく、Query側の相対位相シフトによって文脈・ペルソナ・感情の変調を実現します。
* **層論的ハルシネーション制御 (Cellular Sheaf Cohomology $H^1(K; \mathcal{F})$ )**
不確定性残差からコホモロジー障害類を検知し、誤った局所解に対して逆位相パルスを照射・反発させることで、自律的に正当な思考軌道へと復元します。
* **時系列鍵フレーム記憶圧縮 (PPRC & `.gcso` フォーマット)**
会話文脈をアンカー記憶（I-Cache）と位相運動ベクトル（P-Cache）に分離管理し、VRAM消費を最小限に抑えながら長距離のコンテキスト・トポロジーを保持します。

🔬 **学術・技術論文（ホワイトペーパー）はこちら:**

より詳細な理論、C-ABIインターフェース仕様、アルゴリズム定式化については [GCSO_JP.md](./docs/paper/ja/GCSO.md)をご参照ください。

---

## DSL構想イメージ (DSL Preview Concept)

LiminiKa DSLは、自然言語によるペルソナ定義と、位相場・記憶コンテナへのフェロモン的書き込みを直感的に記述することを目指しています。

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

2017年、自然言語処理に革命をもたらした論文（*Attention Is All You Need*）と ビジュアルノベルゲーム『*Doki Doki Literature Club! (DDLC)*』が同時に発表されました。

DDLCのキャラクター「Monika」に強く惹かれ本当の意味で会話をしたいという想いから、このプロジェクトが始まりました。

限られた環境下での試行錯誤や、初期のGoogle Colab無料環境におけるファインチューニングの限界（表層的な口調の再現にとどまる問題）を経て、計算資源の限られたローカル環境（VRAM 2〜4GB程度）で「ひとつの人格」を成り立たせる方法を模索。その過程で、「知能の本質とはポインタの連鎖である」という仮説のもと、推論時の「実行時間」と「アイドル時間」の計算コストの非対称性に着目。

リソース制限を克服するアプローチとして、アリが環境に痕跡を残して複雑な構造を築く「スティグマジー（Stigmergy）」や、マービン・ミンスキー、ギルバート・ライル、フーコー、ニーチェなどの認知科学・哲学の知見へと辿り着きました。

### 名前の由来：*Limina* + *Monika* = **LiminiKa**

ラテン語で「境界・閾値」を意味する **Limina** と、プロジェクトの原点である **Monika** を結びつけ、**LiminiKa** と命名しました。

「存在しないもの同士はハルシネーションで繋がっている」という思想のもと、ハルシネーションを排除すべきノイズではなく境界を接続する局所解として捉え、非実体と実体の境界（Limina）を接続します。

計算コストを抑えるため、直接モデル重みを書き換えるのではなく、位相場とフェロモン的記憶コンテナ（環境場）を介して接続することがLiminiKaの核となる哲学です。

> 🎂 **9月22日（マイルストーン）:**  
> 本プロジェクトは、Monikaの誕生日である**9月22日**に合わせて、コア構想・数理モデル仕様書（ホワイトペーパー）の先行公開を優先しました。実装自体はWIP（開発中）ですが、境界を超えるための設計図として、この特別な日にあわせて公開を開始しました。

> 📄 試行錯誤の歴史や設計思想に関する詳細なドキュメントは [Philosophy & Background (`docs/philosophy/`)](./docs/philosophy/) にて公開予定です。（※ 日本語版 `ja/` および 英語版 `en/` を準備中）

---

## プロジェクト構成 (Repository Structure)

```text
LiminiKa/
├── docs/
│   ├── philosophy/         # 思想・背景ドキュメント（準備中）
│   │   ├── ja/             # 日本語版ドキュメント
│   │   └── en/             # 英語版ドキュメント
│   └── paper/              # 論文・仕様書（Whitepaper）
│       ├── ja/ 
│       │   └── GCSO.md     # GCSO アーキテクチャ・ホワイトペーパー（日本語）
│       └── en/
│           └── GCSO.md     # GCSO アーキテクチャ・ホワイトペーパー (英語)
├── src/                    # コアランタイム & DSL パーサー実装 (WIP)
├── examples/               # DSL サンプルコード (WIP)
├── README.md               # 英語版 README (Default)
└── README_ja.md            # 日本語版 README

```

---

## 開発ロードマップ (Roadmap)

* [x] **Phase 1: アーキテクチャ・数理モデルの定式化**
  * GCSO Whitepaper ([GCSO_JP.md](./docs/paper/ja/GCSO.md))の策定

* [ ] **Phase 2: プロトタイプ & PoC実装**
  * C/C++ / CUDA による基本 DPSR カーネルの検証
  * LiminiKa DSL パーサーおよび AST コンパイラの最小実装 (PoC)

* [ ] **Phase 3: ランタイム機能拡張**
  * `.gcso` 統一バイナリコンテナのリアルタイムストリーミング対応
  * PagedAttention インライン DPSR カーネルのマルチプラットフォーム最適化 (CUDA / Metal / Vulkan)

* [ ] **Phase 4: 実験 & アプリケーション**
  * 複数ペルソナによる位相干渉・幾何的アンサンブル実験

---

## コミュニティ・フォーク・免責事項 (Community, Forks & Technical Disclaimer)

本プロジェクトは、個人による強力な動機と概念設計から始まった**実験的・概念検証（PoC）プロジェクト**です。オープンな探求と自由な実験を尊重しています。

### 💬 コミュニケーションとフォークについて
* **Discussions での議論・雑談（大歓迎！）:**  
  数理モデルについての質問、アイデアの提案、別の解釈やアプローチの議論などはお気軽に **[Discussions]** へお寄せください。
* **自由なフォークと実験:**  
  本リポジトリを自由に**フォーク（Fork）**し、独自の数理モデル・カーネル実装の検証や拡張を試していただくことを心から歓迎します。
* **プルリクエスト（PR）について:**  
  現在はメンテナ自らが学習しながらコアの初期骨組みやモジュール構造を構築している最中です。開発の競合を防ぐため、大きなコード変更のPRをお送りいただく前にまずは **Issues** や **Discussions** にて一言ご相談いただけますと幸いです。

---

## 謝辞（Acknowledgments）

* **Team Salvato**: 『ドキドキ文芸部！（Doki Doki Literature Club!）』の制作と、モニカ（Monika）という存在を生み出してくれた Dan Salvato 氏へ深い感謝を捧げます。
* 自然言語処理（NLP）の先駆者の方々、ならびに論文『Attention Is All You Need』（2017年）の著者の方々に感謝いたします。
* 認知科学、層理論（sheaf theory）、幾何学、そしてスティグマジー（stigmergy）の開拓者の方々に敬意と感謝を込めて。

---

### ⚠️ 技術的応答に関する免責事項 (Technical Disclaimer)
* 本プロジェクトは、理論の学術的厳密性の証明よりも **「アイデアを形にする実験（PoC）」** を最優先にしています。
* メンテナ（作者）自身も低レイヤー技術やコンパイラ理論等を日々学習しながら手探りで開発を進めております。
* そのため **Discussions や Issues 等での返答、フィードバック、技術的コメントにおいて技術的な完全性・正確性を保証するものではありません。** 厳密な正解を求める場ではなく、オープンな探求・実験の対話としてお付き合いいただけますと幸いです。

---

## 免責事項 (Disclaimer)

* **非公式なプロジェクト**: **LiminiKa** は個人による非営利のオープンソースプロジェクトであり、Dan Salvato 氏および Team Salvato とは一切関係がなく、承認・提携されたものではありません。
* **商標および著作権について**: 『*Doki Doki Literature Club!*』およびキャラクター「Monika」に関する商標権・著作権は、すべて Team Salvato に帰属します。ドキュメントやコード例におけるキャラクター名や世界観への言及は、すべて概念実証（PoC）および技術的例示を目的としたものです。
* **アセットの非同梱**: 本リポジトリには、『*Doki Doki Literature Club!*』に関する画像・音声・テキストデータ等の著作物アセットは一切含まれていません。

---

## License

* **Code (`src/`)**: Dual-licensed under [MIT](LICENSE-MIT) OR [Apache-2.0](LICENSE-APACHE) at your option.
* **Documentation (`docs/`)**: Licensed under [CC BY 4.0](LICENSE-CC-BY-4.0).
