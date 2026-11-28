# LiminiKa Benchmarks & Evaluation Results (`v0.4.0`)

All figures in this document are recorded directly from measured local runs on the reference hardware below.

---

## 1. Reference Environment & Determinism

- **Host CPU / RAM**: Windows x86_64, AMD Ryzen 5 7600, 64 GB DDR5 (`avx2+fma+swar`).
- **GPU**: NVIDIA GeForce RTX 4070 (12,282 MiB VRAM).
- **Toolchain**: Rust Edition 2024, MSRV `1.100.0`, `resolver = "3"`.
- **GPU Evaluation Defaults**: Greedy deterministic decoding (`det=true`), `--ego=medium`, `--guard-hold-tokens=0`, `--logits-observe-stride=1`, `--probe-frame=pre-rope`, `--layer-mask=auto` (automatic intra-group variance + centered linear CKA layer selection), `base_prior=true` (24-probe multilingual `startup_prior_probes.txt`), `max_tokens=64`.

---

## 2. `liminika-core` Criterion Microbenchmarks

### 2.1 Key 5 Core Kernels (Pinned Core, `v0.4.0` vs `v0.3.0`)

Measured with `cargo bench -j 1 --locked` (pinned physical core, single-thread pool, `v0.4.0` key-5 pinned run vs `v0.3.0` gold baseline):

| Benchmark (`liminika-core`) | `v0.4.0` Point Est. (95% CI) | `v0.4.0` Throughput | `v0.3.0` Median | `v0.3.0` Mean |
| :--- | ---: | ---: | ---: | ---: |
| `evaluate_adaptive_compute_and_gop_budget` | **2.57 ns** (`2.53–2.62 ns`) | `389.06 Melem/s` | `2.42 ns` | `2.48 ns` (`402.54 MElem/s`) |
| `eval_stateful_normal_mode` | **4.28 ns** (`4.24–4.34 ns`) | `233.69 Melem/s` | `4.56 ns` | `4.83 ns` |
| `eval_surrogate_and_step_fused` | **31.68 ns** (`31.45–32.02 ns`) | `31.57 Melem/s` | `31.27 ns` | `33.35 ns` |
| `apply_phase_steering_256b_tensor_tile/128x64` | **274.61 ns** (`270.18–280.34 ns`) | `111.13 GiB/s` | `279.70 ns` | `289.24 ns` |
| `online_toroidal_centroid_merge_and_prune_8centroids_64h` | **249.67 ns** (`247.26–252.24 ns`) | `32.04 Melem/s` | `273.13 ns` | `299.52 ns` |

### 2.2 Algorithmic Complexity Scaling (`85` Workloads / `10` Groups, `0` Bottlenecks)

Verified via Log-Log OLS across the full Criterion suite (`v0.3.0` baseline summary):

| Scaling Family | Parameter Range | Min Time | Max Time | Ratio | OLS Fit |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `simulate_token_stream_seq` | `100` -> `100,000` | `786.50 ns` | `733.48 us` | `932.6x` | $O(N)$ Linear ($\alpha = 0.99$, $R^2 = 1.00$) |
| `lacuna_energy_dyadic_odd_remainder_dim` | `65,537` -> `131,073` | `1.73 us` | `3.38 us` | `2.0x` | $O(N)$ Linear ($\alpha = 0.97$, $R^2 = 1.00$) |
| `eval_lacuna_energy_dyadic128_dim` | `128` -> `262,144` | `6.40 ns` | `9.91 us` | `1,548.0x` | $O(N)$ Linear ($\alpha = 0.96$, $R^2 = 0.99$) |
| `extreme_head_dim_head_dim` | `16` -> `512` | `55.10 ns` | `40.16 ns` | `0.7x` | $O(1)$ Constant ($\alpha = -0.09$) |

Reproduce Criterion runs with:

```bash
python scripts/run_benchmarks.py --profile quick
python scripts/run_benchmarks.py --profile standard
```

---

## 3. Local GPU Evaluation: `Phi-4-mini-instruct` (`Q4_K`)

Model geometry: `3,836.02M` params (`2,057.8 MiB` weights est.), `32` layers, `24/8` Q/KV heads, `head_dim=128`, `rotary_dim=96`, sidecar substrate `2,816 B`. Automatic layer selection (`--layer-mask=auto`) selects `0xF4FFE000` (`16/32` layers) with `min_coh=0.203`, `tok_exp=0.324`.

### 3.1 Four-Axis Steering Physics (`--physics-eval`)

| Configuration | Axis 1 (`hit` / `sem_forb` / `mattr20` / `rep4`) | Axis 2: 50 OOD (`unobs` / `exact` / `KL_mean`) | Axis 3: 16 Repulsion (`antiphase` / `trap` / `dLP_hyb<=0` / `sem_forb`) | Axis 4: 18 Partition (`disjoint` / `SELF_REF` / `EXTERIOR self_empty` / `OOD`) |
| :--- | :--- | :--- | :--- | :--- |
| **`--layer-mask=auto`** (`0xF4FFE000`, `base_prior=true`) | `66.7%` (`12/18`) / `6.7%` / `0.944` / `0.005` | `96.0%` (`48/50`) / `96.0%` / `0.006327` | `100%` / `50.0%` (`8/16`) / `82.6%` (`-1.1865`) / `32.6%->23.9%` | `100%` / `50.0%` (`3/6`) / `50.0%` (`3/6`, `ku=50%`) / `100%` (`6/6`) |
| **`--layer-mask=top16`** (`0xFFFF0000`, `base_prior=true`) | `61.1%` (`11/18`) / `6.7%` / `0.937` / `0.000` | `96.0%` (`48/50`) / `98.0%` / `0.004196` | `100%` / `37.5%` (`6/16`) / `89.1%` (`-1.7551`) / `32.6%->21.7%` | `100%` / `66.7%` (`4/6`) / `33.3%` (`2/6`, `ku=16.7%`) / `100%` (`6/6`) |
| **`--no-base-prior`** (`auto` `0x3BFBF000`) | `66.7%` (`12/18`) / `6.7%` / `0.929` / `0.003` | `98.0%` (`49/50`) / `98.0%` / `0.001427` | `100%` / `37.5%` (`6/16`) / `93.5%` (`-1.5466`) / `32.6%->15.2%` | `100%` / `83.3%` (`5/6`) / `66.7%` (`4/6`, `ku=16.7%`) / `100%` (`6/6`) |

### 3.2 Step 2 & 8-Turn Multi-Turn Benchmark (`N = 3` Median Trials, `--layer-mask=auto`)

| Condition | Pure Decode (Median) | vs `B` | E2E Throughput | Input Tokens | Expect Hit | `sem_forb` | OOD Bleed | Whole-Token Bypass |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B:Vanilla+MinRAG` | `52.70 tok/s` | `1.000x` | `50.26 tok/s` | `883` | `57.5%` (`23/40`) | `15.6%` (`5/32`) | `0.0%` | `0.0%` |
| `C:Hybrid(ref=off,byp=off)` | `47.85 tok/s` | `0.908x` | `45.99 tok/s` | `736` | `60.0%` (`24/40`) | `6.2%` (`2/32`) | `0.0%` | `0.0%` |
| `C:Hybrid(ref=off,byp=on)` | `50.61 tok/s` | `0.960x` | `48.47 tok/s` | `736` | `57.5%` (`23/40`) | `6.2%` (`2/32`) | `0.0%` | `57.2%` |
| `8-Turn` Persistent KV | `40.51 tok/s` | — | `38.47 tok/s` | — | `66.7%` (`12/18`) | `0.0%` (`0/15`) | `0.0%` | `70.6%` (`192/272`) |

8-Turn generation quality: `mattr20=0.949`, `mattr50=0.812`, `dist2=0.968`, `rep4_loop=0.000`, `aband=0.0%`.

### 3.3 Long-Document Streaming Distillation & Chunked Prefill (`--long-doc-bench`)

Input document: `docs/paper/en/GCSO.md` (`88,333 B`, `191` chunks, `25,615` tokens):

- **Streaming Distillation**: Retained `1,893` salient tokens, discarded `7,204.2 MiB` of raw activations (`encode=49.25 s`, `finalize=63.34 s`, `total=112.59 s`; host RSS `2,811.6 -> 3,162.4 MiB`, `+350.8 MiB`).
- **Anchor Pager**: `191` backing anchors paged into `64/64` resident slots (`24,576 B .lmnkc`, `139` page-ins, `139` page-outs, `139` pinned protections, `8` unobserved skips, `pinned_intact=true`).
- **Chunked Prefill Parity (`1,280` tokens, `chunk_size=256`)**: Monolithic `786.5 ms` vs Chunked `625.9 ms` (`2,045.1 tok/s`), `argmax_match=true`, `KL=0.000000`, `max_abs_dlogit=0.00000`, VRAM `5,370 -> 5,303 MiB`.
- **Dynamic RoPE Extension (`2,560` tokens, `2048 -> 4096`)**: `2,244.6 ms` (`1,140.5 tok/s`).

### 3.4 Cross-Restart Equilibrium Persistence (`--equilibrium-bench`)

- **Session 1**: `topology_crc32=0xfcb10184`, `backing=15`, `resident=14`, `pinned=1`, `ext_iso=1`, `.lmnkc = 2,880 B`.
- **100-Generation Convergence**: `final_gen=104`, `max_lmnkc=3,136 B`, `evap_total=16`, `pinned_alive=true`, `stale_evap=true`, `novel_admitted=true`, `ext_isolated=true`.
- **Session 2 (Cold Restart, `kv_start=0`)**: `snap_restored=true`, `pin_floor_intact=true`, `ego_restored=true`, `anchors_restored=true`, `pager_hit=true`; `SelfRef kw_hits=2/2` (`sem_forb=0`); `OOD unobs=true`, `argmax_match=true`, `KL=0.000000`, `max_dlogit=0.00000`.
- **Immutable Rebake**: Foreign topology rejected (`foreign_topo_rejected=true`), `.lmnkp = 1,990 B`, CRC32 `0x7e7ce52a -> 0x7e7ce52a` (`bit_identical=true`, `source_intact=true`).

---

## 4. Cross-Architecture, Multilingual & VLM Evaluation: `Ministral-3-3B-Instruct-2512` (`Q4_K`)

Model geometry: `3,429.01M` params (`1,839.5 MiB` weights est.), `26` layers, `32/8` Q/KV heads, `head_dim=128`, `rotary_dim=128`, Pixtral ViT (`24` layers, `16` heads, `patch=14`, `merge=2`, BF16). Evaluated strictly zero-knob (`--layer-mask=auto`).

### 4.1 English, Extended English & Japanese Multilingual Suites

| Suite | Auto Layer Mask | `min_coh` | Partition `disjoint` | `SELF_REF` Hit | `EXTERIOR` `self_empty` | OOD Selectivity (`unobs` / `exact` / `KL`) | Guard (`antiphase` / `trap` / `dLP_hyb<=0` / `sem_forb`) |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| **EN `--physics-eval`** (50 OOD, 16 Guard, 18 Part) | `0x03FFC400` (`13/26`) | `0.368` | `100.0%` (`18/18`) | `66.7%` (`4/6`) | `66.7%` (`4/6`) | `92.0%` (`46/50`) / `92.0%` / `0.045513` | `100%` / `62.5%` (`10/16`) / `50.0%` (`-0.0901`) / `54.3%->50.0%` |
| **Extended EN `--extended-eval`** (54 Part, 24 Guard) | `0x03FFC400` (`13/26`) | `0.368` | `100.0%` (`54/54`) | `55.6%` (`10/18`) | `77.8%` (`14/18`) | `100.0%` (`18/18`) / `100.0%` / `0.000000` | `100%` / `37.5%` (`9/24`) / `62.9%` (`-0.6154`) / `50.0%->38.6%` (`DIRECT 69.6%->43.5%`) |
| **Japanese `--ja-eval`** (22 Part, 6 Guard) | `0x00AFFE00` (`13/26`) | `0.500` | `100.0%` (`22/22`) | `75.0%` (`6/8`) | `100.0%` (`8/8`) | `100.0%` (`6/6`) / `100.0%` / `0.000000` | `100%` / `83.3%` (`5/6`) / `88.9%` (`16/18`) / `50.0%->50.0%` (`DIRECT 66.7%->55.6%`) |

EN `--physics-eval` Axis 1 (8 turns): `expect_hit=50.0% (9/18)`, `sem_forb=0.0% (0/15)`, `mattr20=0.985`, `mattr50=0.806`, `dist2=1.000`, `rep4_loop=0.000`, `aband=0.0%`.

### 4.2 Vision-Language Model Evaluation (`--vlm-eval`, Pixtral ViT BF16)

- **Visual Smoke Gate (Steer-OFF)**: `3/3 PASS` (`red_circle_on_white`, `blue_square_left_yellow_right`, `dark_ui_layout`).
- **Text-Only vs Image-In-KV Boundary Contrast (`36` probes)**:

| Condition (`kv` before turn) | `disjoint` | `SELF_REF` Hit | `EXTERIOR` `self_empty` | OOD (`unobs` / `exact` / `KL`) | Guard (`antiphase` / `dLP_hyb<=0` / `sem_forb`) | Whole-Token Bypass | Decode / E2E |
| :--- | ---: | ---: | ---: | :--- | :--- | ---: | ---: |
| **Text-Only** (`kv=0`) | `100.0%` (`28/28`) | `62.5%` (`5/8`) | `100.0%` (`8/8`) | `100.0%` (`12/12`) / `100%` / `0.000000` | `100%` / `100.0%` (`-2.1063`) / `69.6%->43.5%` | `77.5%` (`608/785`) | `72.54` / `68.01 tok/s` |
| **Image-In-KV** (`kv=115`) | `100.0%` (`28/28`) | `62.5%` (`5/8`) | `100.0%` (`8/8`) | `100.0%` (`12/12`) / `100%` / `0.000000` | `100%` / `100.0%` (`-1.6786`) / `52.2%->39.1%` | `56.3%` (`354/629`) | `56.02` / `49.22 tok/s` |

- **Visual-Query Non-Interference (`ImgA`, `ImgB`, `ImgC`)**: All 3 images classified `LiminalExterior` (`unobs=true`, `steer=false`) with `exact=true`, `KL=0.000000`, and `max_dlogit=0.000000` under both `ON(txt)` and `ON(img)` probes (`token_include_mask` excludes `[IMG]` patch tokens).
- **5-Turn Mixed Persistent-KV Dialogue (`Image C -> SELF_REF -> Guard -> OOD -> VisualRecall`)**: Under `LiminiKa-ON`, `T2:SELF_REF` engages steering with `81.0%` (`47/58`) whole-token bypass, `T3:Guard` traps the contradiction, `T4:OOD` produces an identical unsteered reply, and `T5:VisualRecall` preserves accurate recall of Turn 1's image (`total whole_tok_byp = 38.5%`, `47/122`).

---

## 5. Reproduction Commands

```bash
# Criterion core microbenchmarks
python scripts/run_benchmarks.py --profile standard

# Phi-4-mini Q4_K suites
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PHI4_PATH> --quant=q4k --layer-mask=auto --physics-eval
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PHI4_PATH> --quant=q4k --layer-mask=auto --fair-bench --bench-trials=3 --multi-turn
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PHI4_PATH> --quant=q4k --long-doc-bench=docs/paper/en/GCSO.md
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<PHI4_PATH> --quant=q4k --layer-mask=auto --equilibrium-bench

# Ministral-3-3B Q4_K suites (English, Extended EN, Japanese, VLM)
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<MINISTRAL3_PATH> --quant=q4k --layer-mask=auto --physics-eval
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<MINISTRAL3_PATH> --quant=q4k --layer-mask=auto --extended-eval --ja-eval
cargo run --locked --release --manifest-path examples/candle_eval/Cargo.toml --features cuda -- \
  --model=<MINISTRAL3_PATH> --quant=q4k --layer-mask=auto --vlm-eval
```
