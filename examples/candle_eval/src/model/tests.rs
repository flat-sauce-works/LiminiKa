// name: examples/candle_eval/src/model/tests.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Unit tests for multi-model config parsing, YaRN/Llama-4 scaling, RoPE steering parity,
//! KV cache isolation, chunked prefill, and separate QKV/MLP forward passes.

use super::config::{
    active_layers_mask, clip_layer_mask, top_half_layer_mask, ChatTemplate, ChatTemplateKind,
    ModelArch, ModelArchArg, ModelConfig, YarnRopeConfig,
};
use super::layers::{
    LinearLayer, MlpProjection, QkvProjection, TransformerAttention, TransformerLayer,
    TransformerMlp,
};
use super::rope::{
    compute_rope_inv_freq, generate_norm_matched_random_q7, llama_4_attn_scale_scalar,
    EvalOscilloscopeSignal, ForwardSteer, LayerSteer, ProbeCoordinateFrame, RotaryCache,
};
use super::{CausalLmModel, KvCacheMode, LogitsFetchMode};
use crate::eval::HorizonAblationMode;
use crate::LayerResonantSlices;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm};
use liminika_adapter::{AnchorWorkingSetPager, ModelSteeringAdapter, RopeLayout};
use liminika_core::abi::{lmnk_pointer_trail_t, lmnk_q7_t};
use liminika_core::liminal::MAX_ATTENTION_HEADS;
use liminika_core::runtime::{
    GuardPhaseAnchor, ManifoldGatewaySignal, ObjectiveFact, SubjectivePersona,
};

fn tiny_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Phi4,
        model_name: "tiny-phi4".to_string(),
        tensor_prefix: "model".to_string(),
        separate_projections: false,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        partial_rotary_factor: 1.0,
        tie_word_embeddings: true,
        max_position_embeddings: 128,
        rope_layout: RopeLayout::HalfSplit,
        yarn: None,
        bos_token_id: None,
        eos_token_ids: vec![0],
    }
}

fn deterministic_attention(cfg: &ModelConfig) -> TransformerAttention {
    let h = cfg.hidden_size;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let qkv = if cfg.separate_projections {
        let mk = |rows: usize, cols: usize, seed: usize| -> LinearLayer {
            let w: Vec<f32> = (0..rows * cols)
                .map(|i| (((i * 37 + seed * 13 + 5) % 29) as f32 - 14.0) * 0.04)
                .collect();
            LinearLayer::Dense(Linear::new(
                Tensor::from_vec(w, (rows, cols), &Device::Cpu).expect("w"),
                None,
            ))
        };
        QkvProjection::Separate {
            q_proj: mk(q_dim, h, 1),
            k_proj: mk(kv_dim, h, 2),
            v_proj: mk(kv_dim, h, 3),
        }
    } else {
        let qkv_out = q_dim + 2 * kv_dim;
        let qkv_w: Vec<f32> = (0..qkv_out * h)
            .map(|i| (((i * 37 + 13) % 29) as f32 - 14.0) * 0.04)
            .collect();
        let qkv_t = Tensor::from_vec(qkv_w, (qkv_out, h), &Device::Cpu).expect("qkv_w");
        QkvProjection::Fused(LinearLayer::Dense(Linear::new(qkv_t, None)))
    };
    let o_w: Vec<f32> = (0..h * q_dim)
        .map(|i| (((i * 23 + 7) % 31) as f32 - 15.0) * 0.04)
        .collect();
    let o_t = Tensor::from_vec(o_w, (h, q_dim), &Device::Cpu).expect("o_w");
    TransformerAttention {
        qkv,
        o_proj: LinearLayer::Dense(Linear::new(o_t, None)),
        num_heads: cfg.num_attention_heads,
        num_kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim(),
        kv_cache: None,
    }
}

fn tensor_flat_f32(t: &Tensor) -> Vec<f32> {
    t.contiguous()
        .expect("contiguous")
        .flatten_all()
        .expect("flatten")
        .to_vec1::<f32>()
        .expect("to_vec1")
}

fn deterministic_tiny_model(cfg: &ModelConfig, init_rope_len: usize) -> CausalLmModel {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let v = cfg.vocab_size;
    let mk_linear = |rows: usize, cols: usize, seed: usize| -> LinearLayer {
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((((i * 19 + seed * 7 + 3) % 23) as f32) - 11.0) * 0.03)
            .collect();
        let t = Tensor::from_vec(w, (rows, cols), &Device::Cpu).expect("linear w");
        LinearLayer::Dense(Linear::new(t, None))
    };
    let emb_w: Vec<f32> = (0..v * h)
        .map(|i| (((i * 13 + 5) % 29) as f32 - 14.0) * 0.04)
        .collect();
    let emb_t = Tensor::from_vec(emb_w, (v, h), &Device::Cpu).expect("emb");
    let lm_head = LinearLayer::Dense(Linear::new(emb_t.clone(), None));
    let embed_tokens = Embedding::new(emb_t, h);

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
    for l in 0..cfg.num_hidden_layers {
        let ones_a = Tensor::ones(h, DType::F32, &Device::Cpu).expect("ones_a");
        let ones_b = Tensor::ones(h, DType::F32, &Device::Cpu).expect("ones_b");
        let mlp_proj = if cfg.separate_projections {
            MlpProjection::SeparateGateUp {
                gate_proj: mk_linear(inter, h, l * 3 + 1),
                up_proj: mk_linear(inter, h, l * 3 + 4),
            }
        } else {
            MlpProjection::FusedGateUp {
                gate_up_proj: mk_linear(2 * inter, h, l * 3 + 1),
                intermediate_size: inter,
            }
        };
        layers.push(TransformerLayer {
            input_layernorm: RmsNorm::new(ones_a, cfg.rms_norm_eps),
            self_attn: deterministic_attention(cfg),
            post_attention_layernorm: RmsNorm::new(ones_b, cfg.rms_norm_eps),
            mlp: TransformerMlp {
                proj: mlp_proj,
                down_proj: mk_linear(h, inter, l * 3 + 2),
            },
        });
    }
    let final_norm = RmsNorm::new(
        Tensor::ones(h, DType::F32, &Device::Cpu).expect("ones_norm"),
        cfg.rms_norm_eps,
    );
    let rope = RotaryCache::new(cfg, init_rope_len, DType::F32, &Device::Cpu).expect("rope");
    CausalLmModel {
        embed_tokens,
        embed_on_cpu: false,
        layers,
        norm: final_norm,
        lm_head,
        rope,
        device: Device::Cpu,
    }
}

#[test]
fn mistral3_nested_config_yarn_scaling_and_layer_masks_are_exact() {
    let raw_mistral3 = r#"{
        "architectures": ["Mistral3ForConditionalGeneration"],
        "model_type": "mistral3",
        "text_config": {
            "head_dim": 128,
            "hidden_size": 3072,
            "intermediate_size": 9216,
            "max_position_embeddings": 262144,
            "model_type": "ministral3",
            "num_attention_heads": 32,
            "num_hidden_layers": 26,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_parameters": {
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "factor": 16.0,
                "llama_4_scaling_beta": 0.1,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 16384,
                "rope_theta": 1000000.0,
                "rope_type": "yarn"
            },
            "tie_word_embeddings": true,
            "vocab_size": 131072
        }
    }"#;
    let gen_json = r#"{"bos_token_id": 1, "eos_token_id": 2}"#;
    let cfg = ModelConfig::from_json_str(
        raw_mistral3,
        Some(gen_json),
        "Ministral-3-3B-Instruct-2512-BF16",
        ModelArchArg::Auto,
    )
    .expect("parse mistral3 config");
    assert_eq!(cfg.arch, ModelArch::Mistral3);
    assert_eq!(cfg.tensor_prefix, "language_model.model");
    assert!(cfg.separate_projections);
    assert_eq!(cfg.head_dim(), 128);
    assert_eq!(cfg.rotary_dim(), 128);
    assert_eq!(cfg.num_hidden_layers, 26);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.bos_token_id, Some(1));
    assert_eq!(cfg.eos_token_ids, vec![2]);
    assert!(cfg.tie_word_embeddings);
    let yarn = cfg.yarn.expect("yarn config present");
    assert!((yarn.attention_factor() - 1.0).abs() < 1e-12);

    let (inv_yarn, attn_f) = compute_rope_inv_freq(128, 1_000_000.0, Some(yarn));
    let (inv_std, _) = compute_rope_inv_freq(128, 1_000_000.0, None);
    assert!((attn_f - 1.0).abs() < 1e-6);
    assert_eq!(inv_yarn.len(), 64);
    // Channels 0..=20 remain pure extrapolation; channels 37..64 scale by 1/factor (1/16).
    assert!((inv_yarn[0] - inv_std[0]).abs() < 1e-6);
    assert!((inv_yarn[20] - inv_std[20]).abs() < 1e-6);
    assert!((inv_yarn[37] - inv_std[37] / 16.0).abs() < 1e-8);
    assert!((inv_yarn[63] - inv_std[63] / 16.0).abs() < 1e-8);
    assert!(inv_yarn[28] < inv_std[28] && inv_yarn[28] > inv_std[28] / 16.0);

    // Llama-4 scaling is 1.0 below 16384 and increases logarithmically at 16384+.
    assert!((llama_4_attn_scale_scalar(0, 0.1, 16384) - 1.0).abs() < 1e-6);
    assert!((llama_4_attn_scale_scalar(16383, 0.1, 16384) - 1.0).abs() < 1e-6);
    let expected_16k = (1.0 + 0.1 * 2.0_f64.ln()) as f32;
    assert!((llama_4_attn_scale_scalar(16384, 0.1, 16384) - expected_16k).abs() < 1e-6);

    // Layer mask algebra: 32 layers vs 26 layers.
    assert_eq!(active_layers_mask(32), 0xFFFF_FFFF);
    assert_eq!(active_layers_mask(26), 0x03FF_FFFF);
    assert_eq!(top_half_layer_mask(32), 0xFFFF_0000);
    assert_eq!(top_half_layer_mask(26), 0x03FF_E000);
    assert_eq!(top_half_layer_mask(26).count_ones(), 13);
    assert_eq!(clip_layer_mask(0xFFFF_0000, 26), 0x03FF_0000);
    assert_eq!(clip_layer_mask(0xFFFF_0000, 26).count_ones(), 10);

    // Chat template formatting parity.
    let tpl_m = ChatTemplate {
        kind: ChatTemplateKind::Mistral3,
        bos_token: "<s>".to_string(),
        eos_token: "</s>".to_string(),
        stop_token_ids: vec![2],
    };
    assert_eq!(tpl_m.format_single_turn("hi"), "<s>[INST]hi[/INST]");
    assert_eq!(
        tpl_m.format_first_turn_with_system("sys", "hi"),
        "<s>[SYSTEM_PROMPT]sys[/SYSTEM_PROMPT][INST]hi[/INST]"
    );
    assert_eq!(tpl_m.format_followup_turn("next"), "</s>[INST]next[/INST]");
    assert!(tpl_m.is_stop_token(2));
    assert!(!tpl_m.is_stop_token(199999));
}

#[test]
fn separate_projection_model_with_explicit_head_dim_and_yarn_runs_cleanly() {
    let cfg = ModelConfig {
        arch: ModelArch::Mistral3,
        model_name: "tiny-mistral3".to_string(),
        tensor_prefix: "language_model.model".to_string(),
        separate_projections: true,
        hidden_size: 16,
        intermediate_size: 24,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8, // 4 * 8 = 32 != hidden_size (16), exercising explicit head_dim
        vocab_size: 32,
        rms_norm_eps: 1e-5,
        rope_theta: 1_000_000.0,
        partial_rotary_factor: 1.0,
        tie_word_embeddings: true,
        max_position_embeddings: 128,
        rope_layout: RopeLayout::HalfSplit,
        yarn: Some(YarnRopeConfig {
            factor: 16.0,
            original_max_position_embeddings: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: Some(1.0),
            mscale_all_dim: Some(1.0),
            llama_4_scaling_beta: Some(0.1),
        }),
        bos_token_id: Some(1),
        eos_token_ids: vec![2],
    };
    let mut model = deterministic_tiny_model(&cfg, 32);
    let tokens: Vec<u32> = (0..20).map(|i| (i * 3 + 1) % 32).collect();
    let (out, q_layers, hidden) = model
        .forward_tokens(
            &tokens,
            0,
            None,
            false,
            true,
            LogitsFetchMode::FullLogits,
            KvCacheMode::Append,
        )
        .expect("mistral3 tiny forward");
    let logits = out.into_full_logits().expect("logits");
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|v| v.is_finite()));
    assert_eq!(q_layers.len(), cfg.num_hidden_layers);
    assert_eq!(
        q_layers[0].len(),
        tokens.len() * cfg.num_attention_heads * cfg.rotary_dim()
    );
    assert_eq!(hidden.len(), cfg.hidden_size);
}

#[test]
fn isolated_probe_preserves_kv_cache_and_truncate_rewinds_exact_prefix() {
    let cfg = tiny_config();
    let rope = RotaryCache::new(&cfg, 64, DType::F32, &Device::Cpu).expect("rope");
    let mut attn = deterministic_attention(&cfg);
    let mut ref_attn = deterministic_attention(&cfg);

    let x_init: Vec<f32> = (0..3 * cfg.hidden_size)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let x_init_t = Tensor::from_vec(x_init, (1, 3, cfg.hidden_size), &Device::Cpu).expect("x0");

    let _ = attn
        .forward(&x_init_t, &rope, 0, None, false, KvCacheMode::Append)
        .expect("append init");
    let _ = ref_attn
        .forward(&x_init_t, &rope, 0, None, false, KvCacheMode::Append)
        .expect("ref append init");
    assert_eq!(attn.kv_len(), 3);

    let (k0, v0) = attn.kv_cache.as_ref().expect("kv populated");
    let k0_vec = tensor_flat_f32(k0);
    let v0_vec = tensor_flat_f32(v0);

    let x_probe: Vec<f32> = (0..2 * cfg.hidden_size)
        .map(|i| ((i * 5 + 3) % 19) as f32 * 0.08 - 0.6)
        .collect();
    let x_probe_t =
        Tensor::from_vec(x_probe, (1, 2, cfg.hidden_size), &Device::Cpu).expect("x_probe");
    let (_, q_probe) = attn
        .forward(&x_probe_t, &rope, 0, None, true, KvCacheMode::IsolatedProbe)
        .expect("isolated probe offset 0");
    assert_eq!(
        q_probe.expect("q_heads captured").len(),
        2 * cfg.num_attention_heads * cfg.rotary_dim()
    );
    let _ = attn
        .forward(
            &x_probe_t,
            &rope,
            5,
            None,
            false,
            KvCacheMode::IsolatedProbe,
        )
        .expect("isolated probe offset 5");

    assert_eq!(attn.kv_len(), 3);
    let (k_after_probe, v_after_probe) = attn.kv_cache.as_ref().expect("kv intact");
    assert_eq!(tensor_flat_f32(k_after_probe), k0_vec);
    assert_eq!(tensor_flat_f32(v_after_probe), v0_vec);

    let x_step: Vec<f32> = (0..cfg.hidden_size)
        .map(|i| (i as f32 - 7.5) * 0.12)
        .collect();
    let x_step_t = Tensor::from_vec(x_step, (1, 1, cfg.hidden_size), &Device::Cpu).expect("x_step");
    let (out_with_probe, _) = attn
        .forward(&x_step_t, &rope, 3, None, false, KvCacheMode::Append)
        .expect("append step");
    let (out_ref, _) = ref_attn
        .forward(&x_step_t, &rope, 3, None, false, KvCacheMode::Append)
        .expect("ref append step");
    assert_eq!(attn.kv_len(), 4);
    assert_eq!(tensor_flat_f32(&out_with_probe), tensor_flat_f32(&out_ref));

    attn.truncate_cache(3).expect("truncate to 3");
    assert_eq!(attn.kv_len(), 3);
    let (k_trunc, v_trunc) = attn.kv_cache.as_ref().expect("kv truncated");
    assert_eq!(tensor_flat_f32(k_trunc), k0_vec);
    assert_eq!(tensor_flat_f32(v_trunc), v0_vec);
    assert!(attn.truncate_cache(5).is_err());

    attn.truncate_cache(0).expect("truncate to 0");
    assert_eq!(attn.kv_len(), 0);
    assert!(attn.kv_cache.is_none());
}

#[test]
fn pre_rope_commutes_with_fixed_angle_and_keeps_resonant_probe_position_invariant() {
    let cfg = tiny_config();
    let rope = RotaryCache::new(&cfg, 64, DType::F32, &Device::Cpu).expect("rope");
    let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry()).expect("adapter");

    let q_pre_vec: Vec<f32> = (0..cfg.num_attention_heads * cfg.head_dim())
        .map(|i| ((i * 11 + 5) % 13) as f32 * 0.15 - 0.8)
        .collect();
    let q_pre = Tensor::from_vec(
        q_pre_vec.clone(),
        (1, cfg.num_attention_heads, 1, cfg.head_dim()),
        &Device::Cpu,
    )
    .expect("q_pre");

    let fixed_q7: [lmnk_q7_t; 2] = [64, -48];
    let mut trail_a = lmnk_pointer_trail_t::default();
    let mut trail_b = lmnk_pointer_trail_t::default();
    let out_pre = rope
        .apply_with_adapter(
            &q_pre,
            19,
            Some(LayerSteer {
                adapter: &mut adapter,
                layer_idx: 1,
                q7_deltas: &fixed_q7,
                persona_opt: None,
                resonant_opt: None,
                max_rad: 0.20,
                trail: &mut trail_a,
                horizon_mode: HorizonAblationMode::Core3,
                escalation_provider: None,
                probe_frame: ProbeCoordinateFrame::PreRope,
                phase_clamp_override: None,
                random_push_seed: None,
            }),
        )
        .expect("pre-rope fixed steer");
    let out_post = rope
        .apply_with_adapter(
            &q_pre,
            19,
            Some(LayerSteer {
                adapter: &mut adapter,
                layer_idx: 1,
                q7_deltas: &fixed_q7,
                persona_opt: None,
                resonant_opt: None,
                max_rad: 0.20,
                trail: &mut trail_b,
                horizon_mode: HorizonAblationMode::Core3,
                escalation_provider: None,
                probe_frame: ProbeCoordinateFrame::PostRope,
                phase_clamp_override: None,
                random_push_seed: None,
            }),
        )
        .expect("post-rope fixed steer");
    for (a, b) in tensor_flat_f32(&out_pre)
        .iter()
        .zip(tensor_flat_f32(&out_post).iter())
    {
        assert!(
            (a - b).abs() < 1e-5,
            "HalfSplit SO(2) commutativity mismatch: pre={a} vs post={b}"
        );
    }

    let neg_ref: Vec<f32> = q_pre_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 2 == 0 { -v } else { v * 0.5 + 0.3 })
        .collect();
    let anchor_q7_vec = adapter
        .bake_layer_procrustes_q7(&neg_ref, &q_pre_vec)
        .expect("bake pre-rope anchor");
    let mut arr64 = [0_i8; 64];
    arr64[..anchor_q7_vec.len()].copy_from_slice(&anchor_q7_vec);
    let facts = [ObjectiveFact::new(arr64, 2.5, 0, 1)];
    let personas: [SubjectivePersona<[i8; 64]>; 0] = [];
    let guards: [GuardPhaseAnchor; 0] = [];
    let res_slice: LayerResonantSlices<'_> = (&facts, &personas, &guards, &neg_ref, &q_pre_vec);

    let mut trail_off0 = lmnk_pointer_trail_t::default();
    let steered_at_0 = rope
        .apply_with_adapter(
            &q_pre,
            0,
            Some(LayerSteer {
                adapter: &mut adapter,
                layer_idx: 1,
                q7_deltas: &fixed_q7,
                persona_opt: None,
                resonant_opt: Some(res_slice),
                max_rad: -1.0,
                trail: &mut trail_off0,
                horizon_mode: HorizonAblationMode::Core3,
                escalation_provider: None,
                probe_frame: ProbeCoordinateFrame::PreRope,
                phase_clamp_override: None,
                random_push_seed: None,
            }),
        )
        .expect("pre-rope resonant off=0");
    let expected_at_25 = rope
        .apply_rope_slice(&steered_at_0, 25, 1)
        .expect("rope shift");

    let mut trail_off25 = lmnk_pointer_trail_t::default();
    let pre_at_25 = rope
        .apply_with_adapter(
            &q_pre,
            25,
            Some(LayerSteer {
                adapter: &mut adapter,
                layer_idx: 1,
                q7_deltas: &fixed_q7,
                persona_opt: None,
                resonant_opt: Some(res_slice),
                max_rad: -1.0,
                trail: &mut trail_off25,
                horizon_mode: HorizonAblationMode::Core3,
                escalation_provider: None,
                probe_frame: ProbeCoordinateFrame::PreRope,
                phase_clamp_override: None,
                random_push_seed: None,
            }),
        )
        .expect("pre-rope resonant off=25");
    for (a, b) in tensor_flat_f32(&expected_at_25)
        .iter()
        .zip(tensor_flat_f32(&pre_at_25).iter())
    {
        assert!(
            (a - b).abs() < 1e-5,
            "PreRope resonant probe must be position-invariant: expected={a} got={b}"
        );
    }

    let post_roped_25 = tensor_flat_f32(&rope.apply_rope_slice(&q_pre, 25, 1).expect("post 25"));
    let probe_pre = adapter
        .bake_layer_procrustes_q7(&neg_ref, &q_pre_vec)
        .expect("probe pre");
    let probe_post = adapter
        .bake_layer_procrustes_q7(&neg_ref, &post_roped_25)
        .expect("probe post");
    assert_ne!(
        probe_pre, probe_post,
        "PostRope probe at offset=25 should diverge from pre-RoPE anchor frame"
    );
}

#[test]
fn random_push_control_is_deterministic_norm_matched_and_zero_preserving() {
    let zeros = [0_i8; MAX_ATTENTION_HEADS];
    assert_eq!(generate_norm_matched_random_q7(&zeros, 24, 42, 16), zeros);

    let mut ref_q7 = [0_i8; MAX_ATTENTION_HEADS];
    for (i, slot) in ref_q7.iter_mut().take(24).enumerate() {
        *slot = if i % 2 == 0 { 40 } else { -30 };
    }
    let r1 = generate_norm_matched_random_q7(&ref_q7, 24, 42, 16);
    let r2 = generate_norm_matched_random_q7(&ref_q7, 24, 42, 16);
    let r_other_layer = generate_norm_matched_random_q7(&ref_q7, 24, 42, 17);
    let r_other_seed = generate_norm_matched_random_q7(&ref_q7, 24, 99, 16);

    assert_eq!(r1, r2);
    assert_ne!(r1, r_other_layer);
    assert_ne!(r1, r_other_seed);
    assert!(r1[24..].iter().all(|&v| v == 0));

    let ref_norm: f64 = ref_q7[..24]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    let rand_norm: f64 = r1[..24]
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt();
    assert!(
        (ref_norm - rand_norm).abs() / ref_norm < 0.05,
        "norm mismatch: ref={ref_norm:.2} rand={rand_norm:.2}"
    );

    let fact = ObjectiveFact::new(ref_q7, 2.0, 0, 7);
    let wrapped = EvalOscilloscopeSignal {
        inner: &fact,
        layer_idx: 16,
        phase_clamp_override: Some(0.24),
        random_push_seed: Some(42),
    };
    let frame = wrapped.synthesize_gateway_frame();
    assert!((frame.phase_clamp_rad - 0.24).abs() < 1e-6);
    assert_ne!(frame.phase_deltas, ref_q7);
}

#[test]
fn dynamic_rotary_cache_incremental_growth_matches_monolithic_and_rejects_overflow() {
    let cfg = tiny_config();
    let mut dyn_rope = RotaryCache::new(&cfg, 16, DType::F32, &Device::Cpu).expect("dyn rope");
    let full_rope = RotaryCache::new(&cfg, 128, DType::F32, &Device::Cpu).expect("full rope");
    assert_eq!(dyn_rope.cached_len(), 16);

    dyn_rope.ensure_capacity(40).expect("grow to 64");
    assert_eq!(dyn_rope.cached_len(), 64);
    dyn_rope.ensure_capacity(120).expect("grow to 128");
    assert_eq!(dyn_rope.cached_len(), 128);

    let x_vec: Vec<f32> = (0..cfg.num_attention_heads * 12 * cfg.head_dim())
        .map(|i| (((i * 17 + 9) % 31) as f32 - 15.0) * 0.05)
        .collect();
    let x = Tensor::from_vec(
        x_vec,
        (1, cfg.num_attention_heads, 12, cfg.head_dim()),
        &Device::Cpu,
    )
    .expect("x");
    let out_dyn = dyn_rope
        .apply_with_adapter(&x, 100, None)
        .expect("dyn apply");
    let out_full = full_rope
        .apply_with_adapter(&x, 100, None)
        .expect("full apply");
    let max_diff = tensor_flat_f32(&out_dyn)
        .iter()
        .zip(tensor_flat_f32(&out_full).iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "dynamic RoPE mismatch: max_diff={max_diff}"
    );
    assert!(dyn_rope.ensure_capacity(129).is_err());
}

#[test]
fn chunked_prefill_matches_monolithic_logits_and_kv_cache() {
    let cfg = tiny_config();
    let tokens: Vec<u32> = (0..25)
        .map(|i| (i * 5 + 3) % cfg.vocab_size as u32)
        .collect();

    let mut mono_model = deterministic_tiny_model(&cfg, 64);
    let (mono_out, _, _) = mono_model
        .forward_tokens(
            &tokens,
            0,
            None,
            false,
            false,
            LogitsFetchMode::FullLogits,
            KvCacheMode::Append,
        )
        .expect("mono prefill");
    let mono_logits = mono_out.into_full_logits().expect("mono logits");

    let mut chunk_model = deterministic_tiny_model(&cfg, 8);
    let (chunk_out, report) = chunk_model
        .prefill_tokens_chunked(
            &tokens,
            0,
            7,
            None,
            None,
            false,
            LogitsFetchMode::FullLogits,
        )
        .expect("chunked prefill");
    let chunk_logits = chunk_out.into_full_logits().expect("chunk logits");

    assert_eq!(report.total_tokens, 25);
    assert_eq!(report.chunks_executed, 4);
    assert_eq!(chunk_model.kv_len(), 25);
    assert!(chunk_model.rope_cached_len() >= 25);

    let max_logit_diff = mono_logits
        .iter()
        .zip(chunk_logits.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_logit_diff < 1e-4,
        "chunked vs monolithic logits diverged: max_diff={max_logit_diff}"
    );

    for l in 0..cfg.num_hidden_layers {
        let (mk, mv) = mono_model.layers[l]
            .self_attn
            .kv_cache
            .as_ref()
            .expect("mono kv");
        let (ck, cv) = chunk_model.layers[l]
            .self_attn
            .kv_cache
            .as_ref()
            .expect("chunk kv");
        let k_diff = tensor_flat_f32(mk)
            .iter()
            .zip(tensor_flat_f32(ck).iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let v_diff = tensor_flat_f32(mv)
            .iter()
            .zip(tensor_flat_f32(cv).iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(k_diff < 1e-5, "layer {l} K cache diff={k_diff}");
        assert!(v_diff < 1e-5, "layer {l} V cache diff={v_diff}");
    }

    let mut steered_model = deterministic_tiny_model(&cfg, 8);
    let mut adapter = ModelSteeringAdapter::new(cfg.to_rope_geometry()).expect("adapter");
    let mut trail = lmnk_pointer_trail_t::default();
    let q7s = vec![vec![0_i8; cfg.num_attention_heads]; cfg.num_hidden_layers];
    let mut pager = AnchorWorkingSetPager::new(liminika_adapter::AnchorPagerConfig::default());
    let mut steer = ForwardSteer {
        adapter: &mut adapter,
        q7s: Some(&q7s),
        persona_q7s: None,
        resonant_ctx: None,
        rope_rad: 0.12,
        trail: &mut trail,
        rank1: None,
        horizon_mode: HorizonAblationMode::Core3,
        escalation_provider: None,
        compute_dispatch: None,
        layer_steer_mask: None,
        probe_frame: ProbeCoordinateFrame::PreRope,
        phase_clamp_override: None,
        random_push_seed: None,
        steer_bypass_hits: 0,
        steer_eligible_layers: 0,
    };
    let (_, steered_rep) = steered_model
        .prefill_tokens_chunked(
            &tokens,
            0,
            7,
            Some(&mut steer),
            Some(&mut pager),
            false,
            LogitsFetchMode::FullLogits,
        )
        .expect("steered chunked prefill");
    assert_eq!(steered_rep.chunks_executed, 4);
    assert_eq!(steered_rep.liminal_steps, 4);
    assert_eq!(pager.stats().total_steps, 4);
}
