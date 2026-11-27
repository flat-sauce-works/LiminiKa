// name: examples/candle_eval/src/model/config.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Model configuration parsing, chat template formatting, and safetensors shard resolution
//! supporting both `Phi-4-mini` (fused QKV/MLP) and `Ministral 3 3B` (nested `text_config`,
//! separate projections, explicit `head_dim`, YaRN + Llama-4 attention scaling).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use liminika_adapter::{RopeGeometry, RopeLayout};
use serde::Deserialize;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ModelArch {
    #[default]
    Phi4,
    Mistral3,
}

impl ModelArch {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Phi4 => "phi4",
            Self::Mistral3 => "mistral3",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ModelArchArg {
    #[default]
    Auto,
    Phi4,
    Mistral3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ChatTemplateKind {
    #[default]
    Phi4,
    Mistral3,
}

impl ChatTemplateKind {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Phi4 => "phi4",
            Self::Mistral3 => "mistral3",
        }
    }

    pub(crate) const fn from_arch(arch: ModelArch) -> Self {
        match arch {
            ModelArch::Phi4 => Self::Phi4,
            ModelArch::Mistral3 => Self::Mistral3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ChatTemplateArg {
    #[default]
    Auto,
    Phi4,
    Mistral3,
}

#[derive(Debug, Clone)]
pub(crate) struct ChatTemplate {
    pub(crate) kind: ChatTemplateKind,
    pub(crate) bos_token: String,
    pub(crate) eos_token: String,
    pub(crate) stop_token_ids: Vec<u32>,
}

impl ChatTemplate {
    pub(crate) fn new(kind: ChatTemplateKind, cfg: &ModelConfig, tokenizer: &Tokenizer) -> Self {
        fn push_unique(stop_ids: &mut Vec<u32>, id: u32) {
            if !stop_ids.contains(&id) {
                stop_ids.push(id);
            }
        }
        let mut stop_ids: Vec<u32> = Vec::new();
        for &id in &cfg.eos_token_ids {
            push_unique(&mut stop_ids, id);
        }
        match kind {
            ChatTemplateKind::Phi4 => {
                if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
                    push_unique(&mut stop_ids, id);
                }
                if let Some(id) = tokenizer.token_to_id("<|end|>") {
                    push_unique(&mut stop_ids, id);
                }
                if stop_ids.is_empty() {
                    push_unique(&mut stop_ids, 199999);
                    push_unique(&mut stop_ids, 200020);
                }
                Self {
                    kind,
                    bos_token: String::new(),
                    eos_token: "<|end|>".to_string(),
                    stop_token_ids: stop_ids,
                }
            }
            ChatTemplateKind::Mistral3 => {
                if let Some(id) = tokenizer.token_to_id("</s>") {
                    push_unique(&mut stop_ids, id);
                }
                if stop_ids.is_empty() {
                    push_unique(&mut stop_ids, 2);
                }
                let bos_token = cfg
                    .bos_token_id
                    .and_then(|id| tokenizer.id_to_token(id))
                    .unwrap_or_else(|| "<s>".to_string());
                let eos_token = cfg
                    .eos_token_ids
                    .first()
                    .and_then(|&id| tokenizer.id_to_token(id))
                    .unwrap_or_else(|| "</s>".to_string());
                Self {
                    kind,
                    bos_token,
                    eos_token,
                    stop_token_ids: stop_ids,
                }
            }
        }
    }

    #[inline]
    pub(crate) fn is_stop_token(&self, token_id: u32) -> bool {
        self.stop_token_ids.contains(&token_id)
    }

    pub(crate) fn format_single_turn(&self, user_prompt: &str) -> String {
        match self.kind {
            ChatTemplateKind::Phi4 => format!("<|user|>{user_prompt}<|end|><|assistant|>"),
            ChatTemplateKind::Mistral3 => format!("{}[INST]{user_prompt}[/INST]", self.bos_token),
        }
    }

    pub(crate) fn format_first_turn_with_system(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> String {
        match self.kind {
            ChatTemplateKind::Phi4 => {
                format!("<|system|>{system_prompt}<|end|><|user|>{user_prompt}<|end|><|assistant|>")
            }
            ChatTemplateKind::Mistral3 => format!(
                "{}[SYSTEM_PROMPT]{system_prompt}[/SYSTEM_PROMPT][INST]{user_prompt}[/INST]",
                self.bos_token
            ),
        }
    }

    pub(crate) fn format_followup_turn(&self, user_prompt: &str) -> String {
        match self.kind {
            ChatTemplateKind::Phi4 => format!("<|end|><|user|>{user_prompt}<|end|><|assistant|>"),
            ChatTemplateKind::Mistral3 => format!("{}[INST]{user_prompt}[/INST]", self.eos_token),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct YarnRopeConfig {
    pub(crate) factor: f64,
    pub(crate) original_max_position_embeddings: usize,
    pub(crate) beta_fast: f64,
    pub(crate) beta_slow: f64,
    pub(crate) mscale: Option<f64>,
    pub(crate) mscale_all_dim: Option<f64>,
    pub(crate) llama_4_scaling_beta: Option<f64>,
}

impl YarnRopeConfig {
    /// Computes the YaRN attention scaling factor matching Hugging Face `transformers`
    /// `_compute_yarn_parameters`.
    pub(crate) fn attention_factor(&self) -> f64 {
        fn get_mscale(scale: f64, mscale: f64) -> f64 {
            if scale <= 1.0 {
                1.0
            } else {
                0.1 * mscale * scale.ln() + 1.0
            }
        }
        match (self.mscale, self.mscale_all_dim) {
            (Some(ms), Some(ms_all)) => {
                get_mscale(self.factor, ms) / get_mscale(self.factor, ms_all)
            }
            (Some(ms), None) => get_mscale(self.factor, ms),
            (None, _) => get_mscale(self.factor, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ModelConfig {
    pub(crate) arch: ModelArch,
    pub(crate) model_name: String,
    pub(crate) tensor_prefix: String,
    pub(crate) separate_projections: bool,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) rope_theta: f64,
    pub(crate) partial_rotary_factor: f64,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rope_layout: RopeLayout,
    pub(crate) yarn: Option<YarnRopeConfig>,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EosTokenIdRaw {
    Single(u32),
    Multiple(Vec<u32>),
}

impl EosTokenIdRaw {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(v) => vec![v],
            Self::Multiple(vs) => vs,
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    #[serde(default)]
    rope_type: Option<String>,
    #[serde(rename = "type", default)]
    kind: Option<String>,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default)]
    factor: Option<f64>,
    #[serde(default)]
    original_max_position_embeddings: Option<usize>,
    #[serde(default)]
    beta_fast: Option<f64>,
    #[serde(default)]
    beta_slow: Option<f64>,
    #[serde(default)]
    mscale: Option<f64>,
    #[serde(default)]
    mscale_all_dim: Option<f64>,
    #[serde(default)]
    llama_4_scaling_beta: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RawTextConfig {
    #[serde(default)]
    model_type: Option<String>,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    vocab_size: usize,
    rms_norm_eps: f64,
    #[serde(default)]
    rope_theta: Option<f64>,
    #[serde(default = "default_partial_rotary")]
    partial_rotary_factor: f64,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default)]
    rope_parameters: Option<RawRopeParameters>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<EosTokenIdRaw>,
}

#[derive(Debug, Deserialize)]
struct RawTopConfig {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    architectures: Vec<String>,
    #[serde(default)]
    text_config: Option<RawTextConfig>,
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<EosTokenIdRaw>,
    #[serde(flatten)]
    flat: Option<RawTextConfig>,
}

#[derive(Debug, Deserialize)]
struct RawGenerationConfig {
    #[serde(default)]
    bos_token_id: Option<u32>,
    #[serde(default)]
    eos_token_id: Option<EosTokenIdRaw>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: std::collections::HashMap<String, String>,
}

fn default_partial_rotary() -> f64 {
    1.0
}

fn default_max_position_embeddings() -> usize {
    4096
}

impl ModelConfig {
    pub(crate) fn load_from_dir(
        model_dir: &Path,
        arch_override: ModelArchArg,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let raw_cfg_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let gen_cfg_str = std::fs::read_to_string(model_dir.join("generation_config.json")).ok();
        let fallback_name = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("local-model");
        Self::from_json_str(
            &raw_cfg_str,
            gen_cfg_str.as_deref(),
            fallback_name,
            arch_override,
        )
    }

    pub(crate) fn from_json_str(
        config_json: &str,
        generation_config_json: Option<&str>,
        model_name: &str,
        arch_override: ModelArchArg,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let top: RawTopConfig = serde_json::from_str(config_json)?;
        let is_nested_vlm = top.text_config.is_some();
        let text = top
            .text_config
            .or(top.flat)
            .ok_or("config.json missing transformer text configuration fields")?;

        let detected_arch = match arch_override {
            ModelArchArg::Phi4 => ModelArch::Phi4,
            ModelArchArg::Mistral3 => ModelArch::Mistral3,
            ModelArchArg::Auto => {
                let top_type = top.model_type.as_deref().unwrap_or("");
                let txt_type = text.model_type.as_deref().unwrap_or("");
                let arch_is_mistral3 = top_type.eq_ignore_ascii_case("mistral3")
                    || txt_type.eq_ignore_ascii_case("ministral3")
                    || txt_type.eq_ignore_ascii_case("mistral3")
                    || top
                        .architectures
                        .iter()
                        .any(|a| a.contains("Mistral3") || a.contains("Ministral3"));
                if arch_is_mistral3 {
                    ModelArch::Mistral3
                } else {
                    ModelArch::Phi4
                }
            }
        };

        let head_dim = text
            .head_dim
            .unwrap_or_else(|| text.hidden_size / text.num_attention_heads.max(1));

        let rope_theta = text
            .rope_theta
            .or_else(|| text.rope_parameters.as_ref().and_then(|rp| rp.rope_theta))
            .unwrap_or(10_000.0);

        let yarn = text.rope_parameters.as_ref().and_then(|rp| {
            let rtype = rp.rope_type.as_deref().or(rp.kind.as_deref()).unwrap_or("");
            if rtype.eq_ignore_ascii_case("yarn") {
                Some(YarnRopeConfig {
                    factor: rp.factor.unwrap_or(1.0),
                    original_max_position_embeddings: rp
                        .original_max_position_embeddings
                        .unwrap_or(text.max_position_embeddings),
                    beta_fast: rp.beta_fast.unwrap_or(32.0),
                    beta_slow: rp.beta_slow.unwrap_or(1.0),
                    mscale: rp.mscale,
                    mscale_all_dim: rp.mscale_all_dim,
                    llama_4_scaling_beta: rp.llama_4_scaling_beta,
                })
            } else {
                None
            }
        });

        let gen_cfg = generation_config_json
            .and_then(|s| serde_json::from_str::<RawGenerationConfig>(s).ok());
        let bos_token_id = text
            .bos_token_id
            .or(top.bos_token_id)
            .or_else(|| gen_cfg.as_ref().and_then(|g| g.bos_token_id));
        let mut eos_token_ids = Vec::new();
        for src in [
            text.eos_token_id.map(EosTokenIdRaw::into_vec),
            top.eos_token_id.map(EosTokenIdRaw::into_vec),
            gen_cfg
                .and_then(|g| g.eos_token_id)
                .map(EosTokenIdRaw::into_vec),
        ]
        .into_iter()
        .flatten()
        {
            for id in src {
                if !eos_token_ids.contains(&id) {
                    eos_token_ids.push(id);
                }
            }
        }

        let (tensor_prefix, separate_projections) = match detected_arch {
            ModelArch::Phi4 => ("model".to_string(), false),
            ModelArch::Mistral3 => (
                if is_nested_vlm {
                    "language_model.model".to_string()
                } else {
                    "model".to_string()
                },
                true,
            ),
        };

        Ok(Self {
            arch: detected_arch,
            model_name: model_name.to_string(),
            tensor_prefix,
            separate_projections,
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            num_hidden_layers: text.num_hidden_layers,
            num_attention_heads: text.num_attention_heads,
            num_key_value_heads: text.num_key_value_heads,
            head_dim,
            vocab_size: text.vocab_size,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta,
            partial_rotary_factor: text.partial_rotary_factor,
            tie_word_embeddings: text.tie_word_embeddings,
            max_position_embeddings: text.max_position_embeddings,
            rope_layout: RopeLayout::HalfSplit,
            yarn,
            bos_token_id,
            eos_token_ids,
        })
    }

    pub(crate) const fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub(crate) fn rotary_dim(&self) -> usize {
        let raw = (self.head_dim() as f64 * self.partial_rotary_factor).round() as usize;
        raw - (raw % 2)
    }

    pub(crate) const fn active_layers_mask(&self) -> u64 {
        active_layers_mask(self.num_hidden_layers)
    }

    pub(crate) fn to_rope_geometry(&self) -> RopeGeometry {
        RopeGeometry {
            layout: self.rope_layout,
            head_dim: self.head_dim(),
            rotary_dim: self.rotary_dim(),
            num_q_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            num_layers: self.num_hidden_layers,
            layer_rope_mask: self.active_layers_mask(),
        }
    }
}

pub(crate) use liminika_adapter::{active_layers_mask, clip_layer_mask, top_half_layer_mask};

/// Resolves Hugging Face safetensors weight shards inside `model_dir`, preferring
/// `model.safetensors.index.json` so duplicate `consolidated.safetensors` files are never loaded.
pub(crate) fn resolve_safetensors_files(
    model_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let index: SafetensorsIndex = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let unique: BTreeSet<String> = index.weight_map.into_values().collect();
        if unique.is_empty() {
            return Err(format!("{} contains an empty weight_map", index_path.display()).into());
        }
        let mut paths = Vec::with_capacity(unique.len());
        for rel in unique {
            let p = model_dir.join(&rel);
            if !p.is_file() {
                return Err(format!("missing safetensors shard '{}'", p.display()).into());
            }
            paths.push(p);
        }
        return Ok(paths);
    }
    let single = model_dir.join("model.safetensors");
    if single.is_file() {
        return Ok(vec![single]);
    }
    Err(format!(
        "no model.safetensors.index.json or model.safetensors found in '{}'",
        model_dir.display()
    )
    .into())
}
