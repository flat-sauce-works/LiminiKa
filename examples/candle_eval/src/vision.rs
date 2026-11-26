// name: examples/candle_eval/src/vision.rs
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pixtral vision tower, 2D axial RoPE, spatial patch merger, multimodal projector, and
//! deterministic pure-Rust synthetic RGB test images for VLM boundary evaluation.

use std::path::Path;

use candle_core::{DType, Device, Module, Result as CResult, Tensor, D};
use candle_nn::{linear_no_bias, Linear, RmsNorm, VarBuilder};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::model::{load_rms_norm_on_device, resolve_safetensors_files, CausalLmModel};

const DEFAULT_IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const DEFAULT_IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
const DEFAULT_RESCALE_FACTOR: f32 = 1.0 / 255.0;
const PIXTRAL_RMS_NORM_EPS: f64 = 1e-5;

/// Parsed multimodal vision and projector configuration from `config.json` and
/// `processor_config.json`.
#[derive(Debug, Clone)]
pub(crate) struct VisionConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_channels: usize,
    pub(crate) patch_size: usize,
    pub(crate) spatial_merge_size: usize,
    pub(crate) rope_theta: f64,
    pub(crate) text_hidden_size: usize,
    pub(crate) projector_norm_eps: f64,
    pub(crate) projector_hidden_act: String,
    pub(crate) image_token_id: u32,
    pub(crate) image_break_token_id: u32,
    pub(crate) image_end_token_id: u32,
    pub(crate) image_mean: [f32; 3],
    pub(crate) image_std: [f32; 3],
    pub(crate) rescale_factor: f32,
}

#[derive(Debug, Deserialize)]
struct RawVisionSubConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default = "default_num_channels")]
    num_channels: usize,
    patch_size: usize,
    #[serde(default = "default_vision_rope_theta")]
    rope_theta: f64,
}

#[derive(Debug, Deserialize)]
struct RawTextDimConfig {
    hidden_size: usize,
    #[serde(default = "default_rms_eps")]
    rms_norm_eps: f64,
}

#[derive(Debug, Deserialize)]
struct RawVlmConfig {
    #[serde(default)]
    vision_config: Option<RawVisionSubConfig>,
    #[serde(default)]
    text_config: Option<RawTextDimConfig>,
    #[serde(default = "default_spatial_merge_size")]
    spatial_merge_size: usize,
    #[serde(default = "default_projector_act")]
    projector_hidden_act: String,
    #[serde(default = "default_image_token_id")]
    image_token_index: u32,
}

#[derive(Debug, Deserialize)]
struct RawImageProcessor {
    #[serde(default)]
    image_mean: Option<[f32; 3]>,
    #[serde(default)]
    image_std: Option<[f32; 3]>,
    #[serde(default)]
    rescale_factor: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct RawProcessorConfig {
    #[serde(default)]
    image_token: Option<String>,
    #[serde(default)]
    image_break_token: Option<String>,
    #[serde(default)]
    image_end_token: Option<String>,
    #[serde(default)]
    image_processor: Option<RawImageProcessor>,
}

fn default_num_channels() -> usize {
    3
}

fn default_vision_rope_theta() -> f64 {
    10_000.0
}

fn default_rms_eps() -> f64 {
    PIXTRAL_RMS_NORM_EPS
}

fn default_spatial_merge_size() -> usize {
    2
}

fn default_projector_act() -> String {
    "gelu".to_string()
}

fn default_image_token_id() -> u32 {
    10
}

impl VisionConfig {
    /// Loads and validates `VisionConfig` from `model_dir`. Returns a clear error (never panics)
    /// when called on a text-only checkpoint without `vision_config`.
    pub(crate) fn load_from_dir(
        model_dir: &Path,
        tokenizer: &Tokenizer,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let cfg_str = std::fs::read_to_string(model_dir.join("config.json"))?;
        let proc_str = std::fs::read_to_string(model_dir.join("processor_config.json"))
            .or_else(|_| std::fs::read_to_string(model_dir.join("preprocessor_config.json")))
            .ok();
        let model_label = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("local-model");
        Self::from_json_strs(&cfg_str, proc_str.as_deref(), model_label, Some(tokenizer))
    }

    pub(crate) fn from_json_strs(
        config_json: &str,
        processor_json: Option<&str>,
        model_label: &str,
        tokenizer: Option<&Tokenizer>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let raw: RawVlmConfig = serde_json::from_str(config_json)?;
        let Some(vis) = raw.vision_config else {
            return Err(format!(
                "model '{model_label}' has no vision_config in config.json; \
                 --vlm-eval requires a multimodal VLM checkpoint"
            )
            .into());
        };
        let Some(txt) = raw.text_config else {
            return Err(format!(
                "model '{model_label}' is missing text_config required for multimodal projection"
            )
            .into());
        };
        if vis.patch_size == 0 || raw.spatial_merge_size == 0 || vis.num_attention_heads == 0 {
            return Err(format!(
                "invalid vision geometry in '{model_label}': patch_size={} spatial_merge_size={} heads={}",
                vis.patch_size, raw.spatial_merge_size, vis.num_attention_heads
            )
            .into());
        }
        let head_dim = vis
            .head_dim
            .unwrap_or(vis.hidden_size / vis.num_attention_heads);
        if !head_dim.is_multiple_of(4) {
            return Err(format!(
                "Pixtral 2D axial RoPE requires head_dim divisible by 4, got {head_dim}"
            )
            .into());
        }

        let proc_cfg =
            processor_json.and_then(|s| serde_json::from_str::<RawProcessorConfig>(s).ok());
        let img_proc = proc_cfg.as_ref().and_then(|p| p.image_processor.as_ref());
        let image_mean = img_proc
            .and_then(|p| p.image_mean)
            .unwrap_or(DEFAULT_IMAGE_MEAN);
        let image_std = img_proc
            .and_then(|p| p.image_std)
            .unwrap_or(DEFAULT_IMAGE_STD);
        let rescale_factor = img_proc
            .and_then(|p| p.rescale_factor)
            .unwrap_or(DEFAULT_RESCALE_FACTOR);

        let image_tok_str = proc_cfg
            .as_ref()
            .and_then(|p| p.image_token.as_deref())
            .unwrap_or("[IMG]");
        let break_tok_str = proc_cfg
            .as_ref()
            .and_then(|p| p.image_break_token.as_deref())
            .unwrap_or("[IMG_BREAK]");
        let end_tok_str = proc_cfg
            .as_ref()
            .and_then(|p| p.image_end_token.as_deref())
            .unwrap_or("[IMG_END]");

        let image_token_id = tokenizer
            .and_then(|t| t.token_to_id(image_tok_str))
            .unwrap_or(raw.image_token_index);
        if image_token_id != raw.image_token_index {
            return Err(format!(
                "image token id mismatch: config.json={} vs tokenizer({})={}",
                raw.image_token_index, image_tok_str, image_token_id
            )
            .into());
        }
        let image_break_token_id = tokenizer
            .and_then(|t| t.token_to_id(break_tok_str))
            .unwrap_or(12);
        let image_end_token_id = tokenizer
            .and_then(|t| t.token_to_id(end_tok_str))
            .unwrap_or(13);

        Ok(Self {
            hidden_size: vis.hidden_size,
            intermediate_size: vis.intermediate_size,
            num_hidden_layers: vis.num_hidden_layers,
            num_attention_heads: vis.num_attention_heads,
            head_dim,
            num_channels: vis.num_channels,
            patch_size: vis.patch_size,
            spatial_merge_size: raw.spatial_merge_size,
            rope_theta: vis.rope_theta,
            text_hidden_size: txt.hidden_size,
            projector_norm_eps: txt.rms_norm_eps,
            projector_hidden_act: raw.projector_hidden_act,
            image_token_id,
            image_break_token_id,
            image_end_token_id,
            image_mean,
            image_std,
            rescale_factor,
        })
    }

    pub(crate) const fn grid_unit(&self) -> usize {
        self.patch_size * self.spatial_merge_size
    }
}

/// Pure-Rust deterministic RGB image (`H x W x 3`, row-major `u8`).
#[derive(Debug, Clone)]
pub(crate) struct RgbImage {
    pub(crate) name: &'static str,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) rgb: Vec<u8>,
}

impl RgbImage {
    pub(crate) fn filled(name: &'static str, height: usize, width: usize, color: [u8; 3]) -> Self {
        let mut rgb = Vec::with_capacity(height * width * 3);
        for _ in 0..(height * width) {
            rgb.extend_from_slice(&color);
        }
        Self {
            name,
            height,
            width,
            rgb,
        }
    }

    #[inline]
    pub(crate) fn set_pixel(&mut self, x: usize, y: usize, color: [u8; 3]) {
        if x < self.width && y < self.height {
            let idx = (y * self.width + x) * 3;
            self.rgb[idx] = color[0];
            self.rgb[idx + 1] = color[1];
            self.rgb[idx + 2] = color[2];
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn get_pixel(&self, x: usize, y: usize) -> [u8; 3] {
        let idx = (y * self.width + x) * 3;
        [self.rgb[idx], self.rgb[idx + 1], self.rgb[idx + 2]]
    }

    pub(crate) fn fill_rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, color: [u8; 3]) {
        let y_end = y1.min(self.height);
        let x_end = x1.min(self.width);
        for y in y0..y_end {
            for x in x0..x_end {
                self.set_pixel(x, y, color);
            }
        }
    }

    pub(crate) fn fill_circle(&mut self, cx: i32, cy: i32, radius: i32, color: [u8; 3]) {
        let r2 = i64::from(radius) * i64::from(radius);
        for y in 0..self.height {
            let dy = y as i64 - i64::from(cy);
            for x in 0..self.width {
                let dx = x as i64 - i64::from(cx);
                if dx * dx + dy * dy <= r2 {
                    self.set_pixel(x, y, color);
                }
            }
        }
    }
}

/// Deterministic `224 x 224` (`8 x 8` merged tokens) synthetic test image A:
/// a large crisp red circle centered on a solid white background.
pub(crate) fn make_image_a_red_circle() -> RgbImage {
    let mut img = RgbImage::filled("red_circle_on_white", 224, 224, [255, 255, 255]);
    img.fill_circle(112, 112, 72, [225, 20, 20]);
    img
}

/// Deterministic `224 x 224` synthetic test image B:
/// a solid blue square on the left and a solid yellow square on the right on white background.
pub(crate) fn make_image_b_blue_left_yellow_right() -> RgbImage {
    let mut img = RgbImage::filled("blue_square_left_yellow_right", 224, 224, [255, 255, 255]);
    img.fill_rect(22, 76, 94, 148, [20, 65, 235]);
    img.fill_rect(130, 76, 202, 148, [250, 215, 15]);
    img
}

/// Deterministic `224 x 224` synthetic test image C:
/// dark-gray UI layout with top title bar, left sidebar, and centered green status rectangle.
pub(crate) fn make_image_c_dark_ui_layout() -> RgbImage {
    let mut img = RgbImage::filled("dark_ui_layout", 224, 224, [28, 32, 40]);
    // Top title bar with subtle divider and three window dots.
    img.fill_rect(0, 0, 224, 30, [52, 58, 70]);
    img.fill_rect(0, 30, 224, 32, [74, 82, 98]);
    img.fill_circle(14, 15, 4, [235, 85, 85]);
    img.fill_circle(28, 15, 4, [240, 195, 65]);
    img.fill_circle(42, 15, 4, [65, 205, 105]);
    // Left sidebar with three placeholder navigation items.
    img.fill_rect(0, 32, 56, 224, [40, 46, 56]);
    img.fill_rect(56, 32, 58, 224, [70, 78, 92]);
    img.fill_rect(10, 48, 46, 58, [96, 106, 124]);
    img.fill_rect(10, 70, 46, 80, [84, 94, 110]);
    img.fill_rect(10, 92, 46, 102, [84, 94, 110]);
    // Main panel centered bright green status rectangle.
    img.fill_rect(90, 92, 192, 154, [34, 197, 94]);
    img
}

/// Preprocesses an `RgbImage` into channels-first `[3, H, W]` `f32` normalized values matching
/// `PixtralImageProcessorFast` (`rescale_factor`, `image_mean`, `image_std`).
pub(crate) fn preprocess_rgb_image_vec(
    img: &RgbImage,
    cfg: &VisionConfig,
) -> Result<Vec<f32>, String> {
    let unit = cfg.grid_unit();
    if img.height == 0 || img.width == 0 {
        return Err("image dimensions must be non-zero".to_string());
    }
    if !img.height.is_multiple_of(unit) || !img.width.is_multiple_of(unit) {
        return Err(format!(
            "image '{}' size {}x{} must be divisible by patch_size*spatial_merge_size={unit}",
            img.name, img.height, img.width
        ));
    }
    let hw = img.height * img.width;
    if img.rgb.len() != hw * 3 {
        return Err(format!(
            "image '{}' byte length {} != {}*{}*3",
            img.name,
            img.rgb.len(),
            img.height,
            img.width
        ));
    }
    let mut out = vec![0.0_f32; 3 * hw];
    for p in 0..hw {
        let base = p * 3;
        for c in 0..3 {
            let scaled = f32::from(img.rgb[base + c]) * cfg.rescale_factor;
            out[c * hw + p] = (scaled - cfg.image_mean[c]) / cfg.image_std[c];
        }
    }
    Ok(out)
}

/// Extracts non-overlapping `patch_size x patch_size` patches from `[C, H, W]` and projects them
/// via `weight` `[out_dim, C, P, P]` using `reshape + matmul` (exact mathematical equivalent of
/// stride-`P` bias-free `Conv2d` with zero cuDNN dependency).
pub(crate) fn patch_conv_via_matmul(
    chw: &Tensor,
    weight_ocpp: &Tensor,
    patch_size: usize,
) -> CResult<Tensor> {
    let (c, h, w) = chw.dims3()?;
    let (out_dim, w_c, k_h, k_w) = weight_ocpp.dims4()?;
    if c != w_c || k_h != patch_size || k_w != patch_size {
        return Err(candle_core::Error::Msg(format!(
            "patch_conv shape mismatch: input=[{c},{h},{w}] weight=[{out_dim},{w_c},{k_h},{k_w}] p={patch_size}"
        )));
    }
    if !h.is_multiple_of(patch_size) || !w.is_multiple_of(patch_size) {
        return Err(candle_core::Error::Msg(format!(
            "image {h}x{w} is not divisible by patch_size={patch_size}"
        )));
    }
    let h_p = h / patch_size;
    let w_p = w / patch_size;
    let patch_dim = c * patch_size * patch_size;
    // [C, h_p, P, w_p, P] -> [h_p, w_p, C, P, P] -> [h_p * w_p, C * P * P]
    let patches = chw
        .reshape((c, h_p, patch_size, w_p, patch_size))?
        .permute((1, 3, 0, 2, 4))?
        .contiguous()?
        .reshape((h_p * w_p, patch_dim))?;
    let w_t = weight_ocpp
        .reshape((out_dim, patch_dim))?
        .transpose(0, 1)?
        .contiguous()?;
    patches.matmul(&w_t)
}

/// Computes Pixtral 2D axial RoPE `(cos_half, sin_half)` of shape `[1, 1, h_p * w_p, head_dim / 2]`
/// with even base frequencies assigned to height (`r`) and odd base frequencies to width (`c`).
pub(crate) fn build_pixtral_2d_rope_tables(
    h_patches: usize,
    w_patches: usize,
    head_dim: usize,
    rope_theta: f64,
    dtype: DType,
    device: &Device,
) -> CResult<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let quarter = head_dim / 4;
    let base_freqs: Vec<f64> = (0..half)
        .map(|i| 1.0_f64 / rope_theta.powf((2 * i) as f64 / head_dim as f64))
        .collect();
    let h_freqs: Vec<f64> = (0..quarter).map(|k| base_freqs[2 * k]).collect();
    let w_freqs: Vec<f64> = (0..quarter).map(|k| base_freqs[2 * k + 1]).collect();

    let n_patches = h_patches * w_patches;
    let mut cos_vals = Vec::with_capacity(n_patches * half);
    let mut sin_vals = Vec::with_capacity(n_patches * half);
    for r in 0..h_patches {
        let rf = r as f64;
        for c in 0..w_patches {
            let cf = c as f64;
            for &fh in &h_freqs {
                let ang = rf * fh;
                cos_vals.push(ang.cos() as f32);
                sin_vals.push(ang.sin() as f32);
            }
            for &fw in &w_freqs {
                let ang = cf * fw;
                cos_vals.push(ang.cos() as f32);
                sin_vals.push(ang.sin() as f32);
            }
        }
    }
    let cos = Tensor::from_vec(cos_vals, (1, 1, n_patches, half), device)?.to_dtype(dtype)?;
    let sin = Tensor::from_vec(sin_vals, (1, 1, n_patches, half), device)?.to_dtype(dtype)?;
    Ok((cos, sin))
}

/// Applies HalfSplit (`rotate_half`) 2D RoPE to `x` of shape `[b, num_heads, n_patches, head_dim]`.
pub(crate) fn apply_2d_rope_half_split(
    x: &Tensor,
    cos_half: &Tensor,
    sin_half: &Tensor,
) -> CResult<Tensor> {
    let head_dim = x.dim(D::Minus1)?;
    let half = head_dim / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    let r1 = (x1.broadcast_mul(cos_half)? - x2.broadcast_mul(sin_half)?)?;
    let r2 = (x1.broadcast_mul(sin_half)? + x2.broadcast_mul(cos_half)?)?;
    Tensor::cat(&[&r1, &r2], D::Minus1)
}

/// Performs `Mistral3PatchMerger` spatial unfolding matching PyTorch
/// `F.unfold(x.view(h_p, w_p, d).permute(2, 0, 1).unsqueeze(0), s, stride=s).view(d*s*s, -1).t()`.
///
/// Output shape is `[(h_p / s) * (w_p / s), d * s * s]` with inner channel order `(c, kh, kw)`.
pub(crate) fn unfold_spatial_patches(
    tokens: &Tensor,
    h_patches: usize,
    w_patches: usize,
    spatial_merge_size: usize,
) -> CResult<Tensor> {
    let (n, d) = tokens.dims2()?;
    if n != h_patches * w_patches {
        return Err(candle_core::Error::Msg(format!(
            "unfold_spatial_patches token count {n} != {h_patches}*{w_patches}"
        )));
    }
    if !h_patches.is_multiple_of(spatial_merge_size)
        || !w_patches.is_multiple_of(spatial_merge_size)
    {
        return Err(candle_core::Error::Msg(format!(
            "patch grid {h_patches}x{w_patches} not divisible by spatial_merge_size={spatial_merge_size}"
        )));
    }
    let s = spatial_merge_size;
    let h_m = h_patches / s;
    let w_m = w_patches / s;
    // [h_m, s, w_m, s, d] -> [h_m, w_m, d, s, s] -> [h_m * w_m, d * s * s]
    tokens
        .reshape((h_m, s, w_m, s, d))?
        .permute((0, 2, 4, 1, 3))?
        .contiguous()?
        .reshape((h_m * w_m, d * s * s))
}

/// Expands a single image of size `height x width` into its exact `[IMG]` / `[IMG_BREAK]` /
/// `[IMG_END]` placeholder token ID sequence and returns `(placeholder_ids, img_token_count)`.
pub(crate) fn expand_image_placeholder_tokens(
    height: usize,
    width: usize,
    cfg: &VisionConfig,
) -> Result<(Vec<u32>, usize), String> {
    let unit = cfg.grid_unit();
    if height == 0 || width == 0 || !height.is_multiple_of(unit) || !width.is_multiple_of(unit) {
        return Err(format!(
            "image {height}x{width} must be non-zero and divisible by {unit}"
        ));
    }
    let h_m = height / unit;
    let w_m = width / unit;
    let img_count = h_m * w_m;
    let mut ids = Vec::with_capacity(h_m * (w_m + 1));
    for row in 0..h_m {
        for _ in 0..w_m {
            ids.push(cfg.image_token_id);
        }
        if row + 1 < h_m {
            ids.push(cfg.image_break_token_id);
        } else {
            ids.push(cfg.image_end_token_id);
        }
    }
    Ok((ids, img_count))
}

/// Builds a Mistral3 single-image multimodal chat turn token ID sequence (`<s>[INST]<img_grid><text>[/INST]`
/// for turn 1, or `</s>[INST]<img_grid><text>[/INST]` for follow-up turns) without BPE interference
/// around special image control tokens.
pub(crate) fn build_mistral3_image_turn_tokens(
    tokenizer: &Tokenizer,
    cfg: &VisionConfig,
    img: &RgbImage,
    user_text: &str,
    is_followup: bool,
) -> Result<(Vec<u32>, usize), Box<dyn std::error::Error>> {
    let lead_id = if is_followup {
        tokenizer.token_to_id("</s>").unwrap_or(2)
    } else {
        tokenizer.token_to_id("<s>").unwrap_or(1)
    };
    let inst_open = tokenizer.token_to_id("[INST]").unwrap_or(3);
    let inst_close = tokenizer.token_to_id("[/INST]").unwrap_or(4);
    let (img_ids, img_tok_count) = expand_image_placeholder_tokens(img.height, img.width, cfg)?;
    let text_ids = tokenizer
        .encode(user_text, false)
        .map_err(|e| format!("multimodal text encode error: {e}"))?
        .get_ids()
        .to_vec();

    let mut ids = Vec::with_capacity(3 + img_ids.len() + text_ids.len());
    ids.push(lead_id);
    ids.push(inst_open);
    ids.extend_from_slice(&img_ids);
    ids.extend_from_slice(&text_ids);
    ids.push(inst_close);
    Ok((ids, img_tok_count))
}

/// Builds raw (non-chat-wrapped) multimodal probe token IDs (`<img_grid><text>`) for comparing
/// `IsolatedProbe` behavior with vs without visual embeddings.
pub(crate) fn build_raw_multimodal_probe_tokens(
    tokenizer: &Tokenizer,
    cfg: &VisionConfig,
    img: &RgbImage,
    query_text: &str,
) -> Result<(Vec<u32>, usize), Box<dyn std::error::Error>> {
    let (img_ids, img_tok_count) = expand_image_placeholder_tokens(img.height, img.width, cfg)?;
    let text_ids = tokenizer
        .encode(query_text, false)
        .map_err(|e| format!("multimodal probe encode error: {e}"))?
        .get_ids()
        .to_vec();
    let mut ids = Vec::with_capacity(img_ids.len() + text_ids.len());
    ids.extend_from_slice(&img_ids);
    ids.extend_from_slice(&text_ids);
    Ok((ids, img_tok_count))
}

/// Splices projected `vision_embeds` (`[N_img, hidden]`) into embedded `token_ids` (`[1, L, hidden]`)
/// at every `image_token_id` slot. Returns an explicit error if the `[IMG]` count does not exactly
/// match `vision_embeds.dim(0)` (never pads or truncates).
pub(crate) fn splice_vision_embeddings(
    token_ids: &[u32],
    text_embeds: &Tensor,
    vision_embeds: &Tensor,
    image_token_id: u32,
) -> CResult<Tensor> {
    let (batch, seq_len, hidden) = text_embeds.dims3()?;
    if batch != 1 || seq_len != token_ids.len() {
        return Err(candle_core::Error::Msg(format!(
            "text_embeds shape [{batch}, {seq_len}, {hidden}] mismatches token_ids len {}",
            token_ids.len()
        )));
    }
    let (n_vis, vis_hidden) = vision_embeds.dims2()?;
    if vis_hidden != hidden {
        return Err(candle_core::Error::Msg(format!(
            "vision_embeds hidden dim {vis_hidden} != text hidden dim {hidden}"
        )));
    }
    let placeholder_count = token_ids.iter().filter(|&&id| id == image_token_id).count();
    if placeholder_count != n_vis {
        return Err(candle_core::Error::Msg(format!(
            "[IMG] placeholder count mismatch: found {placeholder_count} placeholders for {n_vis} vision tokens"
        )));
    }
    if n_vis == 0 {
        return Ok(text_embeds.clone());
    }
    let vis_cast = vision_embeds
        .to_device(text_embeds.device())?
        .to_dtype(text_embeds.dtype())?
        .unsqueeze(0)?;

    let mut pieces: Vec<Tensor> = Vec::new();
    let mut pos = 0_usize;
    let mut vis_pos = 0_usize;
    while pos < seq_len {
        let is_img = token_ids[pos] == image_token_id;
        let mut end = pos + 1;
        while end < seq_len && (token_ids[end] == image_token_id) == is_img {
            end += 1;
        }
        let run_len = end - pos;
        if is_img {
            pieces.push(vis_cast.narrow(1, vis_pos, run_len)?);
            vis_pos += run_len;
        } else {
            pieces.push(text_embeds.narrow(1, pos, run_len)?);
        }
        pos = end;
    }
    debug_assert_eq!(vis_pos, n_vis);
    let refs: Vec<&Tensor> = pieces.iter().collect();
    Tensor::cat(&refs, 1)
}

/// Embeds `token_ids` via `model` and splices `vision_embeds` at `[IMG]` positions.
pub(crate) fn build_multimodal_inputs_embeds(
    model: &CausalLmModel,
    token_ids: &[u32],
    vision_embeds: &Tensor,
    image_token_id: u32,
) -> CResult<Tensor> {
    let text_embeds = model.embed_token_ids(token_ids)?;
    splice_vision_embeddings(token_ids, &text_embeds, vision_embeds, image_token_id)
}

struct PixtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl PixtralAttention {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> CResult<Self> {
        let d = cfg.hidden_size;
        Ok(Self {
            q_proj: linear_no_bias(d, d, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(d, d, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(d, d, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(d, d, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
        })
    }

    /// Bidirectional multi-head self-attention (no causal mask) with 2D axial RoPE.
    fn forward(&self, x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> CResult<Tensor> {
        let (b, n_patches, d) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, n_patches, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, n_patches, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, n_patches, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = apply_2d_rope_half_split(&q, cos_half, sin_half)?.contiguous()?;
        let k = apply_2d_rope_half_split(&k, cos_half, sin_half)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?.contiguous()?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&att)?;
        let ctx = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n_patches, d))?;
        self.o_proj.forward(&ctx)
    }
}

struct PixtralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl PixtralMlp {
    fn load(cfg: &VisionConfig, vb: VarBuilder) -> CResult<Self> {
        let d = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        Ok(Self {
            gate_proj: linear_no_bias(d, inter, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(d, inter, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(inter, d, vb.pp("down_proj"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

struct PixtralLayer {
    attention_norm: RmsNorm,
    attention: PixtralAttention,
    ffn_norm: RmsNorm,
    feed_forward: PixtralMlp,
}

impl PixtralLayer {
    fn load(cfg: &VisionConfig, vb: VarBuilder, dev: &Device) -> CResult<Self> {
        Ok(Self {
            attention_norm: load_rms_norm_on_device(
                cfg.hidden_size,
                PIXTRAL_RMS_NORM_EPS,
                vb.pp("attention_norm"),
                dev,
            )?,
            attention: PixtralAttention::load(cfg, vb.pp("attention"))?,
            ffn_norm: load_rms_norm_on_device(
                cfg.hidden_size,
                PIXTRAL_RMS_NORM_EPS,
                vb.pp("ffn_norm"),
                dev,
            )?,
            feed_forward: PixtralMlp::load(cfg, vb.pp("feed_forward"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos_half: &Tensor, sin_half: &Tensor) -> CResult<Tensor> {
        let normed = self.attention_norm.forward(x)?;
        let h = (x + self.attention.forward(&normed, cos_half, sin_half)?)?;
        let ffn_in = self.ffn_norm.forward(&h)?;
        h + self.feed_forward.forward(&ffn_in)?
    }
}

/// Pixtral vision encoder (`vision_tower`) + `Mistral3MultiModalProjector` kept in unquantized
/// `BF16` (CUDA) or `F32` (CPU) so visual features carry zero quantization distortion.
pub(crate) struct VisionModel {
    pub(crate) cfg: VisionConfig,
    patch_conv_weight: Tensor,
    ln_pre: RmsNorm,
    layers: Vec<PixtralLayer>,
    proj_norm: RmsNorm,
    merging_layer: Linear,
    linear_1: Linear,
    linear_2: Linear,
    dtype: DType,
    device: Device,
}

impl VisionModel {
    pub(crate) fn load(
        model_dir: &Path,
        cfg: VisionConfig,
        dev: &Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dtype = if dev.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let files = resolve_safetensors_files(model_dir)?;
        // SAFETY: The safetensors files are read-only model weights on disk.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, dev)? };
        let vtb = vb.pp("vision_tower");
        let patch_conv_weight = vtb.pp("patch_conv").get(
            (
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ),
            "weight",
        )?;
        let ln_pre =
            load_rms_norm_on_device(cfg.hidden_size, PIXTRAL_RMS_NORM_EPS, vtb.pp("ln_pre"), dev)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let layers_vb = vtb.pp("transformer").pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(PixtralLayer::load(&cfg, layers_vb.pp(format!("{i}")), dev)?);
        }

        let mmp = vb.pp("multi_modal_projector");
        let proj_norm =
            load_rms_norm_on_device(cfg.hidden_size, cfg.projector_norm_eps, mmp.pp("norm"), dev)?;
        let merge_in = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merging_layer = linear_no_bias(
            merge_in,
            cfg.hidden_size,
            mmp.pp("patch_merger").pp("merging_layer"),
        )?;
        let linear_1 = linear_no_bias(cfg.hidden_size, cfg.text_hidden_size, mmp.pp("linear_1"))?;
        let linear_2 = linear_no_bias(
            cfg.text_hidden_size,
            cfg.text_hidden_size,
            mmp.pp("linear_2"),
        )?;

        Ok(Self {
            cfg,
            patch_conv_weight,
            ln_pre,
            layers,
            proj_norm,
            merging_layer,
            linear_1,
            linear_2,
            dtype,
            device: dev.clone(),
        })
    }

    pub(crate) const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Encodes a single `RgbImage` into projected LLM embeddings `[N_img, text_hidden_size]`
    /// cast to `target_dtype`.
    pub(crate) fn encode_image(
        &self,
        img: &RgbImage,
        target_dtype: DType,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let chw_vec = preprocess_rgb_image_vec(img, &self.cfg)?;
        let chw = Tensor::from_vec(chw_vec, (3, img.height, img.width), &self.device)?
            .to_dtype(self.dtype)?;
        let h_patches = img.height / self.cfg.patch_size;
        let w_patches = img.width / self.cfg.patch_size;

        let patches = patch_conv_via_matmul(&chw, &self.patch_conv_weight, self.cfg.patch_size)?;
        let mut h = self.ln_pre.forward(&patches.unsqueeze(0)?)?;
        let (cos_half, sin_half) = build_pixtral_2d_rope_tables(
            h_patches,
            w_patches,
            self.cfg.head_dim,
            self.cfg.rope_theta,
            self.dtype,
            &self.device,
        )?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos_half, &sin_half)?;
        }

        let vis_tokens = h.squeeze(0)?;
        let normed = self.proj_norm.forward(&vis_tokens)?;
        let unfolded =
            unfold_spatial_patches(&normed, h_patches, w_patches, self.cfg.spatial_merge_size)?;
        let merged = self.merging_layer.forward(&unfolded)?;
        let h1 = self.linear_1.forward(&merged)?;
        let act = if self.cfg.projector_hidden_act.eq_ignore_ascii_case("silu") {
            candle_nn::ops::silu(&h1)?
        } else {
            // Compute exact erf GELU in F32 for numerical fidelity across BF16/CUDA and CPU.
            h1.to_dtype(DType::F32)?.gelu_erf()?.to_dtype(self.dtype)?
        };
        let projected = self.linear_2.forward(&act)?;
        Ok(projected.to_dtype(target_dtype)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vision_cfg() -> VisionConfig {
        VisionConfig {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            head_dim: 64,
            num_channels: 3,
            patch_size: 14,
            spatial_merge_size: 2,
            rope_theta: 10_000.0,
            text_hidden_size: 3072,
            projector_norm_eps: 1e-5,
            projector_hidden_act: "gelu".to_string(),
            image_token_id: 10,
            image_break_token_id: 12,
            image_end_token_id: 13,
            image_mean: DEFAULT_IMAGE_MEAN,
            image_std: DEFAULT_IMAGE_STD,
            rescale_factor: DEFAULT_RESCALE_FACTOR,
        }
    }

    #[test]
    fn deterministic_synthetic_images_have_expected_geometry_and_colors() {
        let cfg = sample_vision_cfg();
        let img_a = make_image_a_red_circle();
        let img_b = make_image_b_blue_left_yellow_right();
        let img_c = make_image_c_dark_ui_layout();
        for img in [&img_a, &img_b, &img_c] {
            assert_eq!(img.height, 224);
            assert_eq!(img.width, 224);
            assert_eq!(img.height % cfg.grid_unit(), 0);
            assert_eq!(img.width % cfg.grid_unit(), 0);
            assert_eq!(img.rgb.len(), 224 * 224 * 3);
        }
        // Image A: white corner, red center.
        assert_eq!(img_a.get_pixel(0, 0), [255, 255, 255]);
        assert_eq!(img_a.get_pixel(112, 112), [225, 20, 20]);
        // Image B: blue left square, white middle, yellow right square.
        assert_eq!(img_b.get_pixel(58, 112), [20, 65, 235]);
        assert_eq!(img_b.get_pixel(112, 112), [255, 255, 255]);
        assert_eq!(img_b.get_pixel(166, 112), [250, 215, 15]);
        // Image C: top bar, sidebar, dark background, green center rectangle.
        assert_eq!(img_c.get_pixel(120, 10), [52, 58, 70]);
        assert_eq!(img_c.get_pixel(25, 140), [40, 46, 56]);
        assert_eq!(img_c.get_pixel(70, 50), [28, 32, 40]);
        assert_eq!(img_c.get_pixel(140, 120), [34, 197, 94]);
    }

    #[test]
    fn preprocess_rgb_normalizes_channels_first_accurately() {
        let cfg = sample_vision_cfg();
        let img = RgbImage::filled("unit_test", 28, 28, [255, 0, 128]);
        let chw = preprocess_rgb_image_vec(&img, &cfg).expect("preprocess");
        let hw = 28 * 28;
        assert_eq!(chw.len(), 3 * hw);
        let exp_r = (1.0 - cfg.image_mean[0]) / cfg.image_std[0];
        let exp_g = (0.0 - cfg.image_mean[1]) / cfg.image_std[1];
        let exp_b = (128.0 / 255.0 - cfg.image_mean[2]) / cfg.image_std[2];
        assert!((chw[0] - exp_r).abs() < 1e-5);
        assert!((chw[hw] - exp_g).abs() < 1e-5);
        assert!((chw[2 * hw] - exp_b).abs() < 1e-5);
    }

    #[test]
    fn patch_conv_matmul_matches_explicit_non_overlapping_conv2d() {
        let dev = Device::Cpu;
        let (c_in, c_out, p, h, w) = (3, 4, 2, 4, 6);
        let h_p = h / p;
        let w_p = w / p;
        let x_vals: Vec<f32> = (0..(c_in * h * w))
            .map(|i| ((i * 17 + 3) % 29) as f32 * 0.1 - 1.2)
            .collect();
        let w_vals: Vec<f32> = (0..(c_out * c_in * p * p))
            .map(|i| ((i * 13 + 7) % 23) as f32 * 0.05 - 0.5)
            .collect();

        let x_t = Tensor::from_vec(x_vals.clone(), (c_in, h, w), &dev).expect("x");
        let w_t = Tensor::from_vec(w_vals.clone(), (c_out, c_in, p, p), &dev).expect("w");
        let got = patch_conv_via_matmul(&x_t, &w_t, p)
            .expect("patch_conv")
            .to_vec2::<f32>()
            .expect("vec2");
        assert_eq!(got.len(), h_p * w_p);
        assert_eq!(got[0].len(), c_out);

        for r in 0..h_p {
            for c in 0..w_p {
                let patch_idx = r * w_p + c;
                for (o, &got_val) in got[patch_idx].iter().enumerate().take(c_out) {
                    let mut acc = 0.0_f32;
                    for ch in 0..c_in {
                        for kh in 0..p {
                            for kw in 0..p {
                                let x_i = (ch * h + (r * p + kh)) * w + (c * p + kw);
                                let w_i = ((o * c_in + ch) * p + kh) * p + kw;
                                acc += x_vals[x_i] * w_vals[w_i];
                            }
                        }
                    }
                    assert!(
                        (got_val - acc).abs() < 1e-5,
                        "mismatch at patch={patch_idx} o={o}: got={got_val} ref={acc}",
                    );
                }
            }
        }
    }

    #[test]
    fn pixtral_2d_rope_preserves_norm_and_encodes_row_col_coordinates() {
        let dev = Device::Cpu;
        let (h_p, w_p, head_dim) = (4, 6, 16);
        let (cos, sin) =
            build_pixtral_2d_rope_tables(h_p, w_p, head_dim, 10_000.0, DType::F32, &dev)
                .expect("rope2d");
        assert_eq!(
            cos.dims4().expect("cos dims"),
            (1, 1, h_p * w_p, head_dim / 2)
        );
        assert_eq!(
            sin.dims4().expect("sin dims"),
            (1, 1, h_p * w_p, head_dim / 2)
        );

        // Patch (0, 0) has zero angle: cos=1, sin=0 across all dimensions.
        let cos_v = cos
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let sin_v = sin
            .squeeze(0)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        for j in 0..(head_dim / 2) {
            assert!((cos_v[0][j] - 1.0).abs() < 1e-6);
            assert!(sin_v[0][j].abs() < 1e-6);
        }
        // Patch (1, 0) rotates height quarter [0..4] only, leaving width quarter [4..8] at angle 0.
        let row1_idx = w_p;
        assert!((cos_v[row1_idx][0] - 1.0_f32.cos()).abs() < 1e-5);
        for j in (head_dim / 4)..(head_dim / 2) {
            assert!((cos_v[row1_idx][j] - 1.0).abs() < 1e-6);
            assert!(sin_v[row1_idx][j].abs() < 1e-6);
        }
        // Patch (0, 1) leaves height quarter [0..4] at angle 0 and rotates width quarter [4..8].
        for j in 0..(head_dim / 4) {
            assert!((cos_v[1][j] - 1.0).abs() < 1e-6);
            assert!(sin_v[1][j].abs() < 1e-6);
        }
        let expected_w0 = 1.0_f64 / 10_000.0_f64.powf(2.0 / head_dim as f64);
        assert!((cos_v[1][head_dim / 4] - expected_w0.cos() as f32).abs() < 1e-5);

        // Norm preservation across every rotated token vector.
        let x_vals: Vec<f32> = (0..(2 * h_p * w_p * head_dim))
            .map(|i| ((i * 7 + 1) % 19) as f32 * 0.2 - 1.5)
            .collect();
        let x = Tensor::from_vec(x_vals, (1, 2, h_p * w_p, head_dim), &dev).unwrap();
        let rot = apply_2d_rope_half_split(&x, &cos, &sin).unwrap();
        let norm_in = x
            .sqr()
            .unwrap()
            .sum_keepdim(D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let norm_out = rot
            .sqr()
            .unwrap()
            .sum_keepdim(D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(norm_in.len(), norm_out.len());
        for (&ni, &no) in norm_in.iter().zip(norm_out.iter()) {
            assert!((ni - no).abs() < 1e-4);
        }
    }

    #[test]
    fn patch_merger_unfold_matches_channel_first_2x2_axis_order() {
        let dev = Device::Cpu;
        // 4x4 patch grid, d=2 channels, spatial_merge_size=2 -> 2x2=4 merged tokens of dim 8.
        let (h_p, w_p, d, s) = (4, 4, 2, 2);
        let mut vals = Vec::with_capacity(h_p * w_p * d);
        for r in 0..h_p {
            for c in 0..w_p {
                for ch in 0..d {
                    vals.push((ch * 100 + r * 10 + c) as f32);
                }
            }
        }
        let t = Tensor::from_vec(vals, (h_p * w_p, d), &dev).unwrap();
        let merged = unfold_spatial_patches(&t, h_p, w_p, s)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(merged.len(), 4);
        assert_eq!(merged[0].len(), 8);
        // Block (bh=0, bw=1) covers rows 0..2, cols 2..4.
        // Order must be (ch=0, kh=0..2, kw=0..2) then (ch=1, kh=0..2, kw=0..2):
        // ch=0: (0,2)=2, (0,3)=3, (1,2)=12, (1,3)=13; ch=1: 102, 103, 112, 113.
        assert_eq!(
            merged[1],
            vec![2.0, 3.0, 12.0, 13.0, 102.0, 103.0, 112.0, 113.0]
        );
    }

    #[test]
    fn image_placeholder_expansion_and_splice_reject_mismatch() {
        let cfg = sample_vision_cfg();
        let (ids, img_count) = expand_image_placeholder_tokens(224, 224, &cfg).expect("expand");
        assert_eq!(img_count, 64);
        assert_eq!(ids.len(), 8 * 9);
        assert_eq!(ids.iter().filter(|&&id| id == 10).count(), 64);
        assert_eq!(ids.iter().filter(|&&id| id == 12).count(), 7);
        assert_eq!(ids.iter().filter(|&&id| id == 13).count(), 1);
        assert_eq!(*ids.last().unwrap(), 13);

        // Verify splice exact replacement and strict error on count mismatch.
        let dev = Device::Cpu;
        let tok_ids = vec![1_u32, 10, 10, 12, 10, 10, 13, 4];
        let text_emb = Tensor::zeros((1, tok_ids.len(), 3), DType::F32, &dev).unwrap();
        let vis_emb = Tensor::from_vec(
            vec![
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (4, 3),
            &dev,
        )
        .unwrap();
        let spliced = splice_vision_embeddings(&tok_ids, &text_emb, &vis_emb, 10)
            .expect("splice ok")
            .squeeze(0)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        assert_eq!(spliced[0], vec![0.0, 0.0, 0.0]);
        assert_eq!(spliced[1], vec![1.0, 2.0, 3.0]);
        assert_eq!(spliced[2], vec![4.0, 5.0, 6.0]);
        assert_eq!(spliced[3], vec![0.0, 0.0, 0.0]);
        assert_eq!(spliced[4], vec![7.0, 8.0, 9.0]);
        assert_eq!(spliced[5], vec![10.0, 11.0, 12.0]);
        assert_eq!(spliced[6], vec![0.0, 0.0, 0.0]);
        assert_eq!(spliced[7], vec![0.0, 0.0, 0.0]);

        let wrong_vis = Tensor::zeros((3, 3), DType::F32, &dev).unwrap();
        let err = splice_vision_embeddings(&tok_ids, &text_emb, &wrong_vis, 10)
            .expect_err("must reject mismatch");
        assert!(err.to_string().contains("placeholder count mismatch"));
    }

    #[test]
    fn text_only_config_returns_clear_error_for_vlm_eval() {
        let phi4_json = r#"{
            "model_type": "phi3",
            "hidden_size": 3072,
            "intermediate_size": 8192,
            "num_hidden_layers": 32,
            "num_attention_heads": 24,
            "num_key_value_heads": 8,
            "vocab_size": 200064,
            "rms_norm_eps": 1e-5
        }"#;
        let err = VisionConfig::from_json_strs(phi4_json, None, "Phi-4-mini-instruct", None)
            .expect_err("non-VLM must return error");
        assert!(err.to_string().contains("has no vision_config"));
    }
}
