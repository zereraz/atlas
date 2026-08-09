// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for inclusionAI **Ling-3.0-flash** (`bailing_hybrid`).
//!
//! Ling is architecturally a Qwen3.5 hybrid-clone (35 KDA linear-attention +
//! 7 MLA full-attention + MoE + MTP), but uses its OWN tensor names. This
//! module provides the actual layer-assembly loop using Ling's
//! `{lp}.attention.*` names.
//!
//! Ling runs TP=1 on spark in v1 → `supports_tp() = false`.

use anyhow::Result;
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use crate::layer::TransformerLayer;
use crate::layers::{FfnComponent, MoeLayer, Qwen3SsmLayer};
use crate::weight_map::{
    DenseWeight, MtpWeights, SsmWeights, dense, detect_nvfp4_variant, gpu_concat_rows,
    interleave_ba, load_moe_bailing, load_mtp, load_ssm_bailing, quantize_to_nvfp4,
};

/// Ling-3.0-flash (bailing_hybrid).
pub struct BailingHybridWeightLoader;

impl BailingHybridWeightLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for BailingHybridWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelWeightLoader for BailingHybridWeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let layer_types: Vec<LayerType> = if config.layer_types.is_empty() {
            (0..config.num_hidden_layers)
                .map(|i| config.layer_type(i))
                .collect()
        } else {
            config.layer_types.clone()
        };

        let variant = detect_nvfp4_variant(store, config);
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let moe_weights = load_moe_bailing(
                store, &lp, config.num_experts, gpu, config, variant, absmax_k, quantize_k, stream,
                false,
            )?;
            let gate_nvfp4 = quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let moe_layer =
                MoeLayer::new(moe_weights, config.num_experts, Some(gate_nvfp4), gpu, config)?;
            let ffn = FfnComponent::Moe(moe_layer);

            match lt {
                LayerType::FullAttention => {
                    let layer = build_full_attention_bailing(
                        i,
                        store,
                        &lp,
                        gpu,
                        variant,
                        config,
                        layer_kv_dtypes[attn_idx],
                        attn_idx,
                        input_norm,
                        post_attn_norm,
                        ffn,
                    )?;
                    layers.push(layer);
                    attn_idx += 1;
                }
                LayerType::LinearAttention => {
                    let layer = build_linear_attention_bailing(
                        store, &lp, gpu, variant, config, h, absmax_k, quantize_k, stream,
                        input_norm, post_attn_norm, ffn,
                    )?;
                    layers.push(layer);
                }
                LayerType::Moe => unreachable!("Ling has no standalone MoE layer"),
            }

            if (i + 1) % 10 == 0 || i < 5 {
                let free_gb = gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0);
                tracing::info!("Ling[{i}]: loaded layers 0..{i} — {free_gb:.1} GB free");
            }
        }
        Ok(layers)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.word_embeddings.weight"))
            .or_else(|_| dense(store, &format!("{prefix}.embed_tokens.weight")))
            .map_err(|e| anyhow::anyhow!("Ling embedding not found ({e:#})"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = &config.weight_prefix;
        dense(store, &format!("{prefix}.norm.weight"))
            .map_err(|e| anyhow::anyhow!("Ling final norm not found ({e:#})"))
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            self.load_embedding(store, config)
        }
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("Ling: no MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        Ok(Some(load_mtp(store, config.num_experts, gpu, variant)?))
    }
}

/// Ling KDA linear-attention layer assembly (mirrors Qwen3.5's NVFP4 arm but
/// sources weights from `load_ssm_bailing` for Ling's `attention.*` names).
#[allow(clippy::too_many_arguments)]
fn build_linear_attention_bailing(
    store: &WeightStore,
    lp: &str,
    gpu: &dyn GpuBackend,
    variant: crate::weight_map::Nvfp4Variant,
    config: &ModelConfig,
    h: usize,
    absmax_k: spark_runtime::gpu::KernelHandle,
    quantize_k: spark_runtime::gpu::KernelHandle,
    stream: u64,
    input_norm: DenseWeight,
    post_attn_norm: DenseWeight,
    ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    let ssm35 = load_ssm_bailing(store, lp, gpu, variant)?;

    let qkv_rows = config.ssm_qkv_size();
    let z_rows = config.ssm_z_size();
    let qkvz_dense = gpu_concat_rows(
        &ssm35.in_proj_qkv,
        qkv_rows,
        &ssm35.in_proj_z,
        z_rows,
        h,
        gpu,
    )?;

    let nv = config.linear_num_value_heads;
    let nk = config.linear_num_key_heads;
    let ba_dense = interleave_ba(
        &DenseWeight { weight: ssm35.in_proj_a.weight },
        &DenseWeight { weight: ssm35.in_proj_b.weight },
        nv,
        nk,
        h,
        gpu,
    )?;

    let qkvz_size = config.ssm_qkvz_size();
    let qkvz_nvfp4 = quantize_to_nvfp4(&qkvz_dense, qkvz_size, h, gpu, absmax_k, quantize_k, stream)?;
    let qkvz_nvfp4_t = qkvz_nvfp4.transpose_for_gemm(gpu, qkvz_size, h)?;

    let value_dim = nv * config.linear_value_head_dim;
    let out_proj_nvfp4 =
        quantize_to_nvfp4(&ssm35.out_proj, h, value_dim, gpu, absmax_k, quantize_k, stream)?;
    let out_proj_nvfp4_t = out_proj_nvfp4.transpose_for_gemm(gpu, h, value_dim)?;

    let ssm = SsmWeights {
        in_proj_qkvz: qkvz_dense,
        in_proj_ba: ba_dense,
        conv1d: ssm35.conv1d,
        a_log: ssm35.a_log,
        dt_bias: ssm35.dt_bias,
        norm: ssm35.norm,
        out_proj: out_proj_nvfp4,
    };

    let mut layer = Qwen3SsmLayer::new_sequential(
        input_norm,
        ssm,
        post_attn_norm,
        ffn,
        Some(qkvz_nvfp4),
        Some(qkvz_nvfp4_t),
        Some(out_proj_nvfp4_t),
        config,
        gpu,
    )?;
    layer.predequant_for_prefill(gpu, config, stream)?;
    Ok(Box::new(layer))
}

/// Ling MLA full-attention layer assembly.
///
/// Ling has `q_lora_rank = null` (a direct `q_proj`, no `wq_a`/`wq_b` split)
/// and uses absorbed MLA with latent KV (kv_lora_rank=512, rope=64, nope=128,
/// v=128). The absorbed-weight construction (wq_b split, w_uk_t, w_uv,
/// block-diagonals, yarn) mirrors Qwen3.5's `phase_*` MLA helpers and is the
/// remaining careful algebra. Stubbed to fail loudly at load time until wired.
#[allow(clippy::too_many_arguments)]
fn build_full_attention_bailing(
    i: usize,
    _store: &WeightStore,
    lp: &str,
    _gpu: &dyn GpuBackend,
    _variant: crate::weight_map::Nvfp4Variant,
    _config: &ModelConfig,
    _layer_kv_dtype: KvCacheDtype,
    _attn_idx: usize,
    _input_norm: DenseWeight,
    _post_attn_norm: DenseWeight,
    _ffn: FfnComponent,
) -> Result<Box<dyn TransformerLayer>> {
    anyhow::bail!(
        "Ling[{i}]: MLA full-attention assembly ({lp}.attention.*) not yet implemented — \
         see LING_RUST_ATLAS.md"
    )
}
