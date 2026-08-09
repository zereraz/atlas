// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for inclusionAI **Ling-3.0-flash** (`model_type = "bailing_hybrid"`).
//!
//! Architecture (verified against `modeling_bailing_moe_v3.py` on the
//! checkpoint): 42 layers = 35 KDA linear-attention + 7 MLA full-attention
//! (`layer_group_size = 6` → FullAttention at idx where `(i+1) % 6 == 0`),
//! 512-expert sigmoid-routed MoE with DeepSeek-V3-style `expert_bias`, and a
//! single MTP head at layer 42. Quant format is `mxfp4-pack-quantized`.

#![allow(unused_imports)]

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{
    LayerType, ModelConfig, default_one, default_partial_rotary, default_rms_eps,
    default_rope_theta, finalize_config, parse_quantization_config, validate_config,
};

pub(crate) fn parse_bailing_hybrid(raw: &serde_json::Value) -> Result<ModelConfig> {
    let mut config: ModelConfig =
        serde_json::from_value(raw.clone()).context("Failed to parse bailing_hybrid config.json")?;

    config.nested_config = false;

    // ── Attention (MLA) dims ────────────────────────────────────────────────
    // Ling uses MLA: kv_lora_rank / qk_rope_head_dim / qk_nope_head_dim already
    // deserialize into ModelConfig's canonical fields. head_dim for MLA = the
    // KDA head dim (`head_dim: 128`). q_lora_rank is null (no Q down-proj).
    ensure!(
        config.kv_lora_rank > 0,
        "bailing_hybrid: expected kv_lora_rank > 0 (MLA), got {}",
        config.kv_lora_rank
    );

    // ── RoPE ────────────────────────────────────────────────────────────────
    // Ling: rope_theta = 6e6, partial_rotary_factor = 0.5 (rotates the first
    // half of each q/k head). These already deserialize; verify non-default.
    ensure!(
        config.rope_theta > 0.0,
        "bailing_hybrid: rope_theta must be > 0 (Ling uses 6e6)"
    );

    // ── Hybrid layer layout: layer_group_size=6 → full-attn every 6th ───────
    // FullAttention at idx where (i+1) % group == 0 → {5, 11, 17, 23, 29, 35, 41}.
    let group = raw
        .get("layer_group_size")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    if config.full_attention_interval == 1 && group > 1 {
        config.full_attention_interval = group;
    }
    if config.layer_types.is_empty() && group > 1 {
        config.layer_types = (0..config.num_hidden_layers)
            .map(|i| {
                if (i + 1) % group == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect();
    }

    // ── KDA (linear-attention) dims → canonical SSM fields ──────────────────
    // Ling KDA: hidden→q/k/v projections, head_dim=128, conv kernel=4.
    // Qwen3.5 GDN uses in_proj_qkv fused; Ling uses separate q/k/v + f/g/b, but
    // the GROUP/TOTAL dims map identically.
    let head_dim = config.head_dim.max(1);
    let n_heads = config.num_attention_heads;
    if config.linear_num_key_heads == 0 {
        config.linear_num_key_heads = n_heads;
    }
    if config.linear_num_value_heads == 0 {
        config.linear_num_value_heads = n_heads;
    }
    if config.linear_key_head_dim == 0 {
        config.linear_key_head_dim = head_dim;
    }
    if config.linear_value_head_dim == 0 {
        config.linear_value_head_dim = head_dim;
    }
    if config.linear_conv_kernel_dim == 0 {
        config.linear_conv_kernel_dim =
            raw.get("short_conv_kernel_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(4) as usize;
    }

    // ── MoE ─────────────────────────────────────────────────────────────────
    // Ling: num_experts 512, num_experts_per_tok 8, sigmoid router + expert_bias,
    // moe_intermediate_size 768, NO dedicated shared-expert gate tensor.
    config.scoring_func = "sigmoid".to_string();
    config.use_routing_bias = true; // checkpoint ships mlp.gate.expert_bias
    // norm_topk_prob + routed_scaling_factor already deserialize from config.

    // first_k_dense_replace=2: the first 2 layers are dense FFN (no MoE).
    // Qwen35 load loop uses `decoder_sparse_step`/`first_k_dense_replace`-style
    // flags via layer_types, so the dense layers are expressed that way; the
    // weight loader's per-layer MoE-vs-dense decision reads this.

    // ── MTP ─────────────────────────────────────────────────────────────────
    let num_nextn = raw
        .get("num_nextn_predict_layers")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(0) as usize;
    if num_nextn > 0 && config.num_mtp_modules == 0 {
        config.num_mtp_modules = num_nextn;
    }

    // Weight prefix: Ling uses `model.{layers,word_embeddings,norm}`.
    config.weight_prefix = "model".to_string();

    finalize_config(&mut config, raw)?;
    validate_config(&config).context("bailing_hybrid config validation failed")?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bailing_layer_types() {
        let raw = serde_json::json!({
            "model_type": "bailing_hybrid",
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "layer_group_size": 6,
            "head_dim": 128,
            "num_attention_heads": 32,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 128,
            "v_head_dim": 128,
            "rope_theta": 6000000.0,
            "partial_rotary_factor": 0.5,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "n_group": 8,
            "topk_group": 4,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": true,
            "short_conv_kernel_size": 4,
            "num_nextn_predict_layers": 1,
            "quantization_config": { "quant_method": "compressed-tensors",
                                      "format": "mxfp4-pack-quantized" }
        });
        let config = parse_bailing_hybrid(&raw).expect("parse");
        // 42 layers, full-attn every 6th = 7 layers.
        assert_eq!(config.layer_types.len(), 42);
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::FullAttention)
                .count(),
            7
        );
        assert_eq!(
            config
                .layer_types
                .iter()
                .filter(|t| **t == LayerType::LinearAttention)
                .count(),
            35
        );
        assert_eq!(config.layer_types[5], LayerType::FullAttention);
        assert_eq!(config.layer_types[41], LayerType::FullAttention);
        assert_eq!(config.layer_types[0], LayerType::LinearAttention);
        assert_eq!(config.full_attention_interval, 6);
        assert_eq!(config.scoring_func, "sigmoid");
        assert!(config.use_routing_bias);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.num_experts_per_tok, 8);
        assert_eq!(config.num_mtp_modules, 1);
        assert_eq!(config.kv_lora_rank, 512);
        assert_eq!(config.qk_rope_head_dim, 64);
        assert_eq!(config.weight_prefix, "model");
    }
}
