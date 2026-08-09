// SPDX-License-Identifier: AGPL-3.0-only

//! Weight loader for inclusionAI **Ling-3.0-flash** (`bailing_hybrid`).
//!
//! Ling is architecturally a Qwen3.5 hybrid-clone (35 KDA linear-attention +
//! 7 MLA full-attention + MoE + MTP), so we reuse `qwen35::load_layers` for the
//! layer-by-layer weight assembly. The ONLY structural deltas handled here are
//! the tensor naming differences:
//!
//! | item | Qwen3.5 | Ling (Bailing) |
//! |------|---------|----------------|
//! | token embedding | `{prefix}.embed_tokens.weight` | `model.word_embeddings.weight` |
//! | final norm | `{prefix}.norm.weight` | `model.norm.weight` (same) |
//! | lm_head | `lm_head.weight` | `lm_head.weight` (untied) |
//!
//! Ling is **not** TP-aware in this first cut: the MLA + GDN layers use the
//! pre-existing `AttentionWeights`/`SsmWeights` shims but attention projections
//! are stored non-sharded (spark runs TP=1 anyway — see atom_ep2.sh). Mark
//! `supports_tp() = false` so the server fails-fast instead of misloading on
//! `--tp-size > 1`.
//!
//! The full per-layer MLA + KDA assembly lives in `super::qwen35`; Ling only
//! distincts the embeddings/norm/load-surface here. MTP: Ling slots
//! `eh_proj`/`enorm`/`hnorm` + un-quantized dense experts into the existing
//! `load_mtp` shape — its loader reuses the standard `ModelWeightLoader::
//! load_mtp` via the super loader.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::WeightStore;

use super::ModelWeightLoader;
use super::qwen35::Qwen35WeightLoader;
use crate::layer::TransformerLayer;
use crate::weight_map::{DenseWeight, MtpWeights};

/// Ling-3.0-flash (bailing_hybrid). Layer loading is delegated to the
/// Qwen35 engine (architecture is a clone); only embeddings + norms +
/// lm_head differ in tensor naming and are handled explicitly here.
pub struct BailingHybridWeightLoader {
    inner: Qwen35WeightLoader,
}

impl BailingHybridWeightLoader {
    pub fn new() -> Self {
        Self {
            inner: Qwen35WeightLoader,
        }
    }
}

impl Default for BailingHybridWeightLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelWeightLoader for BailingHybridWeightLoader {
    fn supports_tp(&self) -> bool {
        // Ling runs TP=1 on spark in v1; the MLA + GDN builders in atlas
        // already shard full-attention Q/K/V/O across ranks, but KDA/
        // MTP layers are replicated. Marking false forces fail-fast at
        // startup if someone passes --tp-size > 1, instead of misloading.
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        // Ling uses the same MLA/GDN/MoE layout as Qwen3.5 — delegate to
        // the proven qwen35 load loop. `config.weight_prefix` was set to
        // "model" by the bailing parser so `{prefix}.embed_tokens.weight`
        // resolves to `model.embed_tokens.weight`, which Ling does NOT
        // provide (it uses `word_embeddings`) → override embedding below.
        self.inner.load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        // Ling names the token embedding `model.word_embeddings.weight`;
        // Qwen3.5 uses `{prefix}.embed_tokens.weight`. Try Ling first, fall back.
        let prefix = &config.weight_prefix;
        crate::weight_map::dense(store, &format!("{prefix}.word_embeddings.weight"))
            .or_else(|_| self.inner.load_embedding(store, config))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        // Same name as Qwen3.5 (`{prefix}.norm.weight`).
        self.inner.load_final_norm(store, config, gpu)
    }

    fn load_lm_head(&self, store: &WeightStore, config: &ModelConfig) -> Result<DenseWeight> {
        // Ling has a dedicated lm_head (untied).
        if store.contains("lm_head.weight") {
            crate::weight_map::dense(store, "lm_head.weight")
        } else {
            self.inner.load_lm_head(store, config)
        }
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // Ling's MTP block is at layer 42: eh_proj/enorm/hnorm + unquantized
        // BF16 experts + MLA attention (`mtp_use_kda:false` → full-attn, NOT
        // KDA). Reuse the standard load_mtp pipeline via the inner loader.
        self.inner.load_mtp_weights(store, config, gpu)
    }
}
