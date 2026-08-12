// SPDX-License-Identifier: AGPL-3.0-only

//! MXFP4 (microscaling FP4) → BF16 dequantization for Ling-3.0-flash.
//!
//! Ling-3.0-flash-MXFP4 uses the `compressed-tensors` format
//! `mxfp4-pack-quantized`. Unlike NVFP4 (tensor-scaled e2m1), MXFP4 is
//! block-scaled e2m1 with **no per-tensor global scale**:
//!
//! | tensor              | dtype        | shape                  |
//! |---------------------|--------------|------------------------|
//! | `.weight_packed`    | `uint8`      | `[N, K/2]` (2 e2m1/byte)|
//! | `.weight_scale`     | `float8_e4m3`| `[N, ceil(K/GROUP)]`   |
//!
//! Dequant for element (row, col):
//!   `bf16[row,col] = e2m1(nibble[row,col]) * e4m3_to_f32(scale[row, col/GROUP])`
//!
//! where GROUP = 32 (compressed-tensors `group_size` for this checkpoint).
//!
//! We dequant on the CPU (matching the proven `dequant_fp8_blockscaled_to_bf16`
//! path) and upload BF16. This runs once at load time; a follow-up perf cut
//! can move dequant into a CUDA kernel and feed a native MXFP4 grouped GEMM.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::fp8_lut::fp8_e4m3_to_f32;

/// MXFP4 per-group scale in OCP MX is **E8M0** (a plain power-of-two exponent):
///   value = 2^(byte - 127); 0xFF is the MX-NaN sentinel (never produced by a
///   well-formed quantizer for real weights — treat as error)
///
/// Ling-3.0-Flash ships its expert scales as UInt8 in this exact E8M0 form
/// (bytes ~0x78..0x79 → scales ~1/128..1/64 → weights absmax ≈ 0.1).
/// Previously we decoded these bytes as FP8-E4M3 (0x78 → 256!), which is the
/// routed-expert residual-explosion bug: per-#group scales ~33000× too large.
#[inline]
pub(crate) fn e8m0_to_f32(bits: u8) -> f32 {
    debug_assert!(bits != 0xFF, "e8m0 scale: MX-NaN sentinel 0xFF in weight scale");
    let exp = bits as i32 - 127;
    // branch-free pow2: construct float with exponent bits
    f32::from_bits(((exp + 127) as u32) << 23)
}

// Phase B scaffold: `dequant_mxfp4_to_bf16` is exercised by unit tests and will
// be called by the `bailing` weight loader (next commit). Allow dead code until
// the loader lands so `#![deny(warnings)]` doesn't gate the primitive merge.
#[inline]
pub(crate) fn bf16_bytes_from_f32(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    let hi = (bits >> 16) as u16;
    hi.to_le_bytes()
}

/// MXFP4 block-scaling group size for Ling-3.0-flash (compressed-tensors
/// `group_size` under `quantization_config.config_groups.group_0.weights`).
#[allow(dead_code)]
pub const MXFP4_GROUP_SIZE: usize = 32;

/// e2m1 (4-bit float) → f32. Table-driven: 8 magnitude codes (3-bit) × sign.
/// E2M1 has 1 sign, 2 exponent (bias 1), 1 mantissa bit. Values:
///   code 0-7 → 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (sign applied).
#[inline]
pub(crate) fn e2m1_to_f32(nibble: u8) -> f32 {
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if nibble & 0x8 != 0 { -1.0 } else { 1.0 };
    sign * MAG[(nibble & 0x7) as usize]
}

/// Dequant one MXFP4-packed 2D weight to BF16 on GPU.
///
/// `prefix` is the tensor name without `.weight_packed` / `.weight_scale`,
/// e.g. `model.layers.5.mlp.experts.0.gate_proj`.
pub(crate) fn dequant_mxfp4_to_bf16(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    let packed = store.get(&format!("{prefix}.weight_packed"))?;
    let scale = store.get(&format!("{prefix}.weight_scale"))?;

    ensure!(
        packed.dtype == WeightDtype::UInt8,
        "Expected UInt8 (packed e2m1) for {prefix}.weight_packed, got {:?}",
        packed.dtype
    );
    // MXFP4 e4m3 scales are commonly serialized as raw UInt8 bytes
    // (compressed-tensors mxfp4-pack-quantized) — accept either FP8E4M3 or U8.
    ensure!(
        matches!(scale.dtype, WeightDtype::FP8E4M3 | WeightDtype::UInt8),
        "Expected FP8E4M3 or UInt8 for {prefix}.weight_scale, got {:?}",
        scale.dtype
    );
    ensure!(
        packed.shape.len() == 2 && scale.shape.len() == 2,
        "Expected 2D packed+scale for {prefix}, got packed={:?} scale={:?}",
        packed.shape,
        scale.shape
    );

    let n = packed.shape[0];
    let half_k = packed.shape[1];
    let k = half_k * 2;
    let sn = scale.shape[0];
    let sk = scale.shape[1];
    ensure!(sn == n, "{prefix}: scale rows {sn} != weight rows {n}");
    let group = k.div_ceil(sk);
    ensure!(
        group == MXFP4_GROUP_SIZE,
        "{prefix}: inferred MXFP4 group size {group} != {MXFP4_GROUP_SIZE}"
    );

    tracing::debug!(
        "MXFP4 dequant {prefix}: shape=[{n},{k}] groups={sk} (group_size={group})"
    );

    // Download packed weights + per-group scales to host.
    let packed_bytes = packed.byte_size();
    let mut packed_buf = vec![0u8; packed_bytes];
    gpu.copy_d2h(packed.ptr, &mut packed_buf)
        .with_context(|| format!("D2H failed for {prefix}.weight_packed"))?;

    let scale_bytes = scale.byte_size();
    let mut scale_buf = vec![0u8; scale_bytes];
    gpu.copy_d2h(scale.ptr, &mut scale_buf)
        .with_context(|| format!("D2H failed for {prefix}.weight_scale"))?;

    // CPU dequant → BF16 bytes (2 bytes/element).
    let total = n * k;
    let mut bf16_out = vec![0u8; total * 2];
    for row in 0..n {
        let packed_row = row * half_k;
        let scale_row = row * sk;
        let out_row = row * k;
        for col in 0..k {
            let byte = packed_buf[packed_row + col / 2];
            // Little-nibble first: even col = low nibble, odd col = high nibble.
            let nibble = if col % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            // MXFP4 scales are E8M0 (power-of-two), NOT FP8-E4M3. See e8m0_to_f32.
            let s = e8m0_to_f32(scale_buf[scale_row + col / group]);
            let v = e2m1_to_f32(nibble) * s;
            let b = bf16_bytes_from_f32(v);
            bf16_out[(out_row + col) * 2] = b[0];
            bf16_out[(out_row + col) * 2 + 1] = b[1];
        }
    }

    let ptr = gpu.alloc(bf16_out.len())?;
    gpu.copy_h2d(&bf16_out, ptr)
        .with_context(|| format!("H2D upload failed for {prefix} dequant"))?;
    Ok(ptr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lut_matches_spec() {
        // E2M1 magnitudes: code → value (1 sign, 2 exp, 1 mantissa).
        assert_eq!(e2m1_to_f32(0x0), 0.0);
        assert_eq!(e2m1_to_f32(0x1), 0.5);
        assert_eq!(e2m1_to_f32(0x2), 1.0);
        assert_eq!(e2m1_to_f32(0x3), 1.5);
        assert_eq!(e2m1_to_f32(0x4), 2.0);
        assert_eq!(e2m1_to_f32(0x5), 3.0);
        assert_eq!(e2m1_to_f32(0x6), 4.0);
        assert_eq!(e2m1_to_f32(0x7), 6.0);
        // Sign bit.
        assert_eq!(e2m1_to_f32(0x8), -0.0);
        assert_eq!(e2m1_to_f32(0x9), -0.5);
        assert_eq!(e2m1_to_f32(0xF), -6.0);
    }

    #[test]
    fn bf16_bytes_roundtrip() {
        // 1.0f32 = 0x3F800000 → BF16 high half 0x3F80 → bytes [0x80, 0x3F].
        assert_eq!(bf16_bytes_from_f32(1.0), [0x80, 0x3F]);
        // -2.0f32 = 0xC0000000 → BF16 0xC000 → bytes [0x00, 0xC0].
        assert_eq!(bf16_bytes_from_f32(-2.0), [0x00, 0xC0]);
        // 0.5f32 = 0x3F000000 → [0x00, 0x3F].
        assert_eq!(bf16_bytes_from_f32(0.5), [0x00, 0x3F]);
    }

    #[test]
    fn mxfp4_scalar_composition() {
        // MXFP4 scales are E8M0: byte 0x80 (=128) → 2^(128-127) = 2.0
        assert_eq!(e8m0_to_f32(0x80), 2.0);
        // byte 0x7F (=127) → 2^0 = 1.0
        assert_eq!(e8m0_to_f32(0x7F), 1.0);
        // byte 0x78 (=120) — the actual value all over Ling's expert scales → 2^-7 = 1/128
        assert_eq!(e8m0_to_f32(0x78), 1.0 / 128.0);
        // byte 0x79 (=121) → 2^-6 = 1/64
        assert_eq!(e8m0_to_f32(0x79), 1.0 / 64.0);
        // e2m1=1.5 × e8m0(2.0) = 3.0
        let product = e2m1_to_f32(0x3) * e8m0_to_f32(0x80);
        assert!((product - 3.0).abs() < 1e-6, "product={product}");
        // (Keep the E4M3 table pinned too — still used by the FP8 path.)
        assert_eq!(fp8_e4m3_to_f32(0x40), 2.0);
    }
}
