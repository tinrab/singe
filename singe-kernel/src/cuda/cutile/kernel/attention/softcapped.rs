#[cutile::module]
mod kernels {
    use crate::cuda::cutile::kernel::utility::*;
    use cutile::core::*;

    const VECTOR_TILE_SIZE: i32 = 128;

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f32_hd4(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f32::<4>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f32_hd8(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f32::<8>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f32_hd16(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f32::<16>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f32_hd32(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f32::<32>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f32_hd64(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f32::<64>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd4(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<4>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd8(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<8>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd16(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<16>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd32(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<32>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd64(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<64>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn fmha_decode_lse_f32_hd128(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        fmha_decode_lse_f32::<128>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f16_hd4(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f16::<4>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f16_hd8(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f16::<8>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f16_hd16(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f16::<16>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f16_hd32(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f16::<32>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_f16_hd64(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_f16::<64>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_bf16_hd4(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_bf16::<4>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_bf16_hd8(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_bf16::<8>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_bf16_hd16(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_bf16::<16>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_bf16_hd32(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_bf16::<32>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_attention_bf16_hd64(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_attention_bf16::<64>(
            out,
            query,
            key,
            value,
            batch,
            query_len,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_sequence_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_sequence_stride,
            query_head_stride,
            key_head_stride,
            value_head_stride,
            output_head_stride,
            scale,
            causal,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f32_hd4(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f32::<4>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f32_hd8(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f32::<8>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f32_hd16(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f32::<16>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f32_hd32(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f32::<32>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f32_hd64(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f32::<64>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f16_hd4(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f16::<4>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f16_hd8(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f16::<8>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f16_hd16(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f16::<16>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f16_hd32(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f16::<32>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_f16_hd64(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_f16::<64>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_bf16_hd4(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_bf16::<4>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_bf16_hd8(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_bf16::<8>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_bf16_hd16(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_bf16::<16>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_bf16_hd32(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_bf16::<32>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    #[cutile::entry()]
    pub unsafe fn softcapped_window_decode_lse_bf16_hd64(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        softcapped_window_decode_lse_bf16::<64>(
            out,
            lse,
            query,
            key,
            value,
            batch,
            key_len,
            heads,
            kv_heads,
            query_batch_stride,
            key_batch_stride,
            value_batch_stride,
            output_batch_stride,
            query_head_stride,
            key_sequence_stride,
            value_sequence_stride,
            output_head_stride,
            key_head_stride,
            value_head_stride,
            scale,
            window_size,
            soft_cap,
            has_soft_cap,
            output_len,
            output_values_per_batch,
        );
    }

    fn sliding_window_attention_f32<const HEAD_DIM: i32, const WINDOW: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();

        // Sliding-window attention maps one lane to one output scalar.
        // Absolute query/key positions define the local history window before the two-pass softmax.
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let effective_window = if WINDOW == 0 { window } else { WINDOW };
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn fmha_prefill_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn mla_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: i32,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        q_features: i32,
        kv_features: i32,
        qpe_features: i32,
        kpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(q_features, tile_shape)
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * kv_features, tile_shape)
                + kv_head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + query_index * broadcast_scalar(qpe_features, tile_shape)
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + broadcast_scalar(key_index * kpe_features, tile_shape)
                + kv_head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn sparse_mla_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        q_features: i32,
        kv_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(q_features, tile_shape)
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + key_index * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + query_index * broadcast_scalar(qpe_features, tile_shape)
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + key_index * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_split_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_batch_base
                + key_index * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = key_pe_batch_base
                + key_index * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value: *mut f32,
        key_pe: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        key_pe_batch_base: Tile<i32, { [128] }>,
        key_index: i32,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets =
                key_batch_base + broadcast_scalar(key_index * head_dim + dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets =
                key_pe_batch_base + broadcast_scalar(key_index * pe_dim + dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_f32(
        query: *mut f32,
        query_pe: *mut f32,
        key_value_cache: *mut f32,
        key_pe_cache: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_value_cache, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector(query_pe, query_offsets, valid, 0.0f32);
            let key_values = load_vector(key_pe_cache, key_offsets, valid, 0.0f32);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_f16(
        query: *mut f16,
        query_pe: *mut f16,
        key_value_cache: *mut f16,
        key_pe_cache: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_f16_as_f32(query, query_offsets, valid);
            let key_values = load_vector_f16_as_f32(key_value_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_f16_as_f32(query_pe, query_offsets, valid);
            let key_values = load_vector_f16_as_f32(key_pe_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn mla_decode_score_paged_bf16(
        query: *mut bf16,
        query_pe: *mut bf16,
        key_value_cache: *mut bf16,
        key_pe_cache: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        query_pe_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        q_features: i32,
        qpe_features: i32,
        head_dim: i32,
        pe_dim: i32,
        key_value_cache_block_stride: i32,
        key_pe_cache_block_stride: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim_index in 0i32..head_dim {
            let query_offsets = query_batch_base
                + head * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(head_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_bf16_as_f32(query, query_offsets, valid);
            let key_values = load_vector_bf16_as_f32(key_value_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        for dim_index in 0i32..pe_dim {
            let query_offsets = query_pe_batch_base
                + head * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let key_offsets = physical_block
                * broadcast_scalar(key_pe_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(pe_dim, tile_shape)
                + broadcast_scalar(dim_index, tile_shape);
            let query_values = load_vector_bf16_as_f32(query_pe, query_offsets, valid);
            let key_values = load_vector_bf16_as_f32(key_pe_cache, key_offsets, valid);
            score = score + query_values * key_values;
        }
        let _ = q_features;
        let _ = qpe_features;
        score * broadcast_scalar(scale, tile_shape)
    }

    fn fmha_prefill_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        // Pass 1 finds the stabilized row maximum after masking and soft cap.
        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        // Pass 2 accumulates the denominator and one V dimension.
        // It uses the same masked, soft-capped scores.
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn softcapped_window_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_key_valid(valid, query_index, key_index, causal, window_size);
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let score = softcapped_attention_score(score, soft_cap, has_soft_cap);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn softcapped_window_key_valid(
        valid: Tile<bool, { [128] }>,
        query_index: Tile<i32, { [128] }>,
        key_index: i32,
        causal: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let key_tile = broadcast_scalar(key_index, tile_shape);
        let mut key_valid = valid
            & if causal != 0 {
                cmpi(key_tile, query_index, predicate::LessThanOrEqual)
            } else {
                constant(true, tile_shape)
            };
        if window_size > 0 {
            // The local window is symmetric around the query index here.
            // Causal mode above still removes keys to the right.
            key_valid = key_valid
                & cmpi(
                    key_tile,
                    query_index - broadcast_scalar(window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    key_tile,
                    query_index + broadcast_scalar(window_size, tile_shape),
                    predicate::LessThanOrEqual,
                );
        }
        key_valid
    }

    fn softcapped_attention_score(
        score: Tile<f32, { [128] }>,
        soft_cap: f32,
        has_soft_cap: i32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        if has_soft_cap != 0 {
            // Bound logits with cap * tanh(score / cap).
            // This matches Gemma's attention logit soft-capping.
            let cap = broadcast_scalar(soft_cap, tile_shape);
            tanh(score / cap) * cap
        } else {
            score
        }
    }

    fn fmha_decode_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn softcapped_window_decode_lse_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid =
                softcapped_window_decode_key_valid(valid, key_index, query_position, window_size);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_values = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_values, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(
            lse,
            lse_offsets,
            max_score + log(denom),
            valid & has_values & first_dim,
        );
    }

    fn softcapped_window_decode_key_valid(
        valid: Tile<bool, { [128] }>,
        key_index: i32,
        query_position: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let mut key_valid = valid;
        if window_size > 0 {
            key_valid = key_valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    broadcast_scalar(query_position - window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                );
        }
        key_valid
    }

    fn fmha_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        window_size: i32,
        soft_cap: f32,
        has_soft_cap: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let query_position = key_len - 1;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = softcapped_window_decode_split_key_valid(
                valid,
                key_index,
                key_len,
                query_position,
                window_size,
            );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = softcapped_attention_score(
                score * broadcast_scalar(scale, tile_shape),
                soft_cap,
                has_soft_cap,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(valid & has_keys, sum / denom, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn softcapped_window_decode_split_key_valid(
        valid: Tile<bool, { [128] }>,
        key_index: Tile<i32, { [128] }>,
        key_len: i32,
        query_position: i32,
        window_size: i32,
    ) -> Tile<bool, { [128] }> {
        let tile_shape = const_shape![128];
        let mut key_valid = valid
            & cmpi(
                key_index,
                broadcast_scalar(key_len, tile_shape),
                predicate::LessThan,
            );
        if window_size > 0 {
            key_valid = key_valid
                & cmpi(
                    key_index,
                    broadcast_scalar(query_position - window_size, tile_shape),
                    predicate::GreaterThanOrEqual,
                );
        }
        key_valid
    }

    fn attention_sink_decode_splitk_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector(query, query_offsets, key_valid, 0.0f32);
                let key_tile = load_vector(key, key_offsets, key_valid, 0.0f32);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn attention_sink_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn attention_sink_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        sinks: *mut f32,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);
        let start_q_tile = broadcast_scalar(start_q, tile_shape);
        let window_start = if window > 0 {
            broadcast_scalar(start_q + 1 - window, tile_shape)
        } else {
            broadcast_scalar(0i32, tile_shape)
        };

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let first_split = cmpi(split, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let sink = load_vector(sinks, head, valid & first_split, -1.0e20f32);
        max_score = select(
            valid
                & first_split
                & cmpf(
                    sink,
                    max_score,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            sink,
            max_score,
        );

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                )
                & cmpi(key_index, start_q_tile, predicate::LessThanOrEqual)
                & cmpi(key_index, window_start, predicate::GreaterThanOrEqual);
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let sink_weight = exp(sink - max_score);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let sink_weight = select(valid & first_split, sink_weight, zero);
        denom = denom + sink_weight;

        let has_terms = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_terms, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_terms, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn fmha_prefill_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_prefill_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let lse_offsets = lse_offsets * broadcast_scalar(_query_len, tile_shape) + query_index;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_prefill_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn fmha_prefill_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_sink_prefill_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        sinks: *mut f32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        start_q: i32,
        window: i32,
        scale: f32,
        causal: i32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let query_pos = broadcast_scalar(start_q, tile_shape) + query_index;
        let zero_i32: Tile<i32, { [128] }> = constant(0i32, tile_shape);
        let window_start = if window == 0 {
            zero_i32
        } else {
            let raw_window_start = query_pos - broadcast_scalar(window - 1, tile_shape);
            select(
                cmpi(raw_window_start, zero_i32, predicate::GreaterThan),
                raw_window_start,
                zero_i32,
            )
        };
        let sink = load_vector(sinks, head, valid, 0.0f32);

        let mut max_score = sink;
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = exp(sink - max_score);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut key_valid = valid
                & if causal != 0 {
                    cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_pos,
                        predicate::LessThanOrEqual,
                    )
                } else {
                    constant(true, tile_shape)
                };
            if window != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        window_start,
                        predicate::GreaterThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn fmha_decode_lse_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_decode_lse_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, valid);
            sum = sum + weight * value_tile;
        }

        let output = sum / denom;
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(heads, tile_shape) + head;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        store_vector(lse, lse_offsets, max_score + log(denom), valid & first_dim);
    }

    fn fmha_decode_splitk_f16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_f16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_f16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn fmha_decode_splitk_bf16<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        splits: i32,
        kv_len_per_split: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_head_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_head_stride: i32,
        output_split_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        lse_batch_stride: i32,
        lse_head_stride: i32,
        scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let head_split_values = splits * HEAD_DIM;
        let head = batch_output_offset / broadcast_scalar(head_split_values, tile_shape);
        let head_output_offset =
            batch_output_offset - head * broadcast_scalar(head_split_values, tile_shape);
        let split = head_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = head_output_offset - split * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);
        let split_start = split * broadcast_scalar(kv_len_per_split, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for split_offset in 0i32..kv_len_per_split {
            let key_index = split_start + broadcast_scalar(split_offset, tile_shape);
            let key_valid = valid
                & cmpi(
                    key_index,
                    broadcast_scalar(key_len, tile_shape),
                    predicate::LessThan,
                );
            let mut score = constant(0.0f32, tile_shape);
            for dim_index in 0i32..HEAD_DIM {
                let query_offsets = query_batch_base
                    + head * broadcast_scalar(query_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let key_offsets = key_batch_base
                    + key_index * broadcast_scalar(key_sequence_stride, tile_shape)
                    + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                    + broadcast_scalar(dim_index, tile_shape);
                let query_tile = load_vector_bf16_as_f32(query, query_offsets, key_valid);
                let key_tile = load_vector_bf16_as_f32(key, key_offsets, key_valid);
                score = score + query_tile * key_tile;
            }
            score = score * broadcast_scalar(scale, tile_shape);
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + key_index * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let has_keys = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = sum / denom;
        let output = select(valid & has_keys, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + split * broadcast_scalar(output_split_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);

        let lse_offsets = batch_index * broadcast_scalar(lse_batch_stride, tile_shape)
            + head * broadcast_scalar(lse_head_stride, tile_shape)
            + split;
        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_keys, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid & first_dim);
    }

    fn sliding_window_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_f16_from_f32(out, output_offsets, output, valid);
    }

    fn sliding_window_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        key_batch_stride: i32,
        value_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);
        let key_batch_base = batch_index * broadcast_scalar(key_batch_stride, tile_shape);
        let value_batch_base = batch_index * broadcast_scalar(value_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                key_batch_base,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                query_index,
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(key_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = value_batch_base
                + broadcast_scalar(key_index * value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector_bf16_from_f32(out, output_offsets, output, valid);
    }

    fn attention_score_f16<const HEAD_DIM: i32>(
        query: *mut f16,
        key: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_f16_as_f32(query, query_offsets, valid);
            let k = load_vector_f16_as_f32(key, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_bf16<const HEAD_DIM: i32>(
        query: *mut bf16,
        key: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_bf16_as_f32(query, query_offsets, valid);
            let k = load_vector_bf16_as_f32(key, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn sliding_window_attention_paged_kv_f32<const HEAD_DIM: i32, const WINDOW: i32>(
        out: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        block_table: *mut u32,
        _batch: i32,
        _query_len: i32,
        key_len: i32,
        heads: i32,
        kv_heads: i32,
        query_start: i32,
        key_start: i32,
        window: i32,
        query_batch_stride: i32,
        output_batch_stride: i32,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        value_sequence_stride: i32,
        output_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        value_head_stride: i32,
        output_head_stride: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        key_cache_block_stride: i32,
        value_cache_block_stride: i32,
        scale: f32,
        output_scale: f32,
        output_len: i32,
        output_values_per_batch: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(output_values_per_batch, tile_shape);
        let batch_output_offset =
            offsets - batch_index * broadcast_scalar(output_values_per_batch, tile_shape);
        let query_index = batch_output_offset / broadcast_scalar(features, tile_shape);
        let feature = batch_output_offset - query_index * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_abs = query_index + broadcast_scalar(query_start, tile_shape);
        let effective_window = if WINDOW == 0 { window } else { WINDOW };
        let query_batch_base = batch_index * broadcast_scalar(query_batch_stride, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                key_cache_block_stride,
                query_index,
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..key_len {
            let key_abs = key_start + key_index;
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_abs, tile_shape),
                    query_abs,
                    predicate::LessThanOrEqual,
                )
                & cmpi(
                    broadcast_scalar(key_abs + effective_window, tile_shape),
                    query_abs,
                    predicate::GreaterThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                query_sequence_stride,
                key_sequence_stride,
                query_head_stride,
                key_head_stride,
                key_cache_block_stride,
                query_index,
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = exp(score - max_score);
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, weight, zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(value_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(value_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let output = (sum / denom) * broadcast_scalar(output_scale, tile_shape);
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let output = select(valid, output, zero);
        let output_offsets = batch_index * broadcast_scalar(output_batch_stride, tile_shape)
            + query_index * broadcast_scalar(output_sequence_stride, tile_shape)
            + head * broadcast_scalar(output_head_stride, tile_shape)
            + dim;
        store_vector(out, output_offsets, output, valid);
    }

    fn paged_kv_decode_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);
    }

    fn paged_kv_decode_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        query: *mut f16,
        key_cache: *mut f16,
        value_cache: *mut f16,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);
    }

    fn paged_kv_decode_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        query: *mut bf16,
        key_cache: *mut bf16,
        value_cache: *mut bf16,
        actual_seq_lens: *mut i32,
        block_table: *mut u32,
        _batch: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let batch_index = offsets / broadcast_scalar(features, tile_shape);
        let batch_output_offset = offsets - batch_index * broadcast_scalar(features, tile_shape);
        let head = batch_output_offset / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = batch_output_offset - head * broadcast_scalar(HEAD_DIM, tile_shape);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let seq_len = load_i32_vector(actual_seq_lens, batch_index, valid, 0i32);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = constant(0.0f32, tile_shape);
        let mut sum = constant(0.0f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let key_valid = valid
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len,
                    predicate::LessThan,
                );
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                batch_index * broadcast_scalar(features, tile_shape),
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }
        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);
    }

    fn paged_kv_prefill_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key_cache: *mut f32,
        value_cache: *mut f32,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector(value_cache, value_offsets, block_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn paged_kv_prefill_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key_cache: *mut f16,
        value_cache: *mut f16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_f16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn paged_kv_prefill_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key_cache: *mut bf16,
        value_cache: *mut bf16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        block_table: *mut u32,
        batch: i32,
        _total_query_tokens: i32,
        heads: i32,
        kv_heads: i32,
        block_size: i32,
        physical_blocks: i32,
        block_table_batch_stride: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let seq_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(global_token, seq_offset, predicate::GreaterThanOrEqual)
                & cmpi(
                    global_token,
                    seq_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - seq_offset, query_index);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let key_sequence_stride = kv_heads * HEAD_DIM;
        let key_head_stride = HEAD_DIM;
        let key_cache_block_stride = block_size * key_sequence_stride;
        let value_cache_block_stride = key_cache_block_stride;
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            max_score = select(
                block_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..(block_table_batch_stride * block_size) {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let (physical_block, block_token) = paged_block_lookup(
                block_table,
                batch_index,
                key_index,
                block_size,
                block_table_batch_stride,
                key_valid,
            );
            let block_valid = key_valid
                & cmpi(
                    physical_block,
                    broadcast_scalar(0i32, tile_shape),
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    physical_block,
                    broadcast_scalar(physical_blocks, tile_shape),
                    predicate::LessThan,
                );
            let score = attention_score_paged_bf16::<HEAD_DIM>(
                query,
                key_cache,
                query_batch_base,
                physical_block,
                block_token,
                0,
                key_sequence_stride,
                HEAD_DIM,
                key_head_stride,
                key_cache_block_stride,
                broadcast_scalar(0i32, tile_shape),
                head,
                kv_head,
                block_valid,
                scale,
            );
            let weight = select(block_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = physical_block
                * broadcast_scalar(value_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value_cache, value_offsets, block_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_f32<const HEAD_DIM: i32>(
        out: *mut f32,
        lse: *mut f32,
        query: *mut f32,
        key: *mut f32,
        value: *mut f32,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                heads,
                kv_heads,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector(value, value_offsets, key_valid, 0.0f32);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_f16<const HEAD_DIM: i32>(
        out: *mut f16,
        lse: *mut f32,
        query: *mut f16,
        key: *mut f16,
        value: *mut f16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_f16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector_f16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_f16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn ragged_kv_prefill_attention_bf16<const HEAD_DIM: i32>(
        out: *mut bf16,
        lse: *mut f32,
        query: *mut bf16,
        key: *mut bf16,
        value: *mut bf16,
        actual_seq_lens_q: *mut i32,
        actual_seq_lens_kv: *mut i32,
        actual_seq_offsets: *mut i32,
        batch: i32,
        _total_query_tokens: i32,
        total_kv_tokens: i32,
        heads: i32,
        kv_heads: i32,
        causal: i32,
        scale: f32,
        output_len: i32,
    ) {
        let tile_shape = const_shape![128];
        let pid: (i32, i32, i32) = get_tile_block_id();
        let offsets: Tile<i32, { [128] }> =
            iota(tile_shape) + broadcast_scalar(pid.0 * VECTOR_TILE_SIZE, tile_shape);
        let valid = cmpi(
            offsets,
            broadcast_scalar(output_len, tile_shape),
            predicate::LessThan,
        );
        let features = heads * HEAD_DIM;
        let kv_features = kv_heads * HEAD_DIM;
        let global_token = offsets / broadcast_scalar(features, tile_shape);
        let feature = offsets - global_token * broadcast_scalar(features, tile_shape);
        let head = feature / broadcast_scalar(HEAD_DIM, tile_shape);
        let dim = feature - head * broadcast_scalar(HEAD_DIM, tile_shape);

        let mut batch_index = broadcast_scalar(-1i32, tile_shape);
        let mut query_index = broadcast_scalar(0i32, tile_shape);
        let mut seq_offset = broadcast_scalar(0i32, tile_shape);
        let mut seq_len_q = broadcast_scalar(0i32, tile_shape);
        for candidate in 0i32..batch {
            let candidate_tile = broadcast_scalar(candidate, tile_shape);
            let candidate_offset = load_i32_vector(actual_seq_offsets, candidate_tile, valid, 0i32);
            let candidate_seq_len = load_i32_vector(actual_seq_lens_q, candidate_tile, valid, 0i32);
            let in_batch = valid
                & cmpi(
                    global_token,
                    candidate_offset,
                    predicate::GreaterThanOrEqual,
                )
                & cmpi(
                    global_token,
                    candidate_offset + candidate_seq_len,
                    predicate::LessThan,
                );
            batch_index = select(in_batch, candidate_tile, batch_index);
            query_index = select(in_batch, global_token - candidate_offset, query_index);
            seq_offset = select(in_batch, candidate_offset, seq_offset);
            seq_len_q = select(in_batch, candidate_seq_len, seq_len_q);
        }
        let valid_query = valid
            & cmpi(
                batch_index,
                broadcast_scalar(0i32, tile_shape),
                predicate::GreaterThanOrEqual,
            )
            & cmpi(query_index, seq_len_q, predicate::LessThan);
        let seq_len_kv = load_i32_vector(actual_seq_lens_kv, batch_index, valid_query, 0i32);
        let query_group_size = heads / kv_heads;
        let kv_head = head / broadcast_scalar(query_group_size, tile_shape);
        let query_batch_base = global_token * broadcast_scalar(features, tile_shape);

        let mut max_score = constant(-1.0e20f32, tile_shape);
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            max_score = select(
                key_valid
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let zero: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut denom = zero;
        let mut sum = zero;
        for key_index in 0i32..total_kv_tokens {
            let mut key_valid = valid_query
                & cmpi(
                    broadcast_scalar(key_index, tile_shape),
                    seq_len_kv,
                    predicate::LessThan,
                );
            if causal != 0 {
                key_valid = key_valid
                    & cmpi(
                        broadcast_scalar(key_index, tile_shape),
                        query_index,
                        predicate::LessThanOrEqual,
                    );
            }
            let score = attention_score_bf16::<HEAD_DIM>(
                query,
                key,
                query_batch_base,
                seq_offset * broadcast_scalar(kv_features, tile_shape),
                0,
                kv_features,
                HEAD_DIM,
                HEAD_DIM,
                broadcast_scalar(0i32, tile_shape),
                head,
                key_index,
                kv_head,
                key_valid,
                scale,
            );
            let weight = select(key_valid, exp(score - max_score), zero);
            denom = denom + weight;
            let value_offsets = (seq_offset + broadcast_scalar(key_index, tile_shape))
                * broadcast_scalar(kv_features, tile_shape)
                + kv_head * broadcast_scalar(HEAD_DIM, tile_shape)
                + dim;
            let value_tile = load_vector_bf16_as_f32(value, value_offsets, key_valid);
            sum = sum + weight * value_tile;
        }

        let one: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let has_denominator = cmpf(denom, zero, predicate::GreaterThan, cmp_ordering::Ordered);
        let output = select(
            valid_query & has_denominator,
            sum / select(has_denominator, denom, one),
            zero,
        );
        store_vector_bf16_from_f32(out, offsets, output, valid);

        let first_dim = cmpi(dim, broadcast_scalar(0i32, tile_shape), predicate::Equal);
        let lse_offsets = global_token * broadcast_scalar(heads, tile_shape) + head;
        let invalid_lse: Tile<f32, { [128] }> = constant(-1.0e20f32, tile_shape);
        let lse_value = select(has_denominator, max_score + log(denom), invalid_lse);
        store_vector(lse, lse_offsets, lse_value, valid_query & first_dim);
    }

    fn attention_score<const HEAD_DIM: i32>(
        query: *mut f32,
        key: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        key_batch_base: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        key_index: i32,
        kv_head: Tile<i32, { [128] }>,
        _heads: i32,
        _kv_heads: i32,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = key_batch_base
                + broadcast_scalar(key_index * key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector(query, query_offsets, valid, 0.0f32);
            let k = load_vector(key, key_offsets, valid, 0.0f32);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged<const HEAD_DIM: i32>(
        query: *mut f32,
        key_cache: *mut f32,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector(query, query_offsets, valid, 0.0f32);
            let k = load_vector(key_cache, key_offsets, valid, 0.0f32);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged_f16<const HEAD_DIM: i32>(
        query: *mut f16,
        key_cache: *mut f16,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_f16_as_f32(query, query_offsets, valid);
            let k = load_vector_f16_as_f32(key_cache, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn attention_score_paged_bf16<const HEAD_DIM: i32>(
        query: *mut bf16,
        key_cache: *mut bf16,
        query_batch_base: Tile<i32, { [128] }>,
        physical_block: Tile<i32, { [128] }>,
        block_token: Tile<i32, { [128] }>,
        query_sequence_stride: i32,
        key_sequence_stride: i32,
        query_head_stride: i32,
        key_head_stride: i32,
        key_cache_block_stride: i32,
        query_index: Tile<i32, { [128] }>,
        head: Tile<i32, { [128] }>,
        kv_head: Tile<i32, { [128] }>,
        valid: Tile<bool, { [128] }>,
        scale: f32,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let mut score = constant(0.0f32, tile_shape);
        for dim in 0i32..HEAD_DIM {
            let query_offsets = query_batch_base
                + query_index * broadcast_scalar(query_sequence_stride, tile_shape)
                + head * broadcast_scalar(query_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let key_offsets = physical_block * broadcast_scalar(key_cache_block_stride, tile_shape)
                + block_token * broadcast_scalar(key_sequence_stride, tile_shape)
                + kv_head * broadcast_scalar(key_head_stride, tile_shape)
                + broadcast_scalar(dim, tile_shape);
            let q = load_vector_bf16_as_f32(query, query_offsets, valid);
            let k = load_vector_bf16_as_f32(key_cache, key_offsets, valid);
            score = score + q * k;
        }
        score * broadcast_scalar(scale, tile_shape)
    }

    fn paged_block_lookup(
        block_table: *mut u32,
        batch_index: Tile<i32, { [128] }>,
        key_index: i32,
        block_size: i32,
        block_table_batch_stride: i32,
        valid: Tile<bool, { [128] }>,
    ) -> (Tile<i32, { [128] }>, Tile<i32, { [128] }>) {
        let tile_shape = const_shape![128];
        let logical_block = key_index / block_size;
        let block_token = key_index - logical_block * block_size;
        let block_table_offsets = batch_index
            * broadcast_scalar(block_table_batch_stride, tile_shape)
            + broadcast_scalar(logical_block, tile_shape);
        let physical_block_u32 = load_u32_vector(block_table, block_table_offsets, valid, 0u32);
        let physical_block: Tile<i32, { [128] }> = bitcast(physical_block_u32);
        (physical_block, broadcast_scalar(block_token, tile_shape))
    }

    fn softmax_probability_f32(
        scores: *mut f32,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn softmax_probability_f16(
        scores: *mut f16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn softmax_probability_bf16(
        scores: *mut bf16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let mut max_score = constant(-1.0e30f32, tile_shape);
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            max_score = select(
                valid_score
                    & cmpf(
                        score,
                        max_score,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    ),
                score,
                max_score,
            );
        }

        let mut denom = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            denom = denom + select(valid_score, exp(score - max_score), zero_values);
        }
        select(
            valid_input,
            exp(score_value - max_score) / denom,
            zero_values,
        )
    }

    fn sparsemax_probability_f32(
        scores: *mut f32,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score = load_vector(scores, score_base + score_col_tile, valid_score, 0.0f32);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn sparsemax_probability_f16(
        scores: *mut f16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score =
                    load_vector_f16_as_f32(scores, score_base + score_col_tile, valid_score);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn sparsemax_probability_bf16(
        scores: *mut bf16,
        score_base: Tile<i32, { [128] }>,
        score_value: Tile<f32, { [128] }>,
        input_row: Tile<i32, { [128] }>,
        seq_len: i32,
        valid_row: Tile<bool, { [128] }>,
        valid_input: Tile<bool, { [128] }>,
    ) -> Tile<f32, { [128] }> {
        let tile_shape = const_shape![128];
        let zero_values: Tile<f32, { [128] }> = constant(0.0f32, tile_shape);
        let one_values: Tile<f32, { [128] }> = constant(1.0f32, tile_shape);
        let half_values: Tile<f32, { [128] }> = constant(0.5f32, tile_shape);
        let mut x_max = constant(-1.0e30f32, tile_shape);
        let mut x_sum = zero_values;
        let mut support_count = zero_values;
        for score_col in 0i32..seq_len {
            let score_col_tile = broadcast_scalar(score_col, tile_shape);
            let valid_score =
                valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
            let score = load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
            x_sum = x_sum + select(valid_score, score, zero_values);
            support_count = support_count + select(valid_score, one_values, zero_values);
            x_max = select(
                valid_score & cmpf(score, x_max, predicate::GreaterThan, cmp_ordering::Ordered),
                score,
                x_max,
            );
        }

        let mut tau_lo = (x_sum - one_values) / support_count;
        let mut tau_hi = x_max;
        for _ in 0i32..20i32 {
            let tau_mid = half_values * (tau_lo + tau_hi);
            let mut sum_active = zero_values;
            for score_col in 0i32..seq_len {
                let score_col_tile = broadcast_scalar(score_col, tile_shape);
                let valid_score =
                    valid_row & cmpi(score_col_tile, input_row, predicate::LessThanOrEqual);
                let score =
                    load_vector_bf16_as_f32(scores, score_base + score_col_tile, valid_score);
                let active = valid_score
                    & cmpf(
                        score,
                        tau_mid,
                        predicate::GreaterThan,
                        cmp_ordering::Ordered,
                    );
                sum_active = sum_active + select(active, score - tau_mid, zero_values);
            }
            let above_one = cmpf(
                sum_active,
                one_values,
                predicate::GreaterThanOrEqual,
                cmp_ordering::Ordered,
            );
            tau_lo = select(above_one, tau_mid, tau_lo);
            tau_hi = select(above_one, tau_hi, tau_mid);
        }

        let tau = half_values * (tau_lo + tau_hi);
        let value = score_value - tau;
        select(
            valid_input
                & cmpf(
                    value,
                    zero_values,
                    predicate::GreaterThan,
                    cmp_ordering::Ordered,
                ),
            value,
            zero_values,
        )
    }

    fn store_i32_vector(
        out: *mut i32,
        offsets: Tile<i32, { [128] }>,
        values: Tile<i32, { [128] }>,
        mask: Tile<bool, { [128] }>,
    ) {
        let zero_offsets: Tile<i32, { [128] }> = constant(0i32, const_shape![128]);
        let offsets = select(mask, offsets, zero_offsets);
        let out_base: PointerTile<*mut i32, { [] }> = pointer_to_tile(out);
        let out_base: PointerTile<*mut i32, { [1] }> = out_base.reshape(const_shape![1]);
        let out_ptrs: PointerTile<*mut i32, { [128] }> = out_base.broadcast(const_shape![128]);
        let out_ptrs: PointerTile<*mut i32, { [128] }> = out_ptrs.offset_tile(offsets);
        store_ptr_tko(
            out_ptrs,
            values,
            ordering::Weak,
            None::<scope::TileBlock>,
            Some(mask),
            None,
            Latency::<0>,
        );
    }
}

pub use kernels::*;
