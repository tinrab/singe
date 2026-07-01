//! Recurrent/state-space preprocessing helpers.

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;
use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::{Error, Result},
    utility::{checked_element_count, checked_rank4_len, ensure_len},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrentGatedDeltaRuleConfig {
    pub batch: usize,
    pub time: usize,
    pub query_heads: usize,
    pub value_heads: usize,
    pub qk_dim: usize,
    pub value_dim: usize,
    pub use_qk_l2norm: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkGatedDeltaRuleConfig {
    pub batch: usize,
    pub time: usize,
    pub heads: usize,
    pub qk_dim: usize,
    pub value_dim: usize,
    pub chunk_size: usize,
    pub use_qk_l2norm: bool,
}

macro_rules! gated_delta_rule_preprocess_fn {
    ($name:ident, $ty:ty) => {
        /// Prepares Gated Delta Rule recurrent inputs after projection.
        ///
        /// `b_in` and `a_in` are `[rows, hidden]`; `a_log` and `dt_bias` are `[hidden]`.
        /// The kernel writes `beta = sigmoid(b_in)` and `g = -exp(a_log) * softplus(a_in + dt_bias)` using stable softplus.
        pub fn $name(
            stream: &Stream,
            beta_out: &mut impl DeviceSliceMut<$ty>,
            g_out: &mut impl DeviceSliceMut<$ty>,
            b_in: &impl DeviceSlice<$ty>,
            a_in: &impl DeviceSlice<$ty>,
            a_log: &impl DeviceSlice<$ty>,
            dt_bias: &impl DeviceSlice<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, hidden)?;
            ensure_len(beta_out.len(), len)?;
            ensure_len(g_out.len(), len)?;
            ensure_len(b_in.len(), len)?;
            ensure_len(a_in.len(), len)?;
            ensure_len(a_log.len(), hidden)?;
            ensure_len(dt_bias.len(), hidden)?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$name(
                &stream,
                output_pointer(beta_out),
                output_pointer(g_out),
                input_pointer(b_in),
                input_pointer(a_in),
                input_pointer(a_log),
                input_pointer(dt_bias),
                rows,
                hidden,
            )
        }
    };
}

macro_rules! gated_delta_rule_preprocess_g_f32_fn {
    ($name:ident, $ty:ty) => {
        /// Prepares Gated Delta Rule recurrent inputs and stores the gate in f32.
        ///
        /// This is the mixed-output variant of [`gated_delta_rule_preprocess_f16`]
        /// and [`gated_delta_rule_preprocess_bf16`] for recurrent paths that keep
        /// `g = -exp(a_log) * softplus(a_in + dt_bias)` in f32.
        pub fn $name(
            stream: &Stream,
            beta_out: &mut impl DeviceSliceMut<$ty>,
            g_out: &mut impl DeviceSliceMut<f32>,
            b_in: &impl DeviceSlice<$ty>,
            a_in: &impl DeviceSlice<$ty>,
            a_log: &impl DeviceSlice<$ty>,
            dt_bias: &impl DeviceSlice<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            let len = checked_element_count(rows, hidden)?;
            ensure_len(beta_out.len(), len)?;
            ensure_len(g_out.len(), len)?;
            ensure_len(b_in.len(), len)?;
            ensure_len(a_in.len(), len)?;
            ensure_len(a_log.len(), hidden)?;
            ensure_len(dt_bias.len(), hidden)?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$name(
                &stream,
                output_pointer(beta_out),
                output_pointer(g_out),
                input_pointer(b_in),
                input_pointer(a_in),
                input_pointer(a_log),
                input_pointer(dt_bias),
                rows,
                hidden,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
gated_delta_rule_preprocess_fn!(gated_delta_rule_preprocess_f32, f32);
#[cfg(feature = "dtype-f16")]
gated_delta_rule_preprocess_fn!(gated_delta_rule_preprocess_f16, f16);
#[cfg(feature = "dtype-bf16")]
gated_delta_rule_preprocess_fn!(gated_delta_rule_preprocess_bf16, bf16);

#[cfg(feature = "dtype-f16")]
gated_delta_rule_preprocess_g_f32_fn!(gated_delta_rule_preprocess_f16_g_f32, f16);
#[cfg(feature = "dtype-bf16")]
gated_delta_rule_preprocess_g_f32_fn!(gated_delta_rule_preprocess_bf16_g_f32, bf16);

macro_rules! recurrent_gated_delta_rule_no_state_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            query: &impl DeviceSlice<$ty>,
            key: &impl DeviceSlice<$ty>,
            value: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            beta: &impl DeviceSlice<$ty>,
            config: RecurrentGatedDeltaRuleConfig,
        ) -> Result<()> {
            let params = validate_recurrent_gated_delta_rule_no_state(
                out.len(),
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$name(
                &stream,
                output_pointer(out),
                output_pointer(out),
                input_pointer(query),
                input_pointer(key),
                input_pointer(value),
                input_pointer(gate),
                input_pointer(beta),
                input_pointer(query),
                params,
                false,
                false,
            )
        }
    };
}

macro_rules! recurrent_gated_delta_rule_with_state_fn {
    ($name:ident, $base_name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            final_state: &mut impl DeviceSliceMut<$ty>,
            query: &impl DeviceSlice<$ty>,
            key: &impl DeviceSlice<$ty>,
            value: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            beta: &impl DeviceSlice<$ty>,
            initial_state: &impl DeviceSlice<$ty>,
            config: RecurrentGatedDeltaRuleConfig,
        ) -> Result<()> {
            let params = validate_recurrent_gated_delta_rule_with_state(
                out.len(),
                final_state.len(),
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                initial_state.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$base_name(
                &stream,
                output_pointer(out),
                output_pointer(final_state),
                input_pointer(query),
                input_pointer(key),
                input_pointer(value),
                input_pointer(gate),
                input_pointer(beta),
                input_pointer(initial_state),
                params,
                true,
                true,
            )
        }
    };
}

macro_rules! chunk_gated_delta_rule_no_state_fn {
    ($name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            query: &impl DeviceSlice<$ty>,
            key: &impl DeviceSlice<$ty>,
            value: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            beta: &impl DeviceSlice<$ty>,
            config: ChunkGatedDeltaRuleConfig,
        ) -> Result<()> {
            validate_chunk_gated_delta_rule_no_state(
                out.len(),
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$name(
                &stream,
                output_pointer(out),
                output_pointer(out),
                input_pointer(query),
                input_pointer(key),
                input_pointer(value),
                input_pointer(gate),
                input_pointer(beta),
                input_pointer(query),
                config.batch,
                config.time,
                config.heads,
                config.qk_dim,
                config.value_dim,
                config.chunk_size,
                config.use_qk_l2norm,
                false,
                false,
            )
        }
    };
}

macro_rules! chunk_gated_delta_rule_with_state_fn {
    ($name:ident, $base_name:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            final_state: &mut impl DeviceSliceMut<$ty>,
            query: &impl DeviceSlice<$ty>,
            key: &impl DeviceSlice<$ty>,
            value: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            beta: &impl DeviceSlice<$ty>,
            initial_state: &impl DeviceSlice<$ty>,
            config: ChunkGatedDeltaRuleConfig,
        ) -> Result<()> {
            validate_chunk_gated_delta_rule_with_state(
                out.len(),
                final_state.len(),
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                initial_state.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$base_name(
                &stream,
                output_pointer(out),
                output_pointer(final_state),
                input_pointer(query),
                input_pointer(key),
                input_pointer(value),
                input_pointer(gate),
                input_pointer(beta),
                input_pointer(initial_state),
                config.batch,
                config.time,
                config.heads,
                config.qk_dim,
                config.value_dim,
                config.chunk_size,
                config.use_qk_l2norm,
                true,
                true,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
recurrent_gated_delta_rule_no_state_fn!(recurrent_gated_delta_rule_f32, f32);
#[cfg(feature = "dtype-f16")]
recurrent_gated_delta_rule_no_state_fn!(recurrent_gated_delta_rule_f16, f16);
#[cfg(feature = "dtype-bf16")]
recurrent_gated_delta_rule_no_state_fn!(recurrent_gated_delta_rule_bf16, bf16);

#[cfg(feature = "dtype-f32")]
recurrent_gated_delta_rule_with_state_fn!(
    recurrent_gated_delta_rule_f32_with_state,
    recurrent_gated_delta_rule_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
recurrent_gated_delta_rule_with_state_fn!(
    recurrent_gated_delta_rule_f16_with_state,
    recurrent_gated_delta_rule_f16,
    f16
);
#[cfg(feature = "dtype-bf16")]
recurrent_gated_delta_rule_with_state_fn!(
    recurrent_gated_delta_rule_bf16_with_state,
    recurrent_gated_delta_rule_bf16,
    bf16
);

macro_rules! recurrent_gated_delta_rule_decode_with_state_fn {
    ($name:ident, $cutile_fn:ident, $ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$ty>,
            final_state: &mut impl DeviceSliceMut<$ty>,
            query: &impl DeviceSlice<$ty>,
            key: &impl DeviceSlice<$ty>,
            value: &impl DeviceSlice<$ty>,
            gate: &impl DeviceSlice<$ty>,
            beta: &impl DeviceSlice<$ty>,
            initial_state: &impl DeviceSlice<$ty>,
            config: RecurrentGatedDeltaRuleConfig,
        ) -> Result<()> {
            let params = validate_recurrent_gated_delta_rule_decode_with_state(
                out.len(),
                final_state.len(),
                query.len(),
                key.len(),
                value.len(),
                gate.len(),
                beta.len(),
                initial_state.len(),
                config,
            )?;
            let stream = borrowed_stream(stream)?;
            cutile::recurrent::$cutile_fn(
                &stream,
                output_pointer(out),
                output_pointer(final_state),
                input_pointer(query),
                input_pointer(key),
                input_pointer(value),
                input_pointer(gate),
                input_pointer(beta),
                input_pointer(initial_state),
                params,
                true,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
recurrent_gated_delta_rule_decode_with_state_fn!(
    recurrent_gated_delta_rule_decode_f32_with_state,
    recurrent_gated_delta_rule_decode_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
recurrent_gated_delta_rule_decode_with_state_fn!(
    recurrent_gated_delta_rule_decode_f16_with_state,
    recurrent_gated_delta_rule_decode_f16,
    f16
);
#[cfg(feature = "dtype-bf16")]
recurrent_gated_delta_rule_decode_with_state_fn!(
    recurrent_gated_delta_rule_decode_bf16_with_state,
    recurrent_gated_delta_rule_decode_bf16,
    bf16
);

#[cfg(feature = "dtype-f32")]
chunk_gated_delta_rule_no_state_fn!(chunk_gated_delta_rule_f32, f32);
#[cfg(feature = "dtype-f16")]
chunk_gated_delta_rule_no_state_fn!(chunk_gated_delta_rule_f16, f16);
#[cfg(feature = "dtype-bf16")]
chunk_gated_delta_rule_no_state_fn!(chunk_gated_delta_rule_bf16, bf16);

#[cfg(feature = "dtype-f32")]
chunk_gated_delta_rule_with_state_fn!(
    chunk_gated_delta_rule_f32_with_state,
    chunk_gated_delta_rule_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
chunk_gated_delta_rule_with_state_fn!(
    chunk_gated_delta_rule_f16_with_state,
    chunk_gated_delta_rule_f16,
    f16
);
#[cfg(feature = "dtype-bf16")]
chunk_gated_delta_rule_with_state_fn!(
    chunk_gated_delta_rule_bf16_with_state,
    chunk_gated_delta_rule_bf16,
    bf16
);

fn validate_recurrent_gated_delta_rule_no_state(
    out_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    config: RecurrentGatedDeltaRuleConfig,
) -> Result<cutile::recurrent::RecurrentGatedDeltaRule> {
    let params = recurrent_params(config)?;
    ensure_recurrent_lengths(
        out_len, query_len, key_len, value_len, gate_len, beta_len, config,
    )?;
    Ok(params)
}

fn validate_recurrent_gated_delta_rule_with_state(
    out_len: usize,
    final_state_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    initial_state_len: usize,
    config: RecurrentGatedDeltaRuleConfig,
) -> Result<cutile::recurrent::RecurrentGatedDeltaRule> {
    let params = recurrent_params(config)?;
    ensure_recurrent_lengths(
        out_len, query_len, key_len, value_len, gate_len, beta_len, config,
    )?;
    let state_len = checked_rank4_len([
        config.batch,
        config.value_heads,
        config.qk_dim,
        config.value_dim,
    ])?;
    ensure_len(initial_state_len, state_len)?;
    ensure_len(final_state_len, state_len)?;
    Ok(params)
}

fn validate_recurrent_gated_delta_rule_decode_with_state(
    out_len: usize,
    final_state_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    initial_state_len: usize,
    config: RecurrentGatedDeltaRuleConfig,
) -> Result<cutile::recurrent::RecurrentGatedDeltaRule> {
    if config.time != 1 {
        return Err(Error::InvalidLength);
    }
    validate_recurrent_gated_delta_rule_with_state(
        out_len,
        final_state_len,
        query_len,
        key_len,
        value_len,
        gate_len,
        beta_len,
        initial_state_len,
        config,
    )
}

fn recurrent_params(
    config: RecurrentGatedDeltaRuleConfig,
) -> Result<cutile::recurrent::RecurrentGatedDeltaRule> {
    cutile::recurrent::RecurrentGatedDeltaRule::create(
        config.batch,
        config.time,
        config.query_heads,
        config.value_heads,
        config.qk_dim,
        config.value_dim,
        config.use_qk_l2norm,
    )
}

fn ensure_recurrent_lengths(
    out_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    config: RecurrentGatedDeltaRuleConfig,
) -> Result<()> {
    let query_key_len =
        checked_rank4_len([config.batch, config.time, config.query_heads, config.qk_dim])?;
    let value_out_len = checked_rank4_len([
        config.batch,
        config.time,
        config.value_heads,
        config.value_dim,
    ])?;
    let gate_len_expected = checked_element_count(
        checked_element_count(config.batch, config.time)?,
        config.value_heads,
    )?;
    ensure_len(out_len, value_out_len)?;
    ensure_len(query_len, query_key_len)?;
    ensure_len(key_len, query_key_len)?;
    ensure_len(value_len, value_out_len)?;
    ensure_len(gate_len, gate_len_expected)?;
    ensure_len(beta_len, gate_len_expected)?;
    Ok(())
}

fn validate_chunk_gated_delta_rule_no_state(
    out_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    config: ChunkGatedDeltaRuleConfig,
) -> Result<()> {
    if config.chunk_size == 0 {
        return Err(Error::InvalidLength);
    }
    let recurrent_config = chunk_as_recurrent_config(config);
    recurrent_params(recurrent_config)?;
    ensure_recurrent_lengths(
        out_len,
        query_len,
        key_len,
        value_len,
        gate_len,
        beta_len,
        recurrent_config,
    )
}

fn validate_chunk_gated_delta_rule_with_state(
    out_len: usize,
    final_state_len: usize,
    query_len: usize,
    key_len: usize,
    value_len: usize,
    gate_len: usize,
    beta_len: usize,
    initial_state_len: usize,
    config: ChunkGatedDeltaRuleConfig,
) -> Result<()> {
    validate_chunk_gated_delta_rule_no_state(
        out_len, query_len, key_len, value_len, gate_len, beta_len, config,
    )?;
    let state_len =
        checked_rank4_len([config.batch, config.heads, config.qk_dim, config.value_dim])?;
    ensure_len(initial_state_len, state_len)?;
    ensure_len(final_state_len, state_len)?;
    Ok(())
}

fn chunk_as_recurrent_config(config: ChunkGatedDeltaRuleConfig) -> RecurrentGatedDeltaRuleConfig {
    RecurrentGatedDeltaRuleConfig {
        batch: config.batch,
        time: config.time,
        query_heads: config.heads,
        value_heads: config.heads,
        qk_dim: config.qk_dim,
        value_dim: config.value_dim,
        use_qk_l2norm: config.use_qk_l2norm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;

    #[test]
    fn recurrent_gated_delta_rule_validation_accepts_exact_lengths() -> Result<()> {
        let config = recurrent_validation_config();
        validate_recurrent_gated_delta_rule_no_state(168, 60, 60, 168, 24, 24, config)?;
        validate_recurrent_gated_delta_rule_with_state(168, 280, 60, 60, 168, 24, 24, 280, config)?;
        Ok(())
    }

    #[test]
    fn recurrent_gated_delta_rule_validation_rejects_short_gate() {
        let config = recurrent_validation_config();
        assert!(matches!(
            validate_recurrent_gated_delta_rule_no_state(168, 60, 60, 168, 23, 24, config),
            Err(Error::LengthMismatch)
        ));
    }

    #[test]
    fn recurrent_gated_delta_rule_validation_rejects_bad_head_mapping() {
        let mut config = recurrent_validation_config();
        config.query_heads = 3;
        assert!(matches!(
            validate_recurrent_gated_delta_rule_no_state(168, 90, 90, 168, 24, 24, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn recurrent_gated_delta_rule_validation_rejects_too_wide_qk_dim() {
        let mut config = recurrent_validation_config();
        config.qk_dim = 129;
        assert!(matches!(
            validate_recurrent_gated_delta_rule_no_state(168, 1548, 1548, 168, 24, 24, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn recurrent_gated_delta_rule_decode_validation_accepts_time_one() -> Result<()> {
        let config = recurrent_decode_validation_config();
        validate_recurrent_gated_delta_rule_decode_with_state(
            48, 240, 20, 20, 48, 8, 8, 240, config,
        )?;
        Ok(())
    }

    #[test]
    fn recurrent_gated_delta_rule_decode_validation_rejects_time_not_one() {
        let mut config = recurrent_decode_validation_config();
        config.time = 2;
        assert!(matches!(
            validate_recurrent_gated_delta_rule_decode_with_state(
                96, 240, 40, 40, 96, 16, 16, 240, config
            ),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn chunk_gated_delta_rule_validation_accepts_exact_lengths() -> Result<()> {
        let config = chunk_validation_config();
        validate_chunk_gated_delta_rule_no_state(180, 120, 120, 180, 30, 30, config)?;
        validate_chunk_gated_delta_rule_with_state(180, 144, 120, 120, 180, 30, 30, 144, config)
    }

    #[test]
    fn chunk_gated_delta_rule_validation_rejects_zero_chunk_size() {
        let mut config = chunk_validation_config();
        config.chunk_size = 0;
        assert!(matches!(
            validate_chunk_gated_delta_rule_no_state(180, 120, 120, 180, 30, 30, config),
            Err(Error::InvalidLength)
        ));
    }

    #[test]
    fn chunk_gated_delta_rule_validation_rejects_short_final_state() {
        let config = chunk_validation_config();
        assert!(matches!(
            validate_chunk_gated_delta_rule_with_state(
                180, 143, 120, 120, 180, 30, 30, 144, config
            ),
            Err(Error::LengthMismatch)
        ));
    }

    fn recurrent_validation_config() -> RecurrentGatedDeltaRuleConfig {
        RecurrentGatedDeltaRuleConfig {
            batch: 2,
            time: 3,
            query_heads: 2,
            value_heads: 4,
            qk_dim: 5,
            value_dim: 7,
            use_qk_l2norm: true,
        }
    }

    fn recurrent_decode_validation_config() -> RecurrentGatedDeltaRuleConfig {
        RecurrentGatedDeltaRuleConfig {
            batch: 2,
            time: 1,
            query_heads: 2,
            value_heads: 4,
            qk_dim: 5,
            value_dim: 6,
            use_qk_l2norm: true,
        }
    }

    fn chunk_validation_config() -> ChunkGatedDeltaRuleConfig {
        ChunkGatedDeltaRuleConfig {
            batch: 2,
            time: 5,
            heads: 3,
            qk_dim: 4,
            value_dim: 6,
            chunk_size: 2,
            use_qk_l2norm: false,
        }
    }
}
