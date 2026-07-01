use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

use crate::{
    cuda::cutile::{
        DeviceOpExt,
        kernel::recurrent as kernel_recurrent,
        utility::{VectorLaunch, checked_device_pointer},
    },
    error::{Error, Result},
    utility::{checked_element_count, checked_i32_value},
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GatedDeltaRulePreprocess {
    hidden: i32,
    len: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecurrentGatedDeltaRule {
    batch: i32,
    time: i32,
    query_heads: i32,
    value_heads: i32,
    qk_dim: i32,
    value_dim: i32,
    scale: f32,
    use_qk_l2norm: i32,
    grid: (u32, u32, u32),
}

impl GatedDeltaRulePreprocess {
    fn create(rows: usize, hidden: usize) -> Result<Self> {
        if rows == 0 || hidden == 0 {
            return Err(Error::InvalidLength);
        }
        let len = checked_element_count(rows, hidden)?;
        let launch = VectorLaunch::create(len)?;
        Ok(Self {
            hidden: checked_i32_value(hidden)?,
            len: launch.len_i32,
            grid: launch.grid,
        })
    }
}

impl RecurrentGatedDeltaRule {
    pub fn create(
        batch: usize,
        time: usize,
        query_heads: usize,
        value_heads: usize,
        qk_dim: usize,
        value_dim: usize,
        use_qk_l2norm: bool,
    ) -> Result<Self> {
        if batch == 0
            || time == 0
            || query_heads == 0
            || value_heads == 0
            || qk_dim == 0
            || qk_dim > 128
            || value_dim == 0
            || value_heads < query_heads
            || !value_heads.is_multiple_of(query_heads)
        {
            return Err(Error::InvalidLength);
        }
        let batch_value_heads = checked_element_count(batch, value_heads)?;
        let grid_x = u32::try_from(batch_value_heads).map_err(|_| Error::SizeOverflow)?;
        let grid_y = u32::try_from(value_dim).map_err(|_| Error::SizeOverflow)?;
        Ok(Self {
            batch: checked_i32_value(batch)?,
            time: checked_i32_value(time)?,
            query_heads: checked_i32_value(query_heads)?,
            value_heads: checked_i32_value(value_heads)?,
            qk_dim: checked_i32_value(qk_dim)?,
            value_dim: checked_i32_value(value_dim)?,
            scale: 1.0 / (qk_dim as f32).sqrt(),
            use_qk_l2norm: i32::from(use_qk_l2norm),
            grid: (grid_x, grid_y, 1),
        })
    }
}

macro_rules! gated_delta_rule_preprocess_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            beta_out: DevicePointer<$ty>,
            g_out: DevicePointer<$ty>,
            b_in: DevicePointer<$ty>,
            a_in: DevicePointer<$ty>,
            a_log: DevicePointer<$ty>,
            dt_bias: DevicePointer<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            let params = GatedDeltaRulePreprocess::create(rows, hidden)?;
            checked_device_pointer(beta_out)?;
            checked_device_pointer(g_out)?;
            checked_device_pointer(b_in)?;
            checked_device_pointer(a_in)?;
            checked_device_pointer(a_log)?;
            checked_device_pointer(dt_bias)?;
            unsafe {
                kernel_recurrent::$kernel_fn(
                    beta_out,
                    g_out,
                    b_in,
                    a_in,
                    a_log,
                    dt_bias,
                    params.hidden,
                    params.len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! gated_delta_rule_preprocess_g_f32_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            beta_out: DevicePointer<$ty>,
            g_out: DevicePointer<f32>,
            b_in: DevicePointer<$ty>,
            a_in: DevicePointer<$ty>,
            a_log: DevicePointer<$ty>,
            dt_bias: DevicePointer<$ty>,
            rows: usize,
            hidden: usize,
        ) -> Result<()> {
            let params = GatedDeltaRulePreprocess::create(rows, hidden)?;
            checked_device_pointer(beta_out)?;
            checked_device_pointer(g_out)?;
            checked_device_pointer(b_in)?;
            checked_device_pointer(a_in)?;
            checked_device_pointer(a_log)?;
            checked_device_pointer(dt_bias)?;
            unsafe {
                kernel_recurrent::$kernel_fn(
                    beta_out,
                    g_out,
                    b_in,
                    a_in,
                    a_log,
                    dt_bias,
                    params.hidden,
                    params.len,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
gated_delta_rule_preprocess_fn!(
    gated_delta_rule_preprocess_f32,
    f32,
    gated_delta_rule_preprocess_f32
);
#[cfg(feature = "dtype-f16")]
gated_delta_rule_preprocess_fn!(
    gated_delta_rule_preprocess_f16,
    f16,
    gated_delta_rule_preprocess_f16
);
#[cfg(feature = "dtype-bf16")]
gated_delta_rule_preprocess_fn!(
    gated_delta_rule_preprocess_bf16,
    bf16,
    gated_delta_rule_preprocess_bf16
);

#[cfg(feature = "dtype-f16")]
gated_delta_rule_preprocess_g_f32_fn!(
    gated_delta_rule_preprocess_f16_g_f32,
    f16,
    gated_delta_rule_preprocess_f16_g_f32
);
#[cfg(feature = "dtype-bf16")]
gated_delta_rule_preprocess_g_f32_fn!(
    gated_delta_rule_preprocess_bf16_g_f32,
    bf16,
    gated_delta_rule_preprocess_bf16_g_f32
);

macro_rules! recurrent_gated_delta_rule_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            final_state: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            beta: DevicePointer<$ty>,
            initial_state: DevicePointer<$ty>,
            params: RecurrentGatedDeltaRule,
            has_initial_state: bool,
            output_final_state: bool,
        ) -> Result<()> {
            checked_device_pointer(out)?;
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;
            checked_device_pointer(gate)?;
            checked_device_pointer(beta)?;
            if has_initial_state {
                checked_device_pointer(initial_state)?;
            }
            if output_final_state {
                checked_device_pointer(final_state)?;
            }
            unsafe {
                kernel_recurrent::$kernel_fn(
                    out,
                    final_state,
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    initial_state,
                    params.batch,
                    params.time,
                    params.query_heads,
                    params.value_heads,
                    params.qk_dim,
                    params.value_dim,
                    params.scale,
                    i32::from(has_initial_state),
                    i32::from(output_final_state),
                    params.use_qk_l2norm,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
recurrent_gated_delta_rule_fn!(
    recurrent_gated_delta_rule_f32,
    f32,
    recurrent_gated_delta_rule_f32
);
#[cfg(feature = "dtype-f16")]
recurrent_gated_delta_rule_fn!(
    recurrent_gated_delta_rule_f16,
    f16,
    recurrent_gated_delta_rule_f16
);
#[cfg(feature = "dtype-bf16")]
recurrent_gated_delta_rule_fn!(
    recurrent_gated_delta_rule_bf16,
    bf16,
    recurrent_gated_delta_rule_bf16
);

macro_rules! recurrent_gated_delta_rule_decode_fn {
    ($name:ident, $kernel_fn:ident, $ty:ty) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            final_state: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            beta: DevicePointer<$ty>,
            initial_state: DevicePointer<$ty>,
            params: RecurrentGatedDeltaRule,
            output_final_state: bool,
        ) -> Result<()> {
            if params.time != 1 {
                return Err(Error::InvalidLength);
            }
            checked_device_pointer(out)?;
            if output_final_state {
                checked_device_pointer(final_state)?;
            }
            checked_device_pointer(query)?;
            checked_device_pointer(key)?;
            checked_device_pointer(value)?;
            checked_device_pointer(gate)?;
            checked_device_pointer(beta)?;
            checked_device_pointer(initial_state)?;
            unsafe {
                kernel_recurrent::$kernel_fn(
                    out,
                    final_state,
                    query,
                    key,
                    value,
                    gate,
                    beta,
                    initial_state,
                    params.batch,
                    params.query_heads,
                    params.value_heads,
                    params.qk_dim,
                    params.value_dim,
                    params.scale,
                    i32::from(output_final_state),
                    params.use_qk_l2norm,
                )
            }
            .grid(params.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

#[cfg(feature = "dtype-f32")]
recurrent_gated_delta_rule_decode_fn!(
    recurrent_gated_delta_rule_decode_f32,
    recurrent_gated_delta_rule_decode_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
recurrent_gated_delta_rule_decode_fn!(
    recurrent_gated_delta_rule_decode_f16,
    recurrent_gated_delta_rule_decode_f16,
    f16
);
#[cfg(feature = "dtype-bf16")]
recurrent_gated_delta_rule_decode_fn!(
    recurrent_gated_delta_rule_decode_bf16,
    recurrent_gated_delta_rule_decode_bf16,
    bf16
);

macro_rules! chunk_gated_delta_rule_fn {
    ($name:ident, $recurrent_fn:ident, $ty:ty) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            final_state: DevicePointer<$ty>,
            query: DevicePointer<$ty>,
            key: DevicePointer<$ty>,
            value: DevicePointer<$ty>,
            gate: DevicePointer<$ty>,
            beta: DevicePointer<$ty>,
            initial_state: DevicePointer<$ty>,
            batch: usize,
            time: usize,
            heads: usize,
            qk_dim: usize,
            value_dim: usize,
            chunk_size: usize,
            use_qk_l2norm: bool,
            has_initial_state: bool,
            output_final_state: bool,
        ) -> Result<()> {
            if chunk_size == 0 {
                return Err(Error::InvalidLength);
            }
            let params = RecurrentGatedDeltaRule::create(
                batch,
                time,
                heads,
                heads,
                qk_dim,
                value_dim,
                use_qk_l2norm,
            )?;
            $recurrent_fn(
                stream,
                out,
                final_state,
                query,
                key,
                value,
                gate,
                beta,
                initial_state,
                params,
                has_initial_state,
                output_final_state,
            )
        }
    };
}

#[cfg(feature = "dtype-f32")]
chunk_gated_delta_rule_fn!(
    chunk_gated_delta_rule_f32,
    recurrent_gated_delta_rule_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
chunk_gated_delta_rule_fn!(
    chunk_gated_delta_rule_f16,
    recurrent_gated_delta_rule_f16,
    f16
);
#[cfg(feature = "dtype-bf16")]
chunk_gated_delta_rule_fn!(
    chunk_gated_delta_rule_bf16,
    recurrent_gated_delta_rule_bf16,
    bf16
);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    #[cfg(feature = "dtype-bf16")]
    use crate::cpu::recurrent::{bfloat_to_f32, bfloat_vec, round_bfloat_vec};
    #[cfg(feature = "dtype-f16")]
    use crate::cpu::recurrent::{half_to_f32, half_vec, round_half_vec};
    use crate::{cpu::recurrent as cpu_recurrent, error::Result};

    #[test]
    fn gated_delta_rule_preprocess_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let hidden = 5usize;
        let b_host = vec![
            -2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5, -0.75, 0.25, 0.75, 1.25, -2.5, 3.0,
        ];
        let a_host = vec![
            -50.0f32, -0.5, 0.0, 25.0, 100.0, 1.5, -1.5, 2.0, -2.0, 0.25, 0.75, -0.75, 1.25, -1.25,
            0.0,
        ];
        let a_log_host = vec![-0.5f32, 0.0, 0.25, -1.0, 0.5];
        let dt_bias_host = vec![0.1f32, -0.2, 0.3, 0.0, -0.4];
        let (expected_beta, expected_g) = cpu_recurrent::gated_delta_rule_preprocess(
            &b_host,
            &a_host,
            &a_log_host,
            &dt_bias_host,
            rows,
            hidden,
        );

        let b = api::copy_host_vec_to_device(&Arc::new(b_host)).sync_on(&stream)?;
        let a = api::copy_host_vec_to_device(&Arc::new(a_host)).sync_on(&stream)?;
        let a_log = api::copy_host_vec_to_device(&Arc::new(a_log_host)).sync_on(&stream)?;
        let dt_bias = api::copy_host_vec_to_device(&Arc::new(dt_bias_host)).sync_on(&stream)?;
        let beta = api::zeros::<f32>(&[rows * hidden]).sync_on(&stream)?;
        let g = api::zeros::<f32>(&[rows * hidden]).sync_on(&stream)?;

        gated_delta_rule_preprocess_f32(
            &stream,
            beta.device_pointer(),
            g.device_pointer(),
            b.device_pointer(),
            a.device_pointer(),
            a_log.device_pointer(),
            dt_bias.device_pointer(),
            rows,
            hidden,
        )?;

        singe_core::assert_close!(&beta.to_host_vec().sync_on(&stream)?, &expected_beta, 1e-6);
        singe_core::assert_close!(&g.to_host_vec().sync_on(&stream)?, &expected_g, 1e-6);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn gated_delta_rule_preprocess_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let hidden = 5usize;
        let b_host = half_vec(&[
            -2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5, -0.75, 0.25, 0.75, 1.25, -2.5, 3.0,
        ]);
        let a_host = half_vec(&[
            -1.0f32, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0, 0.25, 0.75, -0.75, 1.25, -1.25, 0.0,
        ]);
        let a_log_host = half_vec(&[-0.5f32, 0.0, 0.25, -1.0, 0.5]);
        let dt_bias_host = half_vec(&[0.1f32, -0.2, 0.3, 0.0, -0.4]);
        let (expected_beta, expected_g) = cpu_recurrent::gated_delta_rule_preprocess(
            &half_to_f32(&b_host),
            &half_to_f32(&a_host),
            &half_to_f32(&a_log_host),
            &half_to_f32(&dt_bias_host),
            rows,
            hidden,
        );
        let expected_beta = round_half_vec(&expected_beta);
        let expected_g = round_half_vec(&expected_g);

        let b = api::copy_host_vec_to_device(&Arc::new(b_host)).sync_on(&stream)?;
        let a = api::copy_host_vec_to_device(&Arc::new(a_host)).sync_on(&stream)?;
        let a_log = api::copy_host_vec_to_device(&Arc::new(a_log_host)).sync_on(&stream)?;
        let dt_bias = api::copy_host_vec_to_device(&Arc::new(dt_bias_host)).sync_on(&stream)?;
        let beta = api::zeros::<f16>(&[rows * hidden]).sync_on(&stream)?;
        let g = api::zeros::<f16>(&[rows * hidden]).sync_on(&stream)?;

        gated_delta_rule_preprocess_f16(
            &stream,
            beta.device_pointer(),
            g.device_pointer(),
            b.device_pointer(),
            a.device_pointer(),
            a_log.device_pointer(),
            dt_bias.device_pointer(),
            rows,
            hidden,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&beta.to_host_vec().sync_on(&stream)?),
            &expected_beta,
            1e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&g.to_host_vec().sync_on(&stream)?),
            &expected_g,
            1e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn gated_delta_rule_preprocess_f16_g_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let hidden = 5usize;
        let b_host = half_vec(&[
            -2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5, -0.75, 0.25, 0.75, 1.25, -2.5, 3.0,
        ]);
        let a_host = half_vec(&[
            -1.0f32, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0, 0.25, 0.75, -0.75, 1.25, -1.25, 0.0,
        ]);
        let a_log_host = half_vec(&[-0.5f32, 0.0, 0.25, -1.0, 0.5]);
        let dt_bias_host = half_vec(&[0.1f32, -0.2, 0.3, 0.0, -0.4]);
        let (expected_beta, expected_g) = cpu_recurrent::gated_delta_rule_preprocess(
            &half_to_f32(&b_host),
            &half_to_f32(&a_host),
            &half_to_f32(&a_log_host),
            &half_to_f32(&dt_bias_host),
            rows,
            hidden,
        );
        let expected_beta = round_half_vec(&expected_beta);

        let b = api::copy_host_vec_to_device(&Arc::new(b_host)).sync_on(&stream)?;
        let a = api::copy_host_vec_to_device(&Arc::new(a_host)).sync_on(&stream)?;
        let a_log = api::copy_host_vec_to_device(&Arc::new(a_log_host)).sync_on(&stream)?;
        let dt_bias = api::copy_host_vec_to_device(&Arc::new(dt_bias_host)).sync_on(&stream)?;
        let beta = api::zeros::<f16>(&[rows * hidden]).sync_on(&stream)?;
        let g = api::zeros::<f32>(&[rows * hidden]).sync_on(&stream)?;

        gated_delta_rule_preprocess_f16_g_f32(
            &stream,
            beta.device_pointer(),
            g.device_pointer(),
            b.device_pointer(),
            a.device_pointer(),
            a_log.device_pointer(),
            dt_bias.device_pointer(),
            rows,
            hidden,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&beta.to_host_vec().sync_on(&stream)?),
            &expected_beta,
            1e-3,
        );
        singe_core::assert_close!(&g.to_host_vec().sync_on(&stream)?, &expected_g, 1e-6);
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn gated_delta_rule_preprocess_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let hidden = 5usize;
        let b_host = bfloat_vec(&[
            -2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5, -0.75, 0.25, 0.75, 1.25, -2.5, 3.0,
        ]);
        let a_host = bfloat_vec(&[
            -1.0f32, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0, 0.25, 0.75, -0.75, 1.25, -1.25, 0.0,
        ]);
        let a_log_host = bfloat_vec(&[-0.5f32, 0.0, 0.25, -1.0, 0.5]);
        let dt_bias_host = bfloat_vec(&[0.1f32, -0.2, 0.3, 0.0, -0.4]);
        let (expected_beta, expected_g) = cpu_recurrent::gated_delta_rule_preprocess(
            &bfloat_to_f32(&b_host),
            &bfloat_to_f32(&a_host),
            &bfloat_to_f32(&a_log_host),
            &bfloat_to_f32(&dt_bias_host),
            rows,
            hidden,
        );
        let expected_beta = round_bfloat_vec(&expected_beta);
        let expected_g = round_bfloat_vec(&expected_g);

        let b = api::copy_host_vec_to_device(&Arc::new(b_host)).sync_on(&stream)?;
        let a = api::copy_host_vec_to_device(&Arc::new(a_host)).sync_on(&stream)?;
        let a_log = api::copy_host_vec_to_device(&Arc::new(a_log_host)).sync_on(&stream)?;
        let dt_bias = api::copy_host_vec_to_device(&Arc::new(dt_bias_host)).sync_on(&stream)?;
        let beta = api::zeros::<bf16>(&[rows * hidden]).sync_on(&stream)?;
        let g = api::zeros::<bf16>(&[rows * hidden]).sync_on(&stream)?;

        gated_delta_rule_preprocess_bf16(
            &stream,
            beta.device_pointer(),
            g.device_pointer(),
            b.device_pointer(),
            a.device_pointer(),
            a_log.device_pointer(),
            dt_bias.device_pointer(),
            rows,
            hidden,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&beta.to_host_vec().sync_on(&stream)?),
            &expected_beta,
            4e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&g.to_host_vec().sync_on(&stream)?),
            &expected_g,
            8e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn gated_delta_rule_preprocess_bf16_g_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let rows = 3usize;
        let hidden = 5usize;
        let b_host = bfloat_vec(&[
            -2.0f32, -1.0, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, -1.5, -0.75, 0.25, 0.75, 1.25, -2.5, 3.0,
        ]);
        let a_host = bfloat_vec(&[
            -1.0f32, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5, 2.0, -2.0, 0.25, 0.75, -0.75, 1.25, -1.25, 0.0,
        ]);
        let a_log_host = bfloat_vec(&[-0.5f32, 0.0, 0.25, -1.0, 0.5]);
        let dt_bias_host = bfloat_vec(&[0.1f32, -0.2, 0.3, 0.0, -0.4]);
        let (expected_beta, expected_g) = cpu_recurrent::gated_delta_rule_preprocess(
            &bfloat_to_f32(&b_host),
            &bfloat_to_f32(&a_host),
            &bfloat_to_f32(&a_log_host),
            &bfloat_to_f32(&dt_bias_host),
            rows,
            hidden,
        );
        let expected_beta = round_bfloat_vec(&expected_beta);

        let b = api::copy_host_vec_to_device(&Arc::new(b_host)).sync_on(&stream)?;
        let a = api::copy_host_vec_to_device(&Arc::new(a_host)).sync_on(&stream)?;
        let a_log = api::copy_host_vec_to_device(&Arc::new(a_log_host)).sync_on(&stream)?;
        let dt_bias = api::copy_host_vec_to_device(&Arc::new(dt_bias_host)).sync_on(&stream)?;
        let beta = api::zeros::<bf16>(&[rows * hidden]).sync_on(&stream)?;
        let g = api::zeros::<f32>(&[rows * hidden]).sync_on(&stream)?;

        gated_delta_rule_preprocess_bf16_g_f32(
            &stream,
            beta.device_pointer(),
            g.device_pointer(),
            b.device_pointer(),
            a.device_pointer(),
            a_log.device_pointer(),
            dt_bias.device_pointer(),
            rows,
            hidden,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&beta.to_host_vec().sync_on(&stream)?),
            &expected_beta,
            4e-3,
        );
        singe_core::assert_close!(&g.to_host_vec().sync_on(&stream)?, &expected_g, 1e-6);
        Ok(())
    }

    #[test]
    fn recurrent_gated_delta_rule_f32_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 3usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = vec![0.2f32, -0.1, 0.4, 0.5, 0.3, -0.2, -0.4, 0.6, 0.1];
        let key_host = vec![-0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1];
        let value_host = vec![
            0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1, 0.6, 0.1, -0.3, 0.2, -0.2, 0.4, 0.8,
            -0.5, 0.1, 0.9, -0.6, 0.3, 0.2, -0.7, 0.4, 0.5,
        ];
        let gate_host = vec![-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25];
        let beta_host = vec![0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8];
        let initial_state_host = vec![
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ];
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &query_host,
            &key_host,
            &value_host,
            &gate_host,
            &beta_host,
            Some(&initial_state_host),
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        );

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<f32>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_f32(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-5);
        singe_core::assert_close!(
            &final_state.to_host_vec().sync_on(&stream)?,
            &expected_final_state,
            1e-5,
        );
        Ok(())
    }

    #[test]
    fn recurrent_gated_delta_rule_decode_f32_matches_recurrent_t1() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 1usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = vec![0.2f32, -0.1, 0.4];
        let key_host = vec![-0.3f32, 0.7, 0.2];
        let value_host = vec![0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1];
        let gate_host = vec![-0.2f32, -0.4];
        let beta_host = vec![0.6f32, 0.4];
        let initial_state_host = vec![
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ];
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let expected_out =
            api::zeros::<f32>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let expected_state =
            api::zeros::<f32>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;
        let actual_out =
            api::zeros::<f32>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let actual_state =
            api::zeros::<f32>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_f32(
            &stream,
            expected_out.device_pointer(),
            expected_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;
        recurrent_gated_delta_rule_decode_f32(
            &stream,
            actual_out.device_pointer(),
            actual_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
        )?;

        singe_core::assert_close!(
            &actual_out.to_host_vec().sync_on(&stream)?,
            &expected_out.to_host_vec().sync_on(&stream)?,
            1e-5,
        );
        singe_core::assert_close!(
            &actual_state.to_host_vec().sync_on(&stream)?,
            &expected_state.to_host_vec().sync_on(&stream)?,
            1e-5,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn recurrent_gated_delta_rule_decode_f16_matches_recurrent_t1() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 1usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = half_vec(&[0.2f32, -0.1, 0.4]);
        let key_host = half_vec(&[-0.3f32, 0.7, 0.2]);
        let value_host = half_vec(&[0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1]);
        let gate_host = half_vec(&[-0.2f32, -0.4]);
        let beta_host = half_vec(&[0.6f32, 0.4]);
        let initial_state_host = half_vec(&[
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ]);
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let expected_out =
            api::zeros::<f16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let expected_state =
            api::zeros::<f16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;
        let actual_out =
            api::zeros::<f16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let actual_state =
            api::zeros::<f16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_f16(
            &stream,
            expected_out.device_pointer(),
            expected_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;
        recurrent_gated_delta_rule_decode_f16(
            &stream,
            actual_out.device_pointer(),
            actual_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&actual_out.to_host_vec().sync_on(&stream)?),
            &half_to_f32(&expected_out.to_host_vec().sync_on(&stream)?),
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&actual_state.to_host_vec().sync_on(&stream)?),
            &half_to_f32(&expected_state.to_host_vec().sync_on(&stream)?),
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn recurrent_gated_delta_rule_decode_bf16_matches_recurrent_t1() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 1usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = bfloat_vec(&[0.2f32, -0.1, 0.4]);
        let key_host = bfloat_vec(&[-0.3f32, 0.7, 0.2]);
        let value_host = bfloat_vec(&[0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1]);
        let gate_host = bfloat_vec(&[-0.2f32, -0.4]);
        let beta_host = bfloat_vec(&[0.6f32, 0.4]);
        let initial_state_host = bfloat_vec(&[
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ]);
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let expected_out =
            api::zeros::<bf16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let expected_state =
            api::zeros::<bf16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;
        let actual_out =
            api::zeros::<bf16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let actual_state =
            api::zeros::<bf16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_bf16(
            &stream,
            expected_out.device_pointer(),
            expected_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;
        recurrent_gated_delta_rule_decode_bf16(
            &stream,
            actual_out.device_pointer(),
            actual_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&actual_out.to_host_vec().sync_on(&stream)?),
            &bfloat_to_f32(&expected_out.to_host_vec().sync_on(&stream)?),
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&actual_state.to_host_vec().sync_on(&stream)?),
            &bfloat_to_f32(&expected_state.to_host_vec().sync_on(&stream)?),
            8e-3,
        );
        Ok(())
    }

    #[test]
    fn recurrent_gated_delta_rule_f32_without_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 3usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = vec![0.2f32, -0.1, 0.4, 0.5, 0.3, -0.2, -0.4, 0.6, 0.1];
        let key_host = vec![-0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1];
        let value_host = vec![
            0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1, 0.6, 0.1, -0.3, 0.2, -0.2, 0.4, 0.8,
            -0.5, 0.1, 0.9, -0.6, 0.3, 0.2, -0.7, 0.4, 0.5,
        ];
        let gate_host = vec![-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25];
        let beta_host = vec![0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8];
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            false,
        )?;
        let (expected_out, _) = cpu_recurrent::recurrent_gated_delta_rule(
            &query_host,
            &key_host,
            &value_host,
            &gate_host,
            &beta_host,
            None,
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            false,
        );

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let final_state = api::zeros::<f32>(&[1]).sync_on(&stream)?;
        let initial_state = api::zeros::<f32>(&[1]).sync_on(&stream)?;

        recurrent_gated_delta_rule_f32(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            false,
            false,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn recurrent_gated_delta_rule_f16_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 3usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = half_vec(&[0.2f32, -0.1, 0.4, 0.5, 0.3, -0.2, -0.4, 0.6, 0.1]);
        let key_host = half_vec(&[-0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1]);
        let value_host = half_vec(&[
            0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1, 0.6, 0.1, -0.3, 0.2, -0.2, 0.4, 0.8,
            -0.5, 0.1, 0.9, -0.6, 0.3, 0.2, -0.7, 0.4, 0.5,
        ]);
        let gate_host = half_vec(&[-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25]);
        let beta_host = half_vec(&[0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8]);
        let initial_state_host = half_vec(&[
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ]);
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &half_to_f32(&query_host),
            &half_to_f32(&key_host),
            &half_to_f32(&value_host),
            &half_to_f32(&gate_host),
            &half_to_f32(&beta_host),
            Some(&half_to_f32(&initial_state_host)),
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        );
        let expected_out = round_half_vec(&expected_out);
        let expected_final_state = round_half_vec(&expected_final_state);

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<f16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_f16(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&final_state.to_host_vec().sync_on(&stream)?),
            &expected_final_state,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn recurrent_gated_delta_rule_bf16_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, query_heads, value_heads, qk_dim, value_dim) =
            (1usize, 3usize, 1usize, 2usize, 3usize, 4usize);
        let query_host = bfloat_vec(&[0.2f32, -0.1, 0.4, 0.5, 0.3, -0.2, -0.4, 0.6, 0.1]);
        let key_host = bfloat_vec(&[-0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1]);
        let value_host = bfloat_vec(&[
            0.3f32, -0.2, 0.1, 0.7, -0.4, 0.5, 0.2, -0.1, 0.6, 0.1, -0.3, 0.2, -0.2, 0.4, 0.8,
            -0.5, 0.1, 0.9, -0.6, 0.3, 0.2, -0.7, 0.4, 0.5,
        ]);
        let gate_host = bfloat_vec(&[-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25]);
        let beta_host = bfloat_vec(&[0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8]);
        let initial_state_host = bfloat_vec(&[
            0.05f32, -0.02, 0.03, 0.01, 0.04, 0.02, -0.01, 0.06, -0.03, 0.05, 0.02, -0.04, 0.02,
            0.01, -0.05, 0.03, -0.04, 0.06, 0.01, -0.02, 0.03, -0.01, 0.04, 0.02,
        ]);
        let config = RecurrentGatedDeltaRule::create(
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        )?;
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &bfloat_to_f32(&query_host),
            &bfloat_to_f32(&key_host),
            &bfloat_to_f32(&value_host),
            &bfloat_to_f32(&gate_host),
            &bfloat_to_f32(&beta_host),
            Some(&bfloat_to_f32(&initial_state_host)),
            batch,
            time,
            query_heads,
            value_heads,
            qk_dim,
            value_dim,
            true,
        );
        let expected_out = round_bfloat_vec(&expected_out);
        let expected_final_state = round_bfloat_vec(&expected_final_state);

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[batch * time * value_heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<bf16>(&[batch * value_heads * qk_dim * value_dim]).sync_on(&stream)?;

        recurrent_gated_delta_rule_bf16(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            config,
            true,
            true,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&final_state.to_host_vec().sync_on(&stream)?),
            &expected_final_state,
            8e-3,
        );
        Ok(())
    }

    #[test]
    fn chunk_gated_delta_rule_f32_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, heads, qk_dim, value_dim) = (1usize, 4usize, 2usize, 3usize, 3usize);
        let query_host = vec![
            0.2f32, -0.1, 0.4, -0.3, 0.6, 0.1, 0.5, 0.3, -0.2, 0.4, -0.4, 0.7, -0.6, 0.2, 0.5, 0.1,
            -0.2, 0.3, 0.8, -0.5, 0.2, -0.1, 0.4, 0.6,
        ];
        let key_host = vec![
            -0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1, -0.2, 0.3, 0.5, 0.7, -0.4, 0.1,
            -0.5, 0.6, 0.2, 0.3, -0.7, 0.4, 0.2, 0.1, -0.6,
        ];
        let value_host = vec![
            0.3f32, -0.2, 0.1, -0.4, 0.5, 0.2, 0.6, 0.1, -0.3, -0.2, 0.4, 0.8, 0.1, 0.9, -0.6, 0.2,
            -0.7, 0.4, 0.5, -0.1, 0.6, -0.3, 0.2, 0.7,
        ];
        let gate_host = vec![-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25, -0.35, -0.15];
        let beta_host = vec![0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8, 0.55, 0.45];
        let initial_state_host = vec![
            0.05f32, -0.02, 0.03, 0.04, 0.02, -0.01, -0.03, 0.05, 0.02, 0.02, 0.01, -0.05, -0.04,
            0.06, 0.01, 0.03, -0.01, 0.04,
        ];
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &query_host,
            &key_host,
            &value_host,
            &gate_host,
            &beta_host,
            Some(&initial_state_host),
            batch,
            time,
            heads,
            heads,
            qk_dim,
            value_dim,
            false,
        );

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[batch * time * heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<f32>(&[batch * heads * qk_dim * value_dim]).sync_on(&stream)?;

        chunk_gated_delta_rule_f32(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            batch,
            time,
            heads,
            qk_dim,
            value_dim,
            2,
            false,
            true,
            true,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected_out, 1e-5);
        singe_core::assert_close!(
            &final_state.to_host_vec().sync_on(&stream)?,
            &expected_final_state,
            1e-5,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn chunk_gated_delta_rule_f16_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, heads, qk_dim, value_dim) = (1usize, 4usize, 2usize, 3usize, 3usize);
        let query_host = half_vec(&[
            0.2f32, -0.1, 0.4, -0.3, 0.6, 0.1, 0.5, 0.3, -0.2, 0.4, -0.4, 0.7, -0.6, 0.2, 0.5, 0.1,
            -0.2, 0.3, 0.8, -0.5, 0.2, -0.1, 0.4, 0.6,
        ]);
        let key_host = half_vec(&[
            -0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1, -0.2, 0.3, 0.5, 0.7, -0.4, 0.1,
            -0.5, 0.6, 0.2, 0.3, -0.7, 0.4, 0.2, 0.1, -0.6,
        ]);
        let value_host = half_vec(&[
            0.3f32, -0.2, 0.1, -0.4, 0.5, 0.2, 0.6, 0.1, -0.3, -0.2, 0.4, 0.8, 0.1, 0.9, -0.6, 0.2,
            -0.7, 0.4, 0.5, -0.1, 0.6, -0.3, 0.2, 0.7,
        ]);
        let gate_host = half_vec(&[-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25, -0.35, -0.15]);
        let beta_host = half_vec(&[0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8, 0.55, 0.45]);
        let initial_state_host = half_vec(&[
            0.05f32, -0.02, 0.03, 0.04, 0.02, -0.01, -0.03, 0.05, 0.02, 0.02, 0.01, -0.05, -0.04,
            0.06, 0.01, 0.03, -0.01, 0.04,
        ]);
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &half_to_f32(&query_host),
            &half_to_f32(&key_host),
            &half_to_f32(&value_host),
            &half_to_f32(&gate_host),
            &half_to_f32(&beta_host),
            Some(&half_to_f32(&initial_state_host)),
            batch,
            time,
            heads,
            heads,
            qk_dim,
            value_dim,
            false,
        );
        let expected_out = round_half_vec(&expected_out);
        let expected_final_state = round_half_vec(&expected_final_state);

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[batch * time * heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<f16>(&[batch * heads * qk_dim * value_dim]).sync_on(&stream)?;

        chunk_gated_delta_rule_f16(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            batch,
            time,
            heads,
            qk_dim,
            value_dim,
            2,
            false,
            true,
            true,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&final_state.to_host_vec().sync_on(&stream)?),
            &expected_final_state,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn chunk_gated_delta_rule_bf16_with_state() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let (batch, time, heads, qk_dim, value_dim) = (1usize, 4usize, 2usize, 3usize, 3usize);
        let query_host = bfloat_vec(&[
            0.2f32, -0.1, 0.4, -0.3, 0.6, 0.1, 0.5, 0.3, -0.2, 0.4, -0.4, 0.7, -0.6, 0.2, 0.5, 0.1,
            -0.2, 0.3, 0.8, -0.5, 0.2, -0.1, 0.4, 0.6,
        ]);
        let key_host = bfloat_vec(&[
            -0.3f32, 0.7, 0.2, 0.1, -0.5, 0.4, 0.6, 0.2, -0.1, -0.2, 0.3, 0.5, 0.7, -0.4, 0.1,
            -0.5, 0.6, 0.2, 0.3, -0.7, 0.4, 0.2, 0.1, -0.6,
        ]);
        let value_host = bfloat_vec(&[
            0.3f32, -0.2, 0.1, -0.4, 0.5, 0.2, 0.6, 0.1, -0.3, -0.2, 0.4, 0.8, 0.1, 0.9, -0.6, 0.2,
            -0.7, 0.4, 0.5, -0.1, 0.6, -0.3, 0.2, 0.7,
        ]);
        let gate_host = bfloat_vec(&[-0.2f32, -0.4, -0.1, -0.3, -0.5, -0.25, -0.35, -0.15]);
        let beta_host = bfloat_vec(&[0.6f32, 0.4, 0.7, 0.5, 0.3, 0.8, 0.55, 0.45]);
        let initial_state_host = bfloat_vec(&[
            0.05f32, -0.02, 0.03, 0.04, 0.02, -0.01, -0.03, 0.05, 0.02, 0.02, 0.01, -0.05, -0.04,
            0.06, 0.01, 0.03, -0.01, 0.04,
        ]);
        let (expected_out, expected_final_state) = cpu_recurrent::recurrent_gated_delta_rule(
            &bfloat_to_f32(&query_host),
            &bfloat_to_f32(&key_host),
            &bfloat_to_f32(&value_host),
            &bfloat_to_f32(&gate_host),
            &bfloat_to_f32(&beta_host),
            Some(&bfloat_to_f32(&initial_state_host)),
            batch,
            time,
            heads,
            heads,
            qk_dim,
            value_dim,
            false,
        );
        let expected_out = round_bfloat_vec(&expected_out);
        let expected_final_state = round_bfloat_vec(&expected_final_state);

        let query = api::copy_host_vec_to_device(&Arc::new(query_host)).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(key_host)).sync_on(&stream)?;
        let value = api::copy_host_vec_to_device(&Arc::new(value_host)).sync_on(&stream)?;
        let gate = api::copy_host_vec_to_device(&Arc::new(gate_host)).sync_on(&stream)?;
        let beta = api::copy_host_vec_to_device(&Arc::new(beta_host)).sync_on(&stream)?;
        let initial_state =
            api::copy_host_vec_to_device(&Arc::new(initial_state_host)).sync_on(&stream)?;
        let out = api::zeros::<bf16>(&[batch * time * heads * value_dim]).sync_on(&stream)?;
        let final_state =
            api::zeros::<bf16>(&[batch * heads * qk_dim * value_dim]).sync_on(&stream)?;

        chunk_gated_delta_rule_bf16(
            &stream,
            out.device_pointer(),
            final_state.device_pointer(),
            query.device_pointer(),
            key.device_pointer(),
            value.device_pointer(),
            gate.device_pointer(),
            beta.device_pointer(),
            initial_state.device_pointer(),
            batch,
            time,
            heads,
            qk_dim,
            value_dim,
            2,
            false,
            true,
            true,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &expected_out,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&final_state.to_host_vec().sync_on(&stream)?),
            &expected_final_state,
            8e-3,
        );
        Ok(())
    }
}
