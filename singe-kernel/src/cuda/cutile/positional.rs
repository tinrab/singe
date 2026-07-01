use std::sync::Arc;

#[cfg(feature = "dtype-bf16")]
use cutile::half::bf16;
#[cfg(feature = "dtype-f16")]
use cutile::half::f16;
use cutile::{
    cuda_async::device_buffer::DevicePointer, cuda_core::Stream, tile_kernel::TileKernel,
};

#[cfg(feature = "dtype-f16")]
use crate::cuda::cutile::kernel::f16::positional as kernel_f16;
#[cfg(feature = "dtype-f32")]
use crate::cuda::cutile::kernel::f32::positional as kernel_f32;
#[cfg(feature = "dtype-f64")]
use crate::cuda::cutile::kernel::f64::positional as kernel_f64;
use crate::{
    cuda::{
        cutile::{
            DeviceOpExt,
            kernel::{
                common as kernel_common, positional_structured as kernel_positional_structured,
            },
            utility::{VectorLaunch, checked_device_pointer, raw_vector_grid},
        },
        positional::{
            BatchedRotaryEmbeddingConfig, BnsdQkRopeConfig, BshdRopeConfig,
            InterleavedComplexRopeConfig, MultiaxisRopeConfig, QkBatchedRotaryEmbeddingConfig,
            QkRotaryEmbeddingConfig, RotaryEmbeddingConfig,
        },
    },
    error::{Error, Result},
    utility::{
        checked_element_count, checked_i32_value, checked_rank3_i32, checked_rank4_i32,
        checked_rank4_len,
    },
};

type RotaryConfig = ([i32; 4], [i32; 4], [i32; 2], [i32; 2], i32);
type BatchedRotaryConfig = ([i32; 4], [i32; 4], [i32; 3], [i32; 3], i32, i32);

#[derive(Clone, Copy, Debug)]
struct QkBatchedRotaryConfig {
    q: BatchedRotaryEmbeddingConfig,
    k: BatchedRotaryEmbeddingConfig,
}

struct QkRotaryConfig {
    q_output_dimensions: [i32; 4],
    k_output_dimensions: [i32; 4],
    q_input_strides: [i32; 4],
    k_input_strides: [i32; 4],
    cos_strides: [i32; 2],
    sin_strides: [i32; 2],
    rotary_pairs: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BshdRopeLaunch {
    batch: i32,
    seq_len: i32,
    heads: i32,
    head_dim: i32,
    pair_count: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BnsdQkRopeLaunch {
    batch: i32,
    seq_len: i32,
    query_heads: i32,
    key_heads: i32,
    head_dim: i32,
    trig_batch: i32,
    rotary_pairs: i32,
    q_len: i32,
    k_len: i32,
    total_len: i32,
    q_pair_count: i32,
    total_pair_count: i32,
    grid: (u32, u32, u32),
    pair_grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MultiaxisRopeLaunch {
    batch: i32,
    seq_len: i32,
    query_heads: i32,
    key_heads: i32,
    head_dim: i32,
    section_t: i32,
    section_h: i32,
    pair_count: i32,
    grid: (u32, u32, u32),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct InterleavedComplexRopeLaunch {
    batch: i32,
    seq_len: i32,
    query_heads: i32,
    key_heads: i32,
    head_dim: i32,
    pair_count: i32,
    grid: (u32, u32, u32),
}

fn checked_rotary_config(config: RotaryEmbeddingConfig) -> Result<RotaryConfig> {
    let head_dim = config.output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }

    Ok((
        checked_rank4_i32(config.output_dimensions)?,
        checked_rank4_i32(config.input_strides)?,
        [
            checked_i32_value(config.cos_strides[0])?,
            checked_i32_value(config.cos_strides[1])?,
        ],
        [
            checked_i32_value(config.sin_strides[0])?,
            checked_i32_value(config.sin_strides[1])?,
        ],
        checked_i32_value(config.rotary_pairs)?,
    ))
}

fn checked_batched_rotary_config(
    config: BatchedRotaryEmbeddingConfig,
) -> Result<BatchedRotaryConfig> {
    let output_batch = config.output_dimensions[0];
    if config.trig_batch != 1 && config.trig_batch != output_batch {
        return Err(Error::LengthMismatch);
    }
    let head_dim = config.output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }

    Ok((
        checked_rank4_i32(config.output_dimensions)?,
        checked_rank4_i32(config.input_strides)?,
        checked_rank3_i32(config.cos_strides)?,
        checked_rank3_i32(config.sin_strides)?,
        checked_i32_value(config.trig_batch)?,
        checked_i32_value(config.rotary_pairs)?,
    ))
}

fn checked_qk_rotary_config(config: QkRotaryEmbeddingConfig) -> Result<QkRotaryConfig> {
    if config.q_output_dimensions[0] != config.k_output_dimensions[0]
        || config.q_output_dimensions[2] != config.k_output_dimensions[2]
        || config.q_output_dimensions[3] != config.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let head_dim = config.q_output_dimensions[3];
    if config
        .rotary_pairs
        .checked_mul(2)
        .ok_or(Error::SizeOverflow)?
        > head_dim
    {
        return Err(Error::LengthMismatch);
    }

    Ok(QkRotaryConfig {
        q_output_dimensions: checked_rank4_i32(config.q_output_dimensions)?,
        k_output_dimensions: checked_rank4_i32(config.k_output_dimensions)?,
        q_input_strides: checked_rank4_i32(config.q_input_strides)?,
        k_input_strides: checked_rank4_i32(config.k_input_strides)?,
        cos_strides: [
            checked_i32_value(config.cos_strides[0])?,
            checked_i32_value(config.cos_strides[1])?,
        ],
        sin_strides: [
            checked_i32_value(config.sin_strides[0])?,
            checked_i32_value(config.sin_strides[1])?,
        ],
        rotary_pairs: checked_i32_value(config.rotary_pairs)?,
    })
}

fn checked_qk_batched_rotary_config(
    config: QkBatchedRotaryEmbeddingConfig,
) -> Result<QkBatchedRotaryConfig> {
    if config.q_output_dimensions[0] != config.k_output_dimensions[0]
        || config.q_output_dimensions[2] != config.k_output_dimensions[2]
        || config.q_output_dimensions[3] != config.k_output_dimensions[3]
    {
        return Err(Error::LengthMismatch);
    }
    let q = BatchedRotaryEmbeddingConfig {
        output_dimensions: config.q_output_dimensions,
        input_strides: config.q_input_strides,
        cos_strides: config.cos_strides,
        sin_strides: config.sin_strides,
        trig_batch: config.trig_batch,
        rotary_pairs: config.rotary_pairs,
    };
    let k = BatchedRotaryEmbeddingConfig {
        output_dimensions: config.k_output_dimensions,
        input_strides: config.k_input_strides,
        cos_strides: config.cos_strides,
        sin_strides: config.sin_strides,
        trig_batch: config.trig_batch,
        rotary_pairs: config.rotary_pairs,
    };
    checked_batched_rotary_config(q)?;
    checked_batched_rotary_config(k)?;
    Ok(QkBatchedRotaryConfig { q, k })
}

impl BshdRopeLaunch {
    fn create(config: BshdRopeConfig) -> Result<Self> {
        if config.batch == 0
            || config.seq_len == 0
            || config.heads == 0
            || config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
        {
            return Err(Error::InvalidLength);
        }
        let pair_count = checked_element_count(
            checked_element_count(
                checked_element_count(config.batch, config.seq_len)?,
                config.heads,
            )?,
            config.head_dim / 2,
        )?;
        Ok(Self {
            batch: checked_i32_value(config.batch)?,
            seq_len: checked_i32_value(config.seq_len)?,
            heads: checked_i32_value(config.heads)?,
            head_dim: checked_i32_value(config.head_dim)?,
            pair_count: checked_i32_value(pair_count)?,
            grid: raw_vector_grid(pair_count)?,
        })
    }
}

impl BnsdQkRopeLaunch {
    fn create(config: BnsdQkRopeConfig) -> Result<Self> {
        if config.batch == 0
            || config.seq_len == 0
            || config.query_heads == 0
            || config.key_heads == 0
            || config.head_dim == 0
            || config.trig_batch == 0
            || config.rotary_pairs == 0
            || config.trig_batch != 1 && config.trig_batch != config.batch
        {
            return Err(Error::InvalidLength);
        }
        if config
            .rotary_pairs
            .checked_mul(2)
            .ok_or(Error::SizeOverflow)?
            > config.head_dim
        {
            return Err(Error::LengthMismatch);
        }
        let batch_seq = checked_element_count(config.batch, config.seq_len)?;
        let q_len = checked_element_count(
            checked_element_count(batch_seq, config.query_heads)?,
            config.head_dim,
        )?;
        let k_len = checked_element_count(
            checked_element_count(batch_seq, config.key_heads)?,
            config.head_dim,
        )?;
        let total_len = q_len.checked_add(k_len).ok_or(Error::SizeOverflow)?;
        let q_pair_count = checked_element_count(
            checked_element_count(batch_seq, config.query_heads)?,
            config.rotary_pairs,
        )?;
        let k_pair_count = checked_element_count(
            checked_element_count(batch_seq, config.key_heads)?,
            config.rotary_pairs,
        )?;
        let total_pair_count = q_pair_count
            .checked_add(k_pair_count)
            .ok_or(Error::SizeOverflow)?;
        Ok(Self {
            batch: checked_i32_value(config.batch)?,
            seq_len: checked_i32_value(config.seq_len)?,
            query_heads: checked_i32_value(config.query_heads)?,
            key_heads: checked_i32_value(config.key_heads)?,
            head_dim: checked_i32_value(config.head_dim)?,
            trig_batch: checked_i32_value(config.trig_batch)?,
            rotary_pairs: checked_i32_value(config.rotary_pairs)?,
            q_len: checked_i32_value(q_len)?,
            k_len: checked_i32_value(k_len)?,
            total_len: checked_i32_value(total_len)?,
            q_pair_count: checked_i32_value(q_pair_count)?,
            total_pair_count: checked_i32_value(total_pair_count)?,
            grid: raw_vector_grid(total_len)?,
            pair_grid: raw_vector_grid(total_pair_count)?,
        })
    }
}

impl MultiaxisRopeLaunch {
    fn create(config: MultiaxisRopeConfig) -> Result<Self> {
        if config.batch == 0
            || config.seq_len == 0
            || config.query_heads == 0
            || config.key_heads == 0
            || config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
        {
            return Err(Error::InvalidLength);
        }
        let head_dim_half = config.head_dim / 2;
        let section_sum = config
            .section_t
            .checked_add(config.section_h)
            .and_then(|value| value.checked_add(config.section_w))
            .ok_or(Error::SizeOverflow)?;
        if section_sum != head_dim_half {
            return Err(Error::LengthMismatch);
        }
        let query_pairs = checked_element_count(
            checked_element_count(
                checked_element_count(config.batch, config.seq_len)?,
                config.query_heads,
            )?,
            head_dim_half,
        )?;
        let key_pairs = checked_element_count(
            checked_element_count(
                checked_element_count(config.batch, config.seq_len)?,
                config.key_heads,
            )?,
            head_dim_half,
        )?;
        let pair_count = query_pairs
            .checked_add(key_pairs)
            .ok_or(Error::SizeOverflow)?;
        Ok(Self {
            batch: checked_i32_value(config.batch)?,
            seq_len: checked_i32_value(config.seq_len)?,
            query_heads: checked_i32_value(config.query_heads)?,
            key_heads: checked_i32_value(config.key_heads)?,
            head_dim: checked_i32_value(config.head_dim)?,
            section_t: checked_i32_value(config.section_t)?,
            section_h: checked_i32_value(config.section_h)?,
            pair_count: checked_i32_value(pair_count)?,
            grid: raw_vector_grid(pair_count)?,
        })
    }
}

impl InterleavedComplexRopeLaunch {
    fn create(config: InterleavedComplexRopeConfig) -> Result<Self> {
        if config.batch == 0
            || config.seq_len == 0
            || config.query_heads == 0
            || config.key_heads == 0
            || config.head_dim == 0
            || !config.head_dim.is_multiple_of(2)
        {
            return Err(Error::InvalidLength);
        }
        let head_dim_half = config.head_dim / 2;
        let query_pairs = checked_element_count(
            checked_element_count(
                checked_element_count(config.batch, config.seq_len)?,
                config.query_heads,
            )?,
            head_dim_half,
        )?;
        let key_pairs = checked_element_count(
            checked_element_count(
                checked_element_count(config.batch, config.seq_len)?,
                config.key_heads,
            )?,
            head_dim_half,
        )?;
        let pair_count = query_pairs
            .checked_add(key_pairs)
            .ok_or(Error::SizeOverflow)?;
        Ok(Self {
            batch: checked_i32_value(config.batch)?,
            seq_len: checked_i32_value(config.seq_len)?,
            query_heads: checked_i32_value(config.query_heads)?,
            key_heads: checked_i32_value(config.key_heads)?,
            head_dim: checked_i32_value(config.head_dim)?,
            pair_count: checked_i32_value(pair_count)?,
            grid: raw_vector_grid(pair_count)?,
        })
    }
}

macro_rules! rotary_embedding_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: RotaryEmbeddingConfig,
        ) -> Result<()> {
            let len = checked_rank4_len(config.output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            if config.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }
            let (output_dimensions, input_strides, cos_strides, sin_strides, rotary_pairs) =
                checked_rotary_config(config)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    cos,
                    sin,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    cos_strides[0],
                    cos_strides[1],
                    sin_strides[0],
                    sin_strides[1],
                    rotary_pairs,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! rotary_embedding_positioned_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: RotaryEmbeddingConfig,
            position_start: usize,
        ) -> Result<()> {
            let len = checked_rank4_len(config.output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            if config.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }
            let (output_dimensions, input_strides, cos_strides, sin_strides, rotary_pairs) =
                checked_rotary_config(config)?;
            let position_start = checked_i32_value(position_start)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    cos,
                    sin,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    cos_strides[0],
                    cos_strides[1],
                    sin_strides[0],
                    sin_strides[1],
                    rotary_pairs,
                    position_start,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! rotary_embedding_dynpos_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            position_start: DevicePointer<u32>,
            config: RotaryEmbeddingConfig,
        ) -> Result<()> {
            let len = checked_rank4_len(config.output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            checked_device_pointer(position_start)?;
            if config.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }
            let (output_dimensions, input_strides, cos_strides, sin_strides, rotary_pairs) =
                checked_rotary_config(config)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    cos,
                    sin,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    cos_strides[0],
                    cos_strides[1],
                    sin_strides[0],
                    sin_strides[1],
                    rotary_pairs,
                    position_start,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! rotary_embedding_batched_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            out: DevicePointer<$ty>,
            input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: BatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            let len = checked_rank4_len(config.output_dimensions)?;
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(out)?;
            checked_device_pointer(input)?;
            if config.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }
            let (
                output_dimensions,
                input_strides,
                cos_strides,
                sin_strides,
                trig_batch,
                rotary_pairs,
            ) = checked_batched_rotary_config(config)?;
            let launch = VectorLaunch::create(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    out,
                    input,
                    cos,
                    sin,
                    output_dimensions[0],
                    output_dimensions[1],
                    output_dimensions[2],
                    output_dimensions[3],
                    input_strides[0],
                    input_strides[1],
                    input_strides[2],
                    input_strides[3],
                    cos_strides[0],
                    cos_strides[1],
                    cos_strides[2],
                    sin_strides[0],
                    sin_strides[1],
                    sin_strides[2],
                    trig_batch,
                    rotary_pairs,
                    launch.len_i32,
                )
            }
            .grid(launch.grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! rotary_embedding_qk_fn {
    ($name:ident, $ty:ty, $kernel:ident, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q_out: DevicePointer<$ty>,
            k_out: DevicePointer<$ty>,
            q_input: DevicePointer<$ty>,
            k_input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            let q_len = checked_rank4_len(config.q_output_dimensions)?;
            let k_len = checked_rank4_len(config.k_output_dimensions)?;
            let len = q_len.max(k_len);
            if len == 0 {
                return Ok(());
            }
            checked_device_pointer(q_out)?;
            checked_device_pointer(k_out)?;
            checked_device_pointer(q_input)?;
            checked_device_pointer(k_input)?;
            if config.rotary_pairs > 0 {
                checked_device_pointer(cos)?;
                checked_device_pointer(sin)?;
            }
            let checked = checked_qk_rotary_config(config)?;
            let q_len = checked_i32_value(q_len)?;
            let k_len = checked_i32_value(k_len)?;
            let grid = raw_vector_grid(len)?;
            unsafe {
                $kernel::$kernel_fn(
                    q_out,
                    k_out,
                    q_input,
                    k_input,
                    cos,
                    sin,
                    checked.q_output_dimensions[0],
                    checked.q_output_dimensions[1],
                    checked.q_output_dimensions[2],
                    checked.q_output_dimensions[3],
                    checked.q_input_strides[0],
                    checked.q_input_strides[1],
                    checked.q_input_strides[2],
                    checked.q_input_strides[3],
                    checked.k_output_dimensions[0],
                    checked.k_output_dimensions[1],
                    checked.k_output_dimensions[2],
                    checked.k_output_dimensions[3],
                    checked.k_input_strides[0],
                    checked.k_input_strides[1],
                    checked.k_input_strides[2],
                    checked.k_input_strides[3],
                    checked.cos_strides[0],
                    checked.cos_strides[1],
                    checked.sin_strides[0],
                    checked.sin_strides[1],
                    checked.rotary_pairs,
                    q_len,
                    k_len,
                )
            }
            .grid(grid)
            .enqueue_on(stream)?;
            Ok(())
        }
    };
}

macro_rules! rotary_embedding_qk_batched_fn {
    ($name:ident, $ty:ty, $kernel_fn:ident) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q_out: DevicePointer<$ty>,
            k_out: DevicePointer<$ty>,
            q_input: DevicePointer<$ty>,
            k_input: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: QkBatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            let checked = checked_qk_batched_rotary_config(config)?;
            $kernel_fn(stream, q_out, q_input, cos, sin, checked.q)?;
            $kernel_fn(stream, k_out, k_input, cos, sin, checked.k)
        }
    };
}

macro_rules! rotary_embedding_qk_batched_in_place_fn {
    ($name:ident, $kernel_fn:ident, $ty:ty) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q: DevicePointer<$ty>,
            k: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: QkBatchedRotaryEmbeddingConfig,
        ) -> Result<()> {
            let checked = checked_qk_batched_rotary_config(config)?;
            $kernel_fn(stream, q, q, cos, sin, checked.q)?;
            $kernel_fn(stream, k, k, cos, sin, checked.k)
        }
    };
}

macro_rules! rotary_embedding_qk_in_place_fn {
    ($name:ident, $kernel_fn:ident, $ty:ty) => {
        pub fn $name(
            stream: &Arc<Stream>,
            q: DevicePointer<$ty>,
            k: DevicePointer<$ty>,
            cos: DevicePointer<$ty>,
            sin: DevicePointer<$ty>,
            config: QkRotaryEmbeddingConfig,
        ) -> Result<()> {
            $kernel_fn(stream, q, k, q, k, cos, sin, config)
        }
    };
}

#[cfg(feature = "dtype-f32")]
rotary_embedding_fn!(rotary_embedding_f32, f32, kernel_f32, rotary_embedding_f32);
#[cfg(feature = "dtype-f16")]
rotary_embedding_fn!(rotary_embedding_f16, f16, kernel_f16, rotary_embedding_f16);
#[cfg(feature = "dtype-f64")]
rotary_embedding_fn!(rotary_embedding_f64, f64, kernel_f64, rotary_embedding_f64);

#[cfg(feature = "dtype-f32")]
rotary_embedding_positioned_fn!(
    rotary_embedding_positioned_f32,
    f32,
    kernel_common,
    rotary_embedding_positioned_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_positioned_fn!(
    rotary_embedding_positioned_f16,
    f16,
    kernel_common,
    rotary_embedding_positioned_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_positioned_fn!(
    rotary_embedding_positioned_f64,
    f64,
    kernel_common,
    rotary_embedding_positioned_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_dynpos_f32,
    f32,
    kernel_common,
    rotary_embedding_dynpos_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_dynpos_f16,
    f16,
    kernel_common,
    rotary_embedding_dynpos_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_dynpos_f64,
    f64,
    kernel_common,
    rotary_embedding_dynpos_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_batched_fn!(
    rotary_embedding_batched_f32,
    f32,
    kernel_common,
    rotary_embedding_batched_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_batched_fn!(
    rotary_embedding_batched_f16,
    f16,
    kernel_common,
    rotary_embedding_batched_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_batched_fn!(
    rotary_embedding_batched_f64,
    f64,
    kernel_common,
    rotary_embedding_batched_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_fn!(
    rotary_embedding_chunked_f32,
    f32,
    kernel_common,
    rotary_embedding_chunked_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_fn!(
    rotary_embedding_chunked_f16,
    f16,
    kernel_common,
    rotary_embedding_chunked_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_fn!(
    rotary_embedding_chunked_f64,
    f64,
    kernel_common,
    rotary_embedding_chunked_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_positioned_fn!(
    rotary_embedding_chunked_positioned_f32,
    f32,
    kernel_common,
    rotary_embedding_chunked_positioned_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_positioned_fn!(
    rotary_embedding_chunked_positioned_f16,
    f16,
    kernel_common,
    rotary_embedding_chunked_positioned_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_positioned_fn!(
    rotary_embedding_chunked_positioned_f64,
    f64,
    kernel_common,
    rotary_embedding_chunked_positioned_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_chunked_dynpos_f32,
    f32,
    kernel_common,
    rotary_embedding_chunked_dynpos_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_chunked_dynpos_f16,
    f16,
    kernel_common,
    rotary_embedding_chunked_dynpos_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_dynpos_fn!(
    rotary_embedding_chunked_dynpos_f64,
    f64,
    kernel_common,
    rotary_embedding_chunked_dynpos_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_batched_fn!(
    rotary_embedding_chunked_batched_f32,
    f32,
    kernel_common,
    rotary_embedding_chunked_batched_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_batched_fn!(
    rotary_embedding_chunked_batched_f16,
    f16,
    kernel_common,
    rotary_embedding_chunked_batched_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_batched_fn!(
    rotary_embedding_chunked_batched_f64,
    f64,
    kernel_common,
    rotary_embedding_chunked_batched_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_fn!(
    rotary_embedding_qk_f32,
    f32,
    kernel_common,
    rotary_embedding_qk_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_fn!(
    rotary_embedding_qk_f16,
    f16,
    kernel_common,
    rotary_embedding_qk_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_fn!(
    rotary_embedding_qk_f64,
    f64,
    kernel_common,
    rotary_embedding_qk_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f32,
    f32,
    rotary_embedding_batched_f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f16,
    f16,
    rotary_embedding_batched_f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_batched_fn!(
    rotary_embedding_qk_batched_f64,
    f64,
    rotary_embedding_batched_f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_in_place_fn!(
    rotary_embedding_qk_in_place_f32,
    rotary_embedding_qk_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_in_place_fn!(
    rotary_embedding_qk_in_place_f16,
    rotary_embedding_qk_f16,
    f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_in_place_fn!(
    rotary_embedding_qk_in_place_f64,
    rotary_embedding_qk_f64,
    f64
);

#[cfg(feature = "dtype-f32")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f32,
    rotary_embedding_batched_f32,
    f32
);
#[cfg(feature = "dtype-f16")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f16,
    rotary_embedding_batched_f16,
    f16
);
#[cfg(feature = "dtype-f64")]
rotary_embedding_qk_batched_in_place_fn!(
    rotary_embedding_qk_batched_in_place_f64,
    rotary_embedding_batched_f64,
    f64
);

#[cfg(feature = "dtype-f32")]
pub fn rope_bshd_f32(
    stream: &Arc<Stream>,
    out: DevicePointer<f32>,
    input: DevicePointer<f32>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BshdRopeConfig,
) -> Result<()> {
    let launch = BshdRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rope_bshd_f32(
            input,
            out,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]
pub fn rope_bshd_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    input: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BshdRopeConfig,
) -> Result<()> {
    let launch = BshdRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rope_bshd_f16(
            input,
            out,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]
pub fn rope_bshd_dynpos_f16(
    stream: &Arc<Stream>,
    out: DevicePointer<f16>,
    input: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    position_start: DevicePointer<u32>,
    config: BshdRopeConfig,
) -> Result<()> {
    let launch = BshdRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    checked_device_pointer(position_start)?;
    unsafe {
        kernel_positional_structured::rope_bshd_dynpos_f16(
            input,
            out,
            cos,
            sin,
            position_start,
            launch.batch,
            launch.seq_len,
            launch.heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]
pub fn rope_bshd_bf16(
    stream: &Arc<Stream>,
    out: DevicePointer<bf16>,
    input: DevicePointer<bf16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BshdRopeConfig,
) -> Result<()> {
    let launch = BshdRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(out)?;
    checked_device_pointer(input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rope_bshd_bf16(
            input,
            out,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]

pub fn rotary_embedding_qk_bnsd_bf16(
    stream: &Arc<Stream>,
    q_out: DevicePointer<bf16>,
    k_out: DevicePointer<bf16>,
    q_input: DevicePointer<bf16>,
    k_input: DevicePointer<bf16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_len == 0 {
        return Ok(());
    }
    checked_device_pointer(q_out)?;
    checked_device_pointer(k_out)?;
    checked_device_pointer(q_input)?;
    checked_device_pointer(k_input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_bf16(
            q_out,
            k_out,
            q_input,
            k_input,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_len,
            launch.k_len,
            launch.total_len,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]

pub fn rotary_embedding_qk_bnsd_f32(
    stream: &Arc<Stream>,
    q_out: DevicePointer<f32>,
    k_out: DevicePointer<f32>,
    q_input: DevicePointer<f32>,
    k_input: DevicePointer<f32>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_len == 0 {
        return Ok(());
    }
    checked_device_pointer(q_out)?;
    checked_device_pointer(k_out)?;
    checked_device_pointer(q_input)?;
    checked_device_pointer(k_input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_f32(
            q_out,
            k_out,
            q_input,
            k_input,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_len,
            launch.k_len,
            launch.total_len,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]

pub fn rotary_embedding_qk_bnsd_f16(
    stream: &Arc<Stream>,
    q_out: DevicePointer<f16>,
    k_out: DevicePointer<f16>,
    q_input: DevicePointer<f16>,
    k_input: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_len == 0 {
        return Ok(());
    }
    checked_device_pointer(q_out)?;
    checked_device_pointer(k_out)?;
    checked_device_pointer(q_input)?;
    checked_device_pointer(k_input)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_f16(
            q_out,
            k_out,
            q_input,
            k_input,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_len,
            launch.k_len,
            launch.total_len,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn rotary_embedding_qk_bnsd_in_place_f32(
    stream: &Arc<Stream>,
    q: DevicePointer<f32>,
    k: DevicePointer<f32>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(q)?;
    checked_device_pointer(k)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_in_place_f32(
            q,
            k,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_pair_count,
            launch.total_pair_count,
        )
    }
    .grid(launch.pair_grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]
pub fn rotary_embedding_qk_bnsd_in_place_f16(
    stream: &Arc<Stream>,
    q: DevicePointer<f16>,
    k: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(q)?;
    checked_device_pointer(k)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_in_place_f16(
            q,
            k,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_pair_count,
            launch.total_pair_count,
        )
    }
    .grid(launch.pair_grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]
pub fn rotary_embedding_qk_bnsd_in_place_bf16(
    stream: &Arc<Stream>,
    q: DevicePointer<bf16>,
    k: DevicePointer<bf16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: BnsdQkRopeConfig,
) -> Result<()> {
    let launch = BnsdQkRopeLaunch::create(config)?;
    if launch.total_pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(q)?;
    checked_device_pointer(k)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::rotary_embedding_qk_bnsd_in_place_bf16(
            q,
            k,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.trig_batch,
            launch.rotary_pairs,
            launch.q_pair_count,
            launch.total_pair_count,
        )
    }
    .grid(launch.pair_grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn multiaxis_rope_qk_f32(
    stream: &Arc<Stream>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    let launch = MultiaxisRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::multiaxis_rope_qk_f32(
            query,
            key,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.section_t,
            launch.section_h,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f32")]
pub fn interleaved_complex_rope_qk_f32(
    stream: &Arc<Stream>,
    query: DevicePointer<f32>,
    key: DevicePointer<f32>,
    freqs: DevicePointer<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    let launch = InterleavedComplexRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(freqs)?;
    unsafe {
        kernel_positional_structured::interleaved_complex_rope_qk_f32(
            query,
            key,
            freqs,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-f16")]
pub fn interleaved_complex_rope_qk_f16(
    stream: &Arc<Stream>,
    query: DevicePointer<f16>,
    key: DevicePointer<f16>,
    freqs: DevicePointer<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    let launch = InterleavedComplexRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(freqs)?;
    unsafe {
        kernel_positional_structured::interleaved_complex_rope_qk_f16(
            query,
            key,
            freqs,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]
pub fn interleaved_complex_rope_qk_bf16(
    stream: &Arc<Stream>,
    query: DevicePointer<bf16>,
    key: DevicePointer<bf16>,
    freqs: DevicePointer<f32>,
    config: InterleavedComplexRopeConfig,
) -> Result<()> {
    let launch = InterleavedComplexRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(freqs)?;
    unsafe {
        kernel_positional_structured::interleaved_complex_rope_qk_bf16(
            query,
            key,
            freqs,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use cutile::prelude::*;

    use super::*;
    use crate::{
        cpu::positional::{
            self as cpu_positional, bfloat_to_f32, bfloat_vec, half_to_f32, half_vec,
        },
        error::Result,
    };

    #[test]
    fn rotary_embedding_qk_batched_f32_matches_single_tensor_launches() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = QkBatchedRotaryEmbeddingConfig {
            q_output_dimensions: [2, 3, 4, 8],
            k_output_dimensions: [2, 2, 4, 8],
            q_input_strides: [96, 32, 8, 1],
            k_input_strides: [64, 32, 8, 1],
            cos_strides: [16, 4, 1],
            sin_strides: [16, 4, 1],
            trig_batch: 2,
            rotary_pairs: 4,
        };
        let q_len = config.q_output_dimensions.iter().product();
        let k_len = config.k_output_dimensions.iter().product();
        let trig_len = config.trig_batch * config.q_output_dimensions[2] * config.rotary_pairs;
        let q_host = (0..q_len)
            .map(|index| (index as f32 % 19.0) * 0.075 - 0.7)
            .collect::<Vec<_>>();
        let k_host = (0..k_len)
            .map(|index| (index as f32 % 17.0) * 0.0625 - 0.5)
            .collect::<Vec<_>>();
        let cos_host = (0..trig_len)
            .map(|index| (index as f32 * 0.0375).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..trig_len)
            .map(|index| (index as f32 * 0.0375).sin())
            .collect::<Vec<_>>();

        let q_input = api::copy_host_vec_to_device(&Arc::new(q_host)).sync_on(&stream)?;
        let k_input = api::copy_host_vec_to_device(&Arc::new(k_host)).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;
        let q_expected = api::zeros::<f32>(&[q_len]).sync_on(&stream)?;
        let k_expected = api::zeros::<f32>(&[k_len]).sync_on(&stream)?;
        let q_out = api::zeros::<f32>(&[q_len]).sync_on(&stream)?;
        let k_out = api::zeros::<f32>(&[k_len]).sync_on(&stream)?;

        rotary_embedding_batched_f32(
            &stream,
            q_expected.device_pointer(),
            q_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            BatchedRotaryEmbeddingConfig {
                output_dimensions: config.q_output_dimensions,
                input_strides: config.q_input_strides,
                cos_strides: config.cos_strides,
                sin_strides: config.sin_strides,
                trig_batch: config.trig_batch,
                rotary_pairs: config.rotary_pairs,
            },
        )?;
        rotary_embedding_batched_f32(
            &stream,
            k_expected.device_pointer(),
            k_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            BatchedRotaryEmbeddingConfig {
                output_dimensions: config.k_output_dimensions,
                input_strides: config.k_input_strides,
                cos_strides: config.cos_strides,
                sin_strides: config.sin_strides,
                trig_batch: config.trig_batch,
                rotary_pairs: config.rotary_pairs,
            },
        )?;
        rotary_embedding_qk_batched_f32(
            &stream,
            q_out.device_pointer(),
            k_out.device_pointer(),
            q_input.device_pointer(),
            k_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &q_out.to_host_vec().sync_on(&stream)?,
            &q_expected.to_host_vec().sync_on(&stream)?,
            1e-5,
        );
        singe_core::assert_close!(
            &k_out.to_host_vec().sync_on(&stream)?,
            &k_expected.to_host_vec().sync_on(&stream)?,
            1e-5,
        );
        Ok(())
    }

    #[test]
    fn rope_bshd_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 3,
            heads: 2,
            head_dim: 6,
        };
        let input_host = (0..config.batch * config.seq_len * config.heads * config.head_dim)
            .map(|index| (index as f32 % 17.0) * 0.125 - 1.0)
            .collect::<Vec<_>>();
        let cos_host = (0..config.seq_len * (config.head_dim / 2))
            .map(|index| (index as f32 * 0.17).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.seq_len * (config.head_dim / 2))
            .map(|index| (index as f32 * 0.17).sin())
            .collect::<Vec<_>>();
        let mut expected = input_host.clone();
        let head_dim_half = config.head_dim / 2;
        for batch in 0..config.batch {
            for seq in 0..config.seq_len {
                for head in 0..config.heads {
                    for dim in 0..head_dim_half {
                        let trig_index = seq * head_dim_half + dim;
                        let base = ((batch * config.seq_len + seq) * config.heads + head)
                            * config.head_dim
                            + dim;
                        let imag_index = base + head_dim_half;
                        let real = input_host[base];
                        let imag = input_host[imag_index];
                        expected[base] = real * cos_host[trig_index] - imag * sin_host[trig_index];
                        expected[imag_index] =
                            imag * cos_host[trig_index] + real * sin_host[trig_index];
                    }
                }
            }
        }

        let input = api::copy_host_vec_to_device(&Arc::new(input_host)).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;
        let out = api::zeros::<f32>(&[expected.len()]).sync_on(&stream)?;

        rope_bshd_f32(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(&out.to_host_vec().sync_on(&stream)?, &expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn rope_bshd_dynpos_f16_matches_host_position_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BshdRopeConfig {
            batch: 2,
            seq_len: 1,
            heads: 3,
            head_dim: 8,
        };
        let position_start = 2u32;
        let max_positions = 6usize;
        let input_len = config.batch * config.seq_len * config.heads * config.head_dim;
        let trig_width = config.head_dim / 2;
        let input_f32 = (0..input_len)
            .map(|index| (index as f32 % 23.0) * 0.0625 - 0.75)
            .collect::<Vec<_>>();
        let cos_host = (0..max_positions * trig_width)
            .map(|index| (index as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..max_positions * trig_width)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let trig_start = position_start as usize * trig_width;
        let trig_end = trig_start + config.seq_len * trig_width;

        let input =
            api::copy_host_vec_to_device(&Arc::new(half_vec(&input_f32))).sync_on(&stream)?;
        let cos_full =
            api::copy_host_vec_to_device(&Arc::new(cos_host.clone())).sync_on(&stream)?;
        let sin_full =
            api::copy_host_vec_to_device(&Arc::new(sin_host.clone())).sync_on(&stream)?;
        let cos_slice =
            api::copy_host_vec_to_device(&Arc::new(cos_host[trig_start..trig_end].to_vec()))
                .sync_on(&stream)?;
        let sin_slice =
            api::copy_host_vec_to_device(&Arc::new(sin_host[trig_start..trig_end].to_vec()))
                .sync_on(&stream)?;
        let position =
            api::copy_host_vec_to_device(&Arc::new(vec![position_start])).sync_on(&stream)?;
        let expected = api::zeros::<f16>(&[input_len]).sync_on(&stream)?;
        let out = api::zeros::<f16>(&[input_len]).sync_on(&stream)?;

        rope_bshd_f16(
            &stream,
            expected.device_pointer(),
            input.device_pointer(),
            cos_slice.device_pointer(),
            sin_slice.device_pointer(),
            config,
        )?;
        rope_bshd_dynpos_f16(
            &stream,
            out.device_pointer(),
            input.device_pointer(),
            cos_full.device_pointer(),
            sin_full.device_pointer(),
            position.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&out.to_host_vec().sync_on(&stream)?),
            &half_to_f32(&expected.to_host_vec().sync_on(&stream)?),
            1e-3,
        );
        Ok(())
    }

    #[test]
    fn rotary_embedding_qk_bnsd_f32_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 1,
            rotary_pairs: 3,
        };
        let q_host = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 17.0) * 0.1 - 0.8)
            .collect::<Vec<_>>();
        let k_host = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let mut q_expected = q_host.clone();
        let mut k_expected = k_host.clone();
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );

        let q_input = api::copy_host_vec_to_device(&Arc::new(q_host)).sync_on(&stream)?;
        let k_input = api::copy_host_vec_to_device(&Arc::new(k_host)).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;
        let q_out = api::zeros::<f32>(&[q_expected.len()]).sync_on(&stream)?;
        let k_out = api::zeros::<f32>(&[k_expected.len()]).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_f32(
            &stream,
            q_out.device_pointer(),
            k_out.device_pointer(),
            q_input.device_pointer(),
            k_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(&q_out.to_host_vec().sync_on(&stream)?, &q_expected, 1e-5);
        singe_core::assert_close!(&k_out.to_host_vec().sync_on(&stream)?, &k_expected, 1e-5);
        Ok(())
    }

    #[test]
    fn rotary_embedding_qk_bnsd_in_place_f32_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 2,
            rotary_pairs: 3,
        };
        let mut q_host = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 17.0) * 0.1 - 0.8)
            .collect::<Vec<_>>();
        let mut k_host = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let mut q_expected = q_host.clone();
        let mut k_expected = k_host.clone();
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );

        let q = api::copy_host_vec_to_device(&Arc::new(q_host.split_off(0))).sync_on(&stream)?;
        let k = api::copy_host_vec_to_device(&Arc::new(k_host.split_off(0))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_in_place_f32(
            &stream,
            q.device_pointer(),
            k.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(&q.to_host_vec().sync_on(&stream)?, &q_expected, 1e-5);
        singe_core::assert_close!(&k.to_host_vec().sync_on(&stream)?, &k_expected, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn rotary_embedding_qk_bnsd_f16_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 2,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 2,
            rotary_pairs: 3,
        };
        let q_f32 = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let k_f32 = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 7.0) * 0.1 - 0.35)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.11).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.11).sin())
            .collect::<Vec<_>>();
        let mut q_expected = half_to_f32(&half_vec(&q_f32));
        let mut k_expected = half_to_f32(&half_vec(&k_f32));
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );
        q_expected = half_to_f32(&half_vec(&q_expected));
        k_expected = half_to_f32(&half_vec(&k_expected));

        let q_input = api::copy_host_vec_to_device(&Arc::new(half_vec(&q_f32))).sync_on(&stream)?;
        let k_input = api::copy_host_vec_to_device(&Arc::new(half_vec(&k_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;
        let q_out = api::zeros::<f16>(&[q_expected.len()]).sync_on(&stream)?;
        let k_out = api::zeros::<f16>(&[k_expected.len()]).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_f16(
            &stream,
            q_out.device_pointer(),
            k_out.device_pointer(),
            q_input.device_pointer(),
            k_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&q_out.to_host_vec().sync_on(&stream)?),
            &q_expected,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&k_out.to_host_vec().sync_on(&stream)?),
            &k_expected,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn rotary_embedding_qk_bnsd_in_place_f16_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 2,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 2,
            rotary_pairs: 3,
        };
        let q_f32 = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let k_f32 = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 7.0) * 0.1 - 0.35)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.11).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.11).sin())
            .collect::<Vec<_>>();
        let mut q_expected = half_to_f32(&half_vec(&q_f32));
        let mut k_expected = half_to_f32(&half_vec(&k_f32));
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );
        q_expected = half_to_f32(&half_vec(&q_expected));
        k_expected = half_to_f32(&half_vec(&k_expected));

        let q = api::copy_host_vec_to_device(&Arc::new(half_vec(&q_f32))).sync_on(&stream)?;
        let k = api::copy_host_vec_to_device(&Arc::new(half_vec(&k_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_in_place_f16(
            &stream,
            q.device_pointer(),
            k.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&q.to_host_vec().sync_on(&stream)?),
            &q_expected,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&k.to_host_vec().sync_on(&stream)?),
            &k_expected,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn rotary_embedding_qk_bnsd_bf16_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 2,
            rotary_pairs: 3,
        };
        let q_f32 = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 17.0) * 0.1 - 0.8)
            .collect::<Vec<_>>();
        let k_f32 = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let mut q_expected = bfloat_to_f32(&bfloat_vec(&q_f32));
        let mut k_expected = bfloat_to_f32(&bfloat_vec(&k_f32));
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );
        q_expected = bfloat_to_f32(&bfloat_vec(&q_expected));
        k_expected = bfloat_to_f32(&bfloat_vec(&k_expected));

        let q_input =
            api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&q_f32))).sync_on(&stream)?;
        let k_input =
            api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&k_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;
        let q_out = api::zeros::<bf16>(&[q_expected.len()]).sync_on(&stream)?;
        let k_out = api::zeros::<bf16>(&[k_expected.len()]).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_bf16(
            &stream,
            q_out.device_pointer(),
            k_out.device_pointer(),
            q_input.device_pointer(),
            k_input.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&q_out.to_host_vec().sync_on(&stream)?),
            &q_expected,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&k_out.to_host_vec().sync_on(&stream)?),
            &k_expected,
            8e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn rotary_embedding_qk_bnsd_in_place_bf16_partial_rope() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = BnsdQkRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            trig_batch: 2,
            rotary_pairs: 3,
        };
        let q_f32 = (0..config.batch * config.query_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 17.0) * 0.1 - 0.8)
            .collect::<Vec<_>>();
        let k_f32 = (0..config.batch * config.key_heads * config.seq_len * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let cos_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).cos())
            .collect::<Vec<_>>();
        let sin_host = (0..config.trig_batch * config.seq_len * config.rotary_pairs)
            .map(|index| (index as f32 * 0.13).sin())
            .collect::<Vec<_>>();
        let mut q_expected = bfloat_to_f32(&bfloat_vec(&q_f32));
        let mut k_expected = bfloat_to_f32(&bfloat_vec(&k_f32));
        cpu_positional::rotate_bnsd_qk_rope_reference(
            &mut q_expected,
            &mut k_expected,
            &cos_host,
            &sin_host,
            config,
        );
        q_expected = bfloat_to_f32(&bfloat_vec(&q_expected));
        k_expected = bfloat_to_f32(&bfloat_vec(&k_expected));

        let q = api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&q_f32))).sync_on(&stream)?;
        let k = api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&k_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        rotary_embedding_qk_bnsd_in_place_bf16(
            &stream,
            q.device_pointer(),
            k.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&q.to_host_vec().sync_on(&stream)?),
            &q_expected,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&k.to_host_vec().sync_on(&stream)?),
            &k_expected,
            8e-3,
        );
        Ok(())
    }

    #[test]
    fn interleaved_complex_rope_qk_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = InterleavedComplexRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
        };
        let mut query_host =
            (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
                .map(|index| (index as f32 % 17.0) * 0.125 - 1.0)
                .collect::<Vec<_>>();
        let mut key_host = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.1 - 0.6)
            .collect::<Vec<_>>();
        let freqs_host = (0..config.seq_len * config.head_dim)
            .map(|index| {
                let angle = (index / 2) as f32 * 0.05 + (index % 2) as f32 * 0.2;
                if index % 2 == 0 {
                    angle.cos()
                } else {
                    angle.sin()
                }
            })
            .collect::<Vec<_>>();
        let mut expected_query = query_host.clone();
        let mut expected_key = key_host.clone();
        cpu_positional::interleaved_complex_rope_qk(
            &mut expected_query,
            &mut expected_key,
            &freqs_host,
            config,
        );

        let query =
            api::copy_host_vec_to_device(&Arc::new(query_host.split_off(0))).sync_on(&stream)?;
        let key =
            api::copy_host_vec_to_device(&Arc::new(key_host.split_off(0))).sync_on(&stream)?;
        let freqs = api::copy_host_vec_to_device(&Arc::new(freqs_host)).sync_on(&stream)?;

        interleaved_complex_rope_qk_f32(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            freqs.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &query.to_host_vec().sync_on(&stream)?,
            &expected_query,
            1e-5,
        );
        singe_core::assert_close!(&key.to_host_vec().sync_on(&stream)?, &expected_key, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn interleaved_complex_rope_qk_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = InterleavedComplexRopeConfig {
            batch: 1,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 6,
        };
        let query_f32 = (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
            .map(|index| (index as f32 % 11.0) * 0.125 - 0.5)
            .collect::<Vec<_>>();
        let key_f32 = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 7.0) * 0.1 - 0.35)
            .collect::<Vec<_>>();
        let freqs_host = interleaved_complex_freqs(config);
        let mut expected_query = half_to_f32(&half_vec(&query_f32));
        let mut expected_key = half_to_f32(&half_vec(&key_f32));
        cpu_positional::interleaved_complex_rope_qk(
            &mut expected_query,
            &mut expected_key,
            &freqs_host,
            config,
        );
        expected_query = half_to_f32(&half_vec(&expected_query));
        expected_key = half_to_f32(&half_vec(&expected_key));

        let query =
            api::copy_host_vec_to_device(&Arc::new(half_vec(&query_f32))).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(half_vec(&key_f32))).sync_on(&stream)?;
        let freqs = api::copy_host_vec_to_device(&Arc::new(freqs_host)).sync_on(&stream)?;

        interleaved_complex_rope_qk_f16(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            freqs.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&query.to_host_vec().sync_on(&stream)?),
            &expected_query,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&key.to_host_vec().sync_on(&stream)?),
            &expected_key,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn interleaved_complex_rope_qk_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = InterleavedComplexRopeConfig {
            batch: 1,
            seq_len: 2,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
        };
        let query_f32 = (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.1 - 0.55)
            .collect::<Vec<_>>();
        let key_f32 = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 9.0) * 0.125 - 0.4)
            .collect::<Vec<_>>();
        let freqs_host = interleaved_complex_freqs(config);
        let mut expected_query = bfloat_to_f32(&bfloat_vec(&query_f32));
        let mut expected_key = bfloat_to_f32(&bfloat_vec(&key_f32));
        cpu_positional::interleaved_complex_rope_qk(
            &mut expected_query,
            &mut expected_key,
            &freqs_host,
            config,
        );
        expected_query = bfloat_to_f32(&bfloat_vec(&expected_query));
        expected_key = bfloat_to_f32(&bfloat_vec(&expected_key));

        let query =
            api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&query_f32))).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&key_f32))).sync_on(&stream)?;
        let freqs = api::copy_host_vec_to_device(&Arc::new(freqs_host)).sync_on(&stream)?;

        interleaved_complex_rope_qk_bf16(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            freqs.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&query.to_host_vec().sync_on(&stream)?),
            &expected_query,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&key.to_host_vec().sync_on(&stream)?),
            &expected_key,
            8e-3,
        );
        Ok(())
    }

    #[test]
    fn multiaxis_rope_qk_f32() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = MultiaxisRopeConfig {
            batch: 2,
            seq_len: 3,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            section_t: 1,
            section_h: 2,
            section_w: 1,
        };
        let mut query_host =
            (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
                .map(|index| (index as f32 % 17.0) * 0.125 - 1.0)
                .collect::<Vec<_>>();
        let mut key_host = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.1 - 0.6)
            .collect::<Vec<_>>();
        let (cos_host, sin_host) = trig_tables(config);
        let mut expected_query = query_host.clone();
        let mut expected_key = key_host.clone();
        cpu_positional::multiaxis_rope_qk(
            &mut expected_query,
            &mut expected_key,
            &cos_host,
            &sin_host,
            config,
        );

        let query =
            api::copy_host_vec_to_device(&Arc::new(query_host.split_off(0))).sync_on(&stream)?;
        let key =
            api::copy_host_vec_to_device(&Arc::new(key_host.split_off(0))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        multiaxis_rope_qk_f32(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &query.to_host_vec().sync_on(&stream)?,
            &expected_query,
            1e-5,
        );
        singe_core::assert_close!(&key.to_host_vec().sync_on(&stream)?, &expected_key, 1e-5);
        Ok(())
    }

    #[cfg(feature = "dtype-f16")]
    #[test]
    fn multiaxis_rope_qk_f16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = MultiaxisRopeConfig {
            batch: 1,
            seq_len: 2,
            query_heads: 3,
            key_heads: 2,
            head_dim: 6,
            section_t: 1,
            section_h: 1,
            section_w: 1,
        };
        let query_f32 = (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
            .map(|index| (index as f32 % 11.0) * 0.1 - 0.5)
            .collect::<Vec<_>>();
        let key_f32 = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 7.0) * 0.125 - 0.375)
            .collect::<Vec<_>>();
        let mut query_expected = half_to_f32(&half_vec(&query_f32));
        let mut key_expected = half_to_f32(&half_vec(&key_f32));
        let (cos_host, sin_host) = trig_tables(config);
        cpu_positional::multiaxis_rope_qk(
            &mut query_expected,
            &mut key_expected,
            &cos_host,
            &sin_host,
            config,
        );
        query_expected = half_to_f32(&half_vec(&query_expected));
        key_expected = half_to_f32(&half_vec(&key_expected));

        let query =
            api::copy_host_vec_to_device(&Arc::new(half_vec(&query_f32))).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(half_vec(&key_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        multiaxis_rope_qk_f16(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &half_to_f32(&query.to_host_vec().sync_on(&stream)?),
            &query_expected,
            2e-3,
        );
        singe_core::assert_close!(
            &half_to_f32(&key.to_host_vec().sync_on(&stream)?),
            &key_expected,
            2e-3,
        );
        Ok(())
    }

    #[cfg(feature = "dtype-bf16")]
    #[test]
    fn multiaxis_rope_qk_bf16() -> Result<()> {
        let Ok(device) = Device::new(0) else {
            return Ok(());
        };
        let stream = device.new_stream()?;
        let config = MultiaxisRopeConfig {
            batch: 1,
            seq_len: 2,
            query_heads: 2,
            key_heads: 1,
            head_dim: 8,
            section_t: 2,
            section_h: 1,
            section_w: 1,
        };
        let query_f32 = (0..config.batch * config.seq_len * config.query_heads * config.head_dim)
            .map(|index| (index as f32 % 13.0) * 0.125 - 0.625)
            .collect::<Vec<_>>();
        let key_f32 = (0..config.batch * config.seq_len * config.key_heads * config.head_dim)
            .map(|index| (index as f32 % 9.0) * 0.1 - 0.4)
            .collect::<Vec<_>>();
        let mut query_expected = bfloat_to_f32(&bfloat_vec(&query_f32));
        let mut key_expected = bfloat_to_f32(&bfloat_vec(&key_f32));
        let (cos_host, sin_host) = trig_tables(config);
        cpu_positional::multiaxis_rope_qk(
            &mut query_expected,
            &mut key_expected,
            &cos_host,
            &sin_host,
            config,
        );
        query_expected = bfloat_to_f32(&bfloat_vec(&query_expected));
        key_expected = bfloat_to_f32(&bfloat_vec(&key_expected));

        let query =
            api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&query_f32))).sync_on(&stream)?;
        let key = api::copy_host_vec_to_device(&Arc::new(bfloat_vec(&key_f32))).sync_on(&stream)?;
        let cos = api::copy_host_vec_to_device(&Arc::new(cos_host)).sync_on(&stream)?;
        let sin = api::copy_host_vec_to_device(&Arc::new(sin_host)).sync_on(&stream)?;

        multiaxis_rope_qk_bf16(
            &stream,
            query.device_pointer(),
            key.device_pointer(),
            cos.device_pointer(),
            sin.device_pointer(),
            config,
        )?;

        singe_core::assert_close!(
            &bfloat_to_f32(&query.to_host_vec().sync_on(&stream)?),
            &query_expected,
            8e-3,
        );
        singe_core::assert_close!(
            &bfloat_to_f32(&key.to_host_vec().sync_on(&stream)?),
            &key_expected,
            8e-3,
        );
        Ok(())
    }

    fn trig_tables(config: MultiaxisRopeConfig) -> (Vec<f32>, Vec<f32>) {
        let head_dim_half = config.head_dim / 2;
        let len = 3 * config.batch * config.seq_len * head_dim_half;
        let cos = (0..len)
            .map(|index| ((index as f32 % 19.0) * 0.03).cos())
            .collect::<Vec<_>>();
        let sin = (0..len)
            .map(|index| ((index as f32 % 19.0) * 0.03).sin())
            .collect::<Vec<_>>();
        (cos, sin)
    }

    #[allow(dead_code)]
    fn interleaved_complex_freqs(config: InterleavedComplexRopeConfig) -> Vec<f32> {
        (0..config.seq_len * config.head_dim)
            .map(|index| {
                let angle = (index / 2) as f32 * 0.05 + (index % 2) as f32 * 0.2;
                if index % 2 == 0 {
                    angle.cos()
                } else {
                    angle.sin()
                }
            })
            .collect()
    }
}

#[cfg(feature = "dtype-f16")]
pub fn multiaxis_rope_qk_f16(
    stream: &Arc<Stream>,
    query: DevicePointer<f16>,
    key: DevicePointer<f16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    let launch = MultiaxisRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::multiaxis_rope_qk_f16(
            query,
            key,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.section_t,
            launch.section_h,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}

#[cfg(feature = "dtype-bf16")]
pub fn multiaxis_rope_qk_bf16(
    stream: &Arc<Stream>,
    query: DevicePointer<bf16>,
    key: DevicePointer<bf16>,
    cos: DevicePointer<f32>,
    sin: DevicePointer<f32>,
    config: MultiaxisRopeConfig,
) -> Result<()> {
    let launch = MultiaxisRopeLaunch::create(config)?;
    if launch.pair_count == 0 {
        return Ok(());
    }
    checked_device_pointer(query)?;
    checked_device_pointer(key)?;
    checked_device_pointer(cos)?;
    checked_device_pointer(sin)?;
    unsafe {
        kernel_positional_structured::multiaxis_rope_qk_bf16(
            query,
            key,
            cos,
            sin,
            launch.batch,
            launch.seq_len,
            launch.query_heads,
            launch.key_heads,
            launch.head_dim,
            launch.section_t,
            launch.section_h,
            launch.pair_count,
        )
    }
    .grid(launch.grid)
    .enqueue_on(stream)?;
    Ok(())
}
