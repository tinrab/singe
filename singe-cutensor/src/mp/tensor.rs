use std::{mem::ManuallyDrop, ptr};

use singe_cuda::data_type::{DataType, DataTypeLike};

use crate::{
    error::{Error, Result},
    mp::context::{Context, ContextRef},
    sys, try_ffi,
    utility::{to_i64_vec, to_u32},
};

#[derive(Debug, Clone, Copy)]
pub struct TensorMode {
    pub shape: u64,
    pub element_stride: Option<u64>,
    pub block_size: Option<u64>,
    pub block_stride: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub struct RankPartition<'a> {
    pub ranks_per_mode: &'a [u64],
    pub ranks: Option<&'a [i32]>,
    pub rank_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptorConfig<'a> {
    pub modes: &'a [TensorMode],
    pub partition: RankPartition<'a>,
    pub data_type: DataType,
}

#[derive(Debug)]
pub struct TensorDescriptor<'comm> {
    handle: sys::cutensorMpTensorDescriptor_t,
    context: ContextRef<'comm>,
    shape: Vec<u64>,
    element_stride: Option<Vec<u64>>,
    block_size: Option<Vec<u64>>,
    block_stride: Option<Vec<u64>>,
    ranks_per_mode: Vec<u64>,
    ranks: Option<Vec<i32>>,
    rank_count: u32,
    data_type: DataType,
}

impl<'comm> TensorDescriptor<'comm> {
    pub fn create_for<T: DataTypeLike>(
        context: &Context<'comm>,
        shape: &[u64],
        ranks_per_mode: &[u64],
        rank_count: u32,
    ) -> Result<Self> {
        let modes = shape
            .iter()
            .copied()
            .map(TensorMode::contiguous)
            .collect::<Vec<_>>();
        Self::create(
            context,
            TensorDescriptorConfig {
                modes: &modes,
                partition: RankPartition {
                    ranks_per_mode,
                    ranks: None,
                    rank_count,
                },
                data_type: T::data_type(),
            },
        )
    }

    pub fn create(context: &Context<'comm>, config: TensorDescriptorConfig<'_>) -> Result<Self> {
        let rank = to_u32(config.modes.len(), "modes")?;
        validate_rank(
            config.partition.ranks_per_mode,
            config.modes.len(),
            "ranks_per_mode",
        )?;
        if let Some(ranks) = config.partition.ranks
            && ranks.len() != config.partition.rank_count as usize
        {
            return Err(Error::LengthMismatch {
                name: "ranks".into(),
                expected: config.partition.rank_count as usize,
                actual: ranks.len(),
            });
        }

        let shape = config
            .modes
            .iter()
            .map(|mode| mode.shape)
            .collect::<Vec<_>>();
        let element_stride =
            collect_optional_mode_values(config.modes, "element_stride", |mode| {
                mode.element_stride
            })?;
        let block_size =
            collect_optional_mode_values(config.modes, "block_size", |mode| mode.block_size)?;
        let block_stride =
            collect_optional_mode_values(config.modes, "block_stride", |mode| mode.block_stride)?;

        let extent_i64 = to_i64_vec(&shape, "shape")?;
        let element_stride_i64 = element_stride
            .as_ref()
            .map(|values| to_i64_vec(values, "element_stride"))
            .transpose()?;
        let block_size_i64 = block_size
            .as_ref()
            .map(|values| to_i64_vec(values, "block_size"))
            .transpose()?;
        let block_stride_i64 = block_stride
            .as_ref()
            .map(|values| to_i64_vec(values, "block_stride"))
            .transpose()?;
        let ranks_per_mode_i64 = to_i64_vec(config.partition.ranks_per_mode, "ranks_per_mode")?;

        let mut handle = ptr::null_mut();
        context.bind()?;
        unsafe {
            try_ffi!(sys::cutensorMpCreateTensorDescriptor(
                context.as_raw(),
                &raw mut handle,
                rank,
                extent_i64.as_ptr(),
                element_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_size_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                ranks_per_mode_i64.as_ptr(),
                config.partition.rank_count,
                config.partition.ranks.map_or(ptr::null(), <[i32]>::as_ptr),
                config.data_type.into(),
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            shape,
            element_stride,
            block_size,
            block_stride,
            ranks_per_mode: config.partition.ranks_per_mode.to_vec(),
            ranks: config.partition.ranks.map(<[i32]>::to_vec),
            rank_count: config.partition.rank_count,
            data_type: config.data_type,
        })
    }

    /// Wraps an existing cuTENSORMp tensor descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMp tensor descriptor associated with
    /// `context`. The metadata in `config` must describe the raw descriptor.
    /// The returned value takes ownership of `handle` and destroys it on drop.
    pub unsafe fn from_raw(
        handle: sys::cutensorMpTensorDescriptor_t,
        context: &Context<'comm>,
        config: TensorDescriptorConfig<'_>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        validate_rank(
            config.partition.ranks_per_mode,
            config.modes.len(),
            "ranks_per_mode",
        )?;
        if let Some(ranks) = config.partition.ranks
            && ranks.len() != config.partition.rank_count as usize
        {
            return Err(Error::LengthMismatch {
                name: "ranks".into(),
                expected: config.partition.rank_count as usize,
                actual: ranks.len(),
            });
        }

        let shape = config
            .modes
            .iter()
            .map(|mode| mode.shape)
            .collect::<Vec<_>>();
        let element_stride =
            collect_optional_mode_values(config.modes, "element_stride", |mode| {
                mode.element_stride
            })?;
        let block_size =
            collect_optional_mode_values(config.modes, "block_size", |mode| mode.block_size)?;
        let block_stride =
            collect_optional_mode_values(config.modes, "block_stride", |mode| mode.block_stride)?;

        Ok(Self {
            handle,
            context: context.as_context_ref(),
            shape,
            element_stride,
            block_size,
            block_stride,
            ranks_per_mode: config.partition.ranks_per_mode.to_vec(),
            ranks: config.partition.ranks.map(<[i32]>::to_vec),
            rank_count: config.partition.rank_count,
            data_type: config.data_type,
        })
    }

    pub fn rank(&self) -> u32 {
        self.shape.len() as u32
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn element_stride(&self) -> Option<&[u64]> {
        self.element_stride.as_deref()
    }

    pub fn block_size(&self) -> Option<&[u64]> {
        self.block_size.as_deref()
    }

    pub fn block_stride(&self) -> Option<&[u64]> {
        self.block_stride.as_deref()
    }

    pub fn ranks_per_mode(&self) -> &[u64] {
        &self.ranks_per_mode
    }

    pub fn ranks(&self) -> Option<&[i32]> {
        self.ranks.as_deref()
    }

    pub fn rank_count(&self) -> u32 {
        self.rank_count
    }

    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    pub const fn as_raw(&self) -> sys::cutensorMpTensorDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMp tensor descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    pub fn into_raw(self) -> sys::cutensorMpTensorDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for TensorDescriptor<'_> {
    fn drop(&mut self) {
        if let Err(err) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensormp context before destroying tensor descriptor: {err}"
            );
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cutensorMpDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp tensor descriptor: {err}");
            }
        }
    }
}

fn validate_rank<T>(values: &[T], rank: usize, name: &str) -> Result<()> {
    if values.len() != rank {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected: rank,
            actual: values.len(),
        });
    }
    Ok(())
}

impl TensorMode {
    pub const fn contiguous(shape: u64) -> Self {
        Self {
            shape,
            element_stride: None,
            block_size: None,
            block_stride: None,
        }
    }
}

fn collect_optional_mode_values(
    modes: &[TensorMode],
    name: &str,
    value: impl Fn(&TensorMode) -> Option<u64>,
) -> Result<Option<Vec<u64>>> {
    let values = modes.iter().filter_map(value).collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(None);
    }
    if values.len() != modes.len() {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected: modes.len(),
            actual: values.len(),
        });
    }
    Ok(Some(values))
}
