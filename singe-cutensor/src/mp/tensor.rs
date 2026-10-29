use std::{mem::ManuallyDrop, ptr};

use singe_cuda::data_type::{DataType, DataTypeLike};

use crate::{
    error::{Error, Result},
    mp::context::{Context, ContextRef},
    sys, try_ffi,
    utility::{to_i64_vec, to_u32},
};

/// Global extent and optional rank-local element and block strides for one tensor mode.
///
/// Each optional property must be set for every mode or left unset for all modes.
#[derive(Debug, Clone, Copy)]
pub struct TensorMode {
    /// Global extent of the tensor mode.
    pub shape: u64,
    /// Rank-local stride in elements between adjacent values; `None` selects compact local strides.
    pub element_stride: Option<u64>,
    /// Block-cyclic extent for the mode. `None`, or an explicit zero, selects one contiguous partition per rank along this mode.
    pub block_size: Option<u64>,
    /// Rank-local displacement in elements between adjacent stored blocks; `None` selects ordinary linear addressing without inter-block padding. Explicit entries must be positive and produce a non-overlapping layout supported by the operation.
    pub block_stride: Option<u64>,
}

/// Process-grid dimensions and an optional explicit mapping to communicator ranks.
#[derive(Debug, Clone, Copy)]
pub struct RankPartition<'a> {
    /// Number of processes distributing each tensor mode; one entry is required per mode.
    pub ranks_per_mode: &'a [u64],
    /// Explicit participating communicator rank identifiers, or `None` for `0..rank_count`.
    pub ranks: Option<&'a [i32]>,
    /// Total number of ranks participating in this tensor distribution.
    pub rank_count: u32,
}

/// Borrowed global layout and rank-distribution settings copied into a tensor descriptor.
#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptorConfig<'a> {
    /// Ordered global extents and optional local storage geometry.
    pub modes: &'a [TensorMode],
    /// Process-grid dimensions and communicator rank mapping.
    pub partition: RankPartition<'a>,
    /// Native representation of tensor elements.
    pub data_type: DataType,
}

/// Owns global tensor distribution metadata and retains the multi-process handle and communicator lifetime.
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
    /// Creates a descriptor for `T` with compact local storage and an implicit communicator rank mapping.
    ///
    /// The supplied `ranks_per_mode` determines distribution. Native defaults select contiguous partitions along distributed modes; `rank_count` records the total participating ranks.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode count cannot fit `u32`, process-grid dimensions or explicit rank counts have incorrect lengths, an optional property is specified for only some modes, a geometry value cannot fit native `i64`, CUDA context binding fails, native descriptor creation fails, or the returned handle is null.
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

    /// Creates a distributed tensor descriptor from global mode geometry and a rank partition.
    ///
    /// # Errors
    ///
    /// Returns an error if the mode count cannot fit `u32`, process-grid dimensions or explicit rank counts have incorrect lengths, an optional property is specified for only some modes, a geometry value cannot fit native `i64`, CUDA context binding fails, native descriptor creation fails, or the returned handle is null.
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
        // SAFETY: `cutensorMpCreateTensorDescriptor` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null, process-grid or explicit rank-array lengths are inconsistent, or an optional property is specified for only some modes. The caller must uphold the remaining native descriptor invariants.
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

    /// Returns the number of tensor modes.
    ///
    /// # Panics
    ///
    /// Panics if the stored rank does not fit `u32`. Checked creation excludes this; unsafe adoption must preserve the native rank invariant.
    #[must_use]
    pub fn rank(&self) -> u32 {
        u32::try_from(self.shape.len()).expect("tensor rank is validated on construction")
    }

    /// Returns global tensor extents in descriptor order.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns explicit local element strides, or `None` for native compact strides.
    #[must_use]
    pub fn element_stride(&self) -> Option<&[u64]> {
        self.element_stride.as_deref()
    }

    /// Returns optional per-mode block extents; zero entries select contiguous partition defaults.
    #[must_use]
    pub fn block_size(&self) -> Option<&[u64]> {
        self.block_size.as_deref()
    }

    /// Returns explicit local inter-block element strides, or `None` for linear local addressing.
    #[must_use]
    pub fn block_stride(&self) -> Option<&[u64]> {
        self.block_stride.as_deref()
    }

    /// Returns the process-grid size along each tensor mode.
    #[must_use]
    pub fn ranks_per_mode(&self) -> &[u64] {
        &self.ranks_per_mode
    }

    /// Returns explicit communicator rank identifiers, or `None` for the implicit rank range.
    #[must_use]
    pub fn ranks(&self) -> Option<&[i32]> {
        self.ranks.as_deref()
    }

    /// Returns the total participating rank count.
    #[must_use]
    pub fn rank_count(&self) -> u32 {
        self.rank_count
    }

    /// Returns the native sample data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn context(&self) -> &ContextRef<'comm> {
        &self.context
    }

    /// Returns the borrowed native distributed tensor descriptor.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMpTensorDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMp tensor descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMpTensorDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for TensorDescriptor<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.context.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cutensormp context before destroying tensor descriptor: {error}"
            );
        }

        // SAFETY: `cutensorMpDestroyTensorDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMpDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormp tensor descriptor: {error}");
            }
        }
    }
}

fn validate_rank<T>(values: &[T], rank: usize, name: &str) -> Result<()> {
    if values.len() != rank {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected: rank,
            actual: values.len(),
        });
    }
    Ok(())
}

impl TensorMode {
    /// Creates mode geometry using native compact local storage and contiguous rank partitions.
    #[must_use]
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
            name: name.to_owned().into(),
            expected: modes.len(),
            actual: values.len(),
        });
    }
    Ok(Some(values))
}

/// Realized layout of this rank's shard. Strides and block sizes are in elements.
///
/// Storage size is the minimum byte span covering all reachable local offsets,
/// including holes introduced by explicit strides. Nonparticipating ranks have
/// zero extents and storage size; this query performs no communication.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalTensorLayout {
    /// Number of valid local logical elements per mode, without padding ragged tails.
    pub extents: Vec<i64>,
    /// Realized local element strides, in descriptor mode order.
    pub element_strides: Vec<i64>,
    /// Realized local block sizes.
    pub block_sizes: Vec<i64>,
    /// Realized local block strides.
    pub block_strides: Vec<i64>,
    /// Minimum local storage span in bytes.
    pub storage_bytes: u64,
}

impl TensorDescriptor<'_> {
    /// Queries the calling rank's realized local tensor layout.
    ///
    /// # Errors
    ///
    /// Returns an error if CUDA context binding or a native layout-attribute query fails, or the byte size of a returned layout array overflows `usize`.
    pub fn local_layout(&self) -> Result<LocalTensorLayout> {
        use sys::cutensorMpTensorDescriptorAttribute_t::{
            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_BLOCK_SIZE,
            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_BLOCK_STRIDE,
            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_ELEMENT_STRIDE,
            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_EXTENT,
            CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_STORAGE_SIZE,
            CUTENSORMP_TENSOR_DESCRIPTOR_NUM_MODES,
        };
        self.context.bind()?;
        let mut modes = 0u32;
        // SAFETY: `cutensorMpTensorDescriptorGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMpTensorDescriptorGetAttribute(
                self.context.as_raw(),
                self.handle,
                CUTENSORMP_TENSOR_DESCRIPTOR_NUM_MODES,
                (&raw mut modes).cast(),
                size_of::<u32>() as _
            ))?;
        }
        let array = |attribute| -> Result<Vec<i64>> {
            let mut values = vec![0i64; modes as usize];
            let bytes = values
                .len()
                .checked_mul(size_of::<i64>())
                .ok_or_else(|| Error::out_of_range("local layout"))?;
            // SAFETY: `cutensorMpTensorDescriptorGetAttribute` receives live handles owned by the wrappers and valid output locations.
            unsafe {
                try_ffi!(sys::cutensorMpTensorDescriptorGetAttribute(
                    self.context.as_raw(),
                    self.handle,
                    attribute,
                    values.as_mut_ptr().cast(),
                    bytes as _
                ))?;
            }
            Ok(values)
        };
        let mut storage_bytes = 0u64;
        // SAFETY: `cutensorMpTensorDescriptorGetAttribute` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi!(sys::cutensorMpTensorDescriptorGetAttribute(
                self.context.as_raw(),
                self.handle,
                CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_STORAGE_SIZE,
                (&raw mut storage_bytes).cast(),
                size_of::<u64>() as _
            ))?;
        }
        Ok(LocalTensorLayout {
            extents: array(CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_EXTENT)?,
            element_strides: array(CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_ELEMENT_STRIDE)?,
            block_sizes: array(CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_BLOCK_SIZE)?,
            block_strides: array(CUTENSORMP_TENSOR_DESCRIPTOR_LOCAL_BLOCK_STRIDE)?,
            storage_bytes,
        })
    }
}
