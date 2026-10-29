use std::{mem::ManuallyDrop, ptr};

use singe_cuda::data_type::{DataType, DataTypeLike};

use crate::{
    error::{Error, Result},
    mg::context::{Context, ContextRef},
    sys, try_ffi,
    utility::{to_i64_vec, to_u32},
};

/// A GPU or host-memory location used by a multi-GPU tensor descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorDevice {
    /// CUDA device ordinal for a tensor partition.
    Device(i32),
    /// Ordinary host memory.
    Host,
    /// Pinned host memory.
    HostPinned,
}

impl TensorDevice {
    /// Returns the CUDA ordinal or native host-memory sentinel.
    #[must_use]
    pub const fn as_raw(self) -> i32 {
        match self {
            Self::Device(device) => device,
            Self::Host => sys::cutensorMgHostDevice_t::CUTENSOR_MG_DEVICE_HOST as i32,
            Self::HostPinned => sys::cutensorMgHostDevice_t::CUTENSOR_MG_DEVICE_HOST_PINNED as i32,
        }
    }
}

impl From<i32> for TensorDevice {
    fn from(value: i32) -> Self {
        Self::Device(value)
    }
}

/// Extent, element stride, and block geometry for one distributed tensor mode.
///
/// For each optional property, supply values for all modes or leave every mode unset.
#[derive(Debug, Clone, Copy)]
pub struct TensorMode {
    /// Global extent of this tensor mode.
    pub shape: u64,
    /// Displacement in elements between adjacent values within a block; `None` requests dense layout.
    pub element_stride: Option<u64>,
    /// Block extent in elements; `None` selects one block spanning the mode.
    pub block_size: Option<u64>,
    /// Displacement in elements between adjacent blocks; `None` requests dense block-interleaved layout.
    pub block_stride: Option<u64>,
}

/// Ordered storage locations and optional block-cyclic distribution across tensor modes.
#[derive(Debug, Clone, Copy)]
pub struct DevicePartition<'a> {
    /// Storage locations in column-major distribution order, with the stride-one device dimension first.
    ///
    /// The count includes replicated partitions as well as partitions distributed across modes.
    pub devices: &'a [TensorDevice],
    /// Number of devices distributing each mode's blocks cyclically, or `None` for a non-distributed tensor.
    pub count_per_mode: Option<&'a [i32]>,
}

/// Borrowed layout and storage-distribution settings copied into a native tensor descriptor.
#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptorConfig<'a> {
    /// Ordered tensor-mode extents, strides, and blocks.
    pub modes: &'a [TensorMode],
    /// Storage locations and per-mode device distribution.
    pub partition: DevicePartition<'a>,
    /// Native scalar representation of tensor elements.
    pub data_type: DataType,
}

/// Owns a distributed tensor layout descriptor and retains its multi-GPU context.
#[derive(Debug)]
pub struct TensorDescriptor {
    handle: sys::cutensorMgTensorDescriptor_t,
    context: ContextRef,
    shape: Vec<u64>,
    element_stride: Option<Vec<u64>>,
    block_size: Option<Vec<u64>>,
    block_stride: Option<Vec<u64>>,
    device_count: Option<Vec<i32>>,
    devices: Vec<TensorDevice>,
    data_type: DataType,
}

impl TensorDescriptor {
    /// Creates a dense, unblocked tensor descriptor using the native type of `T`.
    ///
    /// The supplied storage locations are retained in descriptor order. Per-mode device counts are left unspecified, selecting native non-distributed layout.
    ///
    /// # Errors
    ///
    /// Returns an error if rank or device count cannot fit the native count, an optional property is specified for only some modes, the per-mode device counts have the wrong length, an extent or stride cannot fit `i64`, native descriptor creation fails, or the returned handle is null.
    pub fn create_for<T: DataTypeLike>(
        context: &Context,
        shape: &[u64],
        devices: &[TensorDevice],
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
                partition: DevicePartition {
                    devices,
                    count_per_mode: None,
                },
                data_type: T::data_type(),
            },
        )
    }

    /// Creates a tensor descriptor from mode geometry and a device partition.
    ///
    /// # Errors
    ///
    /// Returns an error if rank or device count cannot fit the native count, an optional property is specified for only some modes, the per-mode device counts have the wrong length, an extent or stride cannot fit `i64`, native descriptor creation fails, or the returned handle is null.
    pub fn create(context: &Context, config: TensorDescriptorConfig<'_>) -> Result<Self> {
        let rank = to_u32(config.modes.len(), "modes")?;
        validate_optional_rank_i32(
            config.partition.count_per_mode,
            config.modes.len(),
            "device_count",
        )?;

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
        let devices_raw: Vec<_> = config
            .partition
            .devices
            .iter()
            .map(|device| device.as_raw())
            .collect();
        let num_devices = to_u32(devices_raw.len(), "devices")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `cutensorMgCreateTensorDescriptor` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi!(sys::cutensorMgCreateTensorDescriptor(
                context.as_raw(),
                &raw mut handle,
                rank,
                extent_i64.as_ptr(),
                element_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_size_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                config
                    .partition
                    .count_per_mode
                    .map_or(ptr::null(), <[i32]>::as_ptr),
                num_devices,
                devices_raw.as_ptr(),
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
            device_count: config.partition.count_per_mode.map(<[i32]>::to_vec),
            devices: config.partition.devices.to_vec(),
            data_type: config.data_type,
        })
    }

    /// Wraps an existing cuTENSORMg tensor descriptor.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid cuTENSORMg tensor descriptor associated with
    /// `context`. The metadata in `config` must describe the raw descriptor.
    /// The returned value takes ownership of `handle` and destroys it on drop.
    ///
    /// # Errors
    ///
    /// Returns an error if `handle` is null or an optional stride or block property is specified for only some modes. The caller must establish all remaining raw descriptor invariants.
    pub unsafe fn from_raw(
        handle: sys::cutensorMgTensorDescriptor_t,
        context: &Context,
        config: TensorDescriptorConfig<'_>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
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
            device_count: config.partition.count_per_mode.map(<[i32]>::to_vec),
            devices: config.partition.devices.to_vec(),
            data_type: config.data_type,
        })
    }

    /// Returns the number of tensor modes.
    ///
    /// # Panics
    ///
    /// Panics if the stored rank does not fit `u32`. Checked construction excludes this; unsafe adoption must preserve the native rank invariant.
    #[must_use]
    pub fn rank(&self) -> u32 {
        u32::try_from(self.shape.len()).expect("tensor rank is validated on construction")
    }

    /// Returns global tensor extents in descriptor order.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Returns explicit within-block element strides, or `None` for native dense layout.
    #[must_use]
    pub fn element_stride(&self) -> Option<&[u64]> {
        self.element_stride.as_deref()
    }

    /// Returns explicit per-mode block extents, or `None` for unblocked modes.
    #[must_use]
    pub fn block_size(&self) -> Option<&[u64]> {
        self.block_size.as_deref()
    }

    /// Returns explicit inter-block element strides, or `None` for native dense block-interleaved layout.
    #[must_use]
    pub fn block_stride(&self) -> Option<&[u64]> {
        self.block_stride.as_deref()
    }

    /// Returns explicit block-cyclic device counts per mode, or `None` for non-distributed layout.
    #[must_use]
    pub fn device_count(&self) -> Option<&[i32]> {
        self.device_count.as_deref()
    }

    /// Returns tensor storage locations in descriptor pointer order.
    #[must_use]
    pub fn devices(&self) -> &[TensorDevice] {
        &self.devices
    }

    /// Returns the native tensor sample representation.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn context(&self) -> &ContextRef {
        &self.context
    }

    /// Returns the borrowed native distributed tensor descriptor.
    #[must_use]
    pub const fn as_raw(&self) -> sys::cutensorMgTensorDescriptor_t {
        self.handle
    }

    /// Consumes this descriptor and returns the owned raw cuTENSORMg tensor descriptor.
    ///
    /// The caller becomes responsible for destroying the descriptor.
    #[must_use]
    pub fn into_raw(self) -> sys::cutensorMgTensorDescriptor_t {
        let this = ManuallyDrop::new(self);
        this.handle
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        // SAFETY: `cutensorMgDestroyTensorDescriptor` receives live handles owned by the wrappers.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyTensorDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cutensormg tensor descriptor: {error}");
            }
        }
    }
}

impl TensorMode {
    /// Creates a mode with the given extent and native dense, unblocked layout defaults.
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

fn validate_optional_rank_i32(values: Option<&[i32]>, rank: usize, name: &str) -> Result<()> {
    if let Some(values) = values
        && values.len() != rank
    {
        return Err(Error::LengthMismatch {
            name: name.to_owned().into(),
            expected: rank,
            actual: values.len(),
        });
    }
    Ok(())
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
