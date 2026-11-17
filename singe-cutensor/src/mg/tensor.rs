use std::{mem::ManuallyDrop, ptr};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    data_type::{DataType, DataTypeLike},
    device::Device,
};

use crate::{
    error::{Error, Result},
    mg::context::{Context, ContextRef},
    sys, try_ffi,
    utility::{collect_optional_mode_values, to_i64_vec, to_u32},
};

/// A GPU or host-memory location used by a multi-GPU tensor descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TensorDevice {
    /// CUDA device holding a tensor partition.
    Device(Device),
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
            Self::Device(device) => device.id(),
            Self::Host => sys::cutensorMgHostDevice_t::CUTENSOR_MG_DEVICE_HOST.0,
            Self::HostPinned => sys::cutensorMgHostDevice_t::CUTENSOR_MG_DEVICE_HOST_PINNED.0,
        }
    }
}

impl From<Device> for TensorDevice {
    fn from(value: Device) -> Self {
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
pub struct TensorDescriptorOptions<'a> {
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

/// Transferred native ownership with its original metadata and retained multi-GPU context.
///
/// Destroy `handle` before releasing this record, or reclaim it with [`TensorDescriptor::from_raw`].
/// Dropping this record does not destroy the native handle.
#[derive(Debug)]
pub struct RawTensorDescriptor {
    /// Transferred native descriptor handle; destroy it before releasing the retained context.
    pub handle: sys::cutensorMgTensorDescriptor_t,
    /// Multi-GPU context the descriptor was created in.
    pub context: ContextRef,
    /// Global tensor extents in descriptor order.
    pub shape: Vec<u64>,
    /// Within-block element strides, one per mode, or `None` for native dense layout.
    pub element_stride: Option<Vec<u64>>,
    /// Block extents, one per mode, or `None` for unblocked modes.
    pub block_size: Option<Vec<u64>>,
    /// Inter-block element strides, one per mode, or `None` for native dense block-interleaved layout.
    pub block_stride: Option<Vec<u64>>,
    /// Block-cyclic device counts, one per mode, or `None` for non-distributed layout.
    pub device_count: Option<Vec<i32>>,
    /// Tensor storage locations in descriptor pointer order.
    pub devices: Vec<TensorDevice>,
    /// Native tensor sample representation.
    pub data_type: DataType,
}

impl TensorDescriptor {
    /// Creates a dense, unblocked tensor descriptor using the native type of `T`.
    ///
    /// The supplied storage locations are retained in descriptor order.
    /// Per-mode device counts are left unspecified, selecting native non-distributed layout.
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
            TensorDescriptorOptions {
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
    pub fn create(context: &Context, options: TensorDescriptorOptions<'_>) -> Result<Self> {
        let rank = to_u32(options.modes.len(), "modes")?;
        validate_optional_rank(
            options.partition.count_per_mode,
            options.modes.len(),
            "device_count",
        )?;

        let shape = options
            .modes
            .iter()
            .map(|mode| mode.shape)
            .collect::<Vec<_>>();
        let element_stride =
            collect_optional_mode_values(options.modes, "element_stride", |mode| {
                mode.element_stride
            })?;
        let block_size =
            collect_optional_mode_values(options.modes, "block_size", |mode| mode.block_size)?;
        let block_stride =
            collect_optional_mode_values(options.modes, "block_stride", |mode| mode.block_stride)?;

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
        let devices_raw: Vec<_> = options
            .partition
            .devices
            .iter()
            .map(|device| device.as_raw())
            .collect();
        let num_devices = to_u32(devices_raw.len(), "devices")?;

        let mut handle = ptr::null_mut();
        // SAFETY: `context` keeps the handle live, the per-mode arrays hold `rank` values (`collect_optional_mode_values`, `validate_optional_rank`), and `handle` is a local output.
        unsafe {
            try_ffi!(sys::cutensorMgCreateTensorDescriptor(
                context.as_raw(),
                &raw mut handle,
                rank,
                extent_i64.as_ptr(),
                element_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_size_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                block_stride_i64.as_ref().map_or(ptr::null(), Vec::as_ptr),
                options
                    .partition
                    .count_per_mode
                    .map_or(ptr::null(), <[i32]>::as_ptr),
                num_devices,
                devices_raw.as_ptr(),
                options.data_type.into(),
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
            device_count: options.partition.count_per_mode.map(<[i32]>::to_vec),
            devices: options.partition.devices.to_vec(),
            data_type: options.data_type,
        })
    }

    /// Returns the number of tensor modes.
    ///
    /// # Panics
    ///
    /// Panics if the stored rank does not fit `u32`.
    /// Checked construction and [`Self::from_raw`] exclude this.
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

    /// Consumes this descriptor and returns the native handle without destroying it.
    ///
    /// The returned record retains the multi-GPU context and descriptor metadata.
    /// Destroy its handle before releasing the record, or reclaim it with `from_raw`.
    #[must_use]
    pub fn into_raw(self) -> RawTensorDescriptor {
        let owner = ManuallyDrop::new(self);
        // SAFETY: `owner` is never dropped, so each field is read out exactly once.
        unsafe {
            RawTensorDescriptor {
                handle: owner.handle,
                context: ptr::read(&raw const owner.context),
                shape: ptr::read(&raw const owner.shape),
                element_stride: ptr::read(&raw const owner.element_stride),
                block_size: ptr::read(&raw const owner.block_size),
                block_stride: ptr::read(&raw const owner.block_stride),
                device_count: ptr::read(&raw const owner.device_count),
                devices: ptr::read(&raw const owner.devices),
                data_type: owner.data_type,
            }
        }
    }

    /// Takes ownership of a native cuTENSORMg tensor descriptor with its metadata and multi-GPU context.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a valid descriptor created in `raw.context`, uniquely owned, and described by the record's metadata.
    /// The handle must not be destroyed elsewhere afterward.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null, the rank does not fit `u32`, or a per-mode property has a length other than the rank.
    /// The caller must establish all remaining raw descriptor invariants.
    pub unsafe fn from_raw(raw: RawTensorDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }

        let rank = raw.shape.len();
        to_u32(rank, "shape")?;
        validate_optional_rank(raw.element_stride.as_deref(), rank, "element_stride")?;
        validate_optional_rank(raw.block_size.as_deref(), rank, "block_size")?;
        validate_optional_rank(raw.block_stride.as_deref(), rank, "block_stride")?;
        validate_optional_rank(raw.device_count.as_deref(), rank, "device_count")?;
        Ok(Self {
            handle: raw.handle,
            context: raw.context,
            shape: raw.shape,
            element_stride: raw.element_stride,
            block_size: raw.block_size,
            block_stride: raw.block_stride,
            device_count: raw.device_count,
            devices: raw.devices,
            data_type: raw.data_type,
        })
    }

    // Bounds the elements one partition spans, as if each device held the most blocks any device holds along every mode.
    // That matches the allocation NVIDIA's cuTENSORMg samples make for dense layouts.
    // Returns `None` when the bound needs a stride left to a native default, whose value cuTENSORMg does not document for mixed layouts.
    pub(crate) fn partition_len(&self) -> Result<Option<u64>> {
        let overflow = || Error::out_of_range("tensor partition length");
        let mut blocks_per_device = Vec::with_capacity(self.shape.len());
        let mut block_sizes = Vec::with_capacity(self.shape.len());
        for (index, &extent) in self.shape.iter().enumerate() {
            if extent == 0 {
                return Ok(Some(0));
            }

            let block_size =
                mode_value(self.block_size.as_deref(), index, "block_size")?.unwrap_or(extent);
            if block_size == 0 {
                return Err(Error::out_of_range("block_size"));
            }

            let device_count =
                mode_value(self.device_count.as_deref(), index, "device_count")?.unwrap_or(1);
            let device_count = u64::try_from(device_count)
                .ok()
                .filter(|&count| count > 0)
                .ok_or_else(|| Error::out_of_range("device_count"))?;
            blocks_per_device.push(extent.div_ceil(block_size).div_ceil(device_count));
            block_sizes.push(block_size);
        }

        let element_stride = self.element_stride.as_deref();
        let block_stride = self.block_stride.as_deref();
        if element_stride.is_none() && block_stride.is_none() {
            return blocks_per_device
                .iter()
                .zip(&block_sizes)
                .try_fold(1_u64, |len, (&blocks, &size)| {
                    len.checked_mul(blocks)?.checked_mul(size)
                })
                .map(Some)
                .ok_or_else(overflow);
        }

        let mut last = 0_u64;
        for (index, (&blocks, &size)) in blocks_per_device.iter().zip(&block_sizes).enumerate() {
            let within = size.min(self.shape[index]) - 1;
            if within > 0 {
                let Some(stride) = mode_value(element_stride, index, "element_stride")? else {
                    return Ok(None);
                };
                last = within
                    .checked_mul(stride)
                    .and_then(|offset| last.checked_add(offset))
                    .ok_or_else(overflow)?;
            }

            let across = blocks - 1;
            if across > 0 {
                let Some(stride) = mode_value(block_stride, index, "block_stride")? else {
                    return Ok(None);
                };
                last = across
                    .checked_mul(stride)
                    .and_then(|offset| last.checked_add(offset))
                    .ok_or_else(overflow)?;
            }
        }
        last.checked_add(1).map(Some).ok_or_else(overflow)
    }
}

impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor exclusively owns its non-null handle, and `self.context` keeps the multi-GPU handle live.
        unsafe {
            if let Err(error) = try_ffi!(sys::cutensorMgDestroyTensorDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cutensormg tensor descriptor", &error);
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

fn validate_optional_rank<T>(values: Option<&[T]>, rank: usize, name: &str) -> Result<()> {
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

fn mode_value<T: Copy>(
    values: Option<&[T]>,
    index: usize,
    name: &'static str,
) -> Result<Option<T>> {
    values
        .map(|values| {
            values
                .get(index)
                .copied()
                .ok_or_else(|| Error::out_of_range(name))
        })
        .transpose()
}
