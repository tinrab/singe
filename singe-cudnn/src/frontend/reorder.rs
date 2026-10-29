//! Preparation of `INT8x32` convolution filters and optional FP32 bias values.
//! Reordered filters use [`BackendTensorReordering::Int8x32`](crate::backend::tensor::BackendTensorReordering::Int8x32).

use std::{
    fmt::{self, Debug, Formatter},
    mem::size_of,
    ptr,
};

use singe_cuda::{
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceRegion, DeviceRepr},
};
use singe_cudnn_sys as sys;

use crate::{
    backend::tensor::TensorSpec,
    context::Context,
    data_type::DataType,
    error::{Error, Result},
    try_ffi,
    utility::{to_i32, to_usize},
};

/// Source and destination for one reordering request.
/// Pairing the buffers prevents a requested input from having a missing output.
pub struct ReorderBuffers<'a, T: DeviceRepr> {
    /// Initialized contiguous source storage in the unreordered filter or bias encoding.
    pub source: &'a dyn DeviceInput<T>,
    /// Exclusive destination storage for the reordered filter or bias values.
    pub destination: &'a mut dyn DeviceOutput<T>,
}

impl<T: DeviceRepr> Debug for ReorderBuffers<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReorderBuffers").finish_non_exhaustive()
    }
}

/// Owns a context-independent legacy filter descriptor used by the C++ frontend's `INT8x32` helper.
/// Descriptor construction happens before execution or stream capture.
/// Prepared helpers can be shared across workers using independent cuDNN handles and buffers.
#[derive(Debug)]
pub struct Int8x32FilterReorder {
    descriptor: FilterDescriptor,
    dimensions: Vec<i64>,
    filter_byte_len: usize,
    output_channels: usize,
}

#[derive(Debug)]
struct FilterDescriptor(sys::cudnnFilterDescriptor_t);

// SAFETY: the exclusively owned legacy descriptor has no CUDA-context or thread affinity.
unsafe impl Send for FilterDescriptor {}

// SAFETY: configuration finishes before exposure, and reordering only reads the private descriptor.
unsafe impl Sync for FilterDescriptor {}

impl Drop for FilterDescriptor {
    fn drop(&mut self) {
        // SAFETY: this object exclusively owns the descriptor, including on initialization errors.
        unsafe {
            sys::cudnnDestroyFilterDescriptor(self.0);
        }
    }
}

impl Int8x32FilterReorder {
    /// Creates a reorder descriptor from scalar dimensions `[K, C, R, S]` or `[K, C, D, R, S]`.
    /// `C` is the full input-channel count, not the count of 32-element channel blocks.
    /// Buffers use contiguous NCHW-vectorized storage with 32 INT8 lanes.
    /// Native execution decides support for the requested spatial rank and device.
    ///
    /// # Errors
    /// Returns invalid dimension, size overflow, allocation or native descriptor errors.
    pub fn create(dimensions: impl Into<Vec<i64>>) -> Result<Self> {
        let dimensions = dimensions.into();
        if !matches!(dimensions.len(), 4 | 5)
            || dimensions.iter().any(|&dimension| dimension <= 0)
            || dimensions[1] % 32 != 0
        {
            return Err(Error::InvalidDataShape);
        }
        let native_dimensions = dimensions
            .iter()
            .map(|&dimension| to_i32(dimension, "reorder filter dimension"))
            .collect::<Result<Vec<_>>>()?;
        let output_channels = to_usize(dimensions[0], "reorder output channels")?;
        let mut descriptor = ptr::null_mut();
        // SAFETY: the output is initialized by cuDNN and adopted immediately after successful creation.
        unsafe {
            try_ffi!(sys::cudnnCreateFilterDescriptor(&raw mut descriptor))?;
        }
        if descriptor.is_null() {
            return Err(Error::NullHandle);
        }
        let descriptor = FilterDescriptor(descriptor);
        let mut filter_byte_len = 0;
        // SAFETY: the descriptor and dimension array remain live through native configuration and size queries.
        unsafe {
            try_ffi!(sys::cudnnSetFilterNdDescriptor(
                descriptor.0,
                sys::cudnnDataType_t::CUDNN_DATA_INT8x32,
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C,
                to_i32(native_dimensions.len(), "reorder filter rank")?,
                native_dimensions.as_ptr(),
            ))?;
            try_ffi!(sys::cudnnGetFilterSizeInBytes(
                descriptor.0,
                &raw mut filter_byte_len,
            ))?;
        }
        Ok(Self {
            descriptor,
            dimensions,
            filter_byte_len: to_usize(filter_byte_len, "reorder filter bytes")?,
            output_channels,
        })
    }

    /// Creates the same descriptor from C++ frontend-style blocked filter dimensions.
    /// For two leading dimensions, these are `[K, C/32, spatial...]`.
    /// With three leading dimensions, axis one must be unit padding; K is axis zero and C/32 is axis two.
    /// The tensor must describe contiguous INT8 storage; this conversion does not move data.
    /// Optional vectorization metadata must specify both the 32-lane count and channel axis.
    /// Negative axes resolve against the blocked tensor rank, as in native tensor construction.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported spatial or blocked rank, non-INT8 or ragged storage, nonunit leading padding, invalid vectorization metadata, noncontiguous strides, overflowing channel expansion, invalid logical dimensions, or native descriptor creation, configuration, or size-query failure.
    ///
    /// # Panics
    /// Panics if the validated leading-axis index cannot fit `i64`. The accepted leading ranks are two and three, so valid inputs cannot trigger this conversion failure.
    pub fn create_for_tensor(filter: &TensorSpec, spatial_dimensions: usize) -> Result<Self> {
        if !matches!(spatial_dimensions, 2 | 3) {
            return Err(Error::InvalidDataShape);
        }
        let dimensions = filter.shape.dimensions();
        let leading = dimensions
            .len()
            .checked_sub(spatial_dimensions)
            .ok_or(Error::InvalidDataShape)?;
        if !matches!(leading, 2 | 3)
            || filter.data_type != DataType::I8
            || filter.ragged_offset.is_some()
        {
            return Err(Error::InvalidDataShape);
        }
        if leading == 3 && dimensions[1] != 1 {
            return Err(Error::DescriptorMismatch {
                name: "int8x32 filter leading padding axis".into(),
            });
        }
        if filter.vectorization()?.is_some_and(|(count, axis)| {
            count != 32
                || axis
                    != i64::try_from(leading - 1).expect("tensor rank fits the native index width")
        }) {
            return Err(Error::DescriptorMismatch {
                name: "int8x32 filter vectorization".into(),
            });
        }
        if !filter.shape.is_contiguous() {
            return Err(Error::InvalidDataStrides);
        }
        let channels = dimensions[leading - 1]
            .checked_mul(32)
            .ok_or_else(|| Error::out_of_range("reorder input channels"))?;
        let mut logical_dimensions = vec![dimensions[0], channels];
        logical_dimensions.extend_from_slice(&dimensions[leading..]);
        Self::create(logical_dimensions)
    }

    /// Borrows logical filter dimensions `[K, C, spatial...]`, with C counting scalar channels.
    #[must_use]
    pub fn dimensions(&self) -> &[i64] {
        &self.dimensions
    }

    /// Native storage requirement queried from the configured `INT8x32` filter descriptor.
    #[must_use]
    pub fn filter_byte_len(&self) -> usize {
        self.filter_byte_len
    }

    /// Returns the number of FP32 bias elements, one per output channel.
    #[must_use]
    pub fn bias_element_count(&self) -> usize {
        self.output_channels
    }

    /// Reorders filter bytes and, when requested, one FP32 bias per output channel on `stream`.
    /// Either pair may be omitted, matching the C++ helper; cuDNN reports unsupported combinations.
    /// The cuDNN handle's previous stream binding is restored after the call.
    /// Extra buffer capacity is left untouched.
    ///
    /// # Safety
    /// Retain this descriptor, the context, stream and all buffers through completion and captured replay.
    /// Inputs must be initialized, and writes must remain exclusive until completion.
    /// Order producers outside the regions' declared readiness streams before this call.
    /// Outputs become initialized only after successful execution or replay.
    ///
    /// # Errors
    /// Returns invalid size, context, overlapping buffer, stream or native execution errors.
    pub unsafe fn execute(
        &self,
        context: &mut Context,
        stream: &Stream,
        mut filter: Option<ReorderBuffers<'_, i8>>,
        mut bias: Option<ReorderBuffers<'_, f32>>,
    ) -> Result<()> {
        if filter.is_none() && bias.is_none() {
            return Err(Error::EmptyList {
                name: "reorder buffers".into(),
            });
        }
        if context.cuda_context() != stream.context() {
            return Err(Error::ContextMismatch {
                name: "reorder stream".into(),
            });
        }
        let mut regions = Vec::new();
        if let Some(buffers) = &filter {
            collect_regions(buffers, self.filter_byte_len, &mut regions)?;
        }
        if let Some(buffers) = &bias {
            collect_regions(buffers, self.output_channels, &mut regions)?;
        }
        if regions
            .iter()
            .any(|(region, _)| region.context() != stream.context())
        {
            return Err(Error::ContextMismatch {
                name: "reorder buffer".into(),
            });
        }
        for (index, (region, writes)) in regions.iter().enumerate() {
            let start = region.as_ptr().addr();
            let end = start
                .checked_add(region.byte_len())
                .ok_or(Error::InvalidDataShape)?;
            for (other, other_writes) in &regions[..index] {
                let other_start = other.as_ptr().addr();
                let other_end = other_start
                    .checked_add(other.byte_len())
                    .ok_or(Error::InvalidDataShape)?;
                if (*writes || *other_writes) && start < other_end && other_start < end {
                    return Err(Error::DescriptorMismatch {
                        name: "reorder overlapping buffers".into(),
                    });
                }
            }
        }
        stream.join_regions(
            &regions
                .iter()
                .map(|(region, _)| *region)
                .collect::<Vec<_>>(),
        )?;
        drop(regions);
        let filter_source = filter
            .as_ref()
            .map_or(ptr::null(), |buffers| buffers.source.as_device_ptr().cast());
        let filter_destination = filter.as_mut().map_or(ptr::null_mut(), |buffers| {
            buffers.destination.as_mut_ptr().cast()
        });
        let bias_source = bias
            .as_ref()
            .map_or(ptr::null(), |buffers| buffers.source.as_device_ptr().cast());
        let bias_destination = bias.as_mut().map_or(ptr::null_mut(), |buffers| {
            buffers.destination.as_mut_ptr().cast()
        });
        context.with_stream(Some(stream), |context| {
            context.bind()?;
            // SAFETY: typed buffer capacities, contexts and overlap were validated before entering the native boundary.
            // The caller retains the descriptor and storage through completion or replay.
            unsafe {
                try_ffi!(sys::cudnnReorderFilterAndBias(
                    context.as_raw(),
                    self.descriptor.0,
                    sys::cudnnReorderType_t::CUDNN_DEFAULT_REORDER,
                    filter_source,
                    filter_destination,
                    i32::from(bias.is_some()),
                    bias_source,
                    bias_destination,
                ))
            }
        })
    }
}

fn collect_regions<'a, T: DeviceRepr>(
    buffers: &'a ReorderBuffers<'_, T>,
    elements: usize,
    regions: &mut Vec<(DeviceRegion<'a>, bool)>,
) -> Result<()> {
    let bytes = elements
        .checked_mul(size_of::<T>())
        .ok_or(Error::InvalidDataShape)?;
    for (region, len, writes) in [
        (buffers.source.region(), buffers.source.len(), false),
        (
            buffers.destination.region(),
            buffers.destination.len(),
            true,
        ),
    ] {
        if len < elements {
            return Err(Error::LengthMismatch {
                name: "reorder buffer".into(),
                expected: elements,
                actual: len,
            });
        }
        regions.push((region.slice(..bytes)?, writes));
    }
    Ok(())
}
