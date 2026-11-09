//! CUDA tensor metadata independent of value initialization.

use std::mem::MaybeUninit;

use singe_cuda::view::{DeviceInputView, DeviceOutputView, DeviceRegion, DeviceRepr};

use crate::{
    backend::cuda::utility::cuda_storage,
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    storage::cuda::CudaStorage,
    tensor::{Tensor, TensorDataType},
};

/// CUDA tensor metadata without a value initialization or exclusive access guarantee.
///
/// # Safety
/// Metadata must agree with the returned region, which starts at the logical tensor origin and retains the actual context and allocation borrow.
/// Returning an error for storage that cannot supply a CUDA region is permitted.
pub unsafe trait CudaTensorSource {
    /// Returns the CUDA device that owns the allocation.
    fn device(&self) -> Device;

    /// Returns the tensor element data type.
    fn data_type(&self) -> DataType;

    /// Returns the tensor shape, element strides, and allocation-relative offset.
    fn layout(&self) -> &Layout;

    /// Returns the logical tensor shape.
    fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    /// Returns the logical origin as an element offset into the allocation.
    fn storage_offset(&self) -> usize;

    /// Borrows the CUDA allocation range addressed by this tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the source cannot supply a CUDA region or its layout exceeds the retained allocation.
    /// Implementations report failures to represent the layout or byte offsets as errors.
    fn tensor_region(&self) -> Result<CudaTensorRegion<'_>>;

    /// Returns the number of logical tensor elements.
    fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    /// Returns whether the layout stores logical elements contiguously in row-major order.
    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns the logical element byte count, excluding layout gaps and the storage offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical element byte count overflows `usize`.
    fn byte_size(&self) -> Result<usize> {
        self.shape().byte_size(self.data_type().size_bytes())
    }
}

// SAFETY: Tensor metadata and its CUDA storage describe the same retained allocation.
unsafe impl CudaTensorSource for Tensor {
    fn device(&self) -> Device {
        self.device()
    }

    fn data_type(&self) -> DataType {
        self.data_type()
    }

    fn layout(&self) -> &Layout {
        self.layout()
    }

    fn storage_offset(&self) -> usize {
        self.storage_offset()
    }

    fn tensor_region(&self) -> Result<CudaTensorRegion<'_>> {
        CudaTensorRegion::from_tensor(self)
    }
}

/// Shape and datatype metadata over an actual CUDA allocation range.
/// This descriptor grants neither initialized read access nor exclusive writes.
#[derive(Debug, Clone, Copy)]
pub struct CudaTensorRegion<'a> {
    data_type: DataType,
    layout: &'a Layout,
    region: DeviceRegion<'a>,
}

impl<'a> CudaTensorRegion<'a> {
    /// Checks that a CUDA region covers the shape's complete storage span.
    ///
    /// `region` must start at the logical tensor origin.
    /// The layout offset is metadata and is not added to the supplied region again.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout span cannot be represented or the region is too small.
    pub fn create(
        data_type: DataType,
        layout: &'a Layout,
        region: DeviceRegion<'a>,
    ) -> Result<Self> {
        let bytes = layout
            .clone()
            .with_offset(0)
            .storage_span_byte_size(data_type.size_bytes())?;
        let region = region.slice(..bytes)?;
        Ok(Self {
            data_type,
            layout,
            region,
        })
    }

    /// Borrows allocation metadata from a readable CUDA tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has no CUDA storage, its byte offset overflows, or its layout exceeds the allocation.
    pub fn from_tensor(tensor: &'a Tensor) -> Result<Self> {
        let storage = cuda_storage(tensor)?;
        let start = tensor
            .storage_offset()
            .checked_mul(tensor.data_type().size_bytes())
            .ok_or_else(|| Error::out_of_range("cuda tensor region offset"))?;
        Self::create(
            tensor.data_type(),
            tensor.layout(),
            storage.region(start..storage.byte_size())?,
        )
    }

    /// Returns the CUDA device that owns the allocation.
    #[must_use]
    pub fn device(self) -> Device {
        Device::Cuda(self.region.context().device().id())
    }

    /// Returns the tensor element data type.
    #[must_use]
    pub const fn data_type(self) -> DataType {
        self.data_type
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub fn shape(self) -> &'a Shape {
        self.layout.shape()
    }

    /// Returns the tensor shape, element strides, and allocation-relative offset.
    #[must_use]
    pub fn layout(self) -> &'a Layout {
        self.layout
    }

    /// Returns the number of logical tensor elements.
    #[must_use]
    pub fn element_count(self) -> usize {
        self.layout.element_count()
    }

    /// Returns whether the layout stores logical elements contiguously in row-major order.
    #[must_use]
    pub fn is_contiguous(self) -> bool {
        self.layout.is_contiguous()
    }

    /// Returns the borrowed allocation range, including any gaps between tensor elements.
    /// The range grants no value initialization or exclusive access guarantee.
    #[must_use]
    pub const fn region(self) -> DeviceRegion<'a> {
        self.region
    }

    /// Describes a contiguous typed input without claiming initialized values.
    /// Raw consumers must order valid production before reading this range.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` differs from the declared data type, the layout is not contiguous, or the region fails typed-view alignment or byte-length checks.
    pub fn input<T: TensorDataType + DeviceRepr>(self) -> Result<DeviceInputView<'a, T>> {
        if self.data_type != T::data_type() {
            return Err(Error::data_type_mismatch(
                "cuda tensor input region",
                T::data_type(),
                self.data_type,
            ));
        }

        if !self.is_contiguous() {
            return Err(Error::ContiguousRequired);
        }
        Ok(DeviceInputView::create(self.region)?)
    }
}

/// A contiguous CUDA result allocation that has no readable tensor values yet.
/// Dropping it retains the same cache-reuse and context rules as initialized storage.
#[derive(Debug)]
pub struct PendingCudaTensor {
    storage: CudaStorage<MaybeUninit<u8>>,
    data_type: DataType,
    layout: Layout,
}

impl PendingCudaTensor {
    /// Borrows the pending contiguous allocation as an exclusive typed destination.
    ///
    /// # Errors
    ///
    /// Returns an error if `T` differs from the tensor data type or the allocation fails typed-view alignment, size, or range checks.
    pub fn output<T: TensorDataType + DeviceRepr>(&mut self) -> Result<DeviceOutputView<'_, T>> {
        if T::data_type() != self.data_type {
            return Err(Error::data_type_mismatch(
                "pending cuda tensor output",
                self.data_type,
                T::data_type(),
            ));
        }
        self.storage.output()
    }

    /// Takes pending storage whose capacity exactly matches a contiguous shape of `data_type` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the shape byte count overflows or differs from the storage byte count.
    pub fn create(
        storage: CudaStorage<MaybeUninit<u8>>,
        data_type: DataType,
        shape: Shape,
    ) -> Result<Self> {
        let bytes = shape.byte_size(data_type.size_bytes())?;
        if bytes != storage.byte_size() {
            return Err(Error::length_mismatch(
                "pending cuda tensor bytes",
                bytes,
                storage.byte_size(),
            ));
        }
        Ok(Self {
            storage,
            data_type,
            layout: Layout::new(shape, 0),
        })
    }

    /// Borrows shape and allocation metadata without claiming initialized values.
    #[must_use]
    pub fn region(&self) -> CudaTensorRegion<'_> {
        CudaTensorRegion {
            data_type: self.data_type,
            layout: &self.layout,
            region: self.storage.allocation_region(),
        }
    }

    /// Returns whether the pending tensor has zero logical elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layout.element_count() == 0
    }

    /// Produces a readable tensor after a completed producer has written every element.
    ///
    /// # Safety
    /// Every byte must have been initialized to a valid representation for the declared datatype.
    /// All producers must have completed successfully and no outstanding write may conflict with subsequent tensor access.
    #[must_use]
    pub unsafe fn assume_init(self) -> Tensor {
        // SAFETY: the caller upholds this function's safety contract.
        let storage = unsafe { self.storage.assume_init() };
        // SAFETY: creation validated capacity and shape; the caller establishes value validity.
        unsafe { Tensor::from_parts_unchecked(storage, self.data_type, self.layout) }
    }
}
