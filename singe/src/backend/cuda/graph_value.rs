//! Owning tensor regions used while recording and replaying CUDA graphs.

use std::{mem::MaybeUninit, sync::Arc};

use singe_cuda::{
    error::Error as CudaError,
    submission::{AccessMode, Resources},
    view::{AllocationId, DeviceOutputView, DeviceRegion, DeviceRepr},
};

use crate::{
    backend::cuda::{
        tensor_region::{CudaTensorRegion, CudaTensorSource},
        utility::cuda_storage,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    storage::cuda::CudaStorage,
    tensor::{Tensor, TensorDataType},
};

#[derive(Debug)]
enum CudaGraphValueOwner {
    Allocation(CudaStorage<MaybeUninit<u8>>),
    Tensor(Tensor),
}

/// An owning CUDA tensor region with deferred value validity.
///
/// Graph definitions, reuse pools, and raw callers can retain this value without asserting that any producer has executed.
/// It exposes metadata rather than readable tensor storage.
/// Cloning it does not grant exclusive write access.
#[derive(Debug, Clone)]
pub struct CudaGraphValue {
    pub(crate) inner: Arc<CudaGraphValueInner>,
}

#[derive(Debug)]
pub struct CudaGraphValueInner {
    owner: Arc<CudaGraphValueOwner>,
    data_type: DataType,
    device: Device,
    layout: Layout,
    allocation_id: AllocationId,
}

impl CudaGraphValue {
    pub(crate) fn submission_resources(&self, mode: AccessMode) -> Result<Resources> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(&self.inner));
        resources.access(self.region()?.region(), mode);
        Ok(resources)
    }

    /// Takes pending storage covering every address in the shape, including gaps.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte offset or layout span cannot be represented or the layout exceeds the retained allocation.
    pub fn create(
        storage: CudaStorage<MaybeUninit<u8>>,
        data_type: DataType,
        layout: Layout,
    ) -> Result<Self> {
        let device = Device::Cuda(storage.context().device().id());
        let allocation_id = storage.allocation_region().allocation_id();
        let value = Self {
            inner: Arc::new(CudaGraphValueInner {
                owner: Arc::new(CudaGraphValueOwner::Allocation(storage)),
                data_type,
                device,
                layout,
                allocation_id,
            }),
        };
        value.region()?;
        Ok(value)
    }

    /// Retains an initialized tensor as graph input metadata without copying it.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor has no CUDA storage or its byte offset or layout span cannot be represented within its allocation.
    pub fn from_tensor(tensor: Tensor) -> Result<Self> {
        let storage = cuda_storage(&tensor)?;
        let allocation_id = storage.allocation_region().allocation_id();
        let data_type = tensor.data_type();
        let device = tensor.device();
        let layout = tensor.layout().clone();
        let value = Self {
            inner: Arc::new(CudaGraphValueInner {
                owner: Arc::new(CudaGraphValueOwner::Tensor(tensor)),
                data_type,
                device,
                layout,
                allocation_id,
            }),
        };
        value.region()?;
        Ok(value)
    }

    fn allocation_region(&self) -> Result<DeviceRegion<'_>> {
        Ok(match self.inner.owner.as_ref() {
            CudaGraphValueOwner::Allocation(storage) => storage.allocation_region(),
            CudaGraphValueOwner::Tensor(tensor) => cuda_storage(tensor)?.allocation_region(),
        })
    }

    /// Borrows the range from the logical tensor origin through the last addressed element.
    /// The range includes layout gaps and grants no initialized read or exclusive write access.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte offset or layout span cannot be represented or the layout exceeds the retained allocation.
    pub fn region(&self) -> Result<CudaTensorRegion<'_>> {
        let start = self
            .inner
            .layout
            .storage_offset()
            .checked_mul(self.data_type().size_bytes())
            .ok_or_else(|| Error::out_of_range("cuda graph value byte offset"))?;
        CudaTensorRegion::create(
            self.data_type(),
            self.layout(),
            self.allocation_region()?.slice(start..)?,
        )
    }

    /// Returns the CUDA device that owns the allocation.
    #[must_use]
    pub fn device(&self) -> Device {
        self.inner.device
    }

    /// Returns the tensor element data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.inner.data_type
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.inner.layout.shape()
    }

    /// Returns the logical origin as an element offset into the allocation.
    #[must_use]
    pub fn storage_offset(&self) -> usize {
        self.inner.layout.storage_offset()
    }

    /// Returns the number of logical tensor elements.
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.shape().element_count()
    }

    /// Returns the logical element byte count, excluding layout gaps and the storage offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the logical element byte count overflows `usize`.
    pub fn byte_size(&self) -> Result<usize> {
        self.shape().byte_size(self.data_type().size_bytes())
    }

    /// Returns whether the layout stores logical elements contiguously in row-major order.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }

    /// Returns the tensor shape, element strides, and allocation-relative offset.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.inner.layout
    }

    /// Returns the byte size of a pending allocation, or `None` for a value that retains a tensor.
    pub(crate) fn pending_allocation_bytes(&self) -> Option<usize> {
        match self.inner.owner.as_ref() {
            CudaGraphValueOwner::Allocation(storage) => Some(storage.byte_size()),
            CudaGraphValueOwner::Tensor(_) => None,
        }
    }

    /// Creates an alias of the same pending allocation with another data type and layout, with no access permissions.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout exceeds the retained allocation.
    pub(crate) fn reinterpret(&self, data_type: DataType, layout: Layout) -> Result<Self> {
        let value = Self {
            inner: Arc::new(CudaGraphValueInner {
                owner: Arc::clone(&self.inner.owner),
                data_type,
                device: self.device(),
                layout,
                allocation_id: self.inner.allocation_id,
            }),
        };
        value.region()?;
        Ok(value)
    }

    /// Returns the identity of the retained CUDA allocation.
    #[must_use]
    pub fn allocation_id(&self) -> AllocationId {
        self.inner.allocation_id
    }

    /// Returns whether both values retain the same allocation.
    /// Their addressed tensor regions need not overlap.
    #[must_use]
    pub fn shares_storage(&self, other: &Self) -> bool {
        self.inner.allocation_id == other.inner.allocation_id
    }

    /// Creates an alias within the same retained storage, with no access permissions.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte offset or layout span cannot be represented or the layout exceeds the retained allocation.
    pub fn alias(&self, layout: Layout) -> Result<Self> {
        self.reinterpret(self.data_type(), layout)
    }

    /// Converts uniquely owned storage into a readable tensor.
    ///
    /// # Safety
    /// Every byte in the retained allocation, including padding and prefixes, must contain an initialized representation.
    /// Every element must be valid for its datatype.
    /// Producers must have completed successfully and no outstanding operation may write the allocation.
    ///
    /// # Errors
    ///
    /// Returns an allocation-in-use error if another value, storage alias, or retained submission resource prevents unique ownership.
    pub unsafe fn assume_init(self) -> Result<Tensor> {
        let inner = Arc::try_unwrap(self.inner).map_err(|_| CudaError::AllocationInUse)?;
        let owner = Arc::try_unwrap(inner.owner).map_err(|_| CudaError::AllocationInUse)?;
        let tensor = match owner {
            CudaGraphValueOwner::Allocation(storage) => {
                // SAFETY: the caller establishes completed, valid initialization.
                unsafe {
                    Tensor::from_parts_unchecked(
                        storage.assume_init(),
                        inner.data_type,
                        inner.layout.clone(),
                    )
                }
            }
            CudaGraphValueOwner::Tensor(tensor) => tensor,
        };
        Ok(tensor.alias_with_layout_unrecorded(inner.layout))
    }

    /// Borrows a contiguous typed destination without asserting value initialization.
    ///
    /// # Safety
    /// Keep every alias from conflicting with this writable range until all submitted work completes, including captured replay and error paths.
    ///
    /// # Errors
    ///
    /// Returns an error if the layout exceeds the allocation, offset or span arithmetic fails, `T` differs from the declared type, the layout is not contiguous, or typed-view alignment or size checks fail.
    pub unsafe fn output<T: TensorDataType + DeviceRepr>(
        &mut self,
    ) -> Result<DeviceOutputView<'_, T>> {
        let value = self.region()?;
        if value.data_type() != T::data_type() {
            return Err(Error::data_type_mismatch(
                "cuda graph output",
                T::data_type(),
                value.data_type(),
            ));
        }

        if !value.is_contiguous() {
            return Err(Error::ContiguousRequired);
        }
        // SAFETY: the region checks type and bounds; the caller establishes exclusivity.
        Ok(unsafe { DeviceOutputView::from_raw_parts(value.region(), value.element_count())? })
    }
}

// SAFETY: all constructors validate the retained storage, datatype, shape, and offset.
unsafe impl CudaTensorSource for CudaGraphValue {
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
        self.region()
    }
}

/// Borrowed graph output destinations with caller-established exclusive access.
#[derive(Debug, Clone, Copy)]
pub struct GraphOutputBuffers<'a> {
    values: &'a [CudaGraphValue],
}

impl<'a> GraphOutputBuffers<'a> {
    /// Supplies raw destinations for one replay submission.
    ///
    /// # Safety
    /// Until the submission completes, no alias may access a destination in a conflicting way.
    /// Destinations must not overlap one another, graph-private storage, or replay inputs.
    /// The request retains allocation owners, but does not grant access permission to otherwise aliased tensors.
    #[must_use]
    pub unsafe fn create(values: &'a [CudaGraphValue]) -> Self {
        Self { values }
    }

    /// Returns the output destinations in replay output order.
    #[must_use]
    pub fn values(self) -> &'a [CudaGraphValue] {
        self.values
    }
}
