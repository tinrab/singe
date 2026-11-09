mod allocation;
mod context;
mod lifecycle;
mod operations;

use std::{
    fmt,
    mem::{ManuallyDrop, MaybeUninit},
    ops::Range,
    ptr,
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    event::Event,
    memory::DeviceMemory,
    stream::Stream,
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRegion, DeviceRegionProvider, DeviceRepr,
        DeviceView,
    },
};

pub(crate) use self::{
    allocation::map_cuda_allocation_error,
    context::{cuda_context_for_device, cuda_device_id},
};
use crate::{
    data_type::DataType,
    error::{Error, Result},
    storage::StorageVersion,
    tensor::TensorDataType,
};

pub(crate) trait CudaMemoryRecycler: fmt::Debug + Send + Sync {
    fn recycle(&self, memory: DeviceMemory<MaybeUninit<u8>>, ready_event: Option<Event>);
}

/// Byte initialization states supported by CUDA tensor storage.
/// `u8` describes initialized representation bytes; `MaybeUninit<u8>` describes pending storage with no readable values.
pub trait CudaStorageByte: DeviceRepr + private::Byte {}

mod private {
    use super::{DeviceMemory, DeviceRepr, MaybeUninit};

    pub trait Byte: DeviceRepr {
        fn into_pending(memory: DeviceMemory<Self>) -> DeviceMemory<MaybeUninit<u8>>;
    }

    impl Byte for u8 {
        fn into_pending(memory: DeviceMemory<Self>) -> DeviceMemory<MaybeUninit<u8>> {
            memory.into_uninit()
        }
    }

    impl Byte for MaybeUninit<u8> {
        fn into_pending(memory: DeviceMemory<Self>) -> DeviceMemory<MaybeUninit<u8>> {
            memory
        }
    }
}

impl CudaStorageByte for u8 {}

impl CudaStorageByte for MaybeUninit<u8> {}

/// Device-resident tensor storage backed by CUDA memory.
///
/// Most code interacts with CUDA storage through `Tensor` operations.
/// Lower-level integrations can match `Storage::Cuda`, inspect the owning CUDA context, or pass the device allocation to CUDA library wrappers.
/// `CudaStorage<MaybeUninit<u8>>` owns pending bytes and does not implement [`crate::storage::Storage`] or expose initialized typed views.
/// Completed producers can promote it explicitly with `assume_init`.
#[derive(Debug)]
pub struct CudaStorage<T: CudaStorageByte = u8> {
    data: ManuallyDrop<DeviceMemory<T>>,
    recycler: Option<Arc<dyn CudaMemoryRecycler>>,
    recycle_stream: Option<Arc<Stream>>,
    version: StorageVersion,
}

impl CudaStorage {
    /// Takes ownership of representation bytes holding whole `data_type` elements.
    ///
    /// # Safety
    ///
    /// Every element must have a valid representation for `data_type`, including 0 or 1 for booleans.
    /// Producers must be ordered before subsequent consumers.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte allocation length is not a whole number of elements of the declared data type.
    pub unsafe fn from_raw(data_type: DataType, data: DeviceMemory<u8>) -> Result<Self> {
        ensure_whole_elements(data_type, data.len())?;
        Ok(Self {
            data: ManuallyDrop::new(data),
            recycler: None,
            recycle_stream: None,
            version: StorageVersion::default(),
        })
    }

    pub(crate) unsafe fn from_memory_recyclable(
        data_type: DataType,
        data: DeviceMemory<u8>,
        recycler: Arc<dyn CudaMemoryRecycler>,
        recycle_stream: Option<Arc<Stream>>,
    ) -> Result<Self> {
        if recycle_stream
            .as_ref()
            .is_some_and(|stream| stream.context() != data.context())
        {
            return Err(CudaError::StreamContextMismatch.into());
        }

        // SAFETY: the caller upholds this function's safety contract.
        let mut storage = unsafe { Self::from_raw(data_type, data)? };
        storage.recycler = Some(recycler);
        storage.recycle_stream = recycle_stream;
        Ok(storage)
    }
}

impl<T: CudaStorageByte> CudaStorage<T> {
    fn into_parts(self) -> CudaStorageParts<T> {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: transfer the allocation once, then release the remaining owning fields.
        unsafe {
            let data = ManuallyDrop::take(&mut this.data);
            let recycler = ptr::read(&raw const this.recycler);
            let stream = ptr::read(&raw const this.recycle_stream);
            (data, recycler, stream)
        }
    }
}

impl CudaStorage<MaybeUninit<u8>> {
    /// Borrows pending storage as a destination of `T` elements.
    ///
    /// # Errors
    ///
    /// Returns an error if the allocation is not a whole number of `T` elements or fails typed-view size, alignment, or range checks.
    pub fn output<T: TensorDataType + DeviceRepr>(&mut self) -> Result<DeviceOutputView<'_, T>> {
        let size = T::data_type().size_bytes();
        if !self.byte_size().is_multiple_of(size) {
            return Err(Error::byte_length_mismatch(
                "cuda storage output",
                size,
                self.byte_size(),
            ));
        }

        let region = self.region(0..self.byte_size())?;
        let length = self.byte_size() / size;
        // SAFETY: this exclusive borrow retains the pending allocation and its full byte span.
        Ok(unsafe { DeviceOutputView::from_raw_parts(region, length)? })
    }

    /// Takes ownership of pending representation bytes without granting readable access.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte allocation length is not a whole number of `data_type` elements.
    pub fn from_uninit(data_type: DataType, data: DeviceMemory<MaybeUninit<u8>>) -> Result<Self> {
        ensure_whole_elements(data_type, data.len())?;
        Ok(Self {
            data: ManuallyDrop::new(data),
            recycler: None,
            recycle_stream: None,
            version: StorageVersion::default(),
        })
    }

    pub(crate) fn from_uninit_recyclable(
        data_type: DataType,
        data: DeviceMemory<MaybeUninit<u8>>,
        recycler: Arc<dyn CudaMemoryRecycler>,
        recycle_stream: Option<Arc<Stream>>,
    ) -> Result<Self> {
        if recycle_stream
            .as_ref()
            .is_some_and(|stream| stream.context() != data.context())
        {
            return Err(CudaError::StreamContextMismatch.into());
        }

        let mut storage = Self::from_uninit(data_type, data)?;
        storage.recycler = Some(recycler);
        storage.recycle_stream = recycle_stream;
        Ok(storage)
    }

    /// Promotes a fully produced allocation while retaining its recycler and context.
    ///
    /// # Safety
    /// Every representation byte must have been initialized by completed producers.
    /// Every element must be valid for the data type of each tensor that reads it, including 0 or 1 for booleans.
    /// No outstanding access may conflict with subsequent use.
    #[must_use]
    pub unsafe fn assume_init(self) -> CudaStorage {
        let (data, recycler, recycle_stream) = self.into_parts();
        CudaStorage {
            // SAFETY: the caller upholds this function's safety contract.
            data: ManuallyDrop::new(unsafe { data.assume_init() }),
            recycler,
            recycle_stream,
            version: StorageVersion::default(),
        }
    }
}

type CudaStorageParts<T> = (
    DeviceMemory<T>,
    Option<Arc<dyn CudaMemoryRecycler>>,
    Option<Arc<Stream>>,
);

impl CudaStorage {
    /// Borrows initialized bytes as aligned values accepting every bit pattern.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is invalid, its length is not a whole number of `T` elements, or the selected nonempty range is insufficiently aligned for `T`.
    pub fn view<T: AnyBitPatternDeviceRepr>(
        &self,
        bytes: Range<usize>,
    ) -> Result<DeviceView<'_, T>> {
        Ok(self.data.view().slice(bytes)?.cast()?)
    }
}

impl<T: CudaStorageByte> CudaStorage<T> {
    /// Borrows the complete allocation metadata without granting value access.
    #[must_use]
    pub fn allocation_region(&self) -> DeviceRegion<'_> {
        self.data.region()
    }

    /// Borrows allocation metadata without granting initialized or writable access.
    ///
    /// # Errors
    ///
    /// Returns an error if the byte range is reversed or exceeds the allocation capacity.
    pub fn region(&self, bytes: Range<usize>) -> Result<DeviceRegion<'_>> {
        Ok(self.data.region().slice(bytes)?)
    }

    /// Returns the allocation capacity in bytes, independently of initialization.
    #[must_use]
    pub fn byte_size(&self) -> usize {
        self.data.len()
    }

    /// Returns the CUDA context that owns this allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<CudaContext> {
        self.data.context()
    }

    /// Returns a raw const pointer to the device allocation.
    ///
    /// The pointer is valid only while this storage owns the allocation.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a raw mutable pointer to the device allocation.
    ///
    /// The pointer is valid only while this storage owns the allocation.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

fn ensure_whole_elements(data_type: DataType, bytes: usize) -> Result<()> {
    let size = data_type.size_bytes();
    if size == 0 || !bytes.is_multiple_of(size) {
        return Err(Error::byte_length_mismatch("cuda storage", size, bytes));
    }
    Ok(())
}
