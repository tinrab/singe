//! Transfers with owned pinned staging and explicit pending device values.
use std::{mem::MaybeUninit, sync::Arc};

use crate::{
    context::Context,
    error::{Error, Result},
    graph::owned::{Recordable, Recording},
    memory::{DeviceMemory, HostAllocationFlags, MemoryCopyKind, PinnedHostMemory},
    stream::Stream,
    submission::{AccessMode, DeviceOperation, Execution, Operation, Resources},
    view::{AnyBitPatternDeviceRepr, DeviceRegion, DeviceRegionProvider, DeviceRepr},
};

/// Owned valid device values whose producer may still be pending.
/// This type exposes allocation metadata, not initialized host or device borrows.
#[derive(Debug)]
pub struct DeviceValue<T: DeviceRepr> {
    memory: DeviceMemory<MaybeUninit<T>>,
    producer: Option<Stream>,
}

impl<T: DeviceRepr> DeviceValue<T> {
    /// Adopts fully initialized device memory as a device value with no additional producer readiness stream.
    #[must_use]
    pub fn from_initialized(memory: DeviceMemory<T>) -> Self {
        Self {
            memory: memory.into_uninit(),
            producer: None,
        }
    }

    /// Adopts storage fully initialized by a previously enqueued producer.
    ///
    /// # Safety
    /// The producer on `stream` must write a valid T to every element before consumers read it.
    /// It must retain all other producer resources through completion.
    /// Exclude conflicting accesses outside the submission tracker.
    ///
    /// # Errors
    ///
    /// Returns `Error::StreamContextMismatch` if the memory and producer stream belong to different contexts.
    pub unsafe fn from_pending(
        memory: DeviceMemory<MaybeUninit<T>>,
        stream: &Stream,
    ) -> Result<Self> {
        if memory.context() != stream.context() {
            return Err(Error::StreamContextMismatch);
        }
        Ok(Self {
            memory,
            producer: Some(stream.clone()),
        })
    }

    /// Returns the number of elements in the allocation, whether producer work is pending or complete.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns whether the allocation contains zero elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Completes producing work before exposing initialized device values.
    ///
    /// # Errors
    ///
    /// Propagates errors from synchronizing the declared region, including producer readiness, conflicting tracked work, context validation, or failed access tracking. No initialized memory wrapper is returned on failure.
    pub fn complete(self) -> Result<DeviceMemory<T>> {
        self.region().synchronize()?;
        // SAFETY: construction establishes full value validity; synchronization establishes completion.
        Ok(unsafe { self.memory.assume_init() })
    }

    /// Allocates pinned readback staging before submission or capture.
    ///
    /// # Errors
    ///
    /// Propagates errors from allocating pinned host staging through [`Readback::create`]. No transfer is submitted by this method.
    pub fn readback(self) -> Result<Readback<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        Readback::create(self)
    }
}

unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceValue<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let region = self.memory.region();
        self.producer.as_ref().map_or(region, |stream| {
            unsafe { region.with_stream(stream) }.expect("producer context matches storage")
        })
    }
}

/// A prepared host-to-device transfer; allocation and staging happen at construction.
#[derive(Debug)]
pub struct Upload<T: DeviceRepr> {
    staging: PinnedHostMemory<T>,
    destination: DeviceMemory<MaybeUninit<T>>,
}

impl<T: DeviceRepr> Upload<T> {
    /// Copies `values` into owned pinned host staging and allocates an uninitialized device destination in `context`.
    ///
    /// The CUDA transfer is deferred until this operation is submitted or recorded.
    ///
    /// # Errors
    ///
    /// Returns an error if staging or device allocation fails, byte-size arithmetic overflows, the context cannot be bound, or CUDA rejects the allocation or staging initialization.
    pub fn create(context: &Arc<Context>, values: &[T]) -> Result<Self> {
        Ok(Self {
            staging: PinnedHostMemory::from_slice(context, values)?,
            destination: DeviceMemory::<T>::create_uninit(context, values.len())?,
        })
    }

    /// Pending destination storage for binding consumers before capture.
    #[must_use]
    pub fn destination(&self) -> &DeviceMemory<MaybeUninit<T>> {
        &self.destination
    }
}

// SAFETY: both DMA endpoints are retained before enqueue; the output retains pending validity only.
unsafe impl<T: DeviceRepr + Send + Sync> Operation for Upload<T> {
    type Pending = Self::Output;

    type Output = DeviceValue<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain_pinned(&self.staging, AccessMode::Read);
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.staging.as_ptr(),
                    self.destination.len(),
                    MemoryCopyKind::HostToDevice,
                    stream,
                )
            })?;
            DeviceValue::from_pending(self.destination, execution.stream())
        }
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<T: DeviceRepr + Send + Sync> DeviceOperation for Upload<T> {}

/// Readback returns initialized host values only after successful DMA completion.
#[derive(Debug)]
pub struct Readback<T: AnyBitPatternDeviceRepr> {
    source: Arc<DeviceValue<T>>,
    staging: PinnedHostMemory<MaybeUninit<T>>,
}

impl<T: AnyBitPatternDeviceRepr> Readback<T> {
    /// Retains `source` and allocates pinned host staging for a deferred device-to-host transfer.
    ///
    /// Submission orders the transfer after the source's producer. Host values become available only after successful completion and resolution.
    ///
    /// # Errors
    ///
    /// Returns an error if pinned staging allocation fails, the byte count overflows, the source context cannot be bound, or CUDA rejects the allocation.
    pub fn create(source: impl Into<Arc<DeviceValue<T>>>) -> Result<Self> {
        let source = source.into();
        let staging = PinnedHostMemory::<T>::create_uninit(
            source.memory.context(),
            source.len(),
            HostAllocationFlags::DEFAULT,
        )?;
        Ok(Self { source, staging })
    }
}

unsafe impl<T: AnyBitPatternDeviceRepr + Send + Sync> Operation for Readback<T> {
    type Pending = PinnedHostMemory<MaybeUninit<T>>;

    type Output = Vec<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Pending> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain_pinned(&self.staging, AccessMode::Write);
        resources.access(self.source.region(), AccessMode::Read);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    self.staging.as_ptr().cast_mut().cast(),
                    self.source.memory.as_ptr().cast(),
                    self.source.len(),
                    MemoryCopyKind::DeviceToHost,
                    stream,
                )
            })?;
        }
        Ok(self.staging)
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        // SAFETY: DMA completed successfully and the source contains valid values.
        Ok(unsafe { pending.assume_init() }.as_slice().to_vec())
    }
}

/// A prepared device copy preserving pending values and retaining both allocations.
#[derive(Debug)]
pub struct CopyDevice<T: DeviceRepr> {
    source: DeviceValue<T>,
    destination: DeviceMemory<MaybeUninit<T>>,
}

impl<T: DeviceRepr> CopyDevice<T> {
    /// Consumes `source` and allocates an equally sized device destination in the same context.
    ///
    /// The copy is deferred until submission or recording, and the result remains a pending device value.
    ///
    /// # Errors
    ///
    /// Returns an error if device allocation fails, byte-size arithmetic overflows, the source context cannot be bound, or CUDA rejects the allocation.
    pub fn create(source: DeviceValue<T>) -> Result<Self> {
        let destination = DeviceMemory::<T>::create_uninit(source.memory.context(), source.len())?;
        Ok(Self {
            source,
            destination,
        })
    }

    /// Pending destination storage for binding consumers before capture.
    #[must_use]
    pub fn destination(&self) -> &DeviceMemory<MaybeUninit<T>> {
        &self.destination
    }
}

unsafe impl<T: DeviceRepr + Send + Sync> Operation for CopyDevice<T> {
    type Pending = Self::Output;

    type Output = DeviceValue<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.source.region(), AccessMode::Read);
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.source.memory.as_ptr().cast(),
                    self.source.len(),
                    MemoryCopyKind::DeviceToDevice,
                    stream,
                )
            })?;
            DeviceValue::from_pending(self.destination, execution.stream())
        }
    }

    unsafe fn resolve(pending: Self::Pending) -> Result<Self::Output> {
        Ok(pending)
    }
}

// SAFETY: enqueue returns pending device metadata and resolution is the identity function.
unsafe impl<T: DeviceRepr + Send + Sync> DeviceOperation for CopyDevice<T> {}

unsafe impl<T: DeviceRepr + Send + Sync> Recordable for Upload<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain_pinned(&self.staging, AccessMode::Read);
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
        unsafe {
            recording.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.staging.as_ptr(),
                    self.destination.len(),
                    MemoryCopyKind::HostToDevice,
                    stream,
                )
            })?;
        }
        Ok(self.destination)
    }
}

// SAFETY: capture retains both endpoints and returns pending storage without claiming execution.
unsafe impl<T: DeviceRepr + Send + Sync> Recordable for CopyDevice<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.source.region(), AccessMode::Read);
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: source validity follows DeviceValue construction and replay orders its producer.
        unsafe {
            recording.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    self.destination.as_ptr().cast_mut().cast(),
                    self.source.memory.as_ptr().cast(),
                    self.source.len(),
                    MemoryCopyKind::DeviceToDevice,
                    stream,
                )
            })?;
        }
        Ok(self.destination)
    }
}
