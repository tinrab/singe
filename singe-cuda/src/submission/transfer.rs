//! Transfers with owned pinned staging and explicit pending device values.

use std::{mem::MaybeUninit, sync::Arc};

use crate::{
    context::Context,
    error::{Error, Result},
    graph::owned::{Recordable, Recording},
    memory::{DeviceMemory, MemoryCopyKind, PinnedVec},
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
    ///
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
    /// Propagates errors from synchronizing the declared region, including producer readiness, conflicting tracked work, context validation, or failed access tracking.
    /// No initialized memory wrapper is returned on failure.
    pub fn complete(self) -> Result<DeviceMemory<T>> {
        self.region().synchronize()?;
        // SAFETY: `from_initialized` or the caller of `from_pending` guarantees every element is written with a valid value, and synchronizing the region completed that producer.
        Ok(unsafe { self.memory.assume_init() })
    }

    /// Allocates pinned readback staging before submission or capture.
    ///
    /// # Errors
    ///
    /// Propagates errors from allocating pinned host staging through [`Readback::create`].
    /// No transfer is submitted by this method.
    pub fn readback(self) -> Result<Readback<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        Readback::create(self)
    }
}

// SAFETY: the region is the owned allocation's own region, optionally tied to the producer stream, so it describes the same live memory and context.
unsafe impl<T: DeviceRepr> DeviceRegionProvider for DeviceValue<T> {
    fn region(&self) -> DeviceRegion<'_> {
        let region = self.memory.region();
        self.producer.as_ref().map_or(region, |stream| {
            // SAFETY: `from_pending`'s caller guarantees the producer writes are ordered on `stream`, the stream that establishes readiness.
            unsafe { region.with_stream(stream) }.expect("producer context matches storage")
        })
    }
}

/// A prepared host-to-device transfer; allocation and staging happen at construction.
#[derive(Debug)]
pub struct Upload<T: DeviceRepr> {
    staging: Arc<PinnedVec<T>>,
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
            staging: Arc::new(context.pinned_host().from_slice(values)?),
            destination: DeviceMemory::<T>::create_uninit(context, values.len())?,
        })
    }

    /// Pending destination storage for binding consumers before capture.
    #[must_use]
    pub fn destination(&self) -> &DeviceMemory<MaybeUninit<T>> {
        &self.destination
    }
}

// SAFETY: `enqueue` registers both DMA endpoints and the destination write before copying, and the output is a pending `DeviceValue` that exposes no initialized borrow.
unsafe impl<T: DeviceRepr + Send + Sync> Operation for Upload<T> {
    type Pending = Self::Output;

    type Output = DeviceValue<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain_pinned(&self.staging, AccessMode::Read);
        resources.retain(Arc::clone(self.destination.allocation()));
        // SAFETY: `resources` retains the staging buffer and destination and declares the destination write, the copy runs only on the execution's stream, and both endpoints hold `self.destination.len()` elements.
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

// SAFETY: `Pending` is `Output`, `resolve` returns it unchanged, and `enqueue` schedules the copy without host resolution.
unsafe impl<T: DeviceRepr + Send + Sync> DeviceOperation for Upload<T> {}

/// Readback returns initialized host values only after successful DMA completion.
#[derive(Debug)]
pub struct Readback<T: AnyBitPatternDeviceRepr> {
    source: Arc<DeviceValue<T>>,
    staging: PinnedVec<T>,
}

impl<T: AnyBitPatternDeviceRepr> Readback<T> {
    /// Retains `source` and allocates pinned host staging for a deferred device-to-host transfer.
    ///
    /// Submission orders the transfer after the source's producer.
    /// Host values become available only after successful completion and resolution.
    ///
    /// # Errors
    ///
    /// Returns an error if pinned staging allocation fails, the byte count overflows, the source context cannot be bound, or CUDA rejects the allocation.
    pub fn create(source: impl Into<Arc<DeviceValue<T>>>) -> Result<Self> {
        let source = source.into();
        let staging = source
            .memory
            .context()
            .pinned_host()
            .with_capacity(source.len())?;
        Ok(Self { source, staging })
    }
}

// SAFETY: `enqueue` registers the source, its producer's region, and the staging write before copying, and the pending staging vector stays empty until resolution sets its length.
unsafe impl<T: AnyBitPatternDeviceRepr + Send + Sync> Operation for Readback<T> {
    type Pending = (Arc<PinnedVec<T>>, usize);

    type Output = PinnedVec<T>;

    fn enqueue(mut self, execution: &mut Execution) -> Result<Self::Pending> {
        // Moving the vector into its `Arc` leaves its buffer in place.
        let destination = self.staging.as_mut_ptr();
        let staging = Arc::new(self.staging);
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain_pinned(&staging, AccessMode::Write);
        resources.access(self.source.region(), AccessMode::Read);
        // SAFETY: `resources` retains the source allocation and staging and declares their accesses, `destination` has capacity for `self.source.len()` elements, and the copy runs only on the execution's stream.
        unsafe {
            execution.enqueue(resources, |stream| {
                DeviceMemory::<T>::copy_async(
                    destination,
                    self.source.memory.as_ptr().cast(),
                    self.source.len(),
                    MemoryCopyKind::DeviceToHost,
                    stream,
                )
            })?;
        }
        Ok((staging, self.source.len()))
    }

    unsafe fn resolve((staging, length): Self::Pending) -> Result<Self::Output> {
        // Retirement dropped the ledger's clone before resolution.
        let mut staging =
            Arc::into_inner(staging).ok_or_else(|| Error::ownership_shared("staging buffer"))?;
        // SAFETY: the copy filled `length` values, the capacity `with_capacity` reserved, and any bytes are valid values.
        unsafe { staging.set_len(length) };
        Ok(staging)
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

// SAFETY: `enqueue` registers both allocations and their read and write accesses before copying, and the output is a pending `DeviceValue`.
unsafe impl<T: DeviceRepr + Send + Sync> Operation for CopyDevice<T> {
    type Pending = Self::Output;

    type Output = DeviceValue<T>;

    fn enqueue(self, execution: &mut Execution) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.source.region(), AccessMode::Read);
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: `resources` retains both allocations and declares their accesses, which orders the copy after the source's producer, and `create` sized the destination to the source's length.
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

// SAFETY: `Pending` is `Output`, `resolve` returns it unchanged, and `enqueue` schedules the copy without host resolution.
unsafe impl<T: DeviceRepr + Send + Sync> DeviceOperation for CopyDevice<T> {}

// SAFETY: `record` registers the staging buffer, destination, and destination write before recording the copy, and returns the destination as uninitialized storage.
unsafe impl<T: DeviceRepr + Send + Sync> Recordable for Upload<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain_pinned(&self.staging, AccessMode::Read);
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: the recording retains the staging buffer and destination for every replay, the copy is recorded on the capture stream, and both endpoints hold `self.destination.len()` elements.
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

// SAFETY: `record` registers both allocations and their accesses before recording the copy, and returns the destination as uninitialized storage.
unsafe impl<T: DeviceRepr + Send + Sync> Recordable for CopyDevice<T> {
    type Output = DeviceMemory<MaybeUninit<T>>;

    fn record(self, recording: &mut Recording) -> Result<Self::Output> {
        let mut resources = Resources::new();
        resources.retain(Arc::clone(self.source.memory.allocation()));
        resources.retain(Arc::clone(self.destination.allocation()));
        resources.access(self.source.region(), AccessMode::Read);
        resources.access(self.destination.region(), AccessMode::Write);
        // SAFETY: the recording retains both equally sized allocations for every replay and declares their accesses, the copy is recorded on the capture stream, and `DeviceValue` construction guarantees the source's validity once its producer runs.
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
