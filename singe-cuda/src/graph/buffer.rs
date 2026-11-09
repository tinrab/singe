use std::{mem::MaybeUninit, result::Result as StdResult, sync::Arc};

use crate::{
    context::Context,
    error::Result,
    memory::{DeviceAllocation, DeviceMemory, transfer},
    view::{
        AnyBitPatternDeviceRepr, DeviceOutputView, DeviceRepr, DeviceSlice, DeviceView,
        DeviceViewMut, NoUninitDeviceRepr,
    },
};

use super::{GraphBuffer, RetainedAllocation};

impl<T> GraphBuffer<T>
where
    T: DeviceRepr,
{
    pub(crate) fn from_memory(memory: DeviceMemory<T>) -> Self {
        Self { memory }
    }

    pub(crate) fn retained(&self) -> RetainedAllocation {
        let memory = Arc::clone(self.memory.allocation());
        RetainedAllocation(memory)
    }

    /// Returns the number of exposed elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Returns whether the exposed element range is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Returns the byte length of the exposed element range.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.memory.byte_len()
    }

    /// Returns the context retained by the buffer allocation.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.memory.context()
    }

    /// Returns the CUDA-accessible element pointer without synchronizing graph work.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.memory.as_ptr()
    }

    /// Returns the CUDA-accessible mutable element pointer.
    ///
    /// The caller must order raw accesses against graph launches using this allocation.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.as_mut_ptr()
    }

    /// Copies a host slice into this graph-retained device buffer.
    ///
    /// This updates the stable allocation used by graph-buffer node APIs.
    /// The caller is still responsible for ordering this copy against graph launches that read or write the same allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if `host_slice` does not have the same length as this buffer or if CUDA rejects the copy.
    pub fn copy_from_host(&mut self, host_slice: &[T]) -> Result<()> {
        transfer::copy_from_host(self, host_slice)
    }

    /// Copies this graph-retained device buffer into a host slice.
    ///
    /// # Errors
    ///
    /// Returns an error if `host_slice` does not have the same length as this buffer or if CUDA rejects the copy.
    pub fn copy_to_host(&self, host_slice: &mut [T]) -> Result<()>
    where
        T: AnyBitPatternDeviceRepr,
    {
        let () = T::LAYOUT;
        // SAFETY: `T: AnyBitPatternDeviceRepr`, so any device bytes are valid `T` values.
        unsafe { self.copy_to_host_unchecked(host_slice) }
    }

    /// Copies graph-buffer values whose validity follows from the graph's kernels.
    ///
    /// # Safety
    ///
    /// Every copied element must be initialized and valid for `T`.
    /// The copy must be ordered after all graph writes to this buffer.
    ///
    /// # Errors
    ///
    /// Returns an error for unequal element counts or byte-size overflow.
    /// Also propagates context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    pub unsafe fn copy_to_host_unchecked(&self, host_slice: &mut [T]) -> Result<()> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.memory.copy_to_host_unchecked(host_slice) }
    }

    /// Copies another graph-retained buffer into this buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffers have different lengths or if CUDA rejects the copy.
    pub fn copy_from_buffer<S: DeviceSlice<T> + ?Sized>(&mut self, src: &S) -> Result<()> {
        transfer::copy_device(self, src)
    }

    /// Copies every element into a new host vector and waits for completion.
    ///
    /// # Errors
    ///
    /// Propagates byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// No initialized vector is returned unless the transfer completes.
    pub fn copy_to_host_vec(&self) -> Result<Vec<T>>
    where
        T: AnyBitPatternDeviceRepr,
    {
        self.memory.copy_to_host_vec()
    }

    /// Reads graph output values whose validity follows from the producing kernels.
    ///
    /// # Safety
    ///
    /// All elements must be initialized and valid for `T`, and graph writes must finish before this readback.
    ///
    /// # Errors
    ///
    /// Propagates byte-size, context-binding, stream-capture, CUDA copy, and stream synchronization errors.
    /// No initialized vector is returned unless the transfer completes.
    pub unsafe fn copy_to_host_vec_unchecked(&self) -> Result<Vec<T>> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.memory.copy_to_host_vec_unchecked() }
    }

    /// Returns the allocation retained by graph definitions, executables, and foreign owners.
    #[must_use]
    pub fn allocation(&self) -> &Arc<DeviceAllocation> {
        self.memory.allocation()
    }

    /// Borrows the initialized buffer as a shared device view.
    #[must_use]
    pub fn view(&self) -> DeviceView<'_, T> {
        DeviceView::from_buffer(self)
    }

    /// Borrows the initialized buffer as an exclusive device view.
    pub fn view_mut(&mut self) -> DeviceViewMut<'_, T> {
        DeviceViewMut::from_buffer(self)
    }

    /// Reinterprets initialized bytes without changing graph-retained allocation ownership.
    ///
    /// # Errors
    ///
    /// Returns the original buffer if the destination type is zero-sized, its size does not divide the exposed byte length, or a nonempty range is not aligned for the destination type.
    pub fn cast<U: AnyBitPatternDeviceRepr>(self) -> StdResult<GraphBuffer<U>, Self>
    where
        T: NoUninitDeviceRepr,
    {
        self.memory
            .cast()
            .map(GraphBuffer::from_memory)
            .map_err(Self::from_memory)
    }

    /// Erases initialization while preserving graph-retained allocation ownership.
    #[must_use]
    pub fn into_uninit(self) -> GraphBuffer<MaybeUninit<T>> {
        GraphBuffer::from_memory(self.memory.into_uninit())
    }
}

impl<T: DeviceRepr> GraphBuffer<MaybeUninit<T>> {
    /// Borrows the uninitialized buffer as a write-only destination of `T` values.
    pub fn output(&mut self) -> DeviceOutputView<'_, T> {
        DeviceOutputView::from_output(self)
    }

    /// Marks all values initialized after their producing work completes.
    ///
    /// # Safety
    /// Every element must be valid, and producing work must complete or be ordered before every consumer.
    /// Recording a graph does not establish initialization; only its successful execution can do so.
    #[must_use]
    pub unsafe fn assume_init(self) -> GraphBuffer<T> {
        // SAFETY: the caller upholds this function's safety contract.
        GraphBuffer::from_memory(unsafe { self.memory.assume_init() })
    }

    /// Exposes only a produced prefix while retaining the full graph allocation.
    ///
    /// # Safety
    /// Every prefix element must be valid, with producer completion and ordering satisfying `assume_init`.
    ///
    /// # Errors
    ///
    /// Returns an error if `length` exceeds the currently exposed element count.
    pub unsafe fn assume_init_prefix(self, length: usize) -> Result<GraphBuffer<T>> {
        // SAFETY: the caller upholds this function's safety contract.
        unsafe { self.memory.assume_init_prefix(length) }.map(GraphBuffer::from_memory)
    }
}
