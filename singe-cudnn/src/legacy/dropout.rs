//! Dropout descriptors retaining RNG state, reserve storage, and forward/backward operations.

use std::{
    mem::{ManuallyDrop, MaybeUninit},
    ptr,
    sync::Arc,
};

use singe_core::report_cleanup_failure;
use singe_cuda::{
    context::Context as CudaContext,
    error::Error as CudaError,
    memory::DeviceMemory,
    view::{DeviceOutput, DeviceRange, DeviceRegionProvider, DeviceSlice},
};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::TensorElement,
    error::{Error, Result},
    legacy::tensor::TensorDescriptor,
    try_ffi,
    utility::{check_range, to_size_t, to_usize},
};

/// Owns a native dropout descriptor, its CUDA context, and opaque random-number-generator state.
#[derive(Debug)]
pub struct DropoutDescriptor {
    handle: sys::cudnnDropoutDescriptor_t,
    cuda_context: Arc<CudaContext>,
    rate: f32,
    seed: u64,
    states: DeviceMemory<MaybeUninit<u8>>,
}

/// Transferred dropout descriptor, retaining its actual context and opaque state storage.
/// Destroy `handle` before releasing the owners, or reclaim it with [`DropoutDescriptor::from_raw`].
/// Queued work must finish before either action.
#[derive(Debug)]
pub struct RawDropoutDescriptor {
    /// Transferred native dropout handle; destroy it before releasing its context and state storage.
    pub handle: sys::cudnnDropoutDescriptor_t,
    /// Actual CUDA context retained for descriptor use and destruction.
    pub cuda_context: Arc<CudaContext>,
    /// Drop probability used to configure the native descriptor.
    pub rate: f32,
    /// Seed used to initialize the retained random-number-generator state.
    pub seed: u64,
    /// Owned opaque RNG allocation referenced by the native descriptor; padding may be uninitialized.
    pub states: DeviceMemory<MaybeUninit<u8>>,
}

// SAFETY: cuDNN dropout descriptors have no thread affinity, and the record moves the handle together with the context and state owners it references.
// Using or destroying the handle stays an exclusive operation of the record's single owner.
unsafe impl Send for RawDropoutDescriptor {}

impl DropoutDescriptor {
    /// Returns the state storage size required by the random number generators used by [`dropout_forward`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuDNN cannot query the required state size, or the reported byte count cannot be represented as `usize`.
    pub fn state_size(context: &Context) -> Result<usize> {
        context.bind()?;

        let mut size_in_bytes = 0;
        // SAFETY: `context` owns a live handle bound above, and `size_in_bytes` is a writable local.
        unsafe {
            try_ffi!(sys::cudnnDropoutGetStatesSize(
                context.as_raw(),
                &raw mut size_in_bytes,
            ))?;
        }

        to_usize(size_in_bytes, "dropout state size")
    }

    /// Allocates a dropout descriptor and RNG storage, initializes the state, and waits for setup to complete.
    ///
    /// # Errors
    ///
    /// Returns an error if the rate is outside `[0, 1]` or is NaN, context binding, state-size querying, or allocation fails, cuDNN rejects descriptor creation or initialization, the returned handle is null, or stream lookup or setup synchronization fails.
    pub fn create(context: &Context, rate: f32, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(Error::out_of_range("dropout rate"));
        }

        context.bind()?;

        let state_size = Self::state_size(context)?;
        let states = DeviceMemory::<u8>::create_uninit(context.cuda_context(), state_size.max(1))?;
        let descriptor = Self::allocate(context, rate, seed, states)?;
        // The descriptor and state owners outlive the completion guard, including native failure and unwinding.
        // State padding remains opaque storage.
        context.sync_scope(|| {
            // SAFETY: `descriptor` owns a fresh handle and at least `state_size` bytes of state storage in this context, and `sync_scope` waits for the initialization before returning.
            unsafe {
                try_ffi!(sys::cudnnSetDropoutDescriptor(
                    descriptor.handle,
                    context.as_raw(),
                    rate,
                    descriptor.states.as_mut_ptr().cast(),
                    to_size_t(state_size, "dropout state size")?,
                    seed,
                ))?;
            }
            Ok::<_, Error>(())
        })?;
        Ok(descriptor)
    }

    fn allocate(
        context: &Context,
        rate: f32,
        seed: u64,
        states: DeviceMemory<MaybeUninit<u8>>,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(Error::out_of_range("dropout rate"));
        }

        if states.context() != context.cuda_context() {
            return Err(CudaError::MemoryContextMismatch.into());
        }

        let required = Self::state_size(context)?;
        if states.byte_len() < required {
            return Err(Error::BufferTooSmall {
                name: "dropout states".into(),
                required,
                actual: states.byte_len(),
            });
        }

        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a live local that cuDNN writes the new descriptor into.
        unsafe {
            try_ffi!(sys::cudnnCreateDropoutDescriptor(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle,
            cuda_context: Arc::clone(context.cuda_context()),
            rate,
            seed,
            states,
        })
    }

    /// Returns the retained drop probability.
    #[must_use]
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Returns the retained RNG initialization seed.
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Borrows opaque RNG storage without claiming that padding is initialized.
    #[must_use]
    pub fn states(&self) -> &DeviceMemory<MaybeUninit<u8>> {
        &self.states
    }

    /// Borrows the actual CUDA context retained with the descriptor and RNG allocation.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_context
    }

    pub(crate) fn ensure_context(&self, context: &Context, name: &str) -> Result<()> {
        if self.cuda_context().as_ref() != context.cuda_context().as_ref() {
            return Err(Error::ContextMismatch {
                name: name.to_owned().into(),
            });
        }
        Ok(())
    }

    /// Takes ownership of a raw cuDNN dropout descriptor and its state storage.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be uniquely owned and configured for `raw.cuda_context`, `raw.rate`, `raw.seed`, and the live allocation in `raw.states`.
    /// All previous work using them must be complete.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null ([`Error::NullHandle`]).
    pub unsafe fn from_raw(raw: RawDropoutDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            cuda_context: raw.cuda_context,
            rate: raw.rate,
            seed: raw.seed,
            states: raw.states,
        })
    }

    /// Returns the fields of an initialized dropout descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor or context is invalid.
    pub fn query(context: &Context, descriptor: &Self) -> Result<(f32, Option<*mut ()>, u64)> {
        descriptor.ensure_context(context, "dropout query")?;
        context.bind()?;

        let mut rate = 0.0;
        let mut states = ptr::null_mut();
        let mut seed = 0;
        // SAFETY: `ensure_context` matched the descriptor to the bound handle, both are owned by live wrappers, and the three outputs are writable locals.
        unsafe {
            try_ffi!(sys::cudnnGetDropoutDescriptor(
                descriptor.handle,
                context.as_raw(),
                &raw mut rate,
                &raw mut states,
                &raw mut seed,
            ))?;
        }

        Ok((rate, (!states.is_null()).then_some(states.cast()), seed))
    }

    /// Restores a descriptor from RNG storage produced by cuDNN.
    ///
    /// # Safety
    ///
    /// `states` must contain RNG state that [`DropoutDescriptor::create`] initialized with the same cuDNN version and seed.
    /// All previous uses of the storage must be complete.
    /// Opaque padding need not be exposed as initialized bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the rate is invalid, the state allocation belongs to another CUDA context, the state is too small, binding the context or querying the state size fails, native descriptor creation or restoration fails, a native handle is null, or looking up or synchronizing the stream fails.
    pub unsafe fn restore(
        context: &Context,
        rate: f32,
        seed: u64,
        states: DeviceMemory<MaybeUninit<u8>>,
    ) -> Result<Self> {
        let descriptor = Self::allocate(context, rate, seed, states)?;
        context.sync_scope(|| {
            // SAFETY: `allocate` checked that the owned state storage is in this context and holds the required size, the caller guarantees it holds RNG state for this seed, and `sync_scope` waits before returning.
            unsafe {
                try_ffi!(sys::cudnnRestoreDropoutDescriptor(
                    descriptor.handle,
                    context.as_raw(),
                    rate,
                    descriptor.states.as_mut_ptr().cast(),
                    to_size_t(descriptor.states.byte_len(), "dropout state size")?,
                    seed,
                ))?;
            }
            Ok::<_, Error>(())
        })?;
        Ok(descriptor)
    }

    /// Returns the reserve space size required to run dropout for `tensor_descriptor`.
    ///
    /// The same reserve space must be passed to [`dropout_forward_async`] and [`dropout_backward_async`], and its contents must remain unchanged between those calls.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects `tensor_descriptor` or the reported byte count cannot be represented as `usize`.
    pub fn reserve_space_size<T>(tensor_descriptor: &TensorDescriptor<T>) -> Result<usize> {
        let mut size_in_bytes = 0;
        // SAFETY: `tensor_descriptor` owns a live descriptor, and `size_in_bytes` is a writable local.
        unsafe {
            try_ffi!(sys::cudnnDropoutGetReserveSpaceSize(
                tensor_descriptor.as_raw(),
                &raw mut size_in_bytes,
            ))?;
        }

        to_usize(size_in_bytes, "dropout reserve space size")
    }

    /// Returns the raw cuDNN dropout descriptor.
    ///
    /// The returned descriptor is borrowed and remains valid only while this wrapper is alive.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnDropoutDescriptor_t {
        self.handle
    }

    /// Transfers native ownership together with the actual context and state storage.
    /// The caller must complete queued work before destroying or reclaiming it.
    #[must_use]
    pub fn into_raw(self) -> RawDropoutDescriptor {
        let descriptor = ManuallyDrop::new(self);
        // SAFETY: `ManuallyDrop` suppresses the source destructor, so each owned field is read exactly once into the record that now owns it.
        unsafe {
            RawDropoutDescriptor {
                handle: descriptor.handle,
                cuda_context: ptr::read(&raw const descriptor.cuda_context),
                rate: descriptor.rate,
                seed: descriptor.seed,
                states: ptr::read(&raw const descriptor.states),
            }
        }
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        if let Err(error) = self.cuda_context.bind() {
            report_cleanup_failure(
                "bind the cuda context to destroy",
                "cudnn dropout descriptor",
                &error,
            );
            return;
        }

        // SAFETY: the handle is non-null and owned by this wrapper, its context is bound, and nothing uses it after drop.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyDropoutDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cudnn dropout descriptor", &error);
            }
        }
    }
}

/// Owns device reserve storage holding the mask needed by a matching dropout forward/backward pair.
#[derive(Debug)]
pub struct DropoutReserveSpace {
    storage: DeviceMemory<MaybeUninit<u8>>,
}

impl DropoutReserveSpace {
    /// Allocates reserve storage for dropout on the supplied tensor descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN cannot determine the reserve size, the size cannot be represented as `usize`, or device allocation fails.
    pub fn create<T>(context: &Context, tensor_descriptor: &TensorDescriptor<T>) -> Result<Self> {
        let size = DropoutDescriptor::reserve_space_size(tensor_descriptor)?;
        let storage = DeviceMemory::<u8>::create_uninit(context.cuda_context(), size.max(1))?;
        Ok(Self { storage })
    }

    /// Returns the allocated reserve capacity in bytes, including the one-byte minimum allocation.
    #[must_use]
    pub fn byte_len(&self) -> usize {
        self.storage.byte_len()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.storage.as_mut_ptr().cast()
    }
}

/// Performs forward dropout over `x`, returning results in `y`.
/// If dropout is configured in `dropout_descriptor`, approximately the `rate` fraction of `x` values are replaced by `0`, and the rest are scaled by `1 / (1 - rate)`.
/// Do not run this concurrently with another [`dropout_forward_async`] call using the same states.
///
/// # Safety
///
/// Retain the context, descriptors, RNG state, reserve storage and tensor ranges through completion, submission failure and graph replay.
/// Order allocation readiness and input producers before this call; inputs must be valid when read.
/// Writes must remain exclusive.
/// Forward mutates RNG state; backward requires the unchanged reserve contents from the matching successful forward call.
/// Only descriptor elements are produced; padding and tails remain pending.
///
/// # Errors
///
/// Returns an error if the input and output tensors have different dimensions, if `reserve_space` is smaller than [`DropoutDescriptor::reserve_space_size`], if `dropout_descriptor` was not initialized with valid state storage, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration.
pub unsafe fn dropout_forward_async<T: TensorElement>(
    context: &Context,
    dropout_descriptor: &mut DropoutDescriptor,
    x_descriptor: &TensorDescriptor<T>,
    x: &(impl DeviceRange<T> + ?Sized),
    y_descriptor: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    dropout_descriptor.ensure_context(context, "dropout forward")?;
    context.bind()?;
    context.prepare_regions(&[
        x.region(),
        y.region(),
        reserve_space.storage.region(),
        dropout_descriptor.states.region(),
    ])?;

    check_range!(
        "dropout input storage span",
        x.region().byte_len() >= x_descriptor.byte_len()?
    )?;
    check_range!(
        "dropout output storage span",
        y.region().byte_len() >= y_descriptor.byte_len()?
    )?;

    if x_descriptor.dimensions() != y_descriptor.dimensions() {
        return Err(Error::DescriptorMismatch {
            name: "dropout forward".into(),
        });
    }

    let required_reserve_size = DropoutDescriptor::reserve_space_size(x_descriptor)?;
    if reserve_space.byte_len() < required_reserve_size {
        return Err(Error::BufferTooSmall {
            name: "reserve space".into(),
            required: required_reserve_size,
            actual: reserve_space.byte_len(),
        });
    }

    // SAFETY: the buffers share the handle's context and cover their descriptors, and the reserve holds the size cuDNN requires for `x_descriptor`.
    // The caller retains, orders and initializes the buffers and RNG state through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnDropoutForward(
            context.as_raw(),
            dropout_descriptor.as_raw(),
            x_descriptor.as_raw(),
            x.as_device_ptr().cast(),
            y_descriptor.as_raw(),
            y.as_mut_ptr().cast(),
            reserve_space.as_mut_ptr().cast(),
            to_size_t(reserve_space.byte_len(), "dropout reserve size")?,
        ))?;
    }

    Ok(())
}

/// Performs backward dropout over `dy`, returning results in `dx`.
/// If a value from `x` was propagated to `y` during forward dropout, the corresponding value from `dy` is propagated to `dx`; otherwise, `dx` is set to `0`.
///
/// Better performance is obtained for fully packed tensors.
///
/// # Safety
///
/// Retain the context, descriptors, RNG state, reserve storage and tensor ranges through completion, submission failure and graph replay.
/// Order allocation readiness and input producers before this call; inputs must be valid when read.
/// Writes must remain exclusive.
/// Forward mutates RNG state; backward requires the unchanged reserve contents from the matching successful forward call.
/// Only descriptor elements are produced; padding and tails remain pending.
///
/// # Errors
///
/// Returns an error if the input and output gradients have different dimensions, if `reserve_space` is smaller than [`DropoutDescriptor::reserve_space_size`], if `dropout_descriptor` was not initialized with valid state storage, if the operation fails to launch on the GPU, or if cuDNN does not support the provided configuration.
pub unsafe fn dropout_backward_async<T: TensorElement>(
    context: &Context,
    dropout_descriptor: &DropoutDescriptor,
    output_gradient_descriptor: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceRange<T> + ?Sized),
    input_gradient_descriptor: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    dropout_descriptor.ensure_context(context, "dropout backward")?;
    context.bind()?;
    context.prepare_regions(&[
        output_gradient.region(),
        input_gradient.region(),
        reserve_space.storage.region(),
    ])?;

    check_range!(
        "dropout input storage span",
        output_gradient.region().byte_len() >= output_gradient_descriptor.byte_len()?
    )?;
    check_range!(
        "dropout output storage span",
        input_gradient.region().byte_len() >= input_gradient_descriptor.byte_len()?
    )?;

    if output_gradient_descriptor.dimensions() != input_gradient_descriptor.dimensions() {
        return Err(Error::DescriptorMismatch {
            name: "dropout backward".into(),
        });
    }

    let required_reserve_size = DropoutDescriptor::reserve_space_size(output_gradient_descriptor)?;
    if reserve_space.byte_len() < required_reserve_size {
        return Err(Error::BufferTooSmall {
            name: "reserve space".into(),
            required: required_reserve_size,
            actual: reserve_space.byte_len(),
        });
    }

    // SAFETY: the buffers share the handle's context and cover their descriptors, and the reserve holds the size cuDNN requires.
    // The caller supplies the reserve of the matching forward and retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnDropoutBackward(
            context.as_raw(),
            dropout_descriptor.as_raw(),
            output_gradient_descriptor.as_raw(),
            output_gradient.as_device_ptr().cast(),
            input_gradient_descriptor.as_raw(),
            input_gradient.as_mut_ptr().cast(),
            reserve_space.as_mut_ptr().cast(),
            to_size_t(reserve_space.byte_len(), "dropout reserve size")?,
        ))?;
    }

    Ok(())
}

/// Performs forward dropout over `x` into `y` and waits for it.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`dropout_forward_async`] fails, or the wait fails.
pub fn dropout_forward<T: TensorElement>(
    context: &Context,
    dropout_descriptor: &mut DropoutDescriptor,
    x_descriptor: &TensorDescriptor<T>,
    x: &(impl DeviceSlice<T> + ?Sized),
    y_descriptor: &TensorDescriptor<T>,
    y: &mut (impl DeviceOutput<T> + ?Sized),
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `x` is initialized, the RNG state and reserve storage are borrowed exclusively, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            dropout_forward_async(
                context,
                dropout_descriptor,
                x_descriptor,
                x,
                y_descriptor,
                y,
                reserve_space,
            )
        }
    })
}

/// Performs backward dropout over `output_gradient` into `input_gradient` and waits for it.
///
/// The reserve space must hold the mask of the matching forward call, or the gradients are meaningless.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`dropout_backward_async`] fails, or the wait fails.
pub fn dropout_backward<T: TensorElement>(
    context: &Context,
    dropout_descriptor: &DropoutDescriptor,
    output_gradient_descriptor: &TensorDescriptor<T>,
    output_gradient: &(impl DeviceSlice<T> + ?Sized),
    input_gradient_descriptor: &TensorDescriptor<T>,
    input_gradient: &mut (impl DeviceOutput<T> + ?Sized),
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `output_gradient` is initialized, the reserve bytes are opaque mask storage, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            dropout_backward_async(
                context,
                dropout_descriptor,
                output_gradient_descriptor,
                output_gradient,
                input_gradient_descriptor,
                input_gradient,
                reserve_space,
            )
        }
    })
}

#[cfg(all(test, feature = "testing"))]
mod tests {

    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_dropout_descriptor_create() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = DropoutDescriptor::create(test_context, 0.25, 1234)?;

        assert_eq!(descriptor.rate(), 0.25);
        assert_eq!(descriptor.seed(), 1234);
        assert!(descriptor.states().byte_len() > 0);

        Ok(())
    }

    #[test]
    fn test_dropout_reserve_space_create() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();
        let tensor_descriptor = TensorDescriptor::<f32>::create_contiguous(&[2, 4])?;
        let reserve_space = DropoutReserveSpace::create(test_context, &tensor_descriptor)?;

        assert!(reserve_space.byte_len() > 0);

        Ok(())
    }

    #[test]
    fn test_dropout_forward_backward_smoke() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let shape = [1, 1, 1, 8];
        let count = shape
            .iter()
            .map(|&dimension| usize::try_from(dimension).expect("positive test dimension"))
            .product::<usize>();
        let tensor_descriptor = TensorDescriptor::<f32>::create_contiguous(&shape)?;
        let mut dropout = DropoutDescriptor::create(test_context, 0.5, 7)?;
        let mut reserve_space = DropoutReserveSpace::create(test_context, &tensor_descriptor)?;

        let x_host = vec![1.0f32; count];
        let y_grad_host = vec![1.0f32; count];

        let x = DeviceMemory::from_slice(test_context.cuda_context(), &x_host)?;
        let mut y = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;
        let dy = DeviceMemory::from_slice(test_context.cuda_context(), &y_grad_host)?;
        let mut dx = DeviceMemory::<f32>::create_uninit(test_context.cuda_context(), count)?;

        test.stream().complete(|| -> Result<()> {
            // SAFETY: `x` is initialized, and every owner outlives the wait in `sync_scope`.
            unsafe {
                dropout_forward_async(
                    test_context,
                    &mut dropout,
                    &tensor_descriptor,
                    &x,
                    &tensor_descriptor,
                    &mut y,
                    &mut reserve_space,
                )?;
            }

            // SAFETY: the matching forward wrote `reserve_space` earlier on the same stream, `dy` is initialized, and every owner outlives the wait.
            unsafe {
                dropout_backward_async(
                    test_context,
                    &dropout,
                    &tensor_descriptor,
                    &dy,
                    &tensor_descriptor,
                    &mut dx,
                    &mut reserve_space,
                )?;
            }

            Ok(())
        })?;

        let mut x_grad_host = vec![0.0f32; count];
        // SAFETY: the completed operation wrote every contiguous descriptor element.
        let dx = unsafe { dx.assume_init() };
        dx.copy_to_host(&mut x_grad_host)?;
        assert_eq!(x_grad_host.len(), count);

        Ok(())
    }

    #[test]
    fn test_dropout_descriptor_restore() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = DropoutDescriptor::create(test_context, 0.25, 1234)?;
        let state_size = descriptor.states().byte_len();
        let raw = descriptor.into_raw();
        // SAFETY: `into_raw` transferred the handle, and no queued work uses it.
        unsafe {
            try_ffi!(sys::cudnnDestroyDropoutDescriptor(raw.handle))?;
        }
        // SAFETY: `create` initialized `raw.states` with this seed in this process and waited for it.
        let restored =
            unsafe { DropoutDescriptor::restore(test_context, raw.rate, raw.seed, raw.states)? };

        assert_eq!(restored.rate(), 0.25);
        assert_eq!(restored.seed(), 1234);
        assert_eq!(restored.states().byte_len(), state_size);

        Ok(())
    }
}
