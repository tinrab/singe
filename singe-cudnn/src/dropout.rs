use std::{mem, ptr, sync::Arc};

use singe_cuda::context::Context as CudaContext;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::DataTypeLike,
    error::{Error, Result},
    tensor::TensorDescriptor,
    try_ffi,
    utility::{check_range, to_usize},
};

#[derive(Debug)]
pub struct DropoutDescriptor {
    handle: sys::cudnnDropoutDescriptor_t,
    cuda_ctx: Arc<CudaContext>,
    rate: f32,
    seed: u64,
    states: DeviceMemory<u8>,
}

impl DropoutDescriptor {
    /// Returns the state storage size required by the random number generators used by [`dropout_forward`].
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, cuDNN cannot query
    /// the required state size, or the reported byte count cannot be
    /// represented as `usize`.
    pub fn state_size(ctx: &Context) -> Result<usize> {
        ctx.bind()?;

        let mut size_in_bytes = 0;
        unsafe {
            try_ffi!(sys::cudnnDropoutGetStatesSize(
                ctx.as_raw(),
                &raw mut size_in_bytes,
            ))?;
        }

        to_usize(size_in_bytes, "dropout state size")
    }

    pub fn create(ctx: &Context, rate: f32, seed: u64) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(Error::OutOfRange {
                name: "dropout rate".into(),
            });
        }

        ctx.bind()?;

        let state_size = Self::state_size(ctx)?;
        let states = DeviceMemory::<u8>::create(state_size.max(1))?;
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateDropoutDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnSetDropoutDescriptor(
                handle,
                ctx.as_raw(),
                rate,
                states.as_mut_ptr() as _,
                state_size as _,
                seed,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            rate,
            seed,
            states,
        })
    }

    pub fn rate(&self) -> f32 {
        self.rate
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn states(&self) -> &DeviceMemory<u8> {
        &self.states
    }

    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.cuda_ctx
    }

    pub(crate) fn ensure_context(&self, ctx: &Context, name: &str) -> Result<()> {
        if self.cuda_context().as_ref() != ctx.cuda_context().as_ref() {
            return Err(Error::ContextMismatch { name: name.into() });
        }
        Ok(())
    }

    /// Takes ownership of a raw cuDNN dropout descriptor and its state storage.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cudnnDropoutDescriptor_t` that is not owned by
    /// any other wrapper, must be associated with `ctx`, and must have been
    /// initialized with `states`. The state allocation must remain valid for
    /// the lifetime of the descriptor.
    pub unsafe fn from_raw(
        handle: sys::cudnnDropoutDescriptor_t,
        ctx: &Context,
        rate: f32,
        seed: u64,
        states: DeviceMemory<u8>,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            rate,
            seed,
            states,
        })
    }

    /// Returns the fields of an initialized dropout descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if the descriptor or context is invalid.
    pub fn query(ctx: &Context, descriptor: &Self) -> Result<(f32, Option<*mut ()>, u64)> {
        descriptor.ensure_context(ctx, "dropout query")?;
        ctx.bind()?;

        let mut rate = 0.0;
        let mut states = ptr::null_mut();
        let mut seed = 0;
        unsafe {
            try_ffi!(sys::cudnnGetDropoutDescriptor(
                descriptor.handle,
                ctx.as_raw(),
                &raw mut rate,
                &raw mut states,
                &raw mut seed,
            ))?;
        }

        Ok((rate, (!states.is_null()).then_some(states.cast()), seed))
    }

    pub fn restore(ctx: &Context, rate: f32, seed: u64, states: DeviceMemory<u8>) -> Result<Self> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(Error::OutOfRange {
                name: "dropout rate".into(),
            });
        }

        ctx.bind()?;

        let mut handle = ptr::null_mut();
        let states = states;
        unsafe {
            try_ffi!(sys::cudnnCreateDropoutDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnRestoreDropoutDescriptor(
                handle,
                ctx.as_raw(),
                rate,
                states.as_mut_ptr() as _,
                states.byte_len() as _,
                seed,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self {
            handle,
            cuda_ctx: Arc::clone(ctx.cuda_context()),
            rate,
            seed,
            states,
        })
    }

    /// Returns the reserve space size required to run dropout for `tensor_descriptor`.
    ///
    /// The same reserve space must be passed to [`dropout_forward`] and
    /// [`dropout_backward`], and its contents must remain unchanged between those calls.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects `tensor_descriptor` or the reported
    /// byte count cannot be represented as `usize`.
    pub fn reserve_space_size<T>(tensor_descriptor: &TensorDescriptor<T>) -> Result<usize> {
        let mut size_in_bytes = 0;
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
    /// The returned descriptor is borrowed and remains valid only while this
    /// wrapper is alive.
    pub fn as_raw(&self) -> sys::cudnnDropoutDescriptor_t {
        self.handle
    }

    /// Transfers ownership of the raw cuDNN dropout descriptor and state
    /// storage to the caller.
    ///
    /// The caller becomes responsible for destroying the descriptor with
    /// `cudnnDestroyDropoutDescriptor` and preserving the returned state
    /// allocation for as long as the descriptor may be used.
    pub fn into_raw_parts(self) -> (sys::cudnnDropoutDescriptor_t, DeviceMemory<u8>) {
        let handle = self.handle;
        let states = unsafe { ptr::read(&self.states) };
        mem::forget(self);
        (handle, states)
    }
}

impl Drop for DropoutDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        if let Err(err) = self.cuda_ctx.bind() {
            #[cfg(debug_assertions)]
            eprintln!(
                "failed to bind cuda context before destroying cudnn dropout descriptor: {err}"
            );
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroyDropoutDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn dropout descriptor: {err}");
            }
        }
    }
}

#[derive(Debug)]
pub struct DropoutReserveSpace {
    storage: DeviceMemory<u8>,
}

impl DropoutReserveSpace {
    pub fn create<T>(tensor_descriptor: &TensorDescriptor<T>) -> Result<Self> {
        let size = DropoutDescriptor::reserve_space_size(tensor_descriptor)?;
        let storage = DeviceMemory::<u8>::create(size.max(1))?;
        Ok(Self { storage })
    }

    pub fn byte_len(&self) -> usize {
        self.storage.byte_len()
    }

    fn as_mut_ptr(&self) -> *mut u8 {
        self.storage.as_mut_ptr()
    }
}

/// Performs forward dropout over `x`, returning results in `y`.
/// If dropout is configured in `dropout_descriptor`, approximately the `rate`
/// fraction of `x` values are replaced by `0`, and the rest are scaled by
/// `1 / (1 - rate)`.
/// Do not run this concurrently with another [`dropout_forward`] call using the same states.
///
/// # Errors
///
/// Returns an error if the input and output tensors are incompatible, if an
/// in-place call uses mismatched strides, if `reserve_space` is smaller than
/// [`DropoutDescriptor::reserve_space_size`], if `dropout_descriptor` was not
/// initialized with valid state storage, if the operation fails to launch on the
/// GPU, or if cuDNN does not support the provided configuration.
pub fn dropout_forward<T: DataTypeLike>(
    ctx: &Context,
    dropout_descriptor: &DropoutDescriptor,
    x_descriptor: &TensorDescriptor<T>,
    x: &DeviceMemory<T>,
    y_descriptor: &TensorDescriptor<T>,
    y: &mut DeviceMemory<T>,
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    dropout_descriptor.ensure_context(ctx, "dropout forward")?;
    ctx.bind()?;

    if x_descriptor.dimensions() != y_descriptor.dimensions() {
        return Err(Error::DescriptorMismatch {
            name: "dropout forward".into(),
        });
    }

    let required_reserve_size = DropoutDescriptor::reserve_space_size(x_descriptor)?;
    if reserve_space.byte_len() < required_reserve_size {
        return Err(Error::InsufficientWorkspaceSize {
            required: required_reserve_size,
            actual: reserve_space.byte_len(),
        });
    }

    unsafe {
        try_ffi!(sys::cudnnDropoutForward(
            ctx.as_raw(),
            dropout_descriptor.as_raw(),
            x_descriptor.as_raw(),
            x.as_ptr() as _,
            y_descriptor.as_raw(),
            y.as_mut_ptr() as _,
            reserve_space.as_mut_ptr() as _,
            reserve_space.byte_len() as _,
        ))?;
    }

    Ok(())
}

/// Performs backward dropout over `dy`, returning results in `dx`.
/// If a value from `x` was propagated to `y` during forward dropout, the corresponding
/// value from `dy` is propagated to `dx`; otherwise, `dx` is set to `0`.
///
/// Better performance is obtained for fully packed tensors.
///
/// # Errors
///
/// Returns an error if the input and output gradients are incompatible, if an
/// in-place call uses mismatched strides, if `reserve_space` is smaller than
/// [`DropoutDescriptor::reserve_space_size`], if `dropout_descriptor` was not
/// initialized with valid state storage, if the operation fails to launch on the
/// GPU, or if cuDNN does not support the provided configuration.
pub fn dropout_backward<T: DataTypeLike>(
    ctx: &Context,
    dropout_descriptor: &DropoutDescriptor,
    output_gradient_descriptor: &TensorDescriptor<T>,
    output_gradient: &DeviceMemory<T>,
    input_gradient_descriptor: &TensorDescriptor<T>,
    input_gradient: &mut DeviceMemory<T>,
    reserve_space: &mut DropoutReserveSpace,
) -> Result<()> {
    dropout_descriptor.ensure_context(ctx, "dropout backward")?;
    ctx.bind()?;

    check_range!(
        "dropout backward",
        output_gradient_descriptor.dimensions() == input_gradient_descriptor.dimensions()
    )?;

    let required_reserve_size = DropoutDescriptor::reserve_space_size(output_gradient_descriptor)?;
    if reserve_space.byte_len() < required_reserve_size {
        return Err(Error::InsufficientWorkspaceSize {
            required: required_reserve_size,
            actual: reserve_space.byte_len(),
        });
    }

    unsafe {
        try_ffi!(sys::cudnnDropoutBackward(
            ctx.as_raw(),
            dropout_descriptor.as_raw(),
            output_gradient_descriptor.as_raw(),
            output_gradient.as_ptr() as _,
            input_gradient_descriptor.as_raw(),
            input_gradient.as_mut_ptr() as _,
            reserve_space.as_mut_ptr() as _,
            reserve_space.byte_len() as _,
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    use crate::testing::setup_context;

    #[test]
    fn test_dropout_descriptor_create() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = DropoutDescriptor::create(&test_context, 0.25, 1234)?;

        assert_eq!(descriptor.rate(), 0.25);
        assert_eq!(descriptor.seed(), 1234);
        assert!(descriptor.states().byte_len() > 0);

        Ok(())
    }

    #[test]
    fn test_dropout_reserve_space_create() -> Result<()> {
        let tensor_descriptor = TensorDescriptor::<f32>::create_contiguous(&[2, 4])?;
        let reserve_space = DropoutReserveSpace::create(&tensor_descriptor)?;

        assert!(reserve_space.byte_len() > 0);

        Ok(())
    }

    #[test]
    fn test_dropout_forward_backward_smoke() -> Result<()> {
        let test_context = setup_context()?;

        let shape = [1, 1, 1, 8];
        let count = shape
            .iter()
            .map(|&dimension| dimension as usize)
            .product::<usize>();
        let tensor_descriptor = TensorDescriptor::<f32>::create_contiguous(&shape)?;
        let dropout = DropoutDescriptor::create(&test_context, 0.5, 7)?;
        let mut reserve_space = DropoutReserveSpace::create(&tensor_descriptor)?;

        let x_host = vec![1.0f32; count];
        let dy_host = vec![1.0f32; count];

        let mut x = DeviceMemory::<f32>::create(count)?;
        let mut y = DeviceMemory::<f32>::create(count)?;
        let mut dy = DeviceMemory::<f32>::create(count)?;
        let mut dx = DeviceMemory::<f32>::create(count)?;

        x.copy_from_host(&x_host)?;
        dy.copy_from_host(&dy_host)?;

        dropout_forward(
            &test_context,
            &dropout,
            &tensor_descriptor,
            &x,
            &tensor_descriptor,
            &mut y,
            &mut reserve_space,
        )?;

        dropout_backward(
            &test_context,
            &dropout,
            &tensor_descriptor,
            &dy,
            &tensor_descriptor,
            &mut dx,
            &mut reserve_space,
        )?;

        test_context.stream().synchronize()?;

        let mut dx_host = vec![0.0f32; count];
        dx.copy_to_host(&mut dx_host)?;
        assert_eq!(dx_host.len(), count);

        Ok(())
    }

    #[test]
    fn test_dropout_descriptor_restore() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = DropoutDescriptor::create(&test_context, 0.25, 1234)?;
        let states = DeviceMemory::<u8>::create(descriptor.states().byte_len())?;
        let restored = DropoutDescriptor::restore(&test_context, 0.25, 1234, states)?;

        assert_eq!(restored.rate(), 0.25);
        assert_eq!(restored.seed(), 1234);
        assert_eq!(restored.states().byte_len(), descriptor.states().byte_len());

        Ok(())
    }
}
