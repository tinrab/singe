use std::{marker::PhantomData, mem::MaybeUninit};

use singe_cuda::{
    memory::DeviceMemory,
    stream::{Stream, StreamBinding},
    view::{DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegionProvider, DeviceSlice},
};
use singe_cudnn_sys as sys;

use crate::{
    causal_convolution::{CausalConvolutionShape, CausalScalar, aligned, require},
    error::{Error, Result},
    try_ffi,
    utility::{to_size_t, to_usize},
};

/// Required device storage for long FFT forward and backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LongFftBufferSizes {
    /// Required temporary device workspace capacity in bytes.
    pub workspace_bytes: usize,
    /// Required saved forward-state capacity in bytes, preserved for matching backward execution.
    pub reserve_bytes: usize,
}

/// Queries scratch and saved-state sizes for the long FFT path.
///
/// # Errors
///
/// Returns an error if a long FFT dimension is invalid or the native size query fails.
pub fn long_fft_buffer_sizes<T: CausalScalar>(
    shape: CausalConvolutionShape,
) -> Result<LongFftBufferSizes> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    shape.fft::<T>(true)?;
    let (mut workspace, mut reserve) = (0, 0);
    // SAFETY: the query reads only the dimensions that `fft` accepted and writes the two sizes into the live locals `workspace` and `reserve`.
    unsafe {
        try_ffi!(sys::cudnnLongFFTCausalConv1dGetBufferSizes(
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            &raw mut workspace,
            &raw mut reserve,
        ))?;
    }
    Ok(LongFftBufferSizes {
        workspace_bytes: to_usize(workspace, "workspace")?,
        reserve_bytes: to_usize(reserve, "reserve")?,
    })
}

/// Borrowed workspace and reserve storage for direct long FFT calls.
/// Storage may come from owned buffers, subranges, or externally managed device views.
/// Reserve bytes remain opaque and must be preserved between matching forward and backward calls.
#[derive(Debug)]
pub struct LongFftBuffers<'a> {
    /// Exclusive temporary device scratch storage with the required capacity and 16-byte alignment.
    pub workspace: DeviceOutputView<'a, u8>,
    /// Exclusive opaque forward-state storage, preserved unchanged until matching backward execution.
    pub reserve: DeviceOutputView<'a, u8>,
}

/// Enqueues long FFT causal convolution on `stream` using caller-supplied scratch and reserve storage.
///
/// # Safety
///
/// Scratch and reserve capacity must satisfy [`long_fft_buffer_sizes`] for this type and configuration.
/// Retain all buffers and the stream through completion, failed submission, and replay.
/// Order initialized inputs before execution and keep all writes exclusive.
/// Preserve reserve contents for any matching backward call.
///
/// # Errors
///
/// Returns an error if a dimension, the context, a capacity, or an alignment is invalid, or the native call fails.
pub unsafe fn long_fft_forward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    mut buffers: LongFftBuffers<'_>,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    shape.fft::<T>(true)?;
    stream.context().bind()?;
    stream.join_regions(&[
        x.region(),
        weight.region(),
        y.region(),
        buffers.workspace.region(),
        buffers.reserve.region(),
    ])?;
    require(x, shape.signal_len()?, "x")?;
    require(weight, shape.weight_len()?, "weight")?;
    require(y, shape.signal_len()?, "y")?;
    aligned(x)?;
    aligned(weight)?;
    aligned(y)?;
    aligned(&buffers.workspace)?;
    aligned(&buffers.reserve)?;
    let workspace_bytes = to_size_t(buffers.workspace.len(), "workspace bytes")?;
    let reserve_bytes = to_size_t(buffers.reserve.len(), "reserve bytes")?;
    // SAFETY: `require` and `aligned` checked every tensor and both scratch buffers, and cuDNN receives the scratch lengths it may use.
    // The caller sizes the scratch per `long_fft_buffer_sizes` and retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnLongFFTCausalConv1dForward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            y.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            buffers.workspace.as_mut_ptr().cast(),
            workspace_bytes,
            buffers.reserve.as_mut_ptr().cast(),
            reserve_bytes,
        ))?;
    }
    Ok(())
}

/// Enqueues long FFT gradients on `stream` using reserve state from a matching forward call.
/// Gradient destinations use the same element type as the signal and filter.
///
/// # Safety
///
/// Reserve must come from a matching forward call, ordered before backward execution.
/// Scratch and reserve capacity must satisfy [`long_fft_buffer_sizes`].
/// Retain all buffers and the stream through completion, failed submission, and replay.
/// Order initialized inputs before execution and keep all writes exclusive.
///
/// # Errors
///
/// Returns an error if a dimension, the context, a capacity, or an alignment is invalid, or the native call fails.
pub unsafe fn long_fft_backward_async<T: CausalScalar>(
    dy: &(impl DeviceRange<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T> + ?Sized),
    mut buffers: LongFftBuffers<'_>,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    shape.fft::<T>(true)?;
    stream.context().bind()?;
    stream.join_regions(&[
        dy.region(),
        dx.region(),
        dweight.region(),
        buffers.workspace.region(),
        buffers.reserve.region(),
    ])?;
    require(dy, shape.signal_len()?, "dy")?;
    require(dx, shape.signal_len()?, "dx")?;
    require(dweight, shape.weight_len()?, "dweight")?;
    aligned(dy)?;
    aligned(dx)?;
    aligned(dweight)?;
    aligned(&buffers.workspace)?;
    aligned(&buffers.reserve)?;
    let workspace_bytes = to_size_t(buffers.workspace.len(), "workspace bytes")?;
    let reserve_bytes = to_size_t(buffers.reserve.len(), "reserve bytes")?;
    // SAFETY: `require` and `aligned` checked every tensor and both scratch buffers, and cuDNN receives the scratch lengths it may use.
    // The caller supplies the reserve of a matching forward and retains and orders every buffer through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnLongFFTCausalConv1dBackward(
            stream.as_raw().cast(),
            dy.as_device_ptr().cast(),
            dx.as_mut_ptr().cast(),
            dweight.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            buffers.workspace.as_mut_ptr().cast(),
            workspace_bytes,
            buffers.reserve.as_mut_ptr().cast(),
            reserve_bytes,
        ))?;
    }
    Ok(())
}

/// Full-sequence FFT causal convolution and its matching backward reserve state.
/// The sequence and kernel lengths must be equal powers of two in [4096, 16777216].
/// FFT weights are reversed relative to the direct causal convolution layout.
/// Use [`long_fft_forward_async`] and [`long_fft_backward_async`] for caller-managed scratch and saved-state storage.
#[derive(Debug)]
pub struct LongFftConvolution<T: CausalScalar> {
    shape: CausalConvolutionShape,
    stream: Stream,
    workspace: DeviceMemory<MaybeUninit<u8>>,
    reserve: DeviceMemory<MaybeUninit<u8>>,
    has_forward: bool,
    _element: PhantomData<T>,
}

impl<T: CausalScalar> LongFftConvolution<T> {
    /// Creates storage for the long FFT path, retaining its execution stream.
    ///
    /// # Errors
    ///
    /// Returns an error if a long FFT dimension is invalid or unrepresentable, binding the context fails, the native size query or byte-count conversion fails, or allocating device scratch or reserve storage fails.
    pub fn create(stream: &Stream, shape: CausalConvolutionShape) -> Result<Self> {
        stream.context().bind()?;
        let sizes = long_fft_buffer_sizes::<T>(shape)?;
        Ok(Self {
            shape,
            stream: stream.clone(),
            workspace: DeviceMemory::<u8>::create_uninit(stream.context(), sizes.workspace_bytes)?,
            reserve: DeviceMemory::<u8>::create_uninit(stream.context(), sizes.reserve_bytes)?,
            has_forward: false,
            _element: PhantomData,
        })
    }

    /// Computes causal convolution and stores frequency-domain state for backward.
    ///
    /// # Safety
    ///
    /// Retain this object, its scratch, inputs and destinations through completion, failed submission and replay.
    /// Order initialized inputs before use and keep destination accesses exclusive.
    /// Backward requires a successful matching forward.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is invalid, a tensor is too small, tensor or scratch storage is misaligned, a scratch byte count is unrepresentable, binding the context or ordering the streams fails, or the native convolution fails.
    /// A failed submission clears the saved-forward marker.
    pub unsafe fn forward_async(
        &mut self,
        x: &(impl DeviceRange<T> + ?Sized),
        weight: &(impl DeviceRange<T> + ?Sized),
        y: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        self.has_forward = false;
        // SAFETY: `create` sized the owned scratch and reserve with `long_fft_buffer_sizes` for `self.shape`, and the caller retains `self` and the tensors through completion, per this method's contract.
        unsafe {
            long_fft_forward_async(
                x,
                weight,
                y,
                LongFftBuffers {
                    workspace: self.workspace.output(),
                    reserve: self.reserve.output(),
                },
                self.shape,
                &self.stream,
            )?;
        }

        self.has_forward = true;
        Ok(())
    }

    /// Computes gradients using frequency-domain state saved by the matching forward call.
    ///
    /// # Safety
    ///
    /// Retain this object, its scratch, inputs and destinations through completion, failed submission and replay.
    /// Order initialized inputs before use and keep destination accesses exclusive.
    /// Backward requires a successful matching forward.
    ///
    /// # Errors
    ///
    /// Returns an error if no forward submission has succeeded since creation or the most recent failed forward attempt, or for invalid dimensions, insufficient tensor capacity, misaligned storage, unrepresentable scratch byte counts, context binding or stream-dependency failure, or native gradient failure.
    pub unsafe fn backward_async(
        &mut self,
        dy: &(impl DeviceRange<T> + ?Sized),
        dx: &mut (impl DeviceOutput<T> + ?Sized),
        dweight: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        if !self.has_forward {
            return Err(Error::DescriptorMismatch {
                name: "long fft forward reserve state".into(),
            });
        }

        // SAFETY: `has_forward` shows the owned reserve holds a successful forward's state for `self.shape`, and the caller retains and orders `self` and the tensors through completion, per this method's contract.
        unsafe {
            long_fft_backward_async(
                dy,
                dx,
                dweight,
                LongFftBuffers {
                    workspace: self.workspace.output(),
                    reserve: self.reserve.output(),
                },
                self.shape,
                &self.stream,
            )?;
        }
        Ok(())
    }

    /// Computes causal convolution, stores frequency-domain state for backward, and waits for the retained stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, [`Self::forward_async`] fails, or the wait fails.
    pub fn forward(
        &mut self,
        x: &(impl DeviceSlice<T> + ?Sized),
        weight: &(impl DeviceSlice<T> + ?Sized),
        y: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        let binding = StreamBinding::Borrowed(self.stream.to_borrowed());
        binding.sync_scope(|| {
            // SAFETY: the inputs are initialized, this object owns its scratch, and the borrows outlive the wait in `sync_scope`.
            unsafe { self.forward_async(x, weight, y) }
        })
    }

    /// Computes gradients from the state saved by the matching forward call, and waits for the retained stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream is capturing, [`Self::backward_async`] fails, or the wait fails.
    pub fn backward(
        &mut self,
        dy: &(impl DeviceSlice<T> + ?Sized),
        dx: &mut (impl DeviceOutput<T> + ?Sized),
        dweight: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        let binding = StreamBinding::Borrowed(self.stream.to_borrowed());
        binding.sync_scope(|| {
            // SAFETY: `dy` is initialized, this object owns its saved state, and the borrows outlive the wait in `sync_scope`.
            unsafe { self.backward_async(dy, dx, dweight) }
        })
    }
}
