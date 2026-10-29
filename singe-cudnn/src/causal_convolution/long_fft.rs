use std::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::MaybeUninit,
};

use singe_cuda::{
    memory::DeviceMemory,
    stream::Stream,
    view::{DeviceInput, DeviceOutput},
};
use singe_cudnn_sys as sys;

use crate::{
    causal_convolution::{CausalScalar, Config, aligned, require},
    error::{Error, Result},
    try_ffi,
    utility::to_usize,
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
/// Returns an error for invalid long FFT dimensions or a failed native size query.
pub fn long_fft_buffer_sizes<T: CausalScalar>(config: Config) -> Result<LongFftBufferSizes> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    config.fft::<T>(true)?;
    let (mut workspace, mut reserve) = (0, 0);
    // SAFETY: `cudnnLongFFTCausalConv1dGetBufferSizes` receives valid output locations.
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
pub struct LongFftBuffers<'a> {
    /// Exclusive temporary device scratch storage with the required capacity and 16-byte alignment.
    pub workspace: &'a mut dyn DeviceOutput<MaybeUninit<u8>>,
    /// Exclusive opaque forward-state storage, preserved unchanged until matching backward execution.
    pub reserve: &'a mut dyn DeviceOutput<MaybeUninit<u8>>,
}

impl Debug for LongFftBuffers<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LongFftBuffers").finish_non_exhaustive()
    }
}

/// Computes long FFT causal convolution using caller-supplied scratch and reserve storage.
///
/// # Errors
/// Returns an error for invalid dimensions, context, capacity, or alignment, or a native failure.
///
/// # Safety
/// Scratch and reserve capacity must satisfy [`long_fft_buffer_sizes`] for this type and configuration.
/// Retain all buffers and the stream through completion, failed submission, and replay.
/// Order initialized inputs before execution and keep all writes exclusive.
/// Preserve reserve contents for any matching backward call.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the raw operation consumes the exclusive workspace and reserve borrows"
)]
pub unsafe fn long_fft_forward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    buffers: LongFftBuffers<'_>,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    config.fft::<T>(true)?;
    stream.context().bind()?;
    stream.join_regions(&[
        x.region(),
        weight.region(),
        y.region(),
        buffers.workspace.region(),
        buffers.reserve.region(),
    ])?;
    require(x, config.signal_len()?, "x")?;
    require(weight, config.weight_len()?, "weight")?;
    require(y, config.signal_len()?, "y")?;
    aligned(x)?;
    aligned(weight)?;
    aligned(y)?;
    aligned(buffers.workspace)?;
    aligned(buffers.reserve)?;
    let workspace_bytes = sys::size_t::try_from(buffers.workspace.len())
        .map_err(|_| Error::out_of_range("workspace bytes"))?;
    let reserve_bytes = sys::size_t::try_from(buffers.reserve.len())
        .map_err(|_| Error::out_of_range("reserve bytes"))?;
    // SAFETY: `cudnnLongFFTCausalConv1dForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Computes long FFT gradients using reserve state from a matching forward call.
/// Gradient destinations use the same element type as the signal and filter.
///
/// # Errors
/// Returns an error for invalid dimensions, context, capacity, or alignment, or a native failure.
///
/// # Safety
/// Reserve must come from a matching forward call, ordered before backward execution.
/// Scratch and reserve capacity must satisfy [`long_fft_buffer_sizes`].
/// Retain all buffers and the stream through completion, failed submission, and replay.
/// Order initialized inputs before execution and keep all writes exclusive.
#[expect(
    clippy::needless_pass_by_value,
    reason = "the raw operation consumes the exclusive workspace and reserve borrows"
)]
pub unsafe fn long_fft_backward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    dy: &(impl DeviceInput<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T> + ?Sized),
    buffers: LongFftBuffers<'_>,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    config.fft::<T>(true)?;
    stream.context().bind()?;
    stream.join_regions(&[
        dy.region(),
        dx.region(),
        dweight.region(),
        buffers.workspace.region(),
        buffers.reserve.region(),
    ])?;
    require(dy, config.signal_len()?, "dy")?;
    require(dx, config.signal_len()?, "dx")?;
    require(dweight, config.weight_len()?, "dweight")?;
    aligned(dy)?;
    aligned(dx)?;
    aligned(dweight)?;
    aligned(buffers.workspace)?;
    aligned(buffers.reserve)?;
    let workspace_bytes = sys::size_t::try_from(buffers.workspace.len())
        .map_err(|_| Error::out_of_range("workspace bytes"))?;
    let reserve_bytes = sys::size_t::try_from(buffers.reserve.len())
        .map_err(|_| Error::out_of_range("reserve bytes"))?;
    // SAFETY: `cudnnLongFFTCausalConv1dBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
/// Use [`long_fft_forward`] and [`long_fft_backward`] for caller-managed scratch and saved-state storage.
#[derive(Debug)]
pub struct LongFftConvolution<T: CausalScalar> {
    config: Config,
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
    /// Returns an error for invalid or unrepresentable long FFT dimensions, context binding failure, native size-query or byte-count conversion failure, or device scratch or reserve allocation failure.
    pub fn create(stream: &Stream, config: Config) -> Result<Self> {
        stream.context().bind()?;
        let sizes = long_fft_buffer_sizes::<T>(config)?;
        Ok(Self {
            config,
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
    /// Retain this object, its scratch, inputs and destinations through completion,
    /// failed submission and replay. Order initialized inputs before use and keep
    /// destination accesses exclusive. Backward requires a successful matching forward.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, insufficient tensor capacity, misaligned tensor or scratch storage, unrepresentable scratch byte counts, context binding or stream-dependency failure, or native convolution failure. A failed submission clears the saved-forward marker.
    pub unsafe fn forward_async(
        &mut self,
        x: &(impl DeviceInput<T> + ?Sized),
        weight: &(impl DeviceInput<T> + ?Sized),
        y: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        self.has_forward = false;
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            long_fft_forward(
                &self.stream,
                self.config,
                x,
                weight,
                y,
                LongFftBuffers {
                    workspace: &mut self.workspace,
                    reserve: &mut self.reserve,
                },
            )?;
        }
        self.has_forward = true;
        Ok(())
    }

    /// Computes gradients using frequency-domain state saved by the matching forward call.
    ///
    /// # Safety
    /// Retain this object, its scratch, inputs and destinations through completion,
    /// failed submission and replay. Order initialized inputs before use and keep
    /// destination accesses exclusive. Backward requires a successful matching forward.
    ///
    /// # Errors
    ///
    /// Returns an error if no forward submission has succeeded since creation or the most recent failed forward attempt, or for invalid dimensions, insufficient tensor capacity, misaligned storage, unrepresentable scratch byte counts, context binding or stream-dependency failure, or native gradient failure.
    pub unsafe fn backward_async(
        &mut self,
        dy: &(impl DeviceInput<T> + ?Sized),
        dx: &mut (impl DeviceOutput<T> + ?Sized),
        dweight: &mut (impl DeviceOutput<T> + ?Sized),
    ) -> Result<()> {
        if !self.has_forward {
            return Err(Error::DescriptorMismatch {
                name: "long fft forward reserve state".into(),
            });
        }
        // SAFETY: the caller upholds this function's safety contract.
        unsafe {
            long_fft_backward(
                &self.stream,
                self.config,
                dy,
                dx,
                dweight,
                LongFftBuffers {
                    workspace: &mut self.workspace,
                    reserve: &mut self.reserve,
                },
            )?;
        }
        Ok(())
    }
}
