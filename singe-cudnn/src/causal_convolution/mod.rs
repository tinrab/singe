//! Direct causal depthwise convolutions, including FFT and fused back-to-back paths.
//! These are direct cuDNN operations, matching the C++ causal-convolution samples.
//! FFT filters use FIR order, reversed relative to the direct convolution filters.

mod long_fft;

mod sealed {
    use crate::data_type::{bf16, f16};

    #[expect(
        unnameable_types,
        reason = "seals `CausalScalar`, so callers cannot name or implement it"
    )]
    pub trait Sealed {}

    impl Sealed for f32 {}

    impl Sealed for f64 {}

    impl Sealed for f16 {}

    impl Sealed for bf16 {}
}

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::impl_enum_display;
use singe_cuda::{
    error::Error as CudaError,
    stream::Stream,
    view::{DeviceOutput, DeviceRange, DeviceRepr, DeviceSlice, DeviceSliceMut},
};
use singe_cudnn_sys as sys;

pub use self::long_fft::{
    LongFftBufferSizes, LongFftBuffers, LongFftConvolution, long_fft_backward_async,
    long_fft_buffer_sizes, long_fft_forward_async,
};
use crate::{
    data_type::{DataType, TensorElement, bf16, f16},
    error::{Error, Result},
    try_ffi,
    utility::to_i32,
};

singe_core::native_enum! {
    /// Activation mode for causal conv1d operations.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum Activation: sys::cudnnCausalConv1dActivation_t {
        /// Identity (no activation).
        Identity = CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY,
        /// `SiLU` (Sigmoid Linear Unit) activation.
        Silu = CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU,
    }
}

impl_enum_display!(Activation, { Self::Identity => "CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY", Self::Silu => "CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU" });

/// Supported feature types and their direct-convolution gradient accumulation type.
pub trait CausalScalar: TensorElement + sealed::Sealed {
    /// Accumulation type for direct filter and bias gradients: `f64` for double inputs and `f32` otherwise.
    type Gradient: TensorElement;
}

impl CausalScalar for f32 {
    type Gradient = Self;
}

impl CausalScalar for f64 {
    type Gradient = Self;
}

impl CausalScalar for f16 {
    type Gradient = f32;
}

impl CausalScalar for bf16 {
    type Gradient = f32;
}

/// Dimensions of a contiguous causal depthwise convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalConvolutionShape {
    /// Batch size.
    pub batch: usize,
    /// Number of channels, each convolved independently.
    pub channels: usize,
    /// Sequence length.
    pub sequence: usize,
    /// Convolution kernel width.
    pub kernel: usize,
}

impl CausalConvolutionShape {
    fn dimensions(self) -> Result<[i32; 4]> {
        if [self.batch, self.channels, self.sequence, self.kernel].contains(&0) {
            return Err(Error::InvalidDataShape);
        }
        Ok([
            to_i32(self.batch, "batch")?,
            to_i32(self.channels, "channels")?,
            to_i32(self.sequence, "sequence")?,
            to_i32(self.kernel, "kernel")?,
        ])
    }

    fn signal_len(self) -> Result<usize> {
        product(&[self.batch, self.channels, self.sequence])
    }

    fn weight_len(self) -> Result<usize> {
        product(&[self.channels, self.kernel])
    }

    fn fft<T: CausalScalar>(self, long: bool) -> Result<()> {
        let valid = if long {
            self.sequence == self.kernel && (4096..=16_777_216).contains(&self.kernel)
        } else {
            (128..=16384).contains(&self.kernel)
                && self.sequence >= self.kernel
                && self.sequence.is_multiple_of(self.kernel)
        };
        if !valid
            || !self.kernel.is_power_of_two()
            || (T::data_type() == DataType::F64
                && self.kernel > if long { 8_388_608 } else { 4096 })
        {
            return Err(Error::InvalidDataShape);
        }
        Ok(())
    }
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |a, &b| {
        a.checked_mul(b)
            .ok_or_else(|| Error::out_of_range("buffer size"))
    })
}

fn require<T: DeviceRepr>(
    memory: &(impl DeviceRange<T> + ?Sized),
    count: usize,
    name: &'static str,
) -> Result<()> {
    if memory.len() < count {
        return Err(Error::BufferTooSmall {
            name: name.into(),
            required: count,
            actual: memory.len(),
        });
    }
    Ok(())
}

fn aligned<T: DeviceRepr>(memory: &(impl DeviceRange<T> + ?Sized)) -> Result<()> {
    if !memory.as_device_ptr().addr().is_multiple_of(16) {
        return Err(Error::out_of_range("16-byte buffer alignment"));
    }
    Ok(())
}

/// Enqueues a causal depthwise 1D convolution with an optional `SiLU` activation, `y = act(conv1d_causal(x, weight) + bias)`.
///
/// Causal padding inserts `kernel - 1` zeros on the left and none on the right.
/// Each channel is convolved independently with its own filter.
/// For each batch and channel, the direct weight layout computes:
/// ```text
/// y[t] = act(bias + sum_{j=0}^{kernel - 1} x[t - j] * weight[kernel - 1 - j])
/// ```
/// where `x[t - j]` is zero when `t < j`.
/// For `weight = [w0, w1, w2]`, this is `y[t] = act(bias + x[t - 2] * w0 + x[t - 1] * w1 + x[t] * w2)`.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
///
/// # Errors
///
/// Returns an error if a dimension is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native convolution fails.
pub unsafe fn forward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    bias: &(impl DeviceRange<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), bias.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[x.region(), weight.region(), bias.region(), y.region()])?;

    require(x, shape.signal_len()?, "x")?;
    require(weight, shape.weight_len()?, "weight")?;
    require(bias, shape.channels, "bias")?;
    require(y, shape.signal_len()?, "y")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`.
    // The caller retains, orders and initializes them through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnCausalConv1dForward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            bias.as_device_ptr().cast(),
            y.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            activation.into(),
        ))?;
    }
    Ok(())
}

/// Enqueues the gradients of [`forward_async`].
///
/// Writes `dx` with layout (batch, channels, sequence), and accumulates `dweight` (channels, kernel) and `dbias` (channels) with atomic adds.
/// Zero `dweight` and `dbias` before the call unless accumulation across calls is intended.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error if a dimension is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native convolution fails.
pub unsafe fn backward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    bias: &(impl DeviceRange<T> + ?Sized),
    dy: &(impl DeviceRange<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [
        x.region(),
        weight.region(),
        bias.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
        dbias.region(),
    ]
    .iter()
    .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[
        x.region(),
        weight.region(),
        bias.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
        dbias.region(),
    ])?;

    require(x, shape.signal_len()?, "x")?;
    require(weight, shape.weight_len()?, "weight")?;
    require(bias, shape.channels, "bias")?;
    require(dy, shape.signal_len()?, "dy")?;
    require(dx, shape.signal_len()?, "dx")?;
    require(dweight, shape.weight_len()?, "dweight")?;
    require(dbias, shape.channels, "dbias")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`.
    // The caller retains, orders and initializes them, including the gradient accumulators, through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnCausalConv1dBackward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            bias.as_device_ptr().cast(),
            dy.as_device_ptr().cast(),
            dx.as_mut_ptr().cast(),
            dweight.as_mut_ptr().cast(),
            dbias.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            T::Gradient::data_type().into(),
            activation.into(),
        ))?;
    }
    Ok(())
}

/// Enqueues [`forward_async`] with NWH layouts: `x` and `y` are (batch, sequence, channels), `weight` is (kernel, channels), and `bias` is (channels).
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
///
/// # Errors
///
/// Returns an error if a dimension is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native convolution fails.
pub unsafe fn nwh_forward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    bias: &(impl DeviceRange<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), bias.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[x.region(), weight.region(), bias.region(), y.region()])?;

    require(x, shape.signal_len()?, "x")?;
    require(weight, shape.weight_len()?, "weight")?;
    require(bias, shape.channels, "bias")?;
    require(y, shape.signal_len()?, "y")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`.
    // The caller retains, orders and initializes them through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnCausalConv1dNwhForward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            bias.as_device_ptr().cast(),
            y.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            activation.into(),
        ))?;
    }
    Ok(())
}

/// Enqueues [`backward_async`] with NWH layouts: `x`, `dy` and `dx` are (batch, sequence, channels), `weight` and `dweight` are (kernel, channels), and `bias` and `dbias` are (channels).
/// The weight and bias gradients use [`CausalScalar::Gradient`].
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error if a dimension is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native convolution fails.
pub unsafe fn nwh_backward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    bias: &(impl DeviceRange<T> + ?Sized),
    dy: &(impl DeviceRange<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [
        x.region(),
        weight.region(),
        bias.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
        dbias.region(),
    ]
    .iter()
    .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[
        x.region(),
        weight.region(),
        bias.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
        dbias.region(),
    ])?;

    require(x, shape.signal_len()?, "x")?;
    require(weight, shape.weight_len()?, "weight")?;
    require(bias, shape.channels, "bias")?;
    require(dy, shape.signal_len()?, "dy")?;
    require(dx, shape.signal_len()?, "dx")?;
    require(dweight, shape.weight_len()?, "dweight")?;
    require(dbias, shape.channels, "dbias")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`.
    // The caller retains, orders and initializes them, including the gradient accumulators, through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnCausalConv1dNwhBackward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            bias.as_device_ptr().cast(),
            dy.as_device_ptr().cast(),
            dx.as_mut_ptr().cast(),
            dweight.as_mut_ptr().cast(),
            dbias.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
            T::Gradient::data_type().into(),
            activation.into(),
        ))?;
    }
    Ok(())
}

/// Enqueues a causal depthwise 1D convolution, `y = conv1d_causal(x, weight)`, on the medium FFT path.
///
/// For each batch and channel, the causal convolution is:
/// ```text
/// y[t] = sum_{j=0}^{kernel - 1} x[t - j] * weight[j]
/// ```
/// where `x[t - j]` is zero when `t < j`.
///
/// This FFT weight layout is reversed relative to the direct layout of [`forward_async`].
/// For `weight = [w0, w1, w2]`, this operation computes `y[t] = x[t] * w0 + x[t - 1] * w1 + x[t - 2] * w2`, while [`forward_async`] computes `y[t] = x[t - 2] * w0 + x[t - 1] * w1 + x[t] * w2`.
///
/// The implementation evaluates this convolution in the frequency domain.
/// It transforms zero-padded signal and filter tiles with real-to-complex FFTs, multiplies the transformed signal and filter pointwise, and applies a complex-to-real inverse FFT before writing the causal output samples.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
///
/// # Errors
///
/// Returns an error if a dimension is invalid or unrepresentable, the FFT length is unsupported for the element type, a size overflows, a buffer is too small or not 16-byte aligned, the CUDA contexts differ, binding the context or ordering the streams fails, or the native FFT convolution fails.
pub unsafe fn fft_forward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[x.region(), weight.region(), y.region()])?;

    shape.fft::<T>(false)?;
    require(x, shape.signal_len()?, "x")?;
    aligned(x)?;
    require(weight, shape.weight_len()?, "weight")?;
    aligned(weight)?;
    require(y, shape.signal_len()?, "y")?;
    aligned(y)?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`, and `aligned` checked the 16-byte alignment cuDNN's FFT kernels need.
    // The caller retains, orders and initializes them through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnFFTCausalConv1dForward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            y.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
        ))?;
    }
    Ok(())
}

/// Enqueues the gradients `dx` and `dweight` of [`fft_forward_async`].
///
/// The gradient kernels use the same FFT formulation as the forward path: signal, filter, and output-gradient tiles are transformed to the frequency domain, combined with pointwise products equivalent to convolution or correlation, and inverse-transformed to produce `dx` and `dweight`.
/// `weight` and `dweight` use the FFT weight layout of [`fft_forward_async`], reversed relative to the direct layout.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error if a dimension is invalid or unrepresentable, the FFT length is unsupported for the element type, a size overflows, a buffer is too small or not 16-byte aligned, the CUDA contexts differ, binding the context or ordering the streams fails, or the native FFT convolution fails.
pub unsafe fn fft_backward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weight: &(impl DeviceRange<T> + ?Sized),
    dy: &(impl DeviceRange<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T> + ?Sized),
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    stream.context().bind()?;
    if [
        x.region(),
        weight.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
    ]
    .iter()
    .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[
        x.region(),
        weight.region(),
        dy.region(),
        dx.region(),
        dweight.region(),
    ])?;

    shape.fft::<T>(false)?;
    require(x, shape.signal_len()?, "x")?;
    aligned(x)?;
    require(weight, shape.weight_len()?, "weight")?;
    aligned(weight)?;
    require(dy, shape.signal_len()?, "dy")?;
    aligned(dy)?;
    require(dx, shape.signal_len()?, "dx")?;
    aligned(dx)?;
    require(dweight, shape.weight_len()?, "dweight")?;
    aligned(dweight)?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape`, and `aligned` checked the 16-byte alignment cuDNN's FFT kernels need.
    // The caller retains, orders and initializes them, including the gradient accumulators, through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnFFTCausalConv1dBackward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weight.as_device_ptr().cast(),
            dy.as_device_ptr().cast(),
            dx.as_mut_ptr().cast(),
            dweight.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            T::data_type().into(),
        ))?;
    }
    Ok(())
}

/// Enqueues a fused back-to-back causal convolution: a projection convolution, elementwise gating, then a mixer convolution.
/// The individual convolutions apply no activation.
///
/// - `x` is (batch, channels, 3, sequence), three interleaved projection channels per channel.
/// - `weights_projection` is (channels, 3, kernel), the projection filters of all three.
/// - `weights_mixer` is (channels, `mixer_kernel`), and `skip_bias` is (channels).
/// - `y` is (batch, channels, sequence), the mixer and skip output that [`back_to_back_backward_async`] consumes.
/// - `y_gated` is (batch, channels, sequence), `y` gated by the projection of the second channel, the output of the fused Hyena-SE block.
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
///
/// # Errors
///
/// Returns an error if a convolution dimension or the mixer width is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native fused convolution fails.
pub unsafe fn back_to_back_forward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weights_projection: &(impl DeviceRange<T> + ?Sized),
    weights_mixer: &(impl DeviceRange<T> + ?Sized),
    skip_bias: &(impl DeviceRange<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    y_gated: &mut (impl DeviceOutput<T> + ?Sized),
    shape: CausalConvolutionShape,
    mixer_kernel: usize,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    let mixer = to_i32(mixer_kernel, "mixer kernel")?;
    if mixer <= 0 {
        return Err(Error::InvalidDataShape);
    }

    stream.context().bind()?;
    if [
        x.region(),
        weights_projection.region(),
        weights_mixer.region(),
        skip_bias.region(),
        y.region(),
        y_gated.region(),
    ]
    .iter()
    .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[
        x.region(),
        weights_projection.region(),
        weights_mixer.region(),
        skip_bias.region(),
        y.region(),
        y_gated.region(),
    ])?;

    require(x, product(&[shape.signal_len()?, 3])?, "x")?;
    require(
        weights_projection,
        product(&[shape.weight_len()?, 3])?,
        "weights_projection",
    )?;
    require(
        weights_mixer,
        product(&[shape.channels, mixer_kernel])?,
        "weights_mixer",
    )?;
    require(skip_bias, shape.channels, "skip_bias")?;
    require(y, shape.signal_len()?, "y")?;
    require(y_gated, shape.signal_len()?, "y_gated")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape` and `mixer_kernel`, and `mixer` is positive.
    // The caller retains, orders and initializes them through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnB2BCausalConv1dForward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weights_projection.as_device_ptr().cast(),
            weights_mixer.as_device_ptr().cast(),
            skip_bias.as_device_ptr().cast(),
            y.as_mut_ptr().cast(),
            y_gated.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            mixer,
            T::data_type().into(),
        ))?;
    }
    Ok(())
}

/// Enqueues the gradients of [`back_to_back_forward_async`].
///
/// `x`, the weights, `skip_bias` and `y` are the forward inputs and intermediate output, in the forward layouts, and `dy` is the gradient of `y_gated`.
/// `dx` is (batch, channels, 3, sequence), `dweights_projection` is (channels, 3, kernel), `dweights_mixer` is (channels, `mixer_kernel`), and `dskip_bias` is (channels).
/// The weight and bias gradients use [`CausalScalar::Gradient`].
///
/// # Safety
///
/// Retain the context or stream, descriptors and all buffer ranges through completion, submission failure and replay.
/// Order allocation readiness and input producers before use; values must be valid when read and writes must stay exclusive.
/// Only the specified tensor elements are produced; padding and tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error if a convolution dimension or the mixer width is zero or unrepresentable, a buffer size overflows, a buffer is too small, the CUDA contexts differ, binding the context or ordering the streams fails, or the native fused convolution fails.
pub unsafe fn back_to_back_backward_async<T: CausalScalar>(
    x: &(impl DeviceRange<T> + ?Sized),
    weights_projection: &(impl DeviceRange<T> + ?Sized),
    weights_mixer: &(impl DeviceRange<T> + ?Sized),
    skip_bias: &(impl DeviceRange<T> + ?Sized),
    y: &(impl DeviceRange<T> + ?Sized),
    dy: &(impl DeviceRange<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweights_projection: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dweights_mixer: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dskip_bias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    shape: CausalConvolutionShape,
    mixer_kernel: usize,
    stream: &Stream,
) -> Result<()> {
    let [batch, dim, seq, kernel] = shape.dimensions()?;
    let mixer = to_i32(mixer_kernel, "mixer kernel")?;
    if mixer <= 0 {
        return Err(Error::InvalidDataShape);
    }

    stream.context().bind()?;
    if [
        x.region(),
        weights_projection.region(),
        weights_mixer.region(),
        skip_bias.region(),
        y.region(),
        dy.region(),
        dx.region(),
        dweights_projection.region(),
        dweights_mixer.region(),
        dskip_bias.region(),
    ]
    .iter()
    .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }

    stream.join_regions(&[
        x.region(),
        weights_projection.region(),
        weights_mixer.region(),
        skip_bias.region(),
        y.region(),
        dy.region(),
        dx.region(),
        dweights_projection.region(),
        dweights_mixer.region(),
        dskip_bias.region(),
    ])?;

    require(x, product(&[shape.signal_len()?, 3])?, "x")?;
    require(
        weights_projection,
        product(&[shape.weight_len()?, 3])?,
        "weights_projection",
    )?;
    require(
        weights_mixer,
        product(&[shape.channels, mixer_kernel])?,
        "weights_mixer",
    )?;
    require(skip_bias, shape.channels, "skip_bias")?;
    require(y, shape.signal_len()?, "y")?;
    require(dy, shape.signal_len()?, "dy")?;
    require(dx, product(&[shape.signal_len()?, 3])?, "dx")?;
    require(
        dweights_projection,
        product(&[shape.weight_len()?, 3])?,
        "dweights_projection",
    )?;
    require(
        dweights_mixer,
        product(&[shape.channels, mixer_kernel])?,
        "dweights_mixer",
    )?;
    require(dskip_bias, shape.channels, "dskip_bias")?;
    // SAFETY: every buffer shares the stream's context and `require` checked its length against `shape` and `mixer_kernel`, and `mixer` is positive.
    // The caller retains, orders and initializes them, including the gradient accumulators, through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnB2BCausalConv1dBackward(
            stream.as_raw().cast(),
            x.as_device_ptr().cast(),
            weights_projection.as_device_ptr().cast(),
            weights_mixer.as_device_ptr().cast(),
            skip_bias.as_device_ptr().cast(),
            y.as_device_ptr().cast(),
            dy.as_device_ptr().cast(),
            dx.as_mut_ptr().cast(),
            dweights_projection.as_mut_ptr().cast(),
            dweights_mixer.as_mut_ptr().cast(),
            dskip_bias.as_mut_ptr().cast(),
            batch,
            dim,
            seq,
            kernel,
            mixer,
            T::data_type().into(),
            T::Gradient::data_type().into(),
        ))?;
    }
    Ok(())
}

/// Computes [`forward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`forward_async`] fails, or the wait fails.
pub fn forward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    bias: &(impl DeviceSlice<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { forward_async(x, weight, bias, y, activation, shape, stream) }
    })
}

/// Computes [`backward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`backward_async`] fails, or the wait fails.
pub fn backward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    bias: &(impl DeviceSlice<T> + ?Sized),
    dy: &(impl DeviceSlice<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs and accumulated gradients are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            backward_async(
                x, weight, bias, dy, dx, dweight, dbias, activation, shape, stream,
            )
        }
    })
}

/// Computes [`nwh_forward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`nwh_forward_async`] fails, or the wait fails.
pub fn nwh_forward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    bias: &(impl DeviceSlice<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { nwh_forward_async(x, weight, bias, y, activation, shape, stream) }
    })
}

/// Computes [`nwh_backward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`nwh_backward_async`] fails, or the wait fails.
pub fn nwh_backward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    bias: &(impl DeviceSlice<T> + ?Sized),
    dy: &(impl DeviceSlice<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    activation: Activation,
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs and accumulated gradients are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            nwh_backward_async(
                x, weight, bias, dy, dx, dweight, dbias, activation, shape, stream,
            )
        }
    })
}

/// Computes [`fft_forward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`fft_forward_async`] fails, or the wait fails.
pub fn fft_forward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { fft_forward_async(x, weight, y, shape, stream) }
    })
}

/// Computes [`fft_backward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`fft_backward_async`] fails, or the wait fails.
pub fn fft_backward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weight: &(impl DeviceSlice<T> + ?Sized),
    dy: &(impl DeviceSlice<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceSliceMut<T> + ?Sized),
    shape: CausalConvolutionShape,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs and `dweight` are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe { fft_backward_async(x, weight, dy, dx, dweight, shape, stream) }
    })
}

/// Computes [`back_to_back_forward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`back_to_back_forward_async`] fails, or the wait fails.
pub fn back_to_back_forward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weights_projection: &(impl DeviceSlice<T> + ?Sized),
    weights_mixer: &(impl DeviceSlice<T> + ?Sized),
    skip_bias: &(impl DeviceSlice<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    y_gated: &mut (impl DeviceOutput<T> + ?Sized),
    shape: CausalConvolutionShape,
    mixer_kernel: usize,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            back_to_back_forward_async(
                x,
                weights_projection,
                weights_mixer,
                skip_bias,
                y,
                y_gated,
                shape,
                mixer_kernel,
                stream,
            )
        }
    })
}

/// Computes [`back_to_back_backward_async`] and waits for `stream`.
///
/// # Errors
///
/// Returns an error if `stream` is capturing, [`back_to_back_backward_async`] fails, or the wait fails.
pub fn back_to_back_backward<T: CausalScalar>(
    x: &(impl DeviceSlice<T> + ?Sized),
    weights_projection: &(impl DeviceSlice<T> + ?Sized),
    weights_mixer: &(impl DeviceSlice<T> + ?Sized),
    skip_bias: &(impl DeviceSlice<T> + ?Sized),
    y: &(impl DeviceSlice<T> + ?Sized),
    dy: &(impl DeviceSlice<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweights_projection: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    dweights_mixer: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    dskip_bias: &mut (impl DeviceSliceMut<T::Gradient> + ?Sized),
    shape: CausalConvolutionShape,
    mixer_kernel: usize,
    stream: &Stream,
) -> Result<()> {
    stream.complete(|| {
        // SAFETY: the inputs and accumulated gradients are initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            back_to_back_backward_async(
                x,
                weights_projection,
                weights_mixer,
                skip_bias,
                y,
                dy,
                dx,
                dweights_projection,
                dweights_mixer,
                dskip_bias,
                shape,
                mixer_kernel,
                stream,
            )
        }
    })
}
