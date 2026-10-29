//! Direct causal depthwise convolutions, including FFT and fused back-to-back paths.
//! These are direct cuDNN operations, matching the C++ causal-convolution samples.
//! FFT filters use FIR order, reversed relative to the direct convolution filters.

mod long_fft;

pub use long_fft::{
    LongFftBufferSizes, LongFftBuffers, LongFftConvolution, long_fft_backward,
    long_fft_buffer_sizes, long_fft_forward,
};
use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::{
    error::Error as CudaError,
    stream::Stream,
    view::{DeviceInput, DeviceOutput, DeviceRepr},
};
use singe_cudnn_sys as sys;

use crate::{
    data_type::{DataType, DataTypeLike, bf16, f16},
    error::{Error, Result},
    try_ffi,
    utility::to_i32,
};

/// Activation mode for causal conv1d operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum Activation {
    /// Identity (no activation).
    Identity = sys::cudnnCausalConv1dActivation_t::CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY as _,
    /// `SiLU` (Sigmoid Linear Unit) activation.
    Silu = sys::cudnnCausalConv1dActivation_t::CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU as _,
}

impl_enum_conversion!(sys::cudnnCausalConv1dActivation_t, Activation);

impl_enum_display!(Activation, { Self::Identity => "CUDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY", Self::Silu => "CUDNN_CAUSAL_CONV1D_ACTIVATION_SILU" });

mod sealed {
    use crate::data_type::{bf16, f16};

    pub trait Sealed {}

    impl Sealed for f32 {}

    impl Sealed for f64 {}

    impl Sealed for f16 {}

    impl Sealed for bf16 {}
}

/// Supported feature types and their direct-convolution gradient accumulation type.
pub trait CausalScalar: DataTypeLike + sealed::Sealed {
    /// Accumulation type for direct filter and bias gradients: `f64` for double inputs and `f32` otherwise.
    type Gradient: DataTypeLike;
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
pub struct Config {
    /// Batch size.
    pub batch: usize,
    /// Number of channels, each convolved independently.
    pub channels: usize,
    /// Sequence length.
    pub sequence: usize,
    /// Convolution kernel width.
    pub kernel: usize,
}

impl Config {
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
    memory: &(impl DeviceInput<T> + ?Sized),
    count: usize,
    name: &'static str,
) -> Result<()> {
    if memory.len() < count {
        return Err(Error::LengthMismatch {
            name: name.into(),
            expected: count,
            actual: memory.len(),
        });
    }
    Ok(())
}

fn aligned<T: DeviceRepr>(memory: &(impl DeviceInput<T> + ?Sized)) -> Result<()> {
    if !memory.as_device_ptr().addr().is_multiple_of(16) {
        return Err(Error::out_of_range("16-byte buffer alignment"));
    }
    Ok(())
}

/// Verifies that the `SubquadraticOps` sub-library version matches the core cuDNN version.
///
/// # Errors
///
/// Returns an error if cuDNN reports that the subquadratic-operations library is incompatible with the core library.
pub fn version_check() -> Result<()> {
    // SAFETY: `cudnnSubquadraticOpsVersionCheck` takes no arguments.
    unsafe {
        try_ffi!(sys::cudnnSubquadraticOpsVersionCheck())?;
    }
    Ok(())
}

/// Compute a causal (left-padded) depthwise 1D convolution with optional `SiLU` activation.
///
/// Computes: y = Act( `conv1d_causal(x`, weight) + bias )
///
/// Causal padding inserts (`kernel_size` - 1) zeros on the left and 0 on the right.
/// The convolution is depthwise: each channel is convolved independently with its
/// own 1D filter.
///
/// For each batch and channel, the non-FFT causal conv1d weight layout is:
/// ```text
/// y[t] = Act(bias + sum_{j=0}^{kernelSize - 1} x[t - j] * weight[kernelSize - 1 - j])
/// ```
/// where `x[t - j]` is treated as zero when t < j. For example, for
/// ```text
/// weight = [w0, w1, w2], the non-FFT operation is:
/// ```
/// ```text
/// y[t] = Act(bias + x[t - 2] * w0 + x[t - 1] * w1 + x[t] * w2)
/// ```
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable dimensions, overflowing buffer sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native convolution failure.
pub unsafe fn forward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    bias: &(impl DeviceInput<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), bias.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    stream.join_regions(&[x.region(), weight.region(), bias.region(), y.region()])?;

    require(x, config.signal_len()?, "x")?;
    require(weight, config.weight_len()?, "weight")?;
    require(bias, config.channels, "bias")?;
    require(y, config.signal_len()?, "y")?;
    // SAFETY: `cudnnCausalConv1dForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Compute gradients for causal depthwise 1D convolution.
///
/// Computes:
/// - dx      = dL/dx       (batch, dim, `seq_len`)
/// - dweight = dL/dweight   (dim, `kernel_size`) — accumulated via atomicAdd
/// - dbias   = dL/dbias     (dim,)             — accumulated via atomicAdd
///
/// The caller must zero-initialize dweight and dbias before calling this function
/// if accumulation across multiple calls is not desired.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable dimensions, overflowing buffer sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native convolution failure.
pub unsafe fn backward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    bias: &(impl DeviceInput<T> + ?Sized),
    dy: &(impl DeviceInput<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    activation: Activation,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
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

    require(x, config.signal_len()?, "x")?;
    require(weight, config.weight_len()?, "weight")?;
    require(bias, config.channels, "bias")?;
    require(dy, config.signal_len()?, "dy")?;
    require(dx, config.signal_len()?, "dx")?;
    require(dweight, config.weight_len()?, "dweight")?;
    require(dbias, config.channels, "dbias")?;
    // SAFETY: `cudnnCausalConv1dBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Same as [`forward`] but with NWH input/output layout:
/// x:  (batch, seqLen, dim)
/// y:  (batch, seqLen, dim)
/// The weight layout is (kernelSize, dim); the bias layout is (dim,).
/// Supported data types are DOUBLE, FLOAT, HALF, and BFLOAT16.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable dimensions, overflowing buffer sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native convolution failure.
pub unsafe fn nwh_forward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    bias: &(impl DeviceInput<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    activation: Activation,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), bias.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    stream.join_regions(&[x.region(), weight.region(), bias.region(), y.region()])?;

    require(x, config.signal_len()?, "x")?;
    require(weight, config.weight_len()?, "weight")?;
    require(bias, config.channels, "bias")?;
    require(y, config.signal_len()?, "y")?;
    // SAFETY: `cudnnCausalConv1dNwhForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Same as cudnnCausalConv1dBackward but with NWH tensor layouts:
/// x, dy, dx: (batch, seqLen, dim)
/// weight, dweight: (kernelSize, dim)
/// bias, dbias: (dim,)
/// dweight and dbias use DOUBLE for DOUBLE inputs and FLOAT otherwise.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable dimensions, overflowing buffer sizes, insufficient buffer capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native convolution failure.
pub unsafe fn nwh_backward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    bias: &(impl DeviceInput<T> + ?Sized),
    dy: &(impl DeviceInput<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dbias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    activation: Activation,
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
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

    require(x, config.signal_len()?, "x")?;
    require(weight, config.weight_len()?, "weight")?;
    require(bias, config.channels, "bias")?;
    require(dy, config.signal_len()?, "dy")?;
    require(dx, config.signal_len()?, "dx")?;
    require(dweight, config.weight_len()?, "dweight")?;
    require(dbias, config.channels, "dbias")?;
    // SAFETY: `cudnnCausalConv1dNwhBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Compute a causal depthwise 1D convolution using the medium FFT path.
///
/// Computes: y = `conv1d_causal(x`, weight)
///
/// For each batch and channel, the causal convolution is:
/// ```text
/// y[t] = sum_{j=0}^{kernelSize - 1} x[t - j] * weight[j]
/// ```
/// where `x[t - j]` is treated as zero when t < j.
///
/// This FFT weight layout is reversed relative to the non-FFT causal conv1d
/// weight layout used by [`forward`]. For example, for
/// ```text
/// weight = [w0, w1, w2], this FFT operation is:
/// ```
/// ```text
/// y[t] = x[t] * w0 + x[t - 1] * w1 + x[t - 2] * w2
/// ```
/// while [`forward`] interprets the same weight tensor as:
/// ```text
/// y[t] = x[t - 2] * w0 + x[t - 1] * w1 + x[t] * w2
/// ```
///
/// The implementation evaluates this convolution in the frequency domain. It
/// transforms zero-padded signal and filter tiles with real-to-complex FFTs,
/// multiplies the transformed signal and filter pointwise, and applies a
/// complex-to-real inverse FFT before writing the causal output samples.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
///
/// # Errors
///
/// Returns an error for invalid or unrepresentable dimensions, unsupported FFT lengths for the element type, overflowing sizes, insufficient capacity, buffers without 16-byte alignment, mismatched CUDA contexts, context binding or stream-dependency failure, or native FFT convolution failure.
pub unsafe fn fft_forward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
    stream.context().bind()?;
    if [x.region(), weight.region(), y.region()]
        .iter()
        .any(|region| region.context() != stream.context())
    {
        return Err(CudaError::MemoryContextMismatch.into());
    }
    stream.join_regions(&[x.region(), weight.region(), y.region()])?;

    config.fft::<T>(false)?;
    require(x, config.signal_len()?, "x")?;
    aligned(x)?;
    require(weight, config.weight_len()?, "weight")?;
    aligned(weight)?;
    require(y, config.signal_len()?, "y")?;
    aligned(y)?;
    // SAFETY: `cudnnFFTCausalConv1dForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Compute gradients for medium FFT causal depthwise 1D convolution.
///
/// Computes gradients for: y = `conv1d_causal(x`, weight)
///
/// The gradient kernels use the same FFT formulation as the forward path:
/// signal, filter, and output-gradient tiles are transformed to the frequency
/// domain, combined with pointwise products equivalent to convolution or
/// correlation, and inverse-transformed to produce dx and dweight.
///
/// The weight and dweight tensors use the FFT weight layout documented by
/// cudnnFFTCausalConv1dForward, which is reversed relative to the non-FFT
/// causal conv1d weight layout.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error for invalid or unrepresentable dimensions, unsupported FFT lengths for the element type, overflowing sizes, insufficient capacity, buffers without 16-byte alignment, mismatched CUDA contexts, context binding or stream-dependency failure, or native FFT convolution failure.
pub unsafe fn fft_backward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    x: &(impl DeviceInput<T> + ?Sized),
    weight: &(impl DeviceInput<T> + ?Sized),
    dy: &(impl DeviceInput<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweight: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
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

    config.fft::<T>(false)?;
    require(x, config.signal_len()?, "x")?;
    aligned(x)?;
    require(weight, config.weight_len()?, "weight")?;
    aligned(weight)?;
    require(dy, config.signal_len()?, "dy")?;
    aligned(dy)?;
    require(dx, config.signal_len()?, "dx")?;
    aligned(dx)?;
    require(dweight, config.weight_len()?, "dweight")?;
    aligned(dweight)?;
    // SAFETY: `cudnnFFTCausalConv1dBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Fused back-to-back causal conv1d: projection conv → elementwise gating → mixer conv.
/// No activation is applied to individual conv outputs (identity only).
///
/// x:            (batch, dim, 3, seqLen)   — 3 interleaved channels per dim
/// weightsProj:  (dim, 3, kernelSizeProj)  — projection filters for all 3 channels
/// weightsMixer: (dim, kernelSizeMixer)
/// skipBias:     (dim,)
/// y:            (batch, dim, seqLen) — intermediate mixer + skip output.
/// Saved and consumed by the backward pass.
/// yGated:       (batch, dim, seqLen) — final post-gated output = y * projX1.
/// This is the primary user-facing output of the fused Hyena-SE block.
///
/// Supported data types are DOUBLE, FLOAT, HALF, and BFLOAT16.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable convolution dimensions or mixer width, overflowing buffer sizes, insufficient capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native fused convolution failure.
pub unsafe fn back_to_back_forward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    mixer_kernel: usize,
    x: &(impl DeviceInput<T> + ?Sized),
    weights_projection: &(impl DeviceInput<T> + ?Sized),
    weights_mixer: &(impl DeviceInput<T> + ?Sized),
    skip_bias: &(impl DeviceInput<T> + ?Sized),
    y: &mut (impl DeviceOutput<T> + ?Sized),
    y_gated: &mut (impl DeviceOutput<T> + ?Sized),
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
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

    require(x, product(&[config.signal_len()?, 3])?, "x")?;
    require(
        weights_projection,
        product(&[config.weight_len()?, 3])?,
        "weights_projection",
    )?;
    require(
        weights_mixer,
        product(&[config.channels, mixer_kernel])?,
        "weights_mixer",
    )?;
    require(skip_bias, config.channels, "skip_bias")?;
    require(y, config.signal_len()?, "y")?;
    require(y_gated, config.signal_len()?, "y_gated")?;
    // SAFETY: `cudnnB2BCausalConv1dForward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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

/// Computes gradients for fused back-to-back causal conv1d.
///
/// x:            (batch, dim, 3, seqLen)   — saved input from forward
/// weightsProj:  (dim, 3, kernelSizeProj)
/// weightsMixer: (dim, kernelSizeMixer)
/// skipBias:     (dim,)
/// y:            (batch, dim, seqLen)      — saved intermediate from forward
/// dy:           (batch, dim, seqLen)      — gradient of yGated
/// dx:           (batch, dim, 3, seqLen)   — gradient of x
/// dweightsProj: (dim, 3, kernelSizeProj)  — gradient of weightsProj (dwDataType)
/// dweightsMixer:(dim, kernelSizeMixer)    — gradient of weightsMixer (dwDataType)
/// dskipBias:    (dim,)                    — gradient of skipBias (dwDataType)
///
/// dwDataType must be DOUBLE when dataType is DOUBLE; otherwise it must be FLOAT.
///
/// # Safety
/// Retain the context or stream, descriptors and all buffer ranges through
/// completion, submission failure and replay. Order allocation readiness and
/// input producers before use; values must be valid when read and writes must
/// stay exclusive. Only the specified tensor elements are produced; padding and
/// tails remain unchanged.
/// Atomic gradient accumulators must already be initialized, usually to zero.
///
/// # Errors
///
/// Returns an error for zero or unrepresentable convolution dimensions or mixer width, overflowing buffer sizes, insufficient capacity, mismatched CUDA contexts, context binding or stream-dependency failure, or native fused convolution failure.
pub unsafe fn back_to_back_backward<T: CausalScalar>(
    stream: &Stream,
    config: Config,
    mixer_kernel: usize,
    x: &(impl DeviceInput<T> + ?Sized),
    weights_projection: &(impl DeviceInput<T> + ?Sized),
    weights_mixer: &(impl DeviceInput<T> + ?Sized),
    skip_bias: &(impl DeviceInput<T> + ?Sized),
    y: &(impl DeviceInput<T> + ?Sized),
    dy: &(impl DeviceInput<T> + ?Sized),
    dx: &mut (impl DeviceOutput<T> + ?Sized),
    dweights_projection: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dweights_mixer: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
    dskip_bias: &mut (impl DeviceOutput<T::Gradient> + ?Sized),
) -> Result<()> {
    let [batch, dim, seq, kernel] = config.dimensions()?;
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

    require(x, product(&[config.signal_len()?, 3])?, "x")?;
    require(
        weights_projection,
        product(&[config.weight_len()?, 3])?,
        "weights_projection",
    )?;
    require(
        weights_mixer,
        product(&[config.channels, mixer_kernel])?,
        "weights_mixer",
    )?;
    require(skip_bias, config.channels, "skip_bias")?;
    require(y, config.signal_len()?, "y")?;
    require(dy, config.signal_len()?, "dy")?;
    require(dx, product(&[config.signal_len()?, 3])?, "dx")?;
    require(
        dweights_projection,
        product(&[config.weight_len()?, 3])?,
        "dweights_projection",
    )?;
    require(
        dweights_mixer,
        product(&[config.channels, mixer_kernel])?,
        "dweights_mixer",
    )?;
    require(dskip_bias, config.channels, "dskip_bias")?;
    // SAFETY: `cudnnB2BCausalConv1dBackward` receives live handles owned by the wrappers, valid output locations and pointers into buffers borrowed for the call.
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
