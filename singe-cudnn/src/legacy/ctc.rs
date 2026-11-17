//! Connectionist temporal classification loss descriptors, workspace queries, costs, and gradients.

use std::{mem, ptr};

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_display, report_cleanup_failure};
use singe_cuda::{
    stream::AsStream,
    view::{DeviceOutput, DeviceOutputView, DeviceRange, DeviceRegionProvider, DeviceSlice},
};
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::DataType,
    error::{Error, Result},
    legacy::tensor::TensorDescriptor,
    try_ffi,
    utility::{to_size_t, to_usize},
};

singe_core::native_enum! {
    /// Input normalization mode for a loss function.
    /// Used with [`CtcLossDescriptor`] through [`CtcLossOptions`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum LossNormalizationMode: sys::cudnnLossNormalizationMode_t {
        /// The input probabilities are expected to be normalized probabilities, and output gradients are with respect to the unnormalized probabilities.
        None = CUDNN_LOSS_NORMALIZATION_NONE,
        /// The input probabilities are expected to be unnormalized activations from the previous layer, and output gradients are with respect to the activations.
        /// Internally the probability is computed by softmax normalization.
        Softmax = CUDNN_LOSS_NORMALIZATION_SOFTMAX,
    }
}

impl_enum_display!(LossNormalizationMode, {
    LossNormalizationMode::None => "CUDNN_LOSS_NORMALIZATION_NONE",
    LossNormalizationMode::Softmax => "CUDNN_LOSS_NORMALIZATION_SOFTMAX",
});

singe_core::native_enum! {
    /// Behavior for out-of-boundary (OOB) samples in a [`CtcLossDescriptor`].
    /// OOB samples are samples where `L + R > T` is encountered during gradient calculation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CtcGradMode: sys::cudnnCTCGradMode_t {
        /// Skip writing gradient for OOB samples.
        SkipOobGradient = CUDNN_CTC_SKIP_OOB_GRADIENTS,
        /// Zero the gradient for OOB samples (guarantees finite).
        ZeroOobGradient = CUDNN_CTC_ZERO_OOB_GRADIENTS,
    }
}

impl_enum_display!(CtcGradMode, {
    CtcGradMode::SkipOobGradient => "CUDNN_CTC_SKIP_OOB_GRADIENTS",
    CtcGradMode::ZeroOobGradient => "CUDNN_CTC_ZERO_OOB_GRADIENTS",
});

singe_core::native_enum! {
    /// Algorithm used to execute the CTC loss operation.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
    #[repr(u32)]
    #[non_exhaustive]
    pub enum CtcLossAlgorithm: sys::cudnnCTCLossAlgo_t {
        /// Results are guaranteed to be reproducible.
        Deterministic = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
        /// Results are not guaranteed to be reproducible.
        NonDeterministic = CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC,
    }
}

impl_enum_display!(CtcLossAlgorithm, {
    CtcLossAlgorithm::Deterministic => "CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",
    CtcLossAlgorithm::NonDeterministic => "CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",
});

/// Compute precision, input normalization, and gradient-boundary policy for native CTC loss.
#[derive(Debug, Clone, Copy)]
pub struct CtcLossOptions {
    /// Arithmetic precision requested for the loss computation.
    pub compute_type: DataType,
    /// Whether inputs are normalized probabilities or activations requiring softmax.
    pub normalization_mode: LossNormalizationMode,
    /// Whether gradients for out-of-boundary samples are skipped or written as zero.
    pub grad_mode: CtcGradMode,
    /// Maximum label-sequence length declared for any sample.
    pub max_label_length: i32,
}

/// Owns a configured native CTC loss descriptor and retains its declared settings.
#[derive(Debug)]
pub struct CtcLossDescriptor {
    handle: sys::cudnnCTCLossDescriptor_t,
    config: CtcLossOptions,
}

/// A transferred native CTC loss descriptor with the settings it was configured with.
#[derive(Debug)]
pub struct RawCtcLossDescriptor {
    /// The owned native CTC loss descriptor handle.
    pub handle: sys::cudnnCTCLossDescriptor_t,
    /// The compute and gradient settings configured on `handle`.
    pub options: CtcLossOptions,
}

impl CtcLossDescriptor {
    /// Creates a CTC loss descriptor with the requested compute and gradient policies.
    ///
    /// # Errors
    ///
    /// Returns an error if cuDNN rejects descriptor creation or configuration, or returns a null handle.
    pub fn create(config: CtcLossOptions) -> Result<Self> {
        let mut handle = ptr::null_mut();
        // SAFETY: `handle` is a live local that cuDNN writes the new descriptor into.
        unsafe {
            try_ffi!(sys::cudnnCreateCTCLossDescriptor(&raw mut handle))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        let descriptor = Self { handle, config };
        // SAFETY: `descriptor` owns the non-null handle created above, and the remaining arguments are plain values.
        unsafe {
            try_ffi!(sys::cudnnSetCTCLossDescriptor_v9(
                descriptor.handle,
                config.compute_type.into(),
                config.normalization_mode.into(),
                config.grad_mode.into(),
                config.max_label_length,
            ))?;
        }

        Ok(descriptor)
    }

    /// Returns the retained loss settings without querying cuDNN.
    #[must_use]
    pub fn options(&self) -> CtcLossOptions {
        self.config
    }

    /// Returns the retained arithmetic precision.
    #[must_use]
    pub fn compute_type(&self) -> DataType {
        self.config.compute_type
    }

    /// Returns the retained input-normalization policy.
    #[must_use]
    pub fn normalization_mode(&self) -> LossNormalizationMode {
        self.config.normalization_mode
    }

    /// Returns the retained policy for out-of-boundary sample gradients.
    #[must_use]
    pub fn grad_mode(&self) -> CtcGradMode {
        self.config.grad_mode
    }

    /// Returns the declared maximum label-sequence length.
    #[must_use]
    pub fn max_label_length(&self) -> i32 {
        self.config.max_label_length
    }

    /// Returns the borrowed native descriptor handle, valid while this wrapper lives.
    #[must_use]
    pub fn as_raw(&self) -> sys::cudnnCTCLossDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN CTC loss descriptor handle.
    ///
    /// # Safety
    ///
    /// `raw.handle` must be a live CTC loss descriptor created by cuDNN.
    /// The returned wrapper takes ownership and destroys it on drop; no other owner may destroy or keep using it.
    /// `raw.options` must accurately describe the descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw.handle` is null ([`Error::NullHandle`]).
    #[expect(
        clippy::needless_pass_by_value,
        reason = "the raw value transfers ownership of its handle even though its fields are `Copy`"
    )]
    pub unsafe fn from_raw(raw: RawCtcLossDescriptor) -> Result<Self> {
        if raw.handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            handle: raw.handle,
            config: raw.options,
        })
    }

    /// Releases ownership and returns the raw cuDNN CTC loss descriptor handle with its settings.
    ///
    /// The caller becomes responsible for destroying the handle.
    #[must_use]
    pub fn into_raw(mut self) -> RawCtcLossDescriptor {
        RawCtcLossDescriptor {
            handle: mem::replace(&mut self.handle, ptr::null_mut()),
            options: self.config,
        }
    }
}

impl Drop for CtcLossDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        // SAFETY: the handle is non-null and owned by this wrapper, and nothing uses it after drop.
        unsafe {
            if let Err(error) = try_ffi!(sys::cudnnDestroyCTCLossDescriptor(self.handle)) {
                report_cleanup_failure("destroy", "cudnn ctc loss descriptor", &error);
            }
        }
    }
}

fn validate_labels(
    descriptor: &CtcLossDescriptor,
    probabilities: &TensorDescriptor<f32>,
    gradients: &TensorDescriptor<f32>,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
) -> Result<()> {
    let dims = probabilities.dimensions();
    if descriptor.compute_type() != DataType::F32
        || dims.len() != 3
        || dims != gradients.dimensions()
        || dims.iter().any(|&d| d <= 0)
    {
        return Err(Error::InvalidDataShape);
    }

    let (steps, batch, classes) = (dims[0], to_usize(dims[1], "ctc batch")?, dims[2]);
    if label_lengths.len() != batch
        || input_lengths.len() != batch
        || input_lengths.iter().any(|&n| n <= 0 || n > steps)
    {
        return Err(Error::InvalidDataShape);
    }

    let required = label_lengths.iter().try_fold(0usize, |sum, &n| {
        if n < 0 || n > descriptor.max_label_length() {
            return Err(Error::InvalidDataShape);
        }
        sum.checked_add(to_usize(n, "ctc label length")?)
            .ok_or(Error::InvalidDataShape)
    })?;
    if labels.len() != required || labels.iter().any(|&label| label < 0 || label >= classes) {
        return Err(Error::InvalidDataShape);
    }
    Ok(())
}

/// Returns the amount of GPU workspace memory needed by [`ctc_loss_async`] with the specified algorithm and descriptors.
///
/// # Errors
///
/// Returns an error if the probability and gradient descriptors are incompatible, if `input_lengths` does not agree with the first probability dimension, if any label length exceeds cuDNN's supported range, if the workspace size cannot be represented as `usize`, or if cuDNN does not support the selected data type, compute type, or algorithm.
pub fn ctc_loss_workspace_size(
    context: &Context,
    algorithm: CtcLossAlgorithm,
    descriptor: &CtcLossDescriptor,
    probabilities_descriptor: &TensorDescriptor<f32>,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    gradients_descriptor: &TensorDescriptor<f32>,
) -> Result<usize> {
    context.bind()?;

    validate_labels(
        descriptor,
        probabilities_descriptor,
        gradients_descriptor,
        labels,
        label_lengths,
        input_lengths,
    )?;

    let mut size_in_bytes = 0;
    // SAFETY: the handle and descriptors are owned by their live wrappers, `validate_labels` checked the host label slices against the probability shape, and `size_in_bytes` is a writable local.
    unsafe {
        try_ffi!(sys::cudnnGetCTCLossWorkspaceSize(
            context.as_raw(),
            probabilities_descriptor.as_raw(),
            gradients_descriptor.as_raw(),
            labels.as_ptr(),
            label_lengths.as_ptr(),
            input_lengths.as_ptr(),
            algorithm.into(),
            descriptor.as_raw(),
            &raw mut size_in_bytes,
        ))?;
    }

    to_usize(size_in_bytes, "ctc workspace size")
}

/// Enqueues f32 CTC costs and gradients using host label metadata.
///
/// Host-label CTC cannot run under graph capture, so this function rejects a capturing stream.
///
/// # Safety
///
/// Retain the context, descriptors, host slices and all device ranges through completion and failed submission.
/// Order allocation readiness and initialized probabilities before use; keep costs, gradients and workspace exclusive.
/// Gradient elements skipped by the configured out-of-bounds policy, padding and tails are not initialized by this operation.
///
/// # Errors
///
/// Returns an error if context binding fails, the handle's stream is capturing, any device range belongs to another CUDA context, tensor or cost storage is too small, probability and gradient shapes or host label metadata are invalid, the workspace query or size conversion fails, workspace capacity is insufficient, or cuDNN rejects the loss operation.
pub unsafe fn ctc_loss_async(
    context: &Context,
    algorithm: CtcLossAlgorithm,
    descriptor: &CtcLossDescriptor,
    probabilities_descriptor: &TensorDescriptor<f32>,
    probabilities: &(impl DeviceRange<f32> + ?Sized),
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    costs: &mut (impl DeviceOutput<f32> + ?Sized),
    gradients_descriptor: &TensorDescriptor<f32>,
    gradients: &mut (impl DeviceOutput<f32> + ?Sized),
    mut workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    context.bind()?;
    context.stream_binding().ensure_not_capturing()?;
    let mut regions = vec![probabilities.region(), costs.region(), gradients.region()];
    regions.extend(workspace.as_ref().map(|workspace| workspace.region()));
    context.prepare_regions(&regions)?;

    if probabilities.region().byte_len() < probabilities_descriptor.byte_len()?
        || gradients.region().byte_len() < gradients_descriptor.byte_len()?
        || costs.len() < label_lengths.len()
    {
        return Err(Error::InvalidDataShape);
    }

    validate_labels(
        descriptor,
        probabilities_descriptor,
        gradients_descriptor,
        labels,
        label_lengths,
        input_lengths,
    )?;

    let required_workspace_size = {
        let mut size_in_bytes = 0;
        // SAFETY: the handle and descriptors are owned by their live wrappers, `validate_labels` checked the host label slices against the probability shape, and `size_in_bytes` is a writable local.
        unsafe {
            try_ffi!(sys::cudnnGetCTCLossWorkspaceSize(
                context.as_raw(),
                probabilities_descriptor.as_raw(),
                gradients_descriptor.as_raw(),
                labels.as_ptr(),
                label_lengths.as_ptr(),
                input_lengths.as_ptr(),
                algorithm.into(),
                descriptor.as_raw(),
                &raw mut size_in_bytes,
            ))?;
        }

        to_usize(size_in_bytes, "ctc workspace size")?
    };
    let workspace_size = workspace
        .as_ref()
        .map_or(0, |workspace| workspace.region().byte_len());
    if workspace_size < required_workspace_size {
        return Err(Error::BufferTooSmall {
            name: "workspace".into(),
            required: required_workspace_size,
            actual: workspace_size,
        });
    }

    let workspace_ptr = workspace
        .as_mut()
        .map_or(ptr::null_mut(), |workspace| workspace.as_mut_ptr().cast());

    // SAFETY: the buffers share the handle's context and cover their descriptors, the labels were validated, and the workspace holds at least the queried size.
    // The caller retains and orders every buffer and host slice through completion, per this function's contract.
    unsafe {
        try_ffi!(sys::cudnnCTCLoss(
            context.as_raw(),
            probabilities_descriptor.as_raw(),
            probabilities.as_device_ptr().cast(),
            labels.as_ptr().cast(),
            label_lengths.as_ptr().cast(),
            input_lengths.as_ptr().cast(),
            costs.as_mut_ptr().cast(),
            gradients_descriptor.as_raw(),
            gradients.as_mut_ptr().cast(),
            algorithm.into(),
            descriptor.as_raw(),
            workspace_ptr,
            to_size_t(workspace_size, "ctc workspace size")?,
        ))?;
    }

    Ok(())
}

/// Computes f32 CTC costs and gradients using host label metadata, and waits for them.
///
/// # Errors
///
/// Returns an error if the handle's stream is capturing, [`ctc_loss_async`] fails, or the wait fails.
pub fn ctc_loss(
    context: &Context,
    algorithm: CtcLossAlgorithm,
    descriptor: &CtcLossDescriptor,
    probabilities_descriptor: &TensorDescriptor<f32>,
    probabilities: &(impl DeviceSlice<f32> + ?Sized),
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    costs: &mut (impl DeviceOutput<f32> + ?Sized),
    gradients_descriptor: &TensorDescriptor<f32>,
    gradients: &mut (impl DeviceOutput<f32> + ?Sized),
    workspace: Option<DeviceOutputView<'_, u8>>,
) -> Result<()> {
    context.sync_scope(|| {
        // SAFETY: `probabilities` is initialized, and the borrows outlive the wait in `sync_scope`.
        unsafe {
            ctc_loss_async(
                context,
                algorithm,
                descriptor,
                probabilities_descriptor,
                probabilities,
                labels,
                label_lengths,
                input_lengths,
                costs,
                gradients_descriptor,
                gradients,
                workspace,
            )
        }
    })
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use crate::testing::setup_context;

    #[test]
    fn test_ctc_loss_descriptor_create() -> Result<()> {
        let descriptor = CtcLossDescriptor::create(CtcLossOptions {
            compute_type: DataType::F32,
            normalization_mode: LossNormalizationMode::Softmax,
            grad_mode: CtcGradMode::SkipOobGradient,
            max_label_length: 8,
        })?;

        assert_eq!(descriptor.options().compute_type, DataType::F32);
        assert_eq!(
            descriptor.options().normalization_mode,
            LossNormalizationMode::Softmax
        );
        assert_eq!(descriptor.options().grad_mode, CtcGradMode::SkipOobGradient);
        assert_eq!(descriptor.options().max_label_length, 8);

        Ok(())
    }

    #[test]
    fn test_ctc_loss_workspace_size_query() -> Result<()> {
        let test = setup_context()?;
        let test_context = test.context();

        let descriptor = CtcLossDescriptor::create(CtcLossOptions {
            compute_type: DataType::F32,
            normalization_mode: LossNormalizationMode::Softmax,
            grad_mode: CtcGradMode::SkipOobGradient,
            max_label_length: 4,
        })?;
        let probabilities_descriptor = TensorDescriptor::<f32>::create_contiguous(&[4, 2, 3])?;
        let gradients_descriptor = TensorDescriptor::<f32>::create_contiguous(&[4, 2, 3])?;

        let workspace_size = ctc_loss_workspace_size(
            test_context,
            CtcLossAlgorithm::Deterministic,
            &descriptor,
            &probabilities_descriptor,
            &[0, 1, 0, 1],
            &[2, 2],
            &[4, 4],
            &gradients_descriptor,
        )?;

        assert!(workspace_size > 0);

        Ok(())
    }
}
