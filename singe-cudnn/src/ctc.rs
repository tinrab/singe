use std::ptr;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use singe_core::{impl_enum_conversion, impl_enum_display};
use singe_cuda::memory::DeviceMemory;
use singe_cudnn_sys as sys;

use crate::{
    context::Context,
    data_type::{DataType, DataTypeLike},
    error::{Error, Result},
    tensor::TensorDescriptor,
    try_ffi,
    utility::to_usize,
};

/// Input normalization mode for a loss function.
/// Used with [`CtcLossDescriptor`] through [`CtcLossConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum LossNormalizationMode {
    /// The input probabilities are expected to be normalized probabilities, and output gradients
    /// are with respect to the unnormalized probabilities.
    None = sys::cudnnLossNormalizationMode_t::CUDNN_LOSS_NORMALIZATION_NONE as _,
    /// The input probabilities are expected to be unnormalized activations from the previous
    /// layer, and output gradients are with respect to the activations.
    /// Internally the probability is computed by softmax normalization.
    Softmax = sys::cudnnLossNormalizationMode_t::CUDNN_LOSS_NORMALIZATION_SOFTMAX as _,
}

impl_enum_conversion!(sys::cudnnLossNormalizationMode_t, LossNormalizationMode);

impl_enum_display!(LossNormalizationMode, {
    LossNormalizationMode::None => "CUDNN_LOSS_NORMALIZATION_NONE",
    LossNormalizationMode::Softmax => "CUDNN_LOSS_NORMALIZATION_SOFTMAX",
});

/// Behavior for out-of-boundary (OOB) samples in a [`CtcLossDescriptor`].
/// OOB samples are samples where `L + R > T` is encountered during gradient calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CtcGradMode {
    SkipOobGradient = sys::cudnnCTCGradMode_t::CUDNN_CTC_SKIP_OOB_GRADIENTS as _,
    ZeroOobGradient = sys::cudnnCTCGradMode_t::CUDNN_CTC_ZERO_OOB_GRADIENTS as _,
}

impl_enum_conversion!(sys::cudnnCTCGradMode_t, CtcGradMode);

impl_enum_display!(CtcGradMode, {
    CtcGradMode::SkipOobGradient => "CUDNN_CTC_SKIP_OOB_GRADIENTS",
    CtcGradMode::ZeroOobGradient => "CUDNN_CTC_ZERO_OOB_GRADIENTS",
});

/// Algorithm used to execute the CTC loss operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum CtcLossAlgorithm {
    /// Results are guaranteed to be reproducible.
    Deterministic = sys::cudnnCTCLossAlgo_t::CUDNN_CTC_LOSS_ALGO_DETERMINISTIC as _,
    /// Results are not guaranteed to be reproducible.
    NonDeterministic = sys::cudnnCTCLossAlgo_t::CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC as _,
}

impl_enum_conversion!(sys::cudnnCTCLossAlgo_t, CtcLossAlgorithm);

impl_enum_display!(CtcLossAlgorithm, {
    CtcLossAlgorithm::Deterministic => "CUDNN_CTC_LOSS_ALGO_DETERMINISTIC",
    CtcLossAlgorithm::NonDeterministic => "CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC",
});

#[derive(Debug, Clone, Copy)]
pub struct CtcLossConfig {
    pub compute_type: DataType,
    pub normalization_mode: LossNormalizationMode,
    pub grad_mode: CtcGradMode,
    pub max_label_length: i32,
}

#[derive(Debug)]
pub struct CtcLossDescriptor {
    handle: sys::cudnnCTCLossDescriptor_t,
    config: CtcLossConfig,
}

impl CtcLossDescriptor {
    pub fn create(config: CtcLossConfig) -> Result<Self> {
        let mut handle = ptr::null_mut();
        unsafe {
            try_ffi!(sys::cudnnCreateCTCLossDescriptor(&raw mut handle))?;
            try_ffi!(sys::cudnnSetCTCLossDescriptor_v9(
                handle,
                config.compute_type.into(),
                config.normalization_mode.into(),
                config.grad_mode.into(),
                config.max_label_length,
            ))?;
        }

        if handle.is_null() {
            return Err(Error::NullHandle);
        }

        Ok(Self { handle, config })
    }

    pub fn config(&self) -> CtcLossConfig {
        self.config
    }

    pub fn compute_type(&self) -> DataType {
        self.config.compute_type
    }

    pub fn normalization_mode(&self) -> LossNormalizationMode {
        self.config.normalization_mode
    }

    pub fn grad_mode(&self) -> CtcGradMode {
        self.config.grad_mode
    }

    pub fn max_label_length(&self) -> i32 {
        self.config.max_label_length
    }

    pub fn as_raw(&self) -> sys::cudnnCTCLossDescriptor_t {
        self.handle
    }

    /// Takes ownership of a raw cuDNN CTC loss descriptor handle.
    ///
    /// # Safety
    ///
    /// `handle` must be a valid `cudnnCTCLossDescriptor_t` created by cuDNN.
    /// The returned wrapper takes ownership and will destroy it with
    /// `cudnnDestroyCTCLossDescriptor`; no other owner may destroy or keep
    /// using it. `config` must accurately describe the descriptor.
    pub unsafe fn from_raw(
        handle: sys::cudnnCTCLossDescriptor_t,
        config: CtcLossConfig,
    ) -> Result<Self> {
        if handle.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self { handle, config })
    }

    /// Releases ownership and returns the raw cuDNN CTC loss descriptor handle.
    ///
    /// The caller becomes responsible for destroying the handle.
    pub fn into_raw(self) -> sys::cudnnCTCLossDescriptor_t {
        let handle = self.handle;
        std::mem::forget(self);
        handle
    }
}

impl Drop for CtcLossDescriptor {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }

        unsafe {
            if let Err(err) = try_ffi!(sys::cudnnDestroyCTCLossDescriptor(self.handle)) {
                #[cfg(debug_assertions)]
                eprintln!("failed to destroy cudnn ctc loss descriptor: {err}");
            }
        }
    }
}

/// Returns the amount of GPU workspace memory needed by [`ctc_loss`] with the specified
/// algorithm and descriptors.
///
/// # Errors
///
/// Returns an error if the probability and gradient descriptors are
/// incompatible, if `input_lengths` does not agree with the first probability
/// dimension, if any label length exceeds cuDNN's supported range, if the
/// workspace size cannot be represented as `usize`, or if cuDNN does not support
/// the selected data type, compute type, or algorithm.
pub fn ctc_loss_workspace_size<T: DataTypeLike>(
    ctx: &Context,
    algorithm: CtcLossAlgorithm,
    descriptor: &CtcLossDescriptor,
    probabilities_descriptor: &TensorDescriptor<T>,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    gradients_descriptor: &TensorDescriptor<T>,
) -> Result<usize> {
    ctx.bind()?;

    let labels = labels.to_vec();
    let label_lengths = label_lengths.to_vec();
    let input_lengths = input_lengths.to_vec();

    let mut size_in_bytes = 0;
    unsafe {
        try_ffi!(sys::cudnnGetCTCLossWorkspaceSize(
            ctx.as_raw(),
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

pub fn ctc_loss<T: DataTypeLike>(
    ctx: &Context,
    algorithm: CtcLossAlgorithm,
    descriptor: &CtcLossDescriptor,
    probabilities_descriptor: &TensorDescriptor<T>,
    probabilities: &DeviceMemory<T>,
    labels: &[i32],
    label_lengths: &[i32],
    input_lengths: &[i32],
    costs: &mut DeviceMemory<T>,
    gradients_descriptor: &TensorDescriptor<T>,
    gradients: &mut DeviceMemory<T>,
    workspace: &mut DeviceMemory<u8>,
) -> Result<()> {
    ctx.bind()?;

    let labels = labels.to_vec();
    let label_lengths = label_lengths.to_vec();
    let input_lengths = input_lengths.to_vec();

    let required_workspace_size = {
        let mut size_in_bytes = 0;
        unsafe {
            try_ffi!(sys::cudnnGetCTCLossWorkspaceSize(
                ctx.as_raw(),
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
    if workspace.byte_len() < required_workspace_size {
        return Err(Error::InsufficientWorkspaceSize {
            required: required_workspace_size,
            actual: workspace.byte_len(),
        });
    }

    unsafe {
        try_ffi!(sys::cudnnCTCLoss(
            ctx.as_raw(),
            probabilities_descriptor.as_raw(),
            probabilities.as_ptr() as _,
            labels.as_ptr() as _,
            label_lengths.as_ptr() as _,
            input_lengths.as_ptr() as _,
            costs.as_mut_ptr() as _,
            gradients_descriptor.as_raw(),
            gradients.as_mut_ptr() as _,
            algorithm.into(),
            descriptor.as_raw(),
            workspace.as_mut_ptr() as _,
            workspace.byte_len() as _,
        ))?;
    }

    Ok(())
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    use crate::testing::setup_context;

    #[test]
    fn test_ctc_loss_descriptor_create() -> Result<()> {
        let descriptor = CtcLossDescriptor::create(CtcLossConfig {
            compute_type: DataType::F32,
            normalization_mode: LossNormalizationMode::Softmax,
            grad_mode: CtcGradMode::SkipOobGradient,
            max_label_length: 8,
        })?;

        assert_eq!(descriptor.config().compute_type, DataType::F32);
        assert_eq!(
            descriptor.config().normalization_mode,
            LossNormalizationMode::Softmax
        );
        assert_eq!(descriptor.config().grad_mode, CtcGradMode::SkipOobGradient);
        assert_eq!(descriptor.config().max_label_length, 8);

        Ok(())
    }

    #[test]
    fn test_ctc_loss_workspace_size_query() -> Result<()> {
        let test_context = setup_context()?;

        let descriptor = CtcLossDescriptor::create(CtcLossConfig {
            compute_type: DataType::F32,
            normalization_mode: LossNormalizationMode::Softmax,
            grad_mode: CtcGradMode::SkipOobGradient,
            max_label_length: 4,
        })?;
        let probabilities_descriptor = TensorDescriptor::<f32>::create_contiguous(&[4, 2, 3])?;
        let gradients_descriptor = TensorDescriptor::<f32>::create_contiguous(&[4, 2, 3])?;

        let workspace_size = ctc_loss_workspace_size(
            &test_context,
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
