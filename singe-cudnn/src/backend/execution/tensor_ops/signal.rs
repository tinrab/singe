use std::result::Result as StdResult;

use num_enum::{IntoPrimitive, TryFromPrimitive};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    error::Result,
};

/// Signaling mode for backend signal operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
#[non_exhaustive]
pub enum SignalMode {
    /// The flag variable is updated with the provided signal value atomically.
    Set = sys::cudnnSignalMode_t::CUDNN_SIGNAL_SET as _,
    /// The operation blocks until the flag variable keeps comparing equal to the provided signal value.
    Wait = sys::cudnnSignalMode_t::CUDNN_SIGNAL_WAIT as _,
}

impl_enum_conversion!(sys::cudnnSignalMode_t, SignalMode);

impl_enum_display!(SignalMode, {
    Self::Set => "CUDNN_SIGNAL_SET",
    Self::Wait => "CUDNN_SIGNAL_WAIT",
});

/// Signal attributes and named operands, using native tensors or retained native UIDs.
/// The required input establishes a predecessor; an optional output establishes a successor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalOperationConfig<T> {
    /// Atomic set or equality-wait signaling mode.
    pub mode: SignalMode,
    /// Flag tensor read or updated by the native signaling operation.
    pub flag: T,
    /// Signed native signal value to write or compare.
    pub value: i64,
    /// Required pass-through input establishing the predecessor dependency.
    pub input: T,
    /// Optional pass-through output establishing a successor dependency.
    pub output: Option<T>,
}

impl<T> SignalOperationConfig<T> {
    /// Creates a signal request with its mandatory dependency input and no output.
    pub fn new(mode: SignalMode, flag: T, value: i64, input: T) -> Self {
        Self {
            mode,
            flag,
            value,
            input,
            output: None,
        }
    }

    /// Sets or clears the optional pass-through output without removing the required input.
    #[must_use]
    pub fn with_output(mut self, output: impl Into<Option<T>>) -> Self {
        self.output = output.into();
        self
    }

    /// Borrows owned tensors without changing dependencies, mode or signal value.
    pub fn as_ref(&self) -> SignalOperationConfig<&T> {
        SignalOperationConfig {
            mode: self.mode,
            flag: &self.flag,
            value: self.value,
            input: &self.input,
            output: self.output.as_ref(),
        }
    }

    /// Maps tensor identities while preserving mode, value and operand roles.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> SignalOperationConfig<U> {
        SignalOperationConfig {
            mode: self.mode,
            flag: map(self.flag),
            value: self.value,
            input: map(self.input),
            output: self.output.map(map),
        }
    }

    /// Resolves supplied operands without manufacturing an optional successor.
    ///
    /// # Errors
    ///
    /// Returns the first tensor-mapping error, processing the flag, required input, and optional output in order.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<SignalOperationConfig<U>, E> {
        Ok(SignalOperationConfig {
            mode: self.mode,
            flag: map(self.flag)?,
            value: self.value,
            input: map(self.input)?,
            output: self.output.map(map).transpose()?,
        })
    }
}

/// Finalized native signal operation with retained mode, flag identity, value, and dependency operands.
#[derive(Debug)]
pub struct SignalOperation {
    descriptor: BackendDescriptor,
    config: SignalOperationConfig<BackendTensorUid>,
}

impl SignalOperation {
    /// Creates a signal with its required pass-through input and optional output.
    /// A nonvirtual output's writes complete before a Set operation updates the flag.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects the mode, flag, tensor descriptors or finalization.
    pub fn create(config: SignalOperationConfig<&Tensor>) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationSignal)?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationSignalMode, config.mode)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSignalFlagDesc,
            config.flag.descriptor(),
        )?;
        descriptor.set_attribute_i64(BackendAttributeName::OperationSignalValue, config.value)?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationSignalXDesc,
            config.input.descriptor(),
        )?;
        if let Some(output) = config.output {
            descriptor.set_attribute_descriptor(
                BackendAttributeName::OperationSignalYDesc,
                output.descriptor(),
            )?;
        }
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            config: config.map(Tensor::id),
        })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Retained mode, value and actual native operand identities without native queries.
    #[must_use]
    pub fn config(&self) -> &SignalOperationConfig<BackendTensorUid> {
        &self.config
    }
}
