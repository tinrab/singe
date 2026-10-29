use num_enum::{IntoPrimitive, TryFromPrimitive};

use serde::{Deserialize, Serialize};

use singe_core::{impl_enum_conversion, impl_enum_display};

use singe_cudnn_sys as sys;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        execution::tensor_ops::TensorTransformMetadata,
        tensor::Tensor,
    },
    error::{Error, Result},
};

/// Whether reshape preserves logical element order or only changes the view.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive, Serialize, Deserialize,
)]
#[repr(u32)]
#[non_exhaustive]
pub enum ReshapeMode {
    ///< View-only reshape (no data movement). @since cuDNN 9.22.0
    ViewOnly = sys::cudnnBackendReshapeMode_t::CUDNN_RESHAPE_VIEW_ONLY as _,
    ///< Logical reshape (may involve data movement). @since cuDNN 9.22.0
    Logical = sys::cudnnBackendReshapeMode_t::CUDNN_RESHAPE_LOGICAL as _,
}

impl_enum_conversion!(sys::cudnnBackendReshapeMode_t, ReshapeMode);

impl_enum_display!(ReshapeMode, {
    Self::ViewOnly => "CUDNN_RESHAPE_VIEW_ONLY",
    Self::Logical => "CUDNN_RESHAPE_LOGICAL",
});

/// Native reshape mode and diagnostic scope shared with frontend construction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReshapeConfig {
    mode: ReshapeMode,
    #[serde(default)]
    name: Option<String>,
}

impl ReshapeConfig {
    /// Records logical or view-only reshape mode without a diagnostic scope.
    #[must_use]
    pub fn new(mode: ReshapeMode) -> Self {
        Self { mode, name: None }
    }

    /// Replaces logical or view-only reshape mode while retaining the diagnostic scope.
    #[must_use]
    pub fn with_mode(mut self, mode: ReshapeMode) -> Self {
        self.mode = mode;
        self
    }

    /// Names inferred results `name::Y` without replacing explicit output names.
    /// An empty scope disables inferred naming.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Returns the logical or view-only reshape mode.
    #[must_use]
    pub fn mode(&self) -> ReshapeMode {
        self.mode
    }

    /// Returns the nonempty output-name scope, or `None` when unnamed.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref().filter(|name| !name.is_empty())
    }
}

/// Finalized native reshape operation with retained input/output identities and mode.
#[derive(Debug)]
pub struct ReshapeOperation {
    descriptor: BackendDescriptor,
    metadata: TensorTransformMetadata<ReshapeConfig>,
}

impl ReshapeOperation {
    /// Creates a reshape from independently declared input and output descriptors.
    ///
    /// # Errors
    ///
    /// Returns an error if logical reshape encodings differ, logical element counts differ, or native descriptor creation, attribute assignment, or finalization fails.
    pub fn create(x: &Tensor, y: &Tensor, config: ReshapeConfig) -> Result<Self> {
        if config.mode() == ReshapeMode::Logical && x.data_type() != y.data_type() {
            return Err(Error::DescriptorMismatch {
                name: "reshape data type".into(),
            });
        }
        if !x.has_same_logical_element_count(y) {
            return Err(Error::DescriptorMismatch {
                name: "reshape element count".into(),
            });
        }

        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationReshape)?;
        descriptor.set_attribute_enum(BackendAttributeName::OperationReshapeMode, config.mode())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReshapeXDesc,
            x.descriptor(),
        )?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationReshapeYDesc,
            y.descriptor(),
        )?;
        descriptor.finalize()?;

        Ok(Self {
            descriptor,
            metadata: TensorTransformMetadata {
                input: x.id(),
                output: y.id(),
                config,
            },
        })
    }

    /// Borrows the finalized native operation descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    /// Borrows the retained reshape mode and diagnostic scope.
    #[must_use]
    pub fn config(&self) -> &ReshapeConfig {
        &self.metadata.config
    }

    /// Native tensor identities with the retained mode and diagnostic scope.
    #[must_use]
    pub fn metadata(&self) -> &TensorTransformMetadata<ReshapeConfig> {
        &self.metadata
    }
}
