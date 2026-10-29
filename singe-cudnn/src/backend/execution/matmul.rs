mod problem;

use std::result::Result as StdResult;

pub use problem::MatmulProblemKey;

use crate::{
    backend::{
        attribute::BackendAttributeName,
        descriptor::{BackendDescriptor, BackendDescriptorType},
        tensor::{BackendTensorUid, Tensor},
    },
    data_type::DataType,
    error::{Error, Result},
};

/// Resolved native matmul attributes, independent of operand and result encodings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatmulDescriptorConfig {
    /// Arithmetic precision used by the native matrix multiplication.
    pub compute_type: DataType,
    /// `None` leaves native padding unset; `Some(0.0)` explicitly requests zero padding.
    pub padding_value: Option<f64>,
}

impl MatmulDescriptorConfig {
    /// Selects compute precision without setting the optional native padding attribute.
    #[must_use]
    pub fn new(compute_type: DataType) -> Self {
        Self {
            compute_type,
            padding_value: None,
        }
    }

    /// Replaces the optional native padding value; `None` leaves the attribute unset at descriptor creation.
    #[must_use]
    pub fn with_padding_value(mut self, padding_value: impl Into<Option<f64>>) -> Self {
        self.padding_value = padding_value.into();
        self
    }
}

/// Finalized native matmul descriptor with retained compute and padding attributes.
#[derive(Debug)]
pub struct MatmulDescriptor {
    descriptor: BackendDescriptor,
    config: MatmulDescriptorConfig,
}

impl MatmulDescriptor {
    /// Creates a native descriptor with explicit compute precision and optional padding.
    ///
    /// # Errors
    /// Returns an error if cuDNN rejects creation, attributes or finalization.
    pub fn create(config: MatmulDescriptorConfig) -> Result<Self> {
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::Matmul)?;
        descriptor.set_attribute_enum(BackendAttributeName::MatmulCompType, config.compute_type)?;
        if let Some(padding_value) = config.padding_value {
            descriptor
                .set_attribute_f64(BackendAttributeName::MatmulPaddingValue, padding_value)?;
        }
        descriptor.finalize()?;
        Ok(Self { descriptor, config })
    }

    /// Retained attributes without native queries or allocation.
    #[must_use]
    pub fn config(&self) -> MatmulDescriptorConfig {
        self.config
    }

    /// Borrows the finalized native descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}

/// Optional operands and batch addressing for native matmul construction.
/// cuDNN finalization and engine selection determine supported combinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulOperationConfig<T> {
    /// Optional tensor supplying the output row-count override M.
    pub m_override: Option<T>,
    /// Optional tensor supplying the output column-count override N.
    pub n_override: Option<T>,
    /// Optional tensor supplying the contraction-dimension override K.
    pub k_override: Option<T>,
    /// Optional irregular batch count; must be positive when creating the operation.
    pub irregularly_strided_batch_count: Option<i64>,
}

impl<T> MatmulOperationConfig<T> {
    /// Borrows optional dimension operands without cloning their owners.
    pub fn as_ref(&self) -> MatmulOperationConfig<&T> {
        MatmulOperationConfig {
            m_override: self.m_override.as_ref(),
            n_override: self.n_override.as_ref(),
            k_override: self.k_override.as_ref(),
            irregularly_strided_batch_count: self.irregularly_strided_batch_count,
        }
    }

    /// Maps supplied dimension operands without changing omissions or batch addressing.
    pub fn map<U>(self, mut map: impl FnMut(T) -> U) -> MatmulOperationConfig<U> {
        MatmulOperationConfig {
            m_override: self.m_override.map(&mut map),
            n_override: self.n_override.map(&mut map),
            k_override: self.k_override.map(map),
            irregularly_strided_batch_count: self.irregularly_strided_batch_count,
        }
    }

    /// Resolves supplied dimension operands while preserving native addressing settings.
    ///
    /// # Errors
    ///
    /// Returns the first mapping error, visiting supplied M, N, and K operands in that order and skipping absent operands.
    pub fn try_map<U, E>(
        self,
        mut map: impl FnMut(T) -> StdResult<U, E>,
    ) -> StdResult<MatmulOperationConfig<U>, E> {
        Ok(MatmulOperationConfig {
            m_override: self.m_override.map(&mut map).transpose()?,
            n_override: self.n_override.map(&mut map).transpose()?,
            k_override: self.k_override.map(map).transpose()?,
            irregularly_strided_batch_count: self.irregularly_strided_batch_count,
        })
    }
}

impl<T> Default for MatmulOperationConfig<T> {
    fn default() -> Self {
        Self {
            m_override: None,
            n_override: None,
            k_override: None,
            irregularly_strided_batch_count: None,
        }
    }
}

/// Finalized matmul operands and attributes, independent of frontend tensor identities.
/// Optional dimension overrides remain separate from the concrete A/B/C descriptor shapes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatmulOperationMetadata {
    /// Native identity of the left matrix operand A.
    pub a: BackendTensorUid,
    /// Native identity of the right matrix operand B.
    pub b: BackendTensorUid,
    /// Native identity of the result matrix C.
    pub c: BackendTensorUid,
    /// Retained optional dimension-override identities and irregular batch count.
    pub config: MatmulOperationConfig<BackendTensorUid>,
    /// Retained compute precision and optional padding value.
    pub descriptor: MatmulDescriptorConfig,
}

/// Finalized native matrix multiplication with retained operand identities, math attributes, and plan-cache declaration key.
#[derive(Debug)]
pub struct MatmulOperation {
    descriptor: BackendDescriptor,
    metadata: MatmulOperationMetadata,
    problem_key: MatmulProblemKey,
}

impl MatmulOperation {
    /// Creates native matmul with independently optional dimension overrides and batch addressing.
    ///
    /// # Errors
    /// Returns an error for a nonpositive irregular batch count or native descriptor failure.
    pub fn create(
        matmul: &MatmulDescriptor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        config: MatmulOperationConfig<&Tensor>,
    ) -> Result<Self> {
        if config
            .irregularly_strided_batch_count
            .is_some_and(|count| count <= 0)
        {
            return Err(Error::out_of_range("irregularly_strided_batch_count"));
        }
        let mut descriptor = BackendDescriptor::create(BackendDescriptorType::OperationMatmul)?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulADesc, a.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulBDesc, b.descriptor())?;
        descriptor
            .set_attribute_descriptor(BackendAttributeName::OperationMatmulCDesc, c.descriptor())?;
        descriptor.set_attribute_descriptor(
            BackendAttributeName::OperationMatmulDesc,
            matmul.descriptor(),
        )?;
        for (attribute, tensor) in [
            (
                BackendAttributeName::OperationMatmulGemmMOverrideDesc,
                config.m_override,
            ),
            (
                BackendAttributeName::OperationMatmulGemmNOverrideDesc,
                config.n_override,
            ),
            (
                BackendAttributeName::OperationMatmulGemmKOverrideDesc,
                config.k_override,
            ),
        ] {
            if let Some(tensor) = tensor {
                descriptor.set_attribute_descriptor(attribute, tensor.descriptor())?;
            }
        }
        if let Some(count) = config.irregularly_strided_batch_count {
            descriptor.set_attribute_i64(
                BackendAttributeName::OperationMatmulIrregularlyStridedBatchCount,
                count,
            )?;
        }
        descriptor.finalize()?;
        let problem_key = MatmulProblemKey::new(matmul.config(), [a, b, c], config);
        Ok(Self {
            descriptor,
            metadata: MatmulOperationMetadata {
                a: a.id(),
                b: b.id(),
                c: c.id(),
                config: config.map(Tensor::id),
                descriptor: matmul.config(),
            },
            problem_key,
        })
    }

    /// Retained operand UIDs, dimension overrides, batch addressing and math attributes.
    /// Does not query native descriptors, allocate or require the original tensor handles.
    #[must_use]
    pub fn metadata(&self) -> &MatmulOperationMetadata {
        &self.metadata
    }

    /// Retained declaration key for native plan caching, independent of device addresses.
    /// Scope the cache to the compatible context, complete graph and compilation policy.
    /// Does not query cuDNN or allocate.
    #[must_use]
    pub fn problem_key(&self) -> &MatmulProblemKey {
        &self.problem_key
    }

    /// Borrows the finalized native descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }
}
