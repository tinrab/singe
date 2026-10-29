use crate::{
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph},
    },
};

impl Graph {
    /// Names a newly inferred result without changing caller-declared tensors.
    pub(crate) fn inferred_tensor(
        &mut self,
        mut tensor: TensorSpec,
        scope: Option<&str>,
        port: &str,
    ) -> TensorId {
        tensor.name = scope
            .filter(|name| !name.is_empty())
            .map(|name| format!("{name}::{port}"));
        self.tensor(tensor)
    }

    /// Applies the accumulation placement rule to an existing destination declaration.
    pub(crate) fn validate_accumulation_output(
        &self,
        output: TensorId,
        reads_destination: bool,
        operation: &'static str,
    ) -> Result<()> {
        validate_accumulation_placement(
            self.tensor_spec(output)?.is_virtual(),
            reads_destination,
            operation,
        )
    }
}

fn validate_accumulation_placement(
    is_virtual: bool,
    reads_destination: bool,
    operation: &'static str,
) -> Result<()> {
    if reads_destination && is_virtual {
        return Err(FrontendError::AccumulationRequiresOutput { operation }.into());
    }
    Ok(())
}

/// Placement and encoding of an inferred operation result.
/// Bindable results inherit I/O precision; virtual results inherit intermediate precision.
/// An explicit encoding takes precedence over either policy, independently of math precision.
/// Operation-specific metadata results can use a fixed default encoding, such as INT8 resampling indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorOutputConfig {
    data_type: Option<DataType>,
    is_virtual: bool,
}

impl Default for TensorOutputConfig {
    fn default() -> Self {
        Self {
            data_type: None,
            is_virtual: true,
        }
    }
}

impl TensorOutputConfig {
    /// Creates a virtual result with inherited intermediate encoding, matching C++ inferred outputs.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Selects a concrete storage encoding independently of placement and compute precision.
    /// Passing `None` restores the operation's default encoding selection.
    #[must_use]
    pub fn with_data_type(mut self, data_type: impl Into<Option<DataType>>) -> Self {
        self.data_type = data_type.into();
        self
    }

    /// Creates an internal result whose omitted encoding follows intermediate precision.
    #[must_use]
    pub fn virtual_tensor(mut self) -> Self {
        self.is_virtual = true;
        self
    }

    /// Creates a caller-bound result whose omitted encoding follows I/O precision.
    #[must_use]
    pub fn output_tensor(mut self) -> Self {
        self.is_virtual = false;
        self
    }

    /// Returns the explicit result-encoding override, or `None` for placement-based inheritance.
    #[must_use]
    pub fn data_type(self) -> Option<DataType> {
        self.data_type
    }

    /// Returns whether the inferred result uses virtual intermediate placement.
    #[must_use]
    pub fn is_virtual(self) -> bool {
        self.is_virtual
    }

    pub(crate) fn validate_accumulation(
        self,
        reads_destination: bool,
        operation: &'static str,
    ) -> Result<()> {
        validate_accumulation_placement(self.is_virtual, reads_destination, operation)
    }

    pub(crate) fn configured_data_type(self, policy: DataTypePolicy) -> Option<DataType> {
        self.data_type.or_else(|| {
            if self.is_virtual {
                policy.intermediate()
            } else {
                policy.io()
            }
        })
    }

    pub(crate) fn resolve_data_type(self, policy: DataTypePolicy, fallback: DataType) -> DataType {
        self.configured_data_type(policy).unwrap_or(fallback)
    }

    pub(crate) fn into_spec(
        self,
        policy: DataTypePolicy,
        fallback: DataType,
        shape: Shape,
    ) -> TensorSpec {
        let data_type = self.resolve_data_type(policy, fallback);
        let tensor = TensorSpec::new(data_type, shape);
        if self.is_virtual {
            tensor.virtual_tensor()
        } else {
            tensor
        }
    }
}
