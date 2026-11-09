use crate::{
    backend::registry::{
        BackendExecution, Operation, OperationAttributeKind, OperationAttributes, OutputRole,
        infer::infer_operation, operation_schema_for_attributes, operation_support,
    },
    data_type::DataType,
    error::{Error, Result, UnsupportedTarget},
    value::ValueSpec,
};

/// Canonical opcode, typed attributes, and storage-free operand specifications.
#[derive(Debug, Clone)]
pub struct OperationRequest {
    operation: Operation,
    attributes: OperationAttributes,
    inputs: Vec<ValueSpec>,
}

/// A logical result's relationship to input storage.
/// Physical view compatibility and materialization decisions are checked during execution preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultAlias {
    /// The result requires storage independent of the inputs.
    Independent,
    /// The result is a logical view of the input at the given zero-based index.
    ViewOf(usize),
    /// The result may reuse storage of the input at the given zero-based index.
    MayAlias(usize),
}

/// Logical result specifications, input storage relationships, and computation data types.
/// The specifications do not establish backend support or allocate tensor storage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationInference {
    pub(crate) outputs: Vec<ValueSpec>,
    pub(crate) aliases: Vec<ResultAlias>,
    pub(crate) computation_data_type: DataType,
    pub(crate) kernel_input_data_types: Vec<DataType>,
}

/// A validated logical request, ready for provider/layout capability selection.
#[derive(Debug, Clone)]
pub struct CheckedOperationRequest {
    request: OperationRequest,
    inference: OperationInference,
}

impl OperationRequest {
    /// Creates a request without validating its attributes, operands, or inferred results.
    /// Use `infer` or `check` to validate its logical contract.
    #[must_use]
    pub fn new(
        operation: Operation,
        attributes: OperationAttributes,
        inputs: Vec<ValueSpec>,
    ) -> Self {
        Self {
            operation,
            attributes,
            inputs,
        }
    }

    /// Returns the requested operation code.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the typed operation attributes.
    #[must_use]
    pub fn attributes(&self) -> &OperationAttributes {
        &self.attributes
    }

    /// Returns the operand specifications in operation input order.
    #[must_use]
    pub fn inputs(&self) -> &[ValueSpec] {
        &self.inputs
    }

    /// Infers ordered results and aliases without tensor data, allocation, device state, or backend/provider availability.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute kind or operand count is invalid, shapes or data types are incompatible with the operation, an attribute is out of range, or result inference cannot represent a required shape or size.
    pub fn infer(&self) -> Result<OperationInference> {
        if attribute_kind(&self.attributes) != Some(self.operation.attribute_kind()) {
            return Err(Error::invalid_argument("operation attribute kind"));
        }

        let schema = operation_schema_for_attributes(self.operation, &self.attributes);
        if !schema.operands().accepts_count(self.inputs.len()) {
            return Err(Error::invalid_argument("operation operand count"));
        }
        infer_operation(self, &schema)
    }

    /// Validates logical inference and retains the request with its inferred results.
    /// Backend support and physical layout compatibility require separate execution checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the attribute kind or operand count is invalid, shapes or data types are incompatible with the operation, an attribute is out of range, or result inference cannot represent a required shape or size.
    pub fn check(self) -> Result<CheckedOperationRequest> {
        let inference = self.infer()?;
        Ok(CheckedOperationRequest {
            request: self,
            inference,
        })
    }
}

impl OperationInference {
    /// Returns the inferred result specifications in operation output order.
    #[must_use]
    pub fn outputs(&self) -> &[ValueSpec] {
        &self.outputs
    }

    /// Returns each result's logical relationship to input storage, in output order.
    #[must_use]
    pub fn aliases(&self) -> &[ResultAlias] {
        &self.aliases
    }

    /// Returns the data type used for operation computation.
    #[must_use]
    pub fn computation_data_type(&self) -> DataType {
        self.computation_data_type
    }

    /// Returns the required kernel operand data types in input order.
    /// Execution preparation converts operands whose declared types differ.
    #[must_use]
    pub fn kernel_input_data_types(&self) -> &[DataType] {
        &self.kernel_input_data_types
    }
}

impl CheckedOperationRequest {
    /// Checks catalog execution and dtype capabilities without tensor storage.
    pub(crate) fn validate_execution(&self, execution: BackendExecution) -> Result<()> {
        let schema = operation_schema_for_attributes(self.operation(), self.attributes());
        let support = schema.support();
        let inputs_types = self
            .inputs()
            .iter()
            .map(ValueSpec::data_type)
            .collect::<Vec<_>>();
        let kernel_types = self.inference().kernel_input_data_types();
        let cast_support = operation_support(Operation::Cast);
        let conversions_supported =
            inputs_types
                .iter()
                .zip(kernel_types)
                .all(|(source, target)| {
                    source == target
                        || (cast_support.supports_input_data_type(execution, *source)
                            && cast_support.supports_data_type(execution, *target))
                });
        let output_supported = schema
            .output_roles()
            .iter()
            .zip(self.inference().outputs())
            .all(|(role, output)| match role {
                OutputRole::Output => support.supports_data_type(execution, output.data_type()),
                OutputRole::Indices | OutputRole::Mask | OutputRole::LogSumExp => {
                    output.data_type() == role.data_type(self.inference().computation_data_type())
                }
            });
        let operands_supported = if let OperationAttributes::Scalar(value) = self.attributes() {
            support.supports_scalar_argument(
                execution,
                self.inputs()
                    .first()
                    .ok_or_else(|| Error::missing_input("validate execution"))?,
                *value,
            )
        } else {
            support.supports_operand_data_types(
                execution,
                &schema.operands().roles_for_count(self.inputs().len()),
                kernel_types,
            )
        };
        if !support.supports(execution)
            || !output_supported
            || !operands_supported
            || !conversions_supported
        {
            return Err(Error::Unsupported {
                operation: schema.name().into(),
                target: if support.supports(execution) {
                    UnsupportedTarget::DataType
                } else {
                    UnsupportedTarget::Operation
                },
                details: format!(
                    "execution {execution:?}, input dtypes {inputs_types:?}, results {:?}, providers [{}]",
                    self.inference().outputs(),
                    support.provider_list()
                )
                .into(),
            });
        }
        Ok(())
    }

    /// Returns the requested operation code.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.request.operation
    }

    /// Returns the typed operation attributes.
    #[must_use]
    pub fn attributes(&self) -> &OperationAttributes {
        &self.request.attributes
    }

    /// Returns the operand specifications in operation input order.
    #[must_use]
    pub fn inputs(&self) -> &[ValueSpec] {
        &self.request.inputs
    }

    /// Returns the logical inference retained when this request was checked.
    #[must_use]
    pub fn inference(&self) -> &OperationInference {
        &self.inference
    }
}

fn attribute_kind(attributes: &OperationAttributes) -> Option<OperationAttributeKind> {
    match attributes.kind() {
        OperationAttributeKind::Input | OperationAttributeKind::Constant => None,
        kind => Some(kind),
    }
}
