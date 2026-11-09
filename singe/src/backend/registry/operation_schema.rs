use crate::{
    backend::registry::{
        AliasBehavior, BatchNormalizationAttributes, Conv2dAttributes, Conv3dAttributes,
        DeclaredOutputSchema, LayerNormalizationAttributes, OperandRole, OperandSchema, Operation,
        OperationAttributeKind, OperationAttributes, OperationSupport, OutputRole,
        QuantizationAttributes, ScaledDotProductAttentionAttributes, operation_name,
        operation_support, table,
    },
    data_type::DataType,
    device::Device,
    shape::Shape,
};

/// Registry contract for an operation's operands, outputs, attributes, aliasing, and backend support.
///
/// [`operation_schema`] returns the base contract.
/// [`operation_schema_for_attributes`] specializes operand roles and declared output metadata for supplied attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSchema {
    operation: Operation,
    name: &'static str,
    operands: OperandSchema,
    outputs: &'static [OutputRole],
    attribute_kind: OperationAttributeKind,
    declared_output: DeclaredOutputSchema,
    alias_behavior: AliasBehavior,
    support: OperationSupport,
}

impl OperationSchema {
    /// Returns the operation identified by this schema.
    #[must_use]
    pub fn operation(&self) -> Operation {
        self.operation
    }

    /// Returns the registry name of the operation.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Returns the accepted operand counts and ordered roles.
    #[must_use]
    pub fn operands(&self) -> &OperandSchema {
        &self.operands
    }

    /// Returns the ordered roles of all result tensors.
    #[must_use]
    pub fn output_roles(&self) -> &'static [OutputRole] {
        self.outputs
    }

    /// Returns the number of result tensors.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Returns the role at the zero-based result `index`, or `None` if it is out of range.
    #[must_use]
    pub fn output_role(&self, index: usize) -> Option<OutputRole> {
        self.outputs.get(index).copied()
    }

    /// Returns the output role when the operation has exactly one result, or `None` otherwise.
    #[must_use]
    pub fn single_output_role(&self) -> Option<OutputRole> {
        match self.outputs {
            [role] => Some(*role),
            _ => None,
        }
    }

    /// Returns the attribute payload kind expected by the operation.
    #[must_use]
    pub fn attribute_kind(&self) -> OperationAttributeKind {
        self.attribute_kind
    }

    /// Returns any explicitly declared output data type, device, or shape.
    #[must_use]
    pub fn declared_output(&self) -> &DeclaredOutputSchema {
        &self.declared_output
    }

    /// Returns the storage-sharing contract between results and inputs.
    #[must_use]
    pub fn alias_behavior(&self) -> AliasBehavior {
        self.alias_behavior
    }

    /// Returns the operation's backend, layout, and data-type support contract.
    #[must_use]
    pub fn support(&self) -> OperationSupport {
        self.support
    }

    fn with_operands(mut self, operands: OperandSchema) -> Self {
        self.operands = operands;
        self
    }

    fn with_declared_output(mut self, declared_output: DeclaredOutputSchema) -> Self {
        self.declared_output = declared_output;
        self
    }

    pub(crate) fn with_concrete_operands(self, roles: impl Into<Box<[OperandRole]>>) -> Self {
        self.with_operands(OperandSchema::Concrete(roles.into()))
    }

    pub(crate) fn with_declared_data_type(self, data_type: DataType) -> Self {
        self.with_declared_output(DeclaredOutputSchema::with_data_type(data_type))
    }

    pub(crate) fn with_declared_device(self, device: Device) -> Self {
        self.with_declared_output(DeclaredOutputSchema::with_device(device))
    }

    pub(crate) fn with_declared_shape(self, shape: Shape) -> Self {
        self.with_declared_output(DeclaredOutputSchema::with_shape(shape))
    }
}

/// Returns the base registry schema for `operation`.
///
/// The schema has no explicitly declared output metadata.
/// Optional operand presence is described by its general operand schema.
#[must_use]
pub fn operation_schema(operation: Operation) -> OperationSchema {
    let support = operation_support(operation);
    OperationSchema {
        operation,
        name: operation_name(operation),
        operands: table::operation_operand_schema(operation),
        outputs: table::operation_output_schema(operation),
        attribute_kind: operation.attribute_kind(),
        declared_output: DeclaredOutputSchema::default(),
        alias_behavior: table::operation_alias_behavior(operation),
        support,
    }
}

/// Specializes the base registry schema using `attributes`.
///
/// Convolution and normalization flags determine concrete operand roles.
/// Cast, quantization, copy, and shape attributes supply explicit output metadata.
/// Quantization without an explicit data type declares `F32`.
///
/// This function does not check that the attribute variant matches the operation.
/// The caller must supply attributes for the requested operation.
#[must_use]
pub fn operation_schema_for_attributes(
    operation: Operation,
    attributes: &OperationAttributes,
) -> OperationSchema {
    let schema = operation_schema(operation);
    let schema = match attributes {
        OperationAttributes::Conv2d(Conv2dAttributes { has_bias, .. })
        | OperationAttributes::Conv3d(Conv3dAttributes { has_bias, .. }) => {
            let roles = schema
                .operands()
                .roles_for_count(2 + usize::from(*has_bias));
            schema.with_concrete_operands(roles)
        }
        OperationAttributes::ScaledDotProductAttention(ScaledDotProductAttentionAttributes {
            masked,
            sinks,
            key_lengths,
            ..
        }) if operation == Operation::ScaledDotProductAttention => {
            let mut roles = vec![OperandRole::Input, OperandRole::Key, OperandRole::Value];
            if *masked {
                roles.push(OperandRole::Bias);
            }

            if *sinks {
                roles.push(OperandRole::Sinks);
            }

            if *key_lengths {
                roles.push(OperandRole::Indices);
            }
            schema.with_concrete_operands(roles)
        }
        OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
            has_weight,
            has_bias,
            has_running_mean,
            has_running_var,
            ..
        }) => {
            let mut roles = vec![OperandRole::Input];
            if *has_weight {
                roles.push(OperandRole::Weight);
            }

            if *has_bias {
                roles.push(OperandRole::Bias);
            }

            if *has_running_mean {
                roles.push(OperandRole::RunningMean);
            }

            if *has_running_var {
                roles.push(OperandRole::RunningVar);
            }
            schema.with_concrete_operands(roles)
        }
        OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
            has_weight,
            has_bias,
            ..
        }) => {
            let mut roles = vec![OperandRole::Input];
            if *has_weight {
                roles.push(OperandRole::Weight);
            }

            if *has_bias {
                roles.push(OperandRole::Bias);
            }
            schema.with_concrete_operands(roles)
        }
        _ => schema,
    };

    match attributes {
        OperationAttributes::Cast(data_type) => schema.with_declared_data_type(*data_type),
        OperationAttributes::Quantization(QuantizationAttributes { data_type, .. }) => {
            schema.with_declared_data_type(data_type.unwrap_or(DataType::F32))
        }
        OperationAttributes::CopyTo(device) => schema.with_declared_device(*device),
        OperationAttributes::Shape(shape) => schema.with_declared_shape(shape.clone()),
        _ => schema,
    }
}
