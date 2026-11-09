use crate::{
    backend::registry::{
        OperandRole as OperationInputRole, Operation, OperationAttributes, OperationSchema,
        OutputRole as OperationOutputRole,
    },
    graph::ir::ValueId,
};

/// Identifier for an executable graph operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub usize);

impl OperationId {
    /// Creates an operation id from its graph-local index.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local operation index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// An operation in graph IR.
#[derive(Debug, Clone)]
pub struct OperationNode {
    /// Operation identifier.
    id: OperationId,
    source_location: Option<String>,
    /// Registry opcode for executable graph operations.
    opcode: Option<Operation>,
    /// Declarative operation schema, when this is an executable operation.
    schema: Option<OperationSchema>,
    /// Typed operation attributes with graph edge ids removed.
    attributes: OperationAttributes,
    /// Input values.
    inputs: Vec<ValueId>,
    /// Schema roles for each input value.
    input_roles: Vec<OperationInputRole>,
    /// Result values produced by this operation.
    results: Vec<OperationResult>,
}

impl OperationNode {
    /// Creates an IR operation.
    #[must_use]
    pub fn new(
        id: OperationId,
        opcode: Option<Operation>,
        schema: Option<OperationSchema>,
        attributes: OperationAttributes,
        inputs: Vec<ValueId>,
        input_roles: Vec<OperationInputRole>,
        results: Vec<OperationResult>,
    ) -> Self {
        if let (Some(opcode), Some(schema)) = (opcode, &schema) {
            debug_assert_eq!(
                opcode.attribute_kind(),
                schema.attribute_kind(),
                "operation attributes {:?} must match schema attribute kind {:?}",
                attributes,
                schema.attribute_kind()
            );
        }
        Self {
            id,
            source_location: None,
            opcode,
            schema,
            attributes,
            inputs,
            input_roles,
            results,
        }
    }

    /// Returns import provenance for diagnostics, excluded from semantic identity.
    #[must_use]
    pub fn source_location(&self) -> Option<&str> {
        self.source_location.as_deref()
    }

    pub(crate) fn with_source_location(mut self, source: Option<String>) -> Self {
        self.source_location = source;
        self
    }

    #[cfg(feature = "onnx")]
    pub(crate) fn set_source_location(&mut self, source: Option<String>) {
        self.source_location = source;
    }

    pub(crate) fn replace_input(&mut self, old: ValueId, new: ValueId) {
        for input in &mut self.inputs {
            if *input == old {
                *input = new;
            }
        }
    }

    /// Returns the operation identifier.
    #[must_use]
    pub fn id(&self) -> OperationId {
        self.id
    }

    /// Returns the registry opcode for executable graph operations.
    #[must_use]
    pub fn opcode(&self) -> Option<Operation> {
        self.opcode
    }

    /// Returns typed operation attributes with graph edge ids removed.
    #[must_use]
    pub fn attributes(&self) -> &OperationAttributes {
        &self.attributes
    }

    /// Returns input values.
    #[must_use]
    pub fn inputs(&self) -> &[ValueId] {
        &self.inputs
    }

    /// Returns schema roles for each input value.
    #[must_use]
    pub fn input_roles(&self) -> &[OperationInputRole] {
        &self.input_roles
    }

    /// Returns result values produced by this operation.
    #[must_use]
    pub fn results(&self) -> &[OperationResult] {
        &self.results
    }

    /// Returns result value identifiers produced by this operation.
    pub fn result_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        self.results.iter().map(|result| result.value())
    }

    /// Returns the declarative operation schema.
    #[must_use]
    pub fn schema(&self) -> Option<&OperationSchema> {
        self.schema.as_ref()
    }
}

/// One result produced by a graph IR operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationResult {
    /// Output ordinal within the producing operation.
    index: usize,
    /// Value bound to this output.
    value: ValueId,
    /// Schema role for this output.
    role: OperationOutputRole,
}

impl OperationResult {
    /// Creates an operation result.
    #[must_use]
    pub fn new(index: usize, value: ValueId, role: OperationOutputRole) -> Self {
        Self { index, value, role }
    }

    /// Returns the output ordinal within the producing operation.
    #[must_use]
    pub fn index(self) -> usize {
        self.index
    }

    /// Returns the value bound to this result.
    #[must_use]
    pub fn value(self) -> ValueId {
        self.value
    }

    /// Returns the schema role for this output.
    #[must_use]
    pub fn role(self) -> OperationOutputRole {
        self.role
    }
}
