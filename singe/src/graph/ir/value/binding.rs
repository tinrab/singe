use crate::{
    graph::ir::{OperationId, value::ValueId},
    tensor::Tensor,
};

/// Binding class for a graph value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueBinding {
    kind: ValueBindingKind,
    input_name: Option<String>,
    runtime_overridable: Option<bool>,
    operation_output: Option<(OperationId, usize)>,
}

/// Binding category for a graph value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueBindingKind {
    /// Runtime graph input.
    RuntimeInput,
    /// Input backed by an initializer and still overridable by runtime input.
    InitializerInput,
    /// Input backed by a static initializer and not overridable at run time.
    StaticInput,
    /// Graph constant embedded in the graph.
    Constant,
    /// Output produced by an operation.
    OperationOutput,
}

impl ValueBinding {
    /// Creates a runtime input binding with the supplied name.
    /// The constructor records metadata without validating it against a graph.
    pub fn runtime_input(name: impl Into<String>) -> Self {
        Self {
            kind: ValueBindingKind::RuntimeInput,
            input_name: Some(name.into()),
            runtime_overridable: None,
            operation_output: None,
        }
    }

    /// Creates an initializer input binding with an explicit runtime override flag.
    /// The constructor records metadata without validating it against a graph.
    pub fn initializer_input(name: impl Into<String>, runtime_overridable: bool) -> Self {
        Self {
            kind: ValueBindingKind::InitializerInput,
            input_name: Some(name.into()),
            runtime_overridable: Some(runtime_overridable),
            operation_output: None,
        }
    }

    /// Creates an initializer input binding that cannot be overridden at run time.
    /// The constructor records metadata without validating it against a graph.
    pub fn static_input(name: impl Into<String>) -> Self {
        Self {
            kind: ValueBindingKind::StaticInput,
            input_name: Some(name.into()),
            runtime_overridable: None,
            operation_output: None,
        }
    }

    /// Creates an embedded constant binding with no input name or producer.
    /// The constructor records metadata without validating it against a graph.
    #[must_use]
    pub fn constant() -> Self {
        Self {
            kind: ValueBindingKind::Constant,
            input_name: None,
            runtime_overridable: None,
            operation_output: None,
        }
    }

    /// Creates a result binding for an operation and its zero-based output index.
    /// The constructor records metadata without validating it against a graph.
    #[must_use]
    pub fn produced_by_operation(operation: OperationId, output_index: usize) -> Self {
        Self {
            kind: ValueBindingKind::OperationOutput,
            input_name: None,
            runtime_overridable: None,
            operation_output: Some((operation, output_index)),
        }
    }

    /// Returns the value binding category.
    #[must_use]
    pub fn kind(&self) -> ValueBindingKind {
        self.kind
    }

    /// Returns the input name when this value is backed by a graph input.
    #[must_use]
    pub fn input_name(&self) -> Option<&str> {
        self.input_name.as_deref()
    }

    /// Returns whether this value is backed by any graph input binding.
    #[must_use]
    pub fn is_input(&self) -> bool {
        self.input_name.is_some()
    }

    /// Returns whether this value is backed by an initializer.
    #[must_use]
    pub fn is_initializer_input(&self) -> bool {
        self.kind == ValueBindingKind::InitializerInput
    }

    /// Returns whether this value is backed by a static initializer.
    #[must_use]
    pub fn is_static_input(&self) -> bool {
        self.kind == ValueBindingKind::StaticInput
    }

    /// Returns whether this value is backed by an initializer tensor.
    #[must_use]
    pub fn is_initializer_backed_input(&self) -> bool {
        matches!(
            self.kind,
            ValueBindingKind::InitializerInput | ValueBindingKind::StaticInput
        )
    }

    /// Returns whether runtime input values may override this initializer.
    #[must_use]
    pub fn runtime_overridable(&self) -> Option<bool> {
        self.runtime_overridable
    }

    /// Returns whether this value is an embedded graph constant.
    #[must_use]
    pub fn is_constant(&self) -> bool {
        self.kind == ValueBindingKind::Constant
    }

    /// Returns the producing operation and output index for operation outputs.
    #[must_use]
    pub fn operation_output(&self) -> Option<(OperationId, usize)> {
        self.operation_output
    }
}

/// Required binding details for an IR value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ValueBindingDetails<'a> {
    kind: ValueBindingKind,
    input_name: Option<&'a str>,
}

impl<'a> ValueBindingDetails<'a> {
    pub(crate) fn new(binding: &'a ValueBinding) -> Self {
        Self {
            kind: binding.kind(),
            input_name: binding.input_name(),
        }
    }

    /// Returns the graph input name for input-backed values.
    #[must_use]
    pub fn input_name(self) -> Option<&'a str> {
        self.input_name
    }

    /// Returns whether this value is backed by an initializer tensor.
    #[must_use]
    pub fn is_initializer_backed_input(self) -> bool {
        matches!(
            self.kind,
            ValueBindingKind::InitializerInput | ValueBindingKind::StaticInput
        )
    }

    /// Returns whether this value is backed by a static initializer tensor.
    #[must_use]
    pub fn is_static_input(self) -> bool {
        self.kind == ValueBindingKind::StaticInput
    }

    /// Returns whether this value is an embedded graph constant.
    #[must_use]
    pub fn is_constant(self) -> bool {
        self.kind == ValueBindingKind::Constant
    }
}

/// Initializer-backed graph input projected into IR.
#[derive(Debug, Clone, Copy)]
pub struct InitializerInputBinding<'a> {
    value: ValueId,
    input: &'a str,
    tensor: &'a Tensor,
    is_static: bool,
}

impl<'a> InitializerInputBinding<'a> {
    pub(crate) fn new(value: ValueId, input: &'a str, tensor: &'a Tensor, is_static: bool) -> Self {
        Self {
            value,
            input,
            tensor,
            is_static,
        }
    }

    /// Returns the IR value id for this initializer-backed input.
    #[must_use]
    pub fn value(self) -> ValueId {
        self.value
    }

    /// Returns the graph input name.
    #[must_use]
    pub fn input(self) -> &'a str {
        self.input
    }

    /// Returns the initializer tensor.
    #[must_use]
    pub fn tensor(self) -> &'a Tensor {
        self.tensor
    }

    /// Returns whether the initializer is static and not runtime-overridable.
    #[must_use]
    pub fn is_static(self) -> bool {
        self.is_static
    }
}
