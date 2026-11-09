mod binding;
mod storage;

pub use self::{
    binding::{InitializerInputBinding, ValueBinding, ValueBindingDetails, ValueBindingKind},
    storage::ValueAlias,
};
use crate::{
    data_type::DataType, device::Device, graph::ir::OperationId, layout::Layout, shape::Shape,
};

/// Identifier for a graph value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ValueId(pub usize);

impl ValueId {
    /// Creates a value id from its graph-local index.
    #[must_use]
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the graph-local value index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0
    }
}

/// A typed value in graph IR.
#[derive(Debug, Clone)]
pub struct Value {
    /// Value identifier.
    id: ValueId,
    /// Optional graph-local value name.
    name: Option<String>,
    /// Value data type.
    data_type: DataType,
    /// Value layout.
    layout: Layout,
    /// Device where the value lives.
    device: Device,
    /// How this value is bound or produced.
    binding: ValueBinding,
    /// Alias metadata when this value aliases another value.
    alias: Option<ValueAlias>,
}

impl Value {
    pub(crate) fn set_producer(&mut self, operation: OperationId, index: usize) {
        self.binding = ValueBinding::produced_by_operation(operation, index);
    }

    pub(crate) fn set_metadata(&mut self, data_type: DataType, layout: Layout, device: Device) {
        self.data_type = data_type;
        self.layout = layout;
        self.device = device;
    }

    pub(crate) fn set_binding(&mut self, binding: ValueBinding) {
        self.binding = binding;
    }

    pub(crate) fn set_alias(&mut self, alias: Option<ValueAlias>) {
        self.alias = alias;
    }

    /// Creates an IR value.
    #[must_use]
    pub fn new(
        id: ValueId,
        name: Option<String>,
        data_type: DataType,
        layout: Layout,
        device: Device,
        binding: ValueBinding,
        alias: Option<ValueAlias>,
    ) -> Self {
        Self {
            id,
            name,
            data_type,
            layout,
            device,
            binding,
            alias,
        }
    }

    /// Returns the value identifier.
    #[must_use]
    pub fn id(&self) -> ValueId {
        self.id
    }

    /// Returns the optional graph-local value name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the value data type.
    #[must_use]
    pub fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns the value shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the value layout.
    #[must_use]
    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the device where the value lives.
    #[must_use]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Returns how this value is bound or produced.
    #[must_use]
    pub fn binding(&self) -> &ValueBinding {
        &self.binding
    }

    /// Returns alias metadata when this value aliases another value.
    #[must_use]
    pub fn alias(&self) -> Option<&ValueAlias> {
        self.alias.as_ref()
    }
}
