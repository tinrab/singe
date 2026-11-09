use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        Graph, StateBindingKind,
        ir::ValueId,
        node::{Node, NodeRef},
    },
    layout::Layout,
    tensor::Tensor,
};

impl Graph {
    /// Adds a named graph input.
    ///
    /// The returned node id can be referenced by later graph operations.
    pub fn add_input(
        &mut self,
        name: &str,
        data_type: DataType,
        layout: impl Into<Layout>,
        device: Device,
    ) -> ValueId {
        self.push_node(Node::input_node(
            name.to_string(),
            data_type,
            layout.into(),
            device,
        ))
    }

    /// Iterates over graph input ids, names, and nodes in insertion order.
    pub(crate) fn input_entries(&self) -> impl Iterator<Item = (ValueId, &str, NodeRef<'_>)> + '_ {
        self.input_values()
            .filter_map(|(id, name)| Some((id, name, self.node(id)?)))
    }

    /// Returns one graph input node by name.
    pub(crate) fn input_node(&self, name: &str) -> Option<NodeRef<'_>> {
        self.input_entries()
            .find(|(_, input_name, _)| *input_name == name)
            .map(|(_, _, node)| node)
    }

    pub(crate) fn validate_input_name(&self, name: &str) -> Result<()> {
        if self.input_node(name).is_some() {
            Ok(())
        } else {
            Err(Error::missing_input("validate input name"))
        }
    }
}

/// Runtime guard for one named graph input dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InputDimensionGuard {
    /// Graph input name.
    input: String,
    /// Zero-based input dimension axis.
    axis: usize,
    /// Symbolic dimension name.
    /// Guards with the same name must match at runtime.
    name: String,
    /// Inclusive minimum allowed dimension value.
    min: Option<i64>,
    /// Inclusive maximum allowed dimension value.
    max: Option<i64>,
}

impl InputDimensionGuard {
    /// Creates a named symbolic/ranged dimension guard.
    pub fn new(
        input: impl Into<String>,
        axis: usize,
        name: impl Into<String>,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Self {
        Self {
            input: input.into(),
            axis,
            name: name.into(),
            min,
            max,
        }
    }

    /// Returns the graph input name.
    #[must_use]
    pub fn input(&self) -> &str {
        &self.input
    }

    /// Returns the zero-based input axis guarded by this symbol.
    #[must_use]
    pub fn axis(&self) -> usize {
        self.axis
    }

    /// Returns the symbolic dimension name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the inclusive minimum accepted dimension value.
    #[must_use]
    pub fn min(&self) -> Option<i64> {
        self.min
    }

    /// Returns the inclusive maximum accepted dimension value.
    #[must_use]
    pub fn max(&self) -> Option<i64> {
        self.max
    }
}

impl Graph {
    /// Adds or replaces a runtime guard for one named input dimension.
    ///
    /// Dimensions guarded with the same symbolic name must be equal when the graph runs.
    /// `min` and `max`, when supplied, are inclusive.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not an input, `axis` is out of range, or the range is invalid.
    pub fn add_input_dimension_guard(
        &mut self,
        name: &str,
        axis: usize,
        symbol: &str,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Result<()> {
        if let (Some(min), Some(max)) = (min, max)
            && min > max
        {
            return Err(Error::InvalidArgument {
                name: "graph input dimension guard range".into(),
            });
        }

        let input = self
            .input_node(name)
            .ok_or_else(|| Error::missing_input("add input dimension guard"))?;
        let rank = usize::try_from(input.output_shape().rank())
            .map_err(|_| Error::out_of_range("graph input dimension guard rank".to_string()))?;
        if axis >= rank {
            let value = i64::try_from(axis)
                .map_err(|_| Error::out_of_range("graph input dimension guard axis".to_string()))?;
            return Err(Error::InvalidDimension {
                name: format!("{name} symbolic dimension axis").into(),
                value,
                rank: input.output_shape().rank(),
            });
        }

        self.input_dimension_guards.insert(
            (name.to_string(), axis),
            InputDimensionGuard::new(name, axis, symbol, min, max),
        );
        Ok(())
    }
}

/// Binding class for a named graph input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphInputClass {
    /// Input must be provided by the caller for each run.
    Runtime,
    /// Input has an initializer but callers may override it at run time.
    Initializer,
    /// Input has a fixed initializer and cannot be overridden at run time.
    Static,
}

impl Graph {
    /// Adds or replaces an initializer used when an input is omitted at run time.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a graph input.
    pub fn add_initializer(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        self.validate_input_name(name)?;
        self.initializers.insert(name.to_string(), tensor);
        self.static_initializers.remove(name);
        self.refresh_input_bindings();
        self.state_kinds
            .insert(name.to_owned(), StateBindingKind::Default);
        Ok(())
    }

    /// Adds or replaces a static initializer that cannot be overridden at run time.
    ///
    /// Static initializers are graph inputs backed by fixed tensor storage.
    /// Backends may wire them directly into compiled executables instead of copying them into runtime input placeholders on every replay.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a graph input.
    pub fn add_static_initializer(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        self.validate_input_name(name)?;
        self.initializers.insert(name.to_string(), tensor);
        self.static_initializers.insert(name.to_string());
        self.refresh_input_bindings();
        self.state_kinds
            .insert(name.to_owned(), StateBindingKind::Static);
        Ok(())
    }

    /// Binds a named parameter without freezing it into the logical program.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not identify a declared graph input.
    pub fn add_parameter(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        self.add_static_initializer(name, tensor)?;
        self.state_kinds
            .insert(name.to_owned(), StateBindingKind::Parameter);
        Ok(())
    }

    /// Binds a persistent, read-only buffer for inference.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` does not identify a declared graph input.
    pub fn add_buffer(&mut self, name: &str, tensor: Tensor) -> Result<()> {
        self.add_static_initializer(name, tensor)?;
        self.state_kinds
            .insert(name.to_owned(), StateBindingKind::Buffer);
        Ok(())
    }

    /// Returns whether `name` has any initializer tensor.
    pub(crate) fn has_initializer(&self, name: &str) -> bool {
        self.initializers.contains_key(name)
    }

    /// Returns whether `name` is backed by a static initializer.
    pub(crate) fn is_static_initializer(&self, name: &str) -> bool {
        self.static_initializers.contains(name)
    }

    /// Returns the binding class for graph input `name`.
    ///
    /// # Errors
    ///
    /// Returns an error if `name` is not a graph input.
    pub fn input_class(&self, name: &str) -> Result<GraphInputClass> {
        self.validate_input_name(name)?;
        Ok(if self.is_static_initializer(name) {
            GraphInputClass::Static
        } else if self.has_initializer(name) {
            GraphInputClass::Initializer
        } else {
            GraphInputClass::Runtime
        })
    }
}
