use std::collections::{HashMap, HashSet};

use singe_onnx::Node;

use crate::{
    backend::registry::OperationRequest,
    data_type::DataType,
    device::Device,
    dispatch,
    graph::ir::ValueId,
    onnx::{
        error::{Error, Result},
        importer::{ImportedValue, Importer},
    },
    shape::Shape,
    tensor::Tensor,
};

impl Importer {
    pub(crate) fn value(&self, name: &str) -> Result<&ImportedValue> {
        self.values
            .get(name)
            .ok_or_else(|| Error::MissingValue(name.into()))
    }

    pub(crate) fn required_input(&self, node: &Node, index: usize) -> Result<ImportedValue> {
        let name = node.inputs.get(index).ok_or_else(|| {
            Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                self.opset_version(node.domain.as_deref()),
                format!("missing input {index}"),
            )
        })?;
        self.value(name).cloned()
    }

    pub(crate) fn optional_input(
        &self,
        node: &Node,
        index: usize,
    ) -> Result<Option<ImportedValue>> {
        match node.inputs.get(index) {
            Some(name) if !name.is_empty() => self.value(name).cloned().map(Some),
            _ => Ok(None),
        }
    }

    pub(crate) fn input_meta(&self, node: &Node, index: usize) -> Result<(DataType, Shape)> {
        let input = self.required_input(node, index)?;
        Ok((input.data_type, input.shape))
    }

    pub(crate) fn register_value(&mut self, name: &str, value: ImportedValue) -> Result<()> {
        if name.is_empty() {
            return Ok(());
        }

        if self.values.insert(name.into(), value).is_some() {
            return Err(Error::DuplicateValue(name.into()));
        }
        Ok(())
    }
}

impl ImportedValue {
    pub(crate) fn new(node: ValueId, data_type: DataType, shape: Shape) -> Self {
        Self {
            node,
            data_type,
            shape,
        }
    }
}

impl Importer {
    pub(crate) fn static_input_tensor(&self, name: &str) -> Result<Option<Tensor>> {
        let Some(value) = self.values.get(name) else {
            return Ok(None);
        };
        if let Some(metadata) = self.graph.shape_program.values.get(&value.node)
            && metadata.is_constant()
        {
            return Ok(Some(Tensor::from_vec(
                Device::Cpu,
                metadata.shape()?,
                self.evaluate_shape_value(metadata)?,
            )?));
        }

        // Evaluate only constant ancestors needed for this import-time operand.
        // The operation catalog supplies both inference and execution semantics.
        let mut needed = HashSet::new();
        let mut pending = vec![value.node];
        while let Some(id) = pending.pop() {
            if needed.insert(id) {
                if self
                    .graph
                    .shape_program
                    .values
                    .get(&id)
                    .is_some_and(|value| !value.is_constant())
                {
                    return Err(Error::UnsupportedShape { name: name.into(), reason: "metadata-derived values require shape-program lowering for this operand".into() });
                }

                pending.extend(self.graph.required_node(id)?.inputs());
            }
        }

        let mut tensors = HashMap::new();
        let mut dependencies: HashMap<_, HashSet<String>> = HashMap::new();
        for (id, node) in self
            .graph
            .node_entries()
            .filter(|(id, _)| needed.contains(id))
        {
            let mut names = HashSet::new();
            let tensor = if let Some(tensor) = node.constant_tensor(0) {
                tensor.clone()
            } else if let Some((name, _, _, _)) = node.input_metadata() {
                let Some(tensor) = self.initializer_tensors.get(name) else {
                    continue;
                };
                names.insert(name.to_owned());
                tensor.clone()
            } else {
                let Some(inputs) = node
                    .inputs()
                    .iter()
                    .map(|id| tensors.get(id).cloned())
                    .collect::<Option<Vec<Tensor>>>()
                else {
                    continue;
                };
                for source in node.inputs() {
                    if let Some(ancestors) = dependencies.get(source) {
                        names.extend(ancestors.iter().cloned());
                    }
                }

                let Some(operation) = node.operation() else {
                    continue;
                };
                let request = OperationRequest::new(
                    operation,
                    node.attributes(),
                    inputs.iter().map(Tensor::value_spec).collect(),
                )
                .check()?;
                let outputs = dispatch::execute(&request, &inputs)?;
                let [tensor] = outputs.as_slice() else {
                    return Ok(None);
                };
                tensor.clone()
            };
            tensors.insert(id, tensor);
            dependencies.insert(id, names);
        }
        tensors.remove(&value.node).map_or_else(
            || Ok(None),
            |tensor| {
                self.specialized_initializers
                    .borrow_mut()
                    .extend(dependencies.remove(&value.node).unwrap_or_default());
                Ok(Some(tensor))
            },
        )
    }

    pub(crate) fn optional_static_input_tensor(
        &self,
        node: &Node,
        index: usize,
    ) -> Result<Option<(String, Tensor)>> {
        let Some(name) = node.inputs.get(index) else {
            return Ok(None);
        };
        if name.is_empty() {
            return Ok(None);
        }
        self.static_input_tensor(name)
            .map(|tensor| tensor.map(|tensor| (name.clone(), tensor)))
    }

    pub(crate) fn required_static_input_tensor(
        &self,
        node: &Node,
        index: usize,
        label: &str,
    ) -> Result<(String, Tensor)> {
        self.required_static_input_tensor_with_reasons(
            node,
            index,
            label,
            format!("{label} input cannot be empty"),
            format!("runtime {label} tensors are not supported"),
        )
    }

    pub(crate) fn required_static_input_tensor_with_reasons(
        &self,
        node: &Node,
        index: usize,
        label: &str,
        empty_reason: impl Into<String>,
        runtime_reason: impl Into<String>,
    ) -> Result<(String, Tensor)> {
        let name = node.inputs.get(index).ok_or_else(|| {
            Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                self.opset_version(node.domain.as_deref()),
                format!("missing {label} input"),
            )
        })?;
        if name.is_empty() {
            return Err(Error::unsupported_operator(
                node.domain.as_deref(),
                &node.operator_type,
                self.opset_version(node.domain.as_deref()),
                empty_reason,
            ));
        }
        self.static_input_tensor(name)?
            .map(|tensor| (name.clone(), tensor))
            .ok_or_else(|| {
                Error::unsupported_operator(
                    node.domain.as_deref(),
                    &node.operator_type,
                    self.opset_version(node.domain.as_deref()),
                    runtime_reason,
                )
            })
    }
}
