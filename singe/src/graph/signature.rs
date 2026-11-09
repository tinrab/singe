use std::collections::{HashMap, HashSet};

use crate::{
    error::{Error, Result},
    graph::{
        Graph, InputDimensionGuard,
        guards::{GraphGuards, InputGuard, storage::static_input_storage_guard},
    },
    tensor::Tensor,
    value::ValueSpec,
};

/// Declared input contract, independent of the optimized graph's live values.
///
/// Defaults and static bindings are immutable snapshots.
/// An optimized-away input still requires the same name, metadata, and override policy.
#[derive(Debug, Clone)]
pub struct GraphSignature {
    guards: GraphGuards,
    outputs: Vec<GraphSignatureOutput>,
}

/// One public output projection.
/// Names and ordering survive optimization.
#[derive(Debug, Clone)]
pub struct GraphSignatureOutput {
    name: Option<String>,
    spec: ValueSpec,
}

impl GraphSignatureOutput {
    /// Returns the optional public output name.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the logical output shape and data type.
    #[must_use]
    pub fn spec(&self) -> &ValueSpec {
        &self.spec
    }
}

impl GraphSignature {
    pub(crate) fn create(graph: &Graph) -> Result<Self> {
        let mut inputs = Vec::new();
        let mut input_names = HashSet::new();
        for (_, name, metadata) in graph.input_entries() {
            if !input_names.insert(name.to_owned()) {
                return Err(Error::DuplicateInput);
            }

            let initializer = graph.initializers.get(name).cloned();
            let is_static_initializer = graph.is_static_initializer(name);
            let static_storage = is_static_initializer
                .then(|| {
                    initializer
                        .as_ref()
                        .map(static_input_storage_guard)
                        .ok_or_else(|| Error::missing_input("signature create"))
                })
                .transpose()?;
            let layout = metadata.output_layout().clone();
            let mut dimension_guards = graph
                .input_dimension_guards
                .values()
                .filter(|guard| guard.input() == name)
                .cloned()
                .collect::<Vec<_>>();
            dimension_guards.sort_by_key(InputDimensionGuard::axis);
            let guard = InputGuard {
                name: name.to_owned(),
                data_type: metadata.output_data_type(),
                device: metadata.output_device(),
                layout,
                #[cfg(test)]
                storage_span_bytes: metadata
                    .output_layout()
                    .storage_span_byte_size(metadata.output_data_type().size_bytes())?,
                initializer,
                is_static_initializer,
                static_storage,
                dimension_guards,
            };
            if let Some(initializer) = &guard.initializer {
                guard.check_with_symbol_values(initializer, &mut HashMap::new())?;
            }

            inputs.push(guard);
        }

        let outputs = graph
            .output_projections()
            .iter()
            .map(|output| {
                let node = &graph.required_node(output.node())?;
                let data_type = if let Some(schema) = node.schema() {
                    schema
                        .output_role(output.output_index())
                        .ok_or_else(|| Error::missing_output("signature create"))?
                        .data_type(node.output_data_type())
                } else if output.output_index() == 0 {
                    node.output_data_type()
                } else {
                    return Err(Error::missing_output("signature create"));
                };
                Ok(GraphSignatureOutput {
                    name: output.name().map(ToOwned::to_owned),
                    spec: ValueSpec::new(data_type, node.output_shape().clone()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            guards: GraphGuards {
                inputs,
                #[cfg(test)]
                constants: Vec::new(),
                input_names,
            },
            outputs,
        })
    }

    /// Returns inputs in declaration order, including defaults and static state.
    #[must_use]
    pub fn inputs(&self) -> &[InputGuard] {
        &self.guards.inputs
    }

    /// Returns public output projections in declaration order.
    #[must_use]
    pub fn outputs(&self) -> &[GraphSignatureOutput] {
        &self.outputs
    }

    pub(crate) fn specialize_dimensions(&mut self) {
        for input in &mut self.guards.inputs {
            for guard in &mut input.dimension_guards {
                let dimension = input.layout.dimensions()[guard.axis()];
                *guard = InputDimensionGuard::new(
                    guard.input(),
                    guard.axis(),
                    guard.name(),
                    Some(dimension),
                    Some(dimension),
                );
            }
        }
    }

    pub(crate) fn resolve(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.guards.check_inputs(inputs)
    }
}
