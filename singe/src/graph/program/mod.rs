use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, OnceLock},
};

use crate::{
    error::{Error, Result},
    graph::{
        CaptureInfo, CompiledGraph, Graph, GraphCompileOptions, GraphSignature,
        GraphSignatureOutput, InputDimensionGuard, StateBindingKind, StateBindingSpec,
        StateSnapshot, fingerprint::fingerprint_program,
    },
    state::StateSchema,
    tensor::Tensor,
    value::ValueSpec,
};

/// Input role in a logical program, without allocation or device identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProgramInputBinding {
    /// Caller-supplied runtime input.
    Runtime,
    /// Input supplied by a named state binding with the recorded role.
    State(StateBindingKind),
}

/// Public input name, logical specification, state role, and dimension guards for a program.
#[derive(Debug, Clone)]
pub struct ProgramInput {
    name: String,
    spec: ValueSpec,
    binding: ProgramInputBinding,
    dimensions: Vec<InputDimensionGuard>,
}

impl ProgramInput {
    /// Returns the public input name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical element type and shape, independently of placement and storage layout.
    #[must_use]
    pub fn spec(&self) -> &ValueSpec {
        &self.spec
    }

    /// Returns whether callers or bound state supply this input.
    #[must_use]
    pub fn binding(&self) -> ProgramInputBinding {
        self.binding
    }

    /// Returns dimension constraints in axis order.
    #[must_use]
    pub fn dimension_guards(&self) -> &[InputDimensionGuard] {
        &self.dimensions
    }
}

/// Public names, logical specs, state roles, and dimension constraints.
#[derive(Debug, Clone)]
pub struct ProgramSignature {
    inputs: Vec<ProgramInput>,
    outputs: Vec<GraphSignatureOutput>,
}

impl ProgramSignature {
    /// Returns program inputs in graph input order.
    #[must_use]
    pub fn inputs(&self) -> &[ProgramInput] {
        &self.inputs
    }

    /// Returns output specifications in public output order.
    #[must_use]
    pub fn outputs(&self) -> &[GraphSignatureOutput] {
        &self.outputs
    }
}

#[derive(Debug, Clone)]
struct ProgramDescription {
    graph: Graph,
    signature: ProgramSignature,
    schema: Arc<StateSchema>,
    specialized_state: BTreeMap<String, Vec<u8>>,
    analysis: OnceLock<CompiledGraph>,
}

/// Stable logical identity, excluding bound state, layouts, devices, handles, and allocation addresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramFingerprint(u64);

impl ProgramFingerprint {
    pub(crate) fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying stable 64-bit logical fingerprint.
    #[must_use]
    pub fn value(self) -> u64 {
        self.0
    }
}

/// Immutable logical graph and state schema.
/// Replaceable state allocations and native execution resources belong to snapshots and executors respectively.
#[derive(Debug, Clone)]
pub struct GraphProgram {
    description: Arc<ProgramDescription>,
}

impl GraphProgram {
    // Adopts the schema of `state`, whose names and specifications must match this program's, so state from it binds without republishing.
    pub fn with_state_schema(&self, state: &StateSnapshot) -> Result<Self> {
        let bindings = &state.schema.bindings;
        if bindings.len() != self.description.schema.bindings.len()
            || self.state_schema().any(|binding| {
                bindings.get(binding.name()).is_none_or(|candidate| {
                    candidate.spec != binding.spec || candidate.kind != binding.kind
                })
            })
        {
            return Err(Error::invalid_argument("model program state schema"));
        }

        let mut description = self.description.as_ref().clone();
        description.schema = Arc::clone(&state.schema);
        Ok(Self {
            description: Arc::new(description),
        })
    }

    /// Lifts initializer bindings out of a construction graph.
    /// Bind their tensor values separately with `bind_state` before preparation.
    ///
    /// # Errors
    ///
    /// Returns an error if the source graph signature is invalid or an initializer has no matching input declaration.
    pub fn create(source: &Graph) -> Result<Self> {
        let guarded = GraphSignature::create(source)?;
        let mut bindings = BTreeMap::new();
        for name in source.initializers.keys() {
            let node = source
                .input_node(name)
                .ok_or_else(|| Error::missing_input("program create"))?;
            let kind = source.state_kinds.get(name).copied().unwrap_or_else(|| {
                if source.is_static_initializer(name) {
                    StateBindingKind::Static
                } else {
                    StateBindingKind::Default
                }
            });
            bindings.insert(
                name.clone(),
                StateBindingSpec {
                    id: 0,
                    name: name.clone(),
                    spec: node.output_spec(),
                    kind,
                },
            );
        }

        for (id, binding) in bindings.values_mut().enumerate() {
            binding.id = id;
        }

        let inputs = source
            .input_entries()
            .map(|(_, name, node)| {
                let mut dimensions = source
                    .input_dimension_guards
                    .values()
                    .filter(|guard| guard.input() == name)
                    .cloned()
                    .collect::<Vec<_>>();
                dimensions.sort_by_key(InputDimensionGuard::axis);
                ProgramInput {
                    name: name.to_owned(),
                    spec: node.output_spec(),
                    binding: bindings
                        .get(name)
                        .map_or(ProgramInputBinding::Runtime, |binding| {
                            ProgramInputBinding::State(binding.kind)
                        }),
                    dimensions,
                }
            })
            .collect();
        let signature = ProgramSignature {
            inputs,
            outputs: guarded.outputs().to_vec(),
        };
        let mut graph = source.clone();
        graph.initializers.clear();
        graph.static_initializers.clear();
        graph.state_kinds.clear();
        graph.refresh_input_bindings();
        Ok(Self {
            description: Arc::new(ProgramDescription {
                graph,
                signature,
                schema: Arc::new(StateSchema { bindings }),
                specialized_state: BTreeMap::new(),
                analysis: OnceLock::new(),
            }),
        })
    }

    /// Returns public input and output metadata for this logical program.
    #[must_use]
    pub fn signature(&self) -> &ProgramSignature {
        &self.description.signature
    }

    /// Returns capture metadata when the logical graph contains such metadata.
    #[must_use]
    pub fn capture_info(&self) -> Option<CaptureInfo> {
        self.description.graph.capture_info()
    }

    /// Iterates named state binding specifications in name order.
    ///
    /// Tied names each appear and share a logical binding identity.
    pub fn state_schema(&self) -> impl Iterator<Item = &StateBindingSpec> {
        self.description.schema.bindings.values()
    }

    /// Declares that two parameter or buffer names refer to one logical state.
    /// The returned program has a new schema; snapshots from the original schema cannot be bound to it.
    /// Equal allocations or values never declare a tie.
    ///
    /// # Errors
    ///
    /// Returns an error if either name is absent, their roles differ or are not parameters or buffers, or their logical data types or shapes differ.
    pub fn tie_state(&self, first: &str, second: &str) -> Result<Self> {
        let schema = &self.description.schema;
        let first = schema
            .bindings
            .get(first)
            .ok_or_else(|| Error::missing_input("tie state"))?;
        let second = schema
            .bindings
            .get(second)
            .ok_or_else(|| Error::missing_input("tie state"))?;
        if first.kind != second.kind
            || !matches!(
                first.kind,
                StateBindingKind::Parameter | StateBindingKind::Buffer
            )
            || first.spec != second.spec
        {
            return Err(Error::invalid_argument("incompatible tied state schema"));
        }

        if first.id == second.id {
            return Ok(self.clone());
        }

        let identity = first.id.min(second.id);
        let mut bindings = schema.bindings.clone();
        for binding in bindings.values_mut() {
            if binding.id == first.id || binding.id == second.id {
                binding.id = identity;
            }
        }
        Ok(Self {
            description: Arc::new(ProgramDescription {
                graph: self.description.graph.clone(),
                signature: self.description.signature.clone(),
                schema: Arc::new(StateSchema { bindings }),
                specialized_state: self.description.specialized_state.clone(),
                analysis: self.description.analysis.clone(),
            }),
        })
    }

    /// Computes a stable logical fingerprint, excluding current allocations and device placement.
    ///
    /// # Errors
    ///
    /// Returns an error if lowering the logical metadata graph, analyzing or encoding the program, or reading constant tensor bytes required by fingerprinting fails.
    pub fn fingerprint(&self) -> Result<ProgramFingerprint> {
        fingerprint_program(self)
    }

    pub(crate) fn logical_analysis(&self) -> Result<&CompiledGraph> {
        let analysis = &self.description.analysis;
        if analysis.get().is_none() {
            let compiled = self
                .logical_graph()
                .compile_with_options(GraphCompileOptions::new())?;
            // Another thread may have set it first; both compiled the same graph.
            analysis.set(compiled).ok();
        }
        analysis
            .get()
            .ok_or_else(|| Error::missing_node("logical analysis"))
    }

    /// Validates named tensors and publishes an immutable snapshot against this program's schema.
    ///
    /// Every logical state group must be supplied.
    /// One name can supply an entire tied group; supplied aliases must agree.
    ///
    /// # Errors
    ///
    /// Returns an error for unexpected or missing bindings, incompatible logical metadata, conflicting tied values, failed tensor reads, exhausted state identity numbering, or changed imported specialization values.
    pub fn bind_state(&self, tensors: &HashMap<String, Tensor>) -> Result<StateSnapshot> {
        let state = StateSnapshot::create(Arc::clone(&self.description.schema), tensors)?;
        self.validate_state_schema(&state)?;
        Ok(state)
    }

    /// Records values consumed by import-time specialization as semantic guards.
    #[cfg(feature = "onnx")]
    pub(crate) fn specialize_state(
        &self,
        names: impl IntoIterator<Item = String>,
        state: &StateSnapshot,
    ) -> Result<Self> {
        self.validate_state_schema(state)?;
        let mut specialized_state = self.description.specialized_state.clone();
        for name in names {
            // Constant node outputs are already represented by logical constants.
            if let Some(tensor) = state.tensor(&name) {
                specialized_state.insert(name, tensor.to_bytes()?);
            }
        }
        Ok(Self {
            description: Arc::new(ProgramDescription {
                graph: self.description.graph.clone(),
                signature: self.description.signature.clone(),
                schema: Arc::clone(&self.description.schema),
                specialized_state,
                analysis: self.description.analysis.clone(),
            }),
        })
    }

    pub(crate) fn specialized_state(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.description.specialized_state
    }

    #[cfg_attr(
        not(feature = "onnx"),
        expect(dead_code, reason = "only onnx export binds state into a graph")
    )]
    pub(crate) fn bound_graph(&self, state: &StateSnapshot) -> Result<Graph> {
        self.validate_state_schema(state)?;
        self.bind_graph_state(self.logical_graph().clone(), state)
    }

    /// The caller has validated this immutable publication at preparation or rebinding.
    /// Do not reread specialized values from CUDA during inference.
    pub(crate) fn bind_graph_state(
        &self,
        mut graph: Graph,
        state: &StateSnapshot,
    ) -> Result<Graph> {
        if !Arc::ptr_eq(&self.description.schema, &state.schema) {
            return Err(Error::invalid_argument("state snapshot program schema"));
        }

        for binding in self.state_schema() {
            let tensor = state
                .tensor(binding.name())
                .cloned()
                .ok_or_else(|| Error::missing_input("bind graph state"))?;
            match binding.kind {
                StateBindingKind::Default => graph.add_initializer(binding.name(), tensor)?,
                StateBindingKind::Static => graph.add_static_initializer(binding.name(), tensor)?,
                StateBindingKind::Parameter => graph.add_parameter(binding.name(), tensor)?,
                StateBindingKind::Buffer => graph.add_buffer(binding.name(), tensor)?,
            }
        }
        Ok(graph)
    }

    /// Called after the executor's complete signature guards.
    /// Only tensor metadata enters the shape stage; placement and state binding follow it.
    pub(crate) fn resolve_runtime_shapes(&self, inputs: &HashMap<String, Tensor>) -> Result<Graph> {
        let mut graph = self.logical_graph().clone();
        graph.rewrite_nodes(|_, mut node| {
            if let Some((name, ..)) = node.view().input_metadata()
                && let Some(tensor) = inputs.get(name)
            {
                node.update_input_metadata(tensor);
            }
            Ok(node)
        })?;
        graph.resolve_shape_program()
    }

    pub(crate) fn validate_state_schema(&self, state: &StateSnapshot) -> Result<()> {
        if !Arc::ptr_eq(&self.description.schema, &state.schema) {
            return Err(Error::invalid_argument("state snapshot program schema"));
        }

        for (name, expected) in &self.description.specialized_state {
            let tensor = state
                .tensor(name)
                .ok_or_else(|| Error::missing_input("validate state schema"))?;
            if tensor.to_bytes()? != *expected {
                return Err(Error::unsupported_operation_with_details(
                    "specialized state binding",
                    format!("`{name}` differs from its imported value"),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn logical_graph(&self) -> &Graph {
        &self.description.graph
    }

    pub(crate) fn retained_analyses(&self) -> impl Iterator<Item = &CompiledGraph> {
        self.description.analysis.get().into_iter()
    }
}

impl Graph {
    /// Lifts this graph's initializer bindings into a logical program with a separate state schema.
    ///
    /// # Errors
    ///
    /// Returns an error if constructing the graph signature fails or an initializer has no matching input.
    pub fn program(&self) -> Result<GraphProgram> {
        GraphProgram::create(self)
    }
}

#[cfg(test)]
mod tests;
