mod session;

use std::{
    collections::{HashMap, HashSet},
    mem,
    sync::Arc,
};

pub use self::session::{
    dimension, evaluate_dimensions, is_active, record_reshape, register_constant,
    with_active_session, with_session,
};

use crate::{
    backend::registry::{
        CheckedOperationRequest, Operation, OperationAttributes, StridedAttributes,
    },
    device::Device,
    error::{CaptureError, Error, Result},
    graph::{
        CaptureBindings, CaptureInfo, CaptureOptions, DimensionExpr, Graph, GraphOperationInputs,
        GraphOperationOutput, GraphOperationRequest, GraphOutput, InputDimensionGuard,
        capture::options::CaptureBinding, generators::GraphGenerator, ir::ValueId,
        shape_program::ReshapeExpression,
    },
    observation::{CaptureReadGuard, track_capture_storage},
    random::Generator,
    shape::Shape,
    tensor::{Tensor, TensorId},
    utility::to_i64,
};

#[derive(Debug)]
pub struct CaptureSession {
    // Identifies this session in its dimension expressions, so another session's are refused.
    owner: Arc<()>,
    graph: Graph,
    tensor_nodes: HashMap<TensorId, ValueId>,
    binding_names: HashSet<String>,
    /// Named tensors bound as graph inputs, in binding order.
    inputs: Vec<(String, Tensor)>,
    /// For each storage identity, a recorded tensor that covers the whole storage contiguously.
    storage_bases: HashMap<usize, ValueId>,
    /// For each storage written during capture, the flat graph value holding its current contents.
    written: HashMap<usize, ValueId>,
    /// Generators bound as inputs, with the state tensor and blocks used so far.
    generators: Vec<(GraphGenerator, Tensor)>,
    // Results narrowed to a count read during capture, by tensor, with the values of their bound and of the count.
    counted: HashMap<TensorId, (ValueId, ValueId)>,
    _read_guard: CaptureReadGuard,
}

impl CaptureSession {
    pub(crate) fn record_invocation(
        &mut self,
        request: &CheckedOperationRequest,
        inputs: &[Tensor],
        outputs: &[Tensor],
    ) -> Result<()> {
        let operands = inputs
            .iter()
            .map(|input| self.ensure_tensor(input))
            .collect::<Result<Vec<_>>>()?;
        for (index, output) in outputs.iter().enumerate() {
            let inputs = GraphOperationInputs::for_result(
                request.operation(),
                request.attributes().clone(),
                operands.clone(),
                index,
            )?;
            let node = self
                .graph
                .add_operation_request(GraphOperationRequest::new(
                    None,
                    inputs,
                    Self::operation_output(output),
                ))?;
            self.record_output(output, node)?;
        }
        Ok(())
    }

    fn create(options: &CaptureOptions) -> Result<Self> {
        let mut graph = Graph::new();
        graph.capture_info = Some(CaptureInfo {
            value_specialization: options.allow_value_specialization,
        });
        Ok(Self {
            owner: Arc::new(()),
            graph,
            tensor_nodes: HashMap::new(),
            binding_names: HashSet::new(),
            inputs: Vec::new(),
            storage_bases: HashMap::new(),
            counted: HashMap::new(),
            written: HashMap::new(),
            generators: Vec::new(),
            _read_guard: CaptureReadGuard::enter(options.allow_value_specialization)?,
        })
    }

    fn with_inputs<I, S>(
        inputs: I,
        bindings: CaptureBindings,
        mut options: CaptureOptions,
    ) -> Result<(Self, Vec<Tensor>)>
    where
        I: IntoIterator<Item = (S, Tensor)>,
        S: Into<String>,
    {
        let dimensions = mem::take(&mut options.dimensions);
        let mut session = Self::create(&options)?;
        let mut tensors = Vec::new();
        for (name, tensor) in inputs {
            let tensor = tensor.alias_unrecorded();
            session.add_input(&name.into(), &tensor)?;
            tensors.push(tensor);
        }

        session.add_dimension_guards(&dimensions, &tensors)?;
        for binding in bindings.bindings {
            match binding {
                CaptureBinding::Parameter(name, tensor) => {
                    session.add_input(&name.clone(), &tensor)?;
                    session.graph.add_parameter(&name, tensor)?;
                }
                CaptureBinding::Buffer(name, tensor) => {
                    session.add_input(&name.clone(), &tensor)?;
                    session.graph.add_buffer(&name, tensor)?;
                }
                CaptureBinding::Constant(name, tensor) => {
                    session.bind_constant(name, &tensor)?;
                }
            }
        }
        Ok((session, tensors))
    }

    // Declares run-time axes on the inputs, whose representatives must agree where a symbol repeats.
    fn add_dimension_guards(
        &mut self,
        dimensions: &[InputDimensionGuard],
        tensors: &[Tensor],
    ) -> Result<()> {
        let mut symbols = HashMap::new();
        for dimension in dimensions {
            let position = self
                .inputs
                .iter()
                .position(|(name, _)| name == dimension.input())
                .ok_or_else(|| Error::missing_input("capture dimension guard"))?;
            let value = tensors
                .get(position)
                .and_then(|tensor| tensor.shape().dimensions().get(dimension.axis()))
                .copied()
                .ok_or(Error::DimensionOutOfBounds)?;
            if symbols
                .insert(dimension.name().to_owned(), value)
                .is_some_and(|previous| previous != value)
            {
                return Err(Error::invalid_argument(
                    "inconsistent representative symbolic dimensions",
                ));
            }

            self.graph.add_input_dimension_guard(
                dimension.input(),
                dimension.axis(),
                dimension.name(),
                dimension.min(),
                dimension.max(),
            )?;
        }
        Ok(())
    }

    // Returns an expression for one axis of `tensor`, or `None` for a tensor this capture did not record.
    pub(crate) fn dimension(
        &mut self,
        tensor: &Tensor,
        axis: usize,
    ) -> Result<Option<DimensionExpr>> {
        tensor
            .shape()
            .dimensions()
            .get(axis)
            .ok_or(Error::DimensionOutOfBounds)?;
        if !self
            .written
            .contains_key(&tensor.storage().storage_identity())
            && !self.tensor_nodes.contains_key(&tensor.id())
        {
            return Ok(None);
        }

        let value = self.ensure_tensor(tensor)?;
        Ok(Some(DimensionExpr::axis(
            Some(Arc::clone(&self.owner)),
            value.index(),
            axis,
        )))
    }

    // Evaluates `dimensions` against the shapes this capture has recorded, refusing another capture's expressions.
    pub(crate) fn evaluate_dimensions(&self, dimensions: &[DimensionExpr]) -> Result<Vec<i64>> {
        dimensions
            .iter()
            .map(|dimension| {
                dimension
                    .remap(Some(&self.owner), &Ok)?
                    .evaluate(&mut |value, axis| {
                        self.graph
                            .required_value_shape(ValueId::new(value))?
                            .dimensions()
                            .get(axis)
                            .copied()
                            .ok_or(Error::DimensionOutOfBounds)
                    })
            })
            .collect()
    }

    // Records that `output`, a reshape's result, takes its shape from `dimensions` on each run.
    pub(crate) fn record_reshape(
        &mut self,
        output: &Tensor,
        dimensions: &[DimensionExpr],
    ) -> Result<()> {
        if self.graph.input_dimension_guards.is_empty() {
            return Ok(());
        }

        let node = *self
            .tensor_nodes
            .get(&output.id())
            .ok_or_else(|| Error::from(CaptureError::UnrecordedTensor))?;
        let dimensions = dimensions
            .iter()
            .map(|dimension| dimension.remap(Some(&self.owner), &Ok))
            .collect::<Result<Vec<_>>>()?;
        self.graph.shape_program.reshapes.insert(
            node,
            ReshapeExpression {
                dimensions,
                allow_zero: true,
            },
        );
        Ok(())
    }

    fn add_input(&mut self, name: &str, tensor: &Tensor) -> Result<ValueId> {
        self.require_new_binding(name, tensor)?;
        let node = self.graph.add_input(
            name,
            tensor.data_type(),
            tensor.layout().clone(),
            tensor.device(),
        );
        self.record_output(tensor, node)?;
        self.inputs.push((name.to_owned(), tensor.clone()));
        Ok(node)
    }

    fn require_new_binding(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        if !self.binding_names.insert(name.to_owned()) {
            return Err(Error::DuplicateInput);
        }

        if self.tensor_nodes.contains_key(&tensor.id()) {
            return Err(Error::invalid_argument("duplicate capture tensor binding"));
        }
        Ok(())
    }

    fn bind_constant(&mut self, name: String, tensor: &Tensor) -> Result<ValueId> {
        self.require_new_binding(&name, tensor)?;
        let node = self.graph.add_named_constant(name, tensor.clone());
        self.record_output(tensor, node)?;
        Ok(node)
    }

    fn explicit_constant(&mut self, name: String, tensor: &Tensor) -> Result<Tensor> {
        let tensor = tensor.alias_unrecorded();
        self.bind_constant(name, &tensor)?;
        Ok(tensor)
    }

    pub(crate) fn record_literal(&mut self, tensor: &Tensor) -> Result<()> {
        let node = self.graph.add_constant(tensor.clone());
        self.record_output(tensor, node)
    }

    /// Returns the graph value of `tensor`.
    ///
    /// A tensor whose storage was written during capture is read as a strided view of the storage's current value.
    pub(crate) fn ensure_tensor(&mut self, tensor: &Tensor) -> Result<ValueId> {
        if let Some(&flat) = self.written.get(&tensor.storage().storage_identity()) {
            return self.add_node(
                Operation::AsStrided,
                OperationAttributes::Strided(StridedAttributes::from_layout(tensor.layout())),
                vec![flat],
                tensor,
            );
        }

        if self.counted.contains_key(&tensor.id()) {
            return Err(Error::unsupported_operation(
                "a result whose extent depends on values, which a compiled graph returns but does not compute with",
            ));
        }
        self.tensor_nodes
            .get(&tensor.id())
            .copied()
            .ok_or_else(|| Error::from(CaptureError::UnrecordedTensor))
    }

    // Records `narrowed`, the leading `count` rows of `bound`, as a result a graph returns narrowed after each run.
    pub(crate) fn record_counted(
        &mut self,
        narrowed: &Tensor,
        bound: &Tensor,
        count: &Tensor,
    ) -> Result<()> {
        let bound = self.ensure_tensor(bound)?;
        let count = self.ensure_tensor(count)?;
        self.counted.insert(narrowed.id(), (bound, count));
        Ok(())
    }

    /// Returns the bound state of `generator` and the blocks used before this call, binding it as an input on first use.
    ///
    /// The generator also advances, so capture consumes numbers as eager execution would.
    pub(crate) fn generator_input(
        &mut self,
        generator: &Generator,
        device: Device,
        blocks: u64,
    ) -> Result<(Tensor, u64)> {
        generator.reserve(blocks)?;
        if let Some((bound, state)) = self
            .generators
            .iter_mut()
            .find(|(bound, _)| bound.generator.id() == generator.id() && bound.device == device)
        {
            let offset = bound.blocks;
            bound.blocks += blocks;
            return Ok((state.clone(), offset));
        }

        let mut start = generator.state()?;
        start.offset -= blocks;
        let state = start.tensor(device)?;
        let name = format!("generator.{}.{device}", generator.id());
        self.add_input(&name, &state)?;
        self.generators.push((
            GraphGenerator {
                name,
                generator: generator.clone(),
                device,
                blocks,
            },
            state.clone(),
        ));
        Ok((state, 0))
    }

    /// Records `assign(destination, source)` as a pure scatter into the destination storage's flat value.
    pub(crate) fn record_write(&mut self, destination: &Tensor, source: &Tensor) -> Result<()> {
        let replaces_storage = destination.is_contiguous()
            && destination.storage_offset() == 0
            && destination.byte_size()? == destination.storage().byte_size()
            && source.is_contiguous()
            && source.storage_offset() == 0;
        let source = self.ensure_tensor(source)?;
        let storage = destination.storage();
        let identity = storage.storage_identity();
        let flat_shape = Shape::create([to_i64(
            storage.byte_size() / destination.data_type().size_bytes(),
            "captured storage element count",
        )?])?;
        if !self.written.contains_key(&identity) && !self.storage_bases.contains_key(&identity) {
            return Err(Error::unsupported_operation(
                "capture write to storage that no captured tensor covers contiguously",
            ));
        }

        // A write of the whole storage reads none of it, so the storage's value simply becomes the source.
        if replaces_storage {
            let replaced = self.add_node_with_shape(
                Operation::Reshape,
                OperationAttributes::Shape(flat_shape.clone()),
                vec![source],
                destination,
                flat_shape,
            )?;
            self.written.insert(identity, replaced);
            return Ok(());
        }

        let flat = if let Some(&flat) = self.written.get(&identity) {
            flat
        } else {
            let base = self.storage_bases[&identity];
            self.add_node_with_shape(
                Operation::Reshape,
                OperationAttributes::Shape(flat_shape.clone()),
                vec![base],
                destination,
                flat_shape.clone(),
            )?
        };
        let updated = self.add_node_with_shape(
            Operation::StridedScatter,
            OperationAttributes::Strided(StridedAttributes::from_layout(destination.layout())),
            vec![flat, source],
            destination,
            flat_shape,
        )?;
        self.written.insert(identity, updated);
        Ok(())
    }

    fn add_node(
        &mut self,
        operation: Operation,
        attributes: OperationAttributes,
        operands: Vec<ValueId>,
        like: &Tensor,
    ) -> Result<ValueId> {
        self.add_node_with_shape(operation, attributes, operands, like, like.shape().clone())
    }

    fn add_node_with_shape(
        &mut self,
        operation: Operation,
        attributes: OperationAttributes,
        operands: Vec<ValueId>,
        like: &Tensor,
        shape: Shape,
    ) -> Result<ValueId> {
        let inputs = GraphOperationInputs::for_result(operation, attributes, operands, 0)?;
        self.graph.add_operation_request(GraphOperationRequest::new(
            None,
            inputs,
            GraphOperationOutput::new(like.data_type(), shape, like.device()),
        ))
    }

    fn record_output(&mut self, tensor: &Tensor, node: ValueId) -> Result<()> {
        track_capture_storage(tensor.storage())?;
        self.tensor_nodes.insert(tensor.id(), node);
        let storage = tensor.storage();
        if tensor.is_contiguous()
            && tensor.storage_offset() == 0
            && tensor.shape().element_count() * tensor.data_type().size_bytes()
                == storage.byte_size()
        {
            self.storage_bases
                .entry(storage.storage_identity())
                .or_insert(node);
        }
        Ok(())
    }

    fn operation_output(output: &Tensor) -> GraphOperationOutput {
        GraphOperationOutput::new(output.data_type(), output.shape().clone(), output.device())
    }

    fn finish(mut self, outputs: Vec<Tensor>) -> Result<(Graph, Vec<Tensor>)> {
        let mut counted_outputs = Vec::new();
        let mut counts = Vec::new();
        let mut output_values = Vec::with_capacity(outputs.len());
        for (position, tensor) in outputs.iter().enumerate() {
            let node = match self.counted.get(&tensor.id()) {
                Some(&(bound, count)) => {
                    counted_outputs.push(position);
                    counts.push(count);
                    bound
                }
                None => self.ensure_tensor(tensor)?,
            };
            output_values.push(GraphOutput::new(node, 0));
        }

        let written_inputs = self
            .inputs
            .iter()
            .filter(|(_, tensor)| {
                self.written
                    .contains_key(&tensor.storage().storage_identity())
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut input_writes = Vec::new();
        for (name, tensor) in written_inputs {
            output_values.push(GraphOutput::new(self.ensure_tensor(&tensor)?, 0));
            input_writes.push(name);
        }

        output_values.extend(counts.into_iter().map(|count| GraphOutput::new(count, 0)));
        self.graph.set_output_values(output_values)?;
        self.graph.input_writes = input_writes;
        self.graph.counted_outputs = counted_outputs;
        self.graph.generators = self
            .generators
            .into_iter()
            .map(|(generator, _)| generator)
            .collect();
        Ok((self.graph, outputs))
    }
}
