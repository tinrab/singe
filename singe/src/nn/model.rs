use std::{collections::HashMap, fmt::Debug, sync::Arc};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{Graph, GraphExecutor, GraphInputSpec, GraphPrepareOptions, GraphProgram},
    nn::{InputGroup, ModelState},
    state::StateSnapshot,
    tensor::Tensor,
};

/// A computation over tensors and named state: the shareable description of a model or of one part of it.
///
/// `forward` runs on real tensors eagerly, and on `Device::Meta` tensors or under capture when a program is built, so it is the one definition of the layer.
/// State lives outside the layer, read by key from a [`ModelState`], so one layer serves any number of workers and state publications.
/// A layer with several inputs takes a named input type.
pub trait Layer<Input = Tensor, Output = Tensor>: Debug + Send + Sync {
    /// Runs the layer on `input` with `state`.
    ///
    /// # Errors
    ///
    /// Returns an error if the layer's state lookup, shape or type checks, or an operation fails.
    fn forward(&self, input: &Input, state: &ModelState) -> Result<Output>;
}

impl Tensor {
    // Runs `layer` on this tensor, so a layer's steps read in order: `x.apply(&a, state)?.apply(&b, state)`.
    pub fn apply<L: Layer + ?Sized>(&self, layer: &L, state: &ModelState) -> Result<Self> {
        layer.forward(self, state)
    }
}

/// Heterogeneous layers applied in insertion order.
#[derive(Debug, Default)]
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    /// Creates a sequence with no layers; applying it preserves the input value.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a layer after the existing layers and returns the sequence.
    #[must_use]
    pub fn with(mut self, layer: impl Layer + 'static) -> Self {
        self.layers.push(Box::new(layer));
        self
    }
}

impl Layer for Sequential {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        self.layers.iter().try_fold(input.clone(), |value, layer| {
            value.apply(layer.as_ref(), state)
        })
    }
}

/// A composable layer and its immutable named state.
#[derive(Debug)]
pub struct Model<L> {
    layer: Arc<L>,
    state: ModelState,
}

impl<L> Clone for Model<L> {
    fn clone(&self) -> Self {
        Self {
            layer: Arc::clone(&self.layer),
            state: self.state.clone(),
        }
    }
}

impl<L> Model<L> {
    /// Pairs `layer` with `state`.
    ///
    /// A key that `state` cannot resolve fails when the layer first reads it, which [`Self::build`] does without running kernels.
    #[must_use]
    pub fn new(layer: L, state: ModelState) -> Self {
        Self {
            layer: Arc::new(layer),
            state,
        }
    }

    /// Returns the retained layer description.
    #[must_use]
    pub fn layer(&self) -> &L {
        &self.layer
    }

    /// Returns this model's immutable state.
    #[must_use]
    pub fn state(&self) -> &ModelState {
        &self.state
    }

    /// Returns a model sharing this layer over the replacement state, which must descend from the same builder.
    ///
    /// # Errors
    ///
    /// Returns an error if `state` comes from another builder.
    pub fn with_state(&self, state: ModelState) -> Result<Self> {
        if !self.state.same_owner(&state) {
            return Err(Error::invalid_argument(
                "replacement state comes from another builder",
            ));
        }
        Ok(Self {
            layer: Arc::clone(&self.layer),
            state,
        })
    }

    /// Returns a model with state transferred to `device` and optional conversion of floating tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if state conversion or transfer fails.
    pub fn to(&self, device: Device, floating_dtype: Option<DataType>) -> Result<Self> {
        self.with_state(self.state.to(device, floating_dtype)?)
    }
}

impl<L> Model<L> {
    // Runs the layer eagerly on a group of named inputs, such as an encoder's `TokenBatch`.
    //
    // # Errors
    //
    // Returns what the layer's `forward` returns.
    pub fn forward_group<I: InputGroup>(&self, inputs: &I) -> Result<Tensor>
    where
        L: Layer<I>,
    {
        self.layer.forward(inputs, &self.state)
    }

    // Captures the layer over a group's tensors, as graph inputs named after the group's fields, with the model's state bound, and returns the graph and its eager output.
    // The graph compiles, prepares, exports, and tunes as any captured graph does.
    //
    // # Errors
    //
    // Returns an error if capture or the layer's `forward` fails.
    pub fn capture_group<I: InputGroup>(&self, inputs: &I) -> Result<(Graph, Tensor)>
    where
        L: Layer<I>,
    {
        let (graph, outputs) = Graph::capture_with_bindings(
            inputs.named_tensors(),
            self.state.capture_bindings()?,
            |tensors| {
                Ok(vec![
                    self.layer
                        .forward(&I::from_tensors(tensors)?, &self.state)?,
                ])
            },
        )?;
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::missing_output("captured model"))?;
        Ok((graph, output))
    }
}

impl<L: Layer> Model<L> {
    /// Executes the retained layer eagerly using this model's immutable state.
    ///
    /// # Errors
    ///
    /// Returns an error from layer state lookup, validation, or operation execution.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.layer.forward(input, &self.state)
    }

    /// Constructs a program from input metadata, on `Device::Meta`: no example tensor, tensor storage, parameter copies, or kernels.
    /// Rust metadata branches choose one fixed construction path.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph input or state declarations are invalid, or running the layer on metadata fails.
    pub fn build(&self, input_name: &str, spec: impl Into<GraphInputSpec>) -> Result<ModelProgram> {
        let state = self.state.to(Device::Meta, None)?;
        let graph = Graph::build([(input_name, spec)], state.capture_bindings()?, |inputs| {
            Ok(vec![self.layer.forward(&inputs[0], &state)?])
        })?;
        Ok(ModelProgram {
            program: self.state.program(&graph)?,
            state: self.state.snapshot.clone(),
        })
    }

    /// Captures the same `forward` used eagerly and declares schema ties in the resulting program.
    /// Host-selected paths retain ordinary capture semantics.
    ///
    /// # Errors
    ///
    /// Returns an error if state roles are unsupported, a binding is missing, eager forward execution or graph capture fails, or constructing and binding the captured program schema fails.
    pub fn capture(&self, input_name: &str, example: Tensor) -> Result<ModelProgram> {
        let (graph, _) = self.capture_graph(input_name, example)?;
        Ok(ModelProgram {
            program: self.state.program(&graph)?,
            state: self.state.snapshot.clone(),
        })
    }

    // Captures the layer over one input, as `capture_group` captures a group, and returns the graph and its eager output.
    //
    // # Errors
    //
    // Returns an error if capture or the layer's `forward` fails.
    pub fn capture_graph(&self, input_name: &str, example: Tensor) -> Result<(Graph, Tensor)> {
        let (graph, outputs) = Graph::capture_with_bindings(
            [(input_name, example)],
            self.state.capture_bindings()?,
            |inputs| Ok(vec![self.forward(&inputs[0])?]),
        )?;
        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::missing_output("captured model"))?;
        Ok((graph, output))
    }

    /// Captures this model with the named example input and prepares an executor using `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if capture, state validation, graph compilation, device placement, or native execution-resource preparation fails.
    pub fn prepare(
        &self,
        input_name: &str,
        example: Tensor,
        options: GraphPrepareOptions,
    ) -> Result<GraphExecutor> {
        self.capture(input_name, example)?.prepare(options)
    }
}

/// A model's constructed or captured program and bound state.
/// New state is checked against this program before explicit executor replacement or rebinding.
#[derive(Debug, Clone)]
pub struct ModelProgram {
    program: GraphProgram,
    state: StateSnapshot,
}

impl ModelProgram {
    /// Returns the constructed or captured logical program.
    #[must_use]
    pub fn program(&self) -> &GraphProgram {
        &self.program
    }

    /// Returns the state snapshot bound when this program was constructed.
    #[must_use]
    pub fn state(&self) -> &StateSnapshot {
        &self.state
    }

    /// Binds model state to this program, returning its existing snapshot when schema identities already match.
    ///
    /// Otherwise bindings are validated and published against the program schema.
    ///
    /// # Errors
    ///
    /// Returns an error if bindings are missing or unexpected, logical metadata or tied values conflict, required tensor reads fail, or specialized imported values do not match.
    pub fn bind_state(&self, state: &ModelState) -> Result<StateSnapshot> {
        if Arc::ptr_eq(&self.state.schema, &state.snapshot.schema) {
            return Ok(state.snapshot.clone());
        }
        self.program.bind_state(
            &state
                .snapshot
                .bindings()
                .map(|(name, tensor)| (name.to_owned(), tensor.clone()))
                .collect::<HashMap<_, _>>(),
        )
    }

    /// Prepares an execution instance using this program's bound state and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration, state validation, shape resolution, device placement, compilation, or native resource preparation fails.
    pub fn prepare(&self, options: GraphPrepareOptions) -> Result<GraphExecutor> {
        self.program.prepare(&self.state.clone(), options)
    }
}
