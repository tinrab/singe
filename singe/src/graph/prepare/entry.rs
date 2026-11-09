use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::graph::GraphExecutionMode;
use crate::{
    device::Device,
    error::{Error, Result},
    graph::{
        Graph, GraphProgram,
        prepare::{GraphExecutor, GraphPrepareOptions},
    },
    state::StateSnapshot,
};

impl Graph {
    /// Places constants and state on the target, normalizes the graph once, and prepares an exclusive execution instance.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration has zero cache capacity, state or runtime shape requirements are invalid, graph placement or compilation fails, a required backend or numerical policy is unsupported, or preparing native resources fails.
    pub fn prepare(&self, options: GraphPrepareOptions) -> Result<GraphExecutor> {
        let program = self.program()?;
        let state = program.bind_state(&self.initializers)?;
        program.prepare(&state, options)
    }

    pub(crate) fn placed_on(&self, device: Device) -> Result<Self> {
        if self.values().iter().all(|value| value.device() == device)
            && self
                .initializers
                .values()
                .all(|tensor| tensor.device() == device)
        {
            return Ok(self.clone());
        }

        let mut graph = self.clone();
        for tensor in graph.initializers.values_mut() {
            *tensor = tensor.to_device(device)?;
        }

        graph.rewrite_nodes(|graph, node| {
            let mut node = node.view().placed_on(device)?;
            if let Some((name, _, _, _)) = node.view().input_metadata()
                && let Some(tensor) = graph.initializers.get(name)
            {
                node.replace_output_metadata(tensor.data_type(), tensor.layout().clone(), device);
            }
            Ok(node)
        })?;
        Ok(graph)
    }
}

impl GraphProgram {
    /// Validates bound state, resolves construction shapes, places state and constants, compiles the graph, and prepares an exclusive execution instance.
    ///
    /// The snapshot must belong to this exact program schema and satisfy any imported specialization guards.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration has zero cache capacity, state or runtime shape requirements are invalid, graph placement or compilation fails, a required backend or numerical policy is unsupported, or preparing native resources fails.
    pub fn prepare(
        &self,
        state: &StateSnapshot,
        options: GraphPrepareOptions,
    ) -> Result<GraphExecutor> {
        let started = Instant::now();
        if options.cache_capacity == 0 {
            return Err(Error::invalid_argument("graph cache capacity"));
        }

        #[cfg(feature = "cuda")]
        if options.device.is_cuda() && options.mode == GraphExecutionMode::Compiled {
            options.cuda.validate_next_recording(0)?;
        }

        self.validate_state_schema(state)?;
        let resolved = self.logical_graph().resolve_shape_program()?;
        let state = state.to_device(options.device)?;
        let placed = self
            .bind_graph_state(resolved, &state)?
            .retain_executable_nodes()?
            .placed_on(options.device)?;
        let compiled = placed.compile_with_freeze(options.compile, options.freeze)?;
        let mut executor = GraphExecutor::create(compiled, options, self.clone(), state, placed)?;
        executor.preparation_time = started.elapsed();
        Ok(executor)
    }
}
