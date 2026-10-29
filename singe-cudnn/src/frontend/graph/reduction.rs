use crate::{
    backend::{
        execution::reduction::ReducerAccumulator, reduction::ReduceTensorOperator, tensor::TensorId,
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{Operation, ReductionAxes, ReductionConfig, ReductionOperation},
    },
};

impl Graph {
    /// Infers a scalar FP32 Amax result independently of reduction compute precision.
    pub(crate) fn absolute_max_infer(
        &mut self,
        input: TensorId,
        compute_type: DataType,
    ) -> Result<TensorId> {
        self.reduction(
            input,
            &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax)
                .with_compute_type(compute_type)
                .with_axes(ReductionAxes::All),
            TensorOutputConfig::new()
                .with_data_type(DataType::F32)
                .output_tensor(),
        )
    }

    /// Adds a reduction operation to the graph.
    ///
    /// Reduction modes are controlled by [`ReduceTensorOperator`], with reduced
    /// dimensions inferred from the input and output tensor shapes.
    /// The explicit output keeps its shape, strides and encoding.
    /// Output geometry in `config` applies only to [`Self::reduction`].
    ///
    /// # Errors
    /// Returns an error if an input, output or group-offset tensor is not declared in the graph.
    pub fn reduction_into(
        &mut self,
        input: TensorId,
        output: TensorId,
        config: &ReductionConfig,
    ) -> Result<()> {
        let operation = ReductionOperation::Reduce {
            name: config.name().map(str::to_owned),
            op: config.op(),
            input,
            output,
            compute_type: config.compute_type(),
            is_deterministic: config.is_deterministic(),
            group_offset: config.group_offset(),
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Reduction(operation));
        Ok(())
    }

    /// Adds a reduction operation and creates the output tensor.
    /// Unspecified output geometry preserves input dimensions, matching C++ inference.
    /// Select axes or supply an output shape to reduce particular dimensions.
    /// Result placement selects I/O or intermediate precision unless an encoding is explicit.
    ///
    /// # Errors
    ///
    /// Returns an error if the input or group-offset tensor is unknown, explicit axes are empty, repeated, or out of range, or inferred output strides cannot be represented.
    pub fn reduction(
        &mut self,
        input: TensorId,
        config: &ReductionConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let input_tensor = self.tensor_spec(input)?;
        if let Some(group_offset) = config.group_offset() {
            self.tensor_spec(group_offset)?;
        }
        let output_shape = config.infer_output_shape(&input_tensor.shape)?;
        let output = output.into_spec(
            self.config.data_type_policy,
            input_tensor.data_type,
            output_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "Y");
        if let Err(error) = self.reduction_into(input, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }
}

impl Graph {
    /// Adds a user-defined reduction with explicit dimensions and accumulator identities.
    /// Compilation requires native support for the user-defined reducer attributes;
    /// header availability alone does not guarantee that support in the loaded runtime.
    /// Accumulators refer to tensor IDs within `body`, independently of the enclosing graph.
    /// The body retains its own math defaults and passes through normal graph preparation
    /// with the enclosing compilation's live context, or deviceless preparation.
    ///
    /// # Errors
    /// Returns an error for unknown operands or body tensors, incompatible identity encodings, or invalid reduction dimensions and output geometry.
    pub fn custom_reduction(
        &mut self,
        inputs: Vec<TensorId>,
        outputs: Vec<TensorId>,
        dimensions: Vec<i64>,
        body: Self,
        accumulators: Vec<ReducerAccumulator<TensorId>>,
    ) -> Result<()> {
        let operation = ReductionOperation::Custom {
            inputs,
            outputs,
            dimensions,
            body: Box::new(body),
            accumulators,
        };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Reduction(operation));
        Ok(())
    }
}
