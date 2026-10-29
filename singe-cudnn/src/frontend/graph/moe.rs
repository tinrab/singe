pub use crate::backend::execution::moe::{
    MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors, MoeGroupedMatmulInputs,
    MoeGroupedMatmulTensors,
};
use crate::{
    backend::tensor::{Shape, TensorId},
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::{Graph, TensorOutputConfig},
        operation::{
            MoeGroupedMatmulBackwardConfig, MoeGroupedMatmulConfig, MoeGroupedMatmulOperation,
            MoeGroupedMatmulRouting, Operation,
        },
    },
    utility::check_range,
};

impl Graph {
    /// Records an expert-grouped matmul into an explicit output tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or routing tensors, non-3D token or weight layouts, incompatible contracted dimensions, unsupported integer routing encodings or ranks, an invalid scatter top-k count, or an output-dimension mismatch.
    pub fn moe_grouped_matmul_into(
        &mut self,
        tensors: MoeGroupedMatmulTensors,
        config: MoeGroupedMatmulConfig,
    ) -> Result<()> {
        let operation = MoeGroupedMatmulOperation::Forward { tensors, config };
        operation.validate_graph(self)?;
        self.operations.push(Operation::MoeGroupedMatmul(operation));
        Ok(())
    }

    /// Infers the routed output with placement and encoding independent of matmul precision.
    /// A configured scope names the result `name::Output`, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or routing tensors, non-3D token or weight layouts, incompatible contracted dimensions, unsupported integer routing encodings or ranks, an invalid scatter top-k count, or an output-dimension mismatch. Inferred output strides must also fit the native layout representation.
    pub fn moe_grouped_matmul(
        &mut self,
        inputs: MoeGroupedMatmulInputs,
        config: MoeGroupedMatmulConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let token_tensor = self.tensor_spec(inputs.token)?;
        let output_dimensions =
            self.infer_moe_grouped_matmul_output_dimensions(inputs, config.routing())?;
        let output = output.into_spec(
            self.config.data_type_policy,
            token_tensor.data_type,
            Shape::contiguous(output_dimensions)?,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "Output");
        if let Err(error) =
            self.moe_grouped_matmul_into(MoeGroupedMatmulTensors::new(inputs, output), config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn infer_moe_grouped_matmul_output_dimensions(
        &self,
        inputs: MoeGroupedMatmulInputs,
        routing: MoeGroupedMatmulRouting<TensorId>,
    ) -> Result<[i64; 3]> {
        let token_tensor = self.tensor_spec(inputs.token)?;
        let weight_tensor = self.tensor_spec(inputs.weight)?;
        self.validate_tensor_rank(inputs.token, 3, "moe grouped matmul token rank")?;
        self.validate_tensor_rank(inputs.weight, 3, "moe grouped matmul weight rank")?;
        self.validate_tensor_data_type_supported(
            inputs.first_token_offset,
            &[DataType::I32, DataType::I64],
            "moe grouped matmul first token offset data type",
        )?;
        if weight_tensor.shape.dimensions()[1] != token_tensor.shape.dimensions()[2] {
            return Err(Error::ShapeMismatch {
                name: "moe grouped matmul contracted dim".into(),
                expected: vec![token_tensor.shape.dimensions()[2]],
                actual: vec![weight_tensor.shape.dimensions()[1]],
            });
        }

        if let Some(token_index) = routing.token_index() {
            self.validate_tensor_data_type(
                token_index,
                DataType::I32,
                "moe grouped matmul token index data type",
            )?;
            self.validate_tensor_min_rank(token_index, 2, "moe grouped matmul token index rank")?;
        }
        let output_tokens = match routing {
            MoeGroupedMatmulRouting::Gather { token_index } => {
                self.tensor_spec(token_index)?.shape.dimensions()[1]
            }
            MoeGroupedMatmulRouting::None | MoeGroupedMatmulRouting::Scatter { .. } => {
                token_tensor.shape.dimensions()[1]
            }
        };

        if let MoeGroupedMatmulRouting::Scatter {
            token_ks, top_k, ..
        } = routing
        {
            self.validate_tensor_data_type(
                token_ks,
                DataType::I32,
                "moe grouped matmul token ks data type",
            )?;
            check_range!("moe top k", top_k > 0)?;
        }

        Ok([1, output_tokens, weight_tensor.shape.dimensions()[2]])
    }

    /// Records grouped weight gradients with a caller-declared expert dimension and layout.
    /// Expert count is independent of offset-table length, including terminal boundaries and shared experts.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensors, invalid activation, gradient, or offset ranks, unsupported offset encoding, mismatched token counts, a nonpositive expert count, or a weight-gradient shape mismatch.
    pub fn moe_grouped_matmul_backward_into(
        &mut self,
        tensors: MoeGroupedMatmulBackwardTensors,
        config: MoeGroupedMatmulBackwardConfig,
    ) -> Result<()> {
        let operation = MoeGroupedMatmulOperation::Backward { tensors, config };
        operation.validate_graph(self)?;
        self.operations.push(Operation::MoeGroupedMatmul(operation));
        Ok(())
    }

    /// Infers column-major weight gradients.
    /// Set the configuration's expert-count hint for terminal boundaries or shared experts.
    /// Without a hint, inference uses offset-table length, matching C++'s legacy inference.
    /// Requested placement selects I/O or intermediate precision unless the result encoding is explicit.
    /// A configured scope names the gradient `name::DWeight`, matching C++.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown tensors, invalid activation, gradient, or offset ranks, unsupported offset encoding, mismatched token counts, a nonpositive expert count, or a weight-gradient shape mismatch. Inferred column-major gradient strides must also be representable.
    pub fn moe_grouped_matmul_backward(
        &mut self,
        inputs: MoeGroupedMatmulBackwardInputs,
        config: MoeGroupedMatmulBackwardConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let token_tensor = self.tensor_spec(inputs.token)?;
        let [experts, rows, columns] = self
            .infer_moe_grouped_matmul_backward_weight_gradient_dimensions(
                inputs,
                config.expert_count(),
            )?;
        let shape = Shape::contiguous([experts, columns, rows])?.transpose_last_two()?;
        let weight_gradient =
            output.into_spec(self.config.data_type_policy, token_tensor.data_type, shape);
        let checkpoint = self.mutation_checkpoint();
        let weight_gradient = self.inferred_tensor(weight_gradient, config.name(), "DWeight");
        if let Err(error) = self.moe_grouped_matmul_backward_into(
            MoeGroupedMatmulBackwardTensors::new(inputs, weight_gradient),
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(weight_gradient)
    }

    pub(crate) fn infer_moe_grouped_matmul_backward_weight_gradient_dimensions(
        &self,
        inputs: MoeGroupedMatmulBackwardInputs,
        expert_count: Option<i64>,
    ) -> Result<[i64; 3]> {
        let output_gradient_tensor = self.tensor_spec(inputs.output_gradient)?;
        let token_tensor = self.tensor_spec(inputs.token)?;
        let first_token_offset_tensor = self.tensor_spec(inputs.first_token_offset)?;

        self.validate_tensor_rank(
            inputs.output_gradient,
            3,
            "moe grouped matmul backward output rank",
        )?;
        self.validate_tensor_rank(inputs.token, 3, "moe grouped matmul backward token rank")?;
        self.validate_tensor_min_rank(
            inputs.first_token_offset,
            1,
            "moe grouped matmul backward first token offset rank",
        )?;
        self.validate_tensor_data_type_supported(
            inputs.first_token_offset,
            &[DataType::I32, DataType::I64],
            "moe grouped matmul backward first token offset",
        )?;

        let token_dims = token_tensor.shape.dimensions();
        let output_gradient_dims = output_gradient_tensor.shape.dimensions();
        if output_gradient_dims[1] != token_dims[1] {
            let expected_output_gradient_dimensions = [
                output_gradient_dims[0],
                token_dims[1],
                output_gradient_dims[2],
            ];
            self.validate_tensor_dimensions(
                inputs.output_gradient,
                &expected_output_gradient_dimensions,
                "moe grouped matmul backward output gradient shape",
            )?;
        }

        let expert_count =
            expert_count.unwrap_or_else(|| first_token_offset_tensor.shape.dimensions()[0]);
        check_range!("moe expert count", expert_count > 0)?;
        Ok([expert_count, token_dims[2], output_gradient_dims[2]])
    }
}
