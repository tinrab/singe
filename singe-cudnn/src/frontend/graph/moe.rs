use crate::{
    data_type::DataType,
    error::{Error, Result},
    execution::advanced::MoeGroupedMatmulMode,
    frontend::{
        graph::Graph,
        operation::{MoeGroupedMatmulBackwardConfig, MoeGroupedMatmulConfig, Operation},
        support,
    },
    tensor::{Shape, TensorId, TensorSpec},
    utility::check_range,
    version,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGroupedMatmulInputs {
    pub token: TensorId,
    pub weight: TensorId,
    pub first_token_offset: TensorId,
}

impl MoeGroupedMatmulInputs {
    pub fn new(token: TensorId, weight: TensorId, first_token_offset: TensorId) -> Self {
        Self {
            token,
            weight,
            first_token_offset,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGroupedMatmulTensors {
    pub inputs: MoeGroupedMatmulInputs,
    pub output: TensorId,
}

impl MoeGroupedMatmulTensors {
    pub fn new(inputs: MoeGroupedMatmulInputs, output: TensorId) -> Self {
        Self { inputs, output }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGroupedMatmulBackwardInputs {
    pub output_gradient: TensorId,
    pub token: TensorId,
    pub first_token_offset: TensorId,
}

impl MoeGroupedMatmulBackwardInputs {
    pub fn new(output_gradient: TensorId, token: TensorId, first_token_offset: TensorId) -> Self {
        Self {
            output_gradient,
            token,
            first_token_offset,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeGroupedMatmulBackwardTensors {
    pub inputs: MoeGroupedMatmulBackwardInputs,
    pub weight_gradient: TensorId,
}

impl MoeGroupedMatmulBackwardTensors {
    pub fn new(inputs: MoeGroupedMatmulBackwardInputs, weight_gradient: TensorId) -> Self {
        Self {
            inputs,
            weight_gradient,
        }
    }
}

impl Graph {
    pub fn moe_grouped_matmul(
        &mut self,
        tensors: MoeGroupedMatmulTensors,
        config: MoeGroupedMatmulConfig,
    ) -> Result<()> {
        self.validate_moe_grouped_matmul_support_surface_for_version(version()?.raw())?;

        let inputs = tensors.inputs;
        let token_tensor = self.tensor_config(inputs.token)?.clone();
        let expected_output_shape =
            self.infer_moe_grouped_matmul_output_dimensions(inputs, config)?;

        self.validate_tensor_data_type(
            tensors.output,
            token_tensor.data_type,
            "moe grouped matmul output",
        )?;
        self.validate_tensor_dimensions(
            tensors.output,
            expected_output_shape.dimensions(),
            "moe grouped matmul output shape",
        )?;

        self.operations.push(Operation::MoeGroupedMatmul {
            token: inputs.token,
            weight: inputs.weight,
            first_token_offset: inputs.first_token_offset,
            output: tensors.output,
            config,
        });
        Ok(())
    }

    pub fn moe_grouped_matmul_infer(
        &mut self,
        inputs: MoeGroupedMatmulInputs,
        config: MoeGroupedMatmulConfig,
    ) -> Result<TensorId> {
        let token_tensor = self.tensor_config(inputs.token)?.clone();
        let output_dimensions = self.infer_moe_grouped_matmul_output_dimensions(inputs, config)?;
        let checkpoint = self.mutation_checkpoint();
        let output = self.tensor(TensorSpec::new(token_tensor.data_type, output_dimensions));
        if let Err(error) =
            self.moe_grouped_matmul(MoeGroupedMatmulTensors::new(inputs, output), config)
        {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    pub(crate) fn infer_moe_grouped_matmul_output_dimensions(
        &self,
        inputs: MoeGroupedMatmulInputs,
        config: MoeGroupedMatmulConfig,
    ) -> Result<Shape> {
        let token_tensor = self.tensor_config(inputs.token)?;
        let weight_tensor = self.tensor_config(inputs.weight)?;
        self.validate_tensor_data_type(
            inputs.weight,
            token_tensor.data_type,
            "moe grouped matmul weight data type",
        )?;
        self.validate_tensor_rank(inputs.token, 3, "moe grouped matmul token rank")?;
        self.validate_tensor_rank(inputs.weight, 3, "moe grouped matmul weight rank")?;
        self.validate_tensor_data_type(
            inputs.first_token_offset,
            DataType::I32,
            "moe grouped matmul first token offset data type",
        )?;
        if weight_tensor.shape.dimensions()[1] != token_tensor.shape.dimensions()[2] {
            return Err(Error::DescriptorMismatch {
                name: "moe grouped matmul contracted dim".into(),
            });
        }

        let output_tokens = match config.mode() {
            MoeGroupedMatmulMode::None => token_tensor.shape.dimensions()[1],
            MoeGroupedMatmulMode::Gather => {
                let token_index = config.token_index().ok_or(Error::DescriptorMismatch {
                    name: "moe grouped matmul token index".into(),
                })?;
                let token_index_tensor = self.tensor_config(token_index)?;
                self.validate_tensor_data_type(
                    token_index,
                    DataType::I32,
                    "moe grouped matmul token index data type",
                )?;
                self.validate_tensor_min_rank(
                    token_index,
                    2,
                    "moe grouped matmul token index rank",
                )?;
                token_index_tensor.shape.dimensions()[1]
            }
            MoeGroupedMatmulMode::Scatter => {
                let token_index = config.token_index().ok_or(Error::DescriptorMismatch {
                    name: "moe grouped matmul token index".into(),
                })?;
                self.validate_tensor_data_type(
                    token_index,
                    DataType::I32,
                    "moe grouped matmul token index data type",
                )?;
                self.validate_tensor_min_rank(
                    token_index,
                    2,
                    "moe grouped matmul token index rank",
                )?;
                token_tensor.shape.dimensions()[1]
            }
        };

        if matches!(config.mode(), MoeGroupedMatmulMode::Scatter) {
            let token_ks = config.token_ks().ok_or(Error::DescriptorMismatch {
                name: "moe grouped matmul token ks".into(),
            })?;
            self.validate_tensor_data_type(
                token_ks,
                DataType::I32,
                "moe grouped matmul token ks data type",
            )?;
            let top_k = config.top_k().ok_or(Error::DescriptorMismatch {
                name: "moe grouped matmul top_k".into(),
            })?;
            if top_k <= 0 {
                check_range!("top_k", false)?;
            }
        } else if let Some(top_k) = config.top_k()
            && top_k <= 0
        {
            check_range!("top_k", false)?;
        }

        Shape::contiguous([1, output_tokens, weight_tensor.shape.dimensions()[2]])
    }

    pub(crate) fn validate_moe_grouped_matmul_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::MOE_GROUPED_MATMUL.require_descriptor_match(cudnn_version)
    }

    pub fn moe_grouped_matmul_backward(
        &mut self,
        tensors: MoeGroupedMatmulBackwardTensors,
        config: MoeGroupedMatmulBackwardConfig,
    ) -> Result<()> {
        self.moe_grouped_matmul_backward_for_version(version()?.raw(), tensors, config)
    }

    pub(crate) fn moe_grouped_matmul_backward_for_version(
        &mut self,
        cudnn_version: u64,
        tensors: MoeGroupedMatmulBackwardTensors,
        config: MoeGroupedMatmulBackwardConfig,
    ) -> Result<()> {
        self.validate_moe_grouped_matmul_backward_support_surface_for_version(cudnn_version)?;
        let inputs = tensors.inputs;
        let expected = self.infer_moe_grouped_matmul_backward_weight_gradient_shape(inputs)?;
        self.validate_tensor_data_type(
            tensors.weight_gradient,
            self.tensor_config(inputs.token)?.data_type,
            "moe grouped matmul backward weight gradient",
        )?;
        self.validate_tensor_dimensions(
            tensors.weight_gradient,
            expected.dimensions(),
            "moe grouped matmul backward weight gradient shape",
        )?;
        self.operations.push(Operation::MoeGroupedMatmulBackward {
            output_gradient: inputs.output_gradient,
            token: inputs.token,
            first_token_offset: inputs.first_token_offset,
            weight_gradient: tensors.weight_gradient,
            config,
        });
        Ok(())
    }

    pub fn moe_grouped_matmul_backward_infer(
        &mut self,
        inputs: MoeGroupedMatmulBackwardInputs,
        config: MoeGroupedMatmulBackwardConfig,
    ) -> Result<TensorId> {
        self.moe_grouped_matmul_backward_infer_for_version(version()?.raw(), inputs, config)
    }

    pub(crate) fn moe_grouped_matmul_backward_infer_for_version(
        &mut self,
        cudnn_version: u64,
        inputs: MoeGroupedMatmulBackwardInputs,
        config: MoeGroupedMatmulBackwardConfig,
    ) -> Result<TensorId> {
        self.validate_moe_grouped_matmul_backward_support_surface_for_version(cudnn_version)?;
        let token_tensor = self.tensor_config(inputs.token)?.clone();
        let shape = self.infer_moe_grouped_matmul_backward_weight_gradient_shape(inputs)?;
        let checkpoint = self.mutation_checkpoint();
        let weight_gradient = self.tensor(TensorSpec::new(token_tensor.data_type, shape));
        if let Err(error) = self.moe_grouped_matmul_backward_for_version(
            cudnn_version,
            MoeGroupedMatmulBackwardTensors::new(inputs, weight_gradient),
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(weight_gradient)
    }

    pub(crate) fn validate_moe_grouped_matmul_backward_support_surface_for_version(
        &self,
        cudnn_version: u64,
    ) -> Result<()> {
        support::MOE_GROUPED_MATMUL_BACKWARD.require_descriptor_match(cudnn_version)
    }

    fn infer_moe_grouped_matmul_backward_weight_gradient_shape(
        &self,
        inputs: MoeGroupedMatmulBackwardInputs,
    ) -> Result<Shape> {
        let output_gradient_tensor = self.tensor_config(inputs.output_gradient)?;
        let token_tensor = self.tensor_config(inputs.token)?;
        let first_token_offset_tensor = self.tensor_config(inputs.first_token_offset)?;

        if output_gradient_tensor.data_type != token_tensor.data_type {
            return Err(Error::FrontendTensorDataTypeMismatch {
                tensor_id: inputs.output_gradient,
                operation: "moe grouped matmul backward output gradient".into(),
                expected: token_tensor.data_type,
                actual: output_gradient_tensor.data_type,
            });
        }
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
        self.validate_tensor_data_type(
            inputs.first_token_offset,
            DataType::I32,
            "moe grouped matmul backward first token offset",
        )?;

        let token_dims = token_tensor.shape.dimensions();
        let output_gradient_dims = output_gradient_tensor.shape.dimensions();
        if output_gradient_dims[1] != token_dims[1] {
            return Err(Error::FrontendTensorDimensionsMismatch {
                tensor_id: inputs.output_gradient,
                operation: "moe grouped matmul backward output gradient shape".into(),
                expected: vec![
                    output_gradient_dims[0],
                    token_dims[1],
                    output_gradient_dims[2],
                ],
                actual: output_gradient_dims.to_vec(),
            });
        }

        Shape::contiguous([
            first_token_offset_tensor.shape.dimensions()[0],
            token_dims[2],
            output_gradient_dims[2],
        ])
    }
}
