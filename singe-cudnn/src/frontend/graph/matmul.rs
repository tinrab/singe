use crate::{
    backend::{
        pointwise::PointwiseMode,
        reduction::ReduceTensorOperator,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::attention_heads_compatible,
        error::Error as FrontendError,
        graph::{BlockScaleDequantizeInputs, BlockScaleQuantizeOutputs, Graph, TensorOutputConfig},
        operation::{
            BlockScaleDequantizeConfig, BlockScaleQuantizeConfig, MatmulConfig, MatmulOperation,
            Operation, PointwiseOperation, ReductionConfig,
        },
    },
    math::NanPropagation,
};

/// FP8 matmul operands and dequantization multipliers applied to their product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8Inputs {
    /// Left matrix operand tensor identity.
    pub a: TensorId,
    /// Right matrix operand tensor identity.
    pub b: TensorId,
    /// Dequantization multiplier for the left operand, broadcast against the product.
    pub descale_a: TensorId,
    /// Dequantization multiplier for the right operand, broadcast against the product.
    pub descale_b: TensorId,
}

/// FP8 matmul input group and an explicitly declared dequantized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8Tensors {
    /// Matrix operands and their dequantization multipliers.
    pub inputs: MatmulFp8Inputs,
    /// Result tensor identity or its independent placement and encoding policy.
    pub output: TensorId,
}

impl MatmulFp8Tensors {
    /// Records FP8 matmul inputs and an explicit dequantized output.
    #[must_use]
    pub fn new(inputs: MatmulFp8Inputs, output: TensorId) -> Self {
        Self { inputs, output }
    }
}

/// FP8 matmul inputs, output quantization scale, and explicit result and absolute-maximum destinations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8QuantizeTensors {
    /// Matrix operands and their dequantization multipliers.
    pub inputs: MatmulFp8Inputs,
    /// Output quantization multiplier tensor identity.
    pub scale_output: TensorId,
    /// Quantized result and absolute-maximum destinations.
    pub outputs: MatmulFp8QuantizeOutputs,
}

impl MatmulFp8QuantizeTensors {
    /// Records FP8 inputs, the output quantization multiplier, and explicit result and absolute-maximum outputs.
    #[must_use]
    pub fn new(
        inputs: MatmulFp8Inputs,
        scale_output: TensorId,
        outputs: MatmulFp8QuantizeOutputs,
    ) -> Self {
        Self {
            inputs,
            scale_output,
            outputs,
        }
    }
}

/// Quantized matmul and Amax results shared by explicit construction and inference.
/// With `TensorOutputConfig`, the fields select their independent inference configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8QuantizeOutputs<T = TensorId> {
    /// Result tensor identity or its independent placement and encoding policy.
    pub output: T,
    /// Absolute maximum of the dequantized product before output scaling, or its inference policy.
    pub absolute_max_output: T,
}

impl<T> MatmulFp8QuantizeOutputs<T> {
    /// Records result and absolute-maximum identities or their independent inference policies.
    pub fn new(output: T, absolute_max_output: T) -> Self {
        Self {
            output,
            absolute_max_output,
        }
    }
}

/// Block-quantized matrix operands with their independent block-scale tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleMatmulInputs {
    /// Left matrix operand tensor identity.
    pub a: TensorId,
    /// Right matrix operand tensor identity.
    pub b: TensorId,
    /// Block-scale tensor for the left matrix operand.
    pub scale_a: TensorId,
    /// Block-scale tensor for the right matrix operand.
    pub scale_b: TensorId,
}

impl Graph {
    /// Adds a matmul operation with an explicit output tensor.
    ///
    /// The last two dimensions of `a` and `b` are interpreted as matrix
    /// dimensions and earlier dimensions as batch dimensions.
    ///
    /// # Errors
    /// Returns an error for unknown operands or dimension-override tensors,
    /// incompatible ranks or a result shape inconsistent with the requested overrides.
    pub fn matmul_into(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        config: MatmulConfig,
    ) -> Result<()> {
        let operation = MatmulOperation { a, b, c, config };
        operation.validate_graph(self)?;
        self.operations.push(Operation::Matmul(operation));
        Ok(())
    }

    /// Adds a matmul operation and creates the output tensor.
    /// Output placement selects I/O or intermediate encoding independently of matmul precision.
    /// An explicit result encoding takes precedence over either policy.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or dimension-override tensors, incompatible ranks or result geometry, or an inferred contiguous layout whose strides cannot be represented.
    pub fn matmul(
        &mut self,
        a: TensorId,
        b: TensorId,
        config: MatmulConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let output_shape = self.infer_matmul_output_shape(a, b, &config, "matmul shapes")?;
        let output = output.into_spec(
            self.config.data_type_policy,
            self.matmul_default_result_type(&config),
            output_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "C");
        if let Err(error) = self.matmul_into(a, b, output, config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds an FP8 matmul followed by descale pointwise operations.
    ///
    /// FP8 matmul support in cuDNN frontend uses explicit scale/descale tensors
    /// in supported fusion patterns.
    /// Intermediate storage and pointwise precision follow their graph policies.
    /// Epilogue math inherits the current graph compute policy during preparation, with an `F32` fallback.
    /// The matmul configuration's compute override applies to the native multiplication.
    /// `FastFloatForFp8` selects native matmul computation and defaults to `F32` storage.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or scales, incompatible matrix or result geometry, unrepresentable inferred strides, or invalid pointwise broadcast or reduction shapes in the dequantization composition.
    pub fn matmul_fp8_into(
        &mut self,
        tensors: MatmulFp8Tensors,
        config: MatmulConfig,
    ) -> Result<()> {
        let MatmulFp8Tensors {
            inputs:
                MatmulFp8Inputs {
                    a,
                    b,
                    descale_a,
                    descale_b,
                },
            output,
        } = tensors;
        let intermediate_type =
            self.effective_intermediate_data_type(self.matmul_default_result_type(&config));
        self.tensor_spec(descale_a)?;
        self.tensor_spec(descale_b)?;
        let matmul_shape = self
            .validate_matmul_output_shape(a, b, output, &config, true, "matmul fp8 output")?
            .clone();
        let inferred_product =
            self.infer_matmul_output_shape(a, b, &config, "matmul fp8 shapes")?;
        let mut product_dimensions = matmul_shape.dimensions().to_vec();
        if product_dimensions.len() == 4 {
            product_dimensions[1] = inferred_product.dimensions()[1];
        }
        let product_shape = Shape::contiguous(product_dimensions)?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let matmul_output = self
                .tensor(TensorSpec::new(intermediate_type, product_shape.clone()).virtual_tensor());
            self.matmul_into(a, b, matmul_output, config)?;
            let matmul_output = if product_shape.dimensions() == matmul_shape.dimensions() {
                matmul_output
            } else {
                let reduced = self.tensor(
                    TensorSpec::new(intermediate_type, matmul_shape.clone()).virtual_tensor(),
                );
                self.reduction_into(
                    matmul_output,
                    reduced,
                    &ReductionConfig::new(ReduceTensorOperator::Add),
                )?;
                reduced
            };
            let scaled_a = self.tensor(
                TensorSpec::new(
                    intermediate_type,
                    self.tensor_spec(matmul_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Mul,
                lhs: matmul_output,
                rhs: descale_a,
                output: scaled_a,
                compute_type: None,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Mul,
                lhs: scaled_a,
                rhs: descale_b,
                output,
                compute_type: None,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
            Ok(())
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Adds an FP8 matmul and creates the output tensor.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or scales, incompatible matrix or result geometry, unrepresentable inferred strides, or invalid pointwise broadcast or reduction shapes in the dequantization composition.
    pub fn matmul_fp8(
        &mut self,
        inputs: MatmulFp8Inputs,
        config: MatmulConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let MatmulFp8Inputs { a, b, .. } = inputs;
        let output_shape = self.infer_matmul_output_shape(a, b, &config, "matmul fp8 shapes")?;
        let output = output.into_spec(
            self.config.data_type_policy,
            self.matmul_default_result_type(&config),
            output_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "C");
        if let Err(error) = self.matmul_fp8_into(MatmulFp8Tensors::new(inputs, output), config) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(output)
    }

    /// Adds an FP8 matmul, descales the inputs, and quantizes the output.
    /// The caller-declared Amax tensor controls its reduction geometry and encoding.
    /// Explicit outputs use the same named group returned by inference.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or scales, incompatible matrix or result geometry, unrepresentable inferred strides, or invalid pointwise broadcast or reduction shapes in the dequantization composition. Also returns an error if output scaling cannot broadcast or the absolute-maximum destination has invalid reduction geometry.
    pub fn matmul_fp8_quantize_into(
        &mut self,
        tensors: MatmulFp8QuantizeTensors,
        config: MatmulConfig,
    ) -> Result<()> {
        let MatmulFp8QuantizeTensors {
            inputs:
                MatmulFp8Inputs {
                    a,
                    b,
                    descale_a,
                    descale_b,
                },
            scale_output,
            outputs:
                MatmulFp8QuantizeOutputs {
                    output,
                    absolute_max_output,
                },
        } = tensors;
        let intermediate_type =
            self.effective_intermediate_data_type(self.matmul_default_result_type(&config));
        let pre_scale_shape = self
            .validate_matmul_output_shape(
                a,
                b,
                output,
                &config,
                true,
                "matmul fp8 quantize output",
            )?
            .clone();
        self.tensor_spec(absolute_max_output)?;
        self.tensor_spec(scale_output)?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let pre_scale =
                self.tensor(TensorSpec::new(intermediate_type, pre_scale_shape).virtual_tensor());
            self.matmul_fp8_into(
                MatmulFp8Tensors::new(
                    MatmulFp8Inputs {
                        a,
                        b,
                        descale_a,
                        descale_b,
                    },
                    pre_scale,
                ),
                config,
            )?;
            self.pointwise_into(PointwiseOperation::Binary {
                name: None,
                mode: PointwiseMode::Mul,
                lhs: pre_scale,
                rhs: scale_output,
                output,
                compute_type: None,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            })?;
            self.reduction_into(
                pre_scale,
                absolute_max_output,
                &ReductionConfig::new(ReduceTensorOperator::AbsoluteMax),
            )?;
            Ok(())
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Creates quantized output and scalar Amax tensors with independent placement and encoding.
    /// Result encoding follows the requested placement; Amax falls back to FP32 when its policy is unset.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown operands or scales, incompatible matrix or result geometry, unrepresentable inferred strides, or invalid pointwise broadcast or reduction shapes in the dequantization composition. Also returns an error if output scaling cannot broadcast or the absolute-maximum destination has invalid reduction geometry.
    pub fn matmul_fp8_quantize(
        &mut self,
        inputs: MatmulFp8Inputs,
        scale_output: TensorId,
        config: MatmulConfig,
        outputs: MatmulFp8QuantizeOutputs<TensorOutputConfig>,
    ) -> Result<MatmulFp8QuantizeOutputs> {
        let MatmulFp8Inputs { a, b, .. } = inputs;
        let output_shape = self.infer_matmul_output_shape(a, b, &config, "matmul fp8 shapes")?;
        let absolute_max_shape = output_shape.unit_like()?;
        let output = outputs.output.into_spec(
            self.config.data_type_policy,
            self.matmul_default_result_type(&config),
            output_shape,
        );
        let absolute_max_output = outputs.absolute_max_output.into_spec(
            self.config.data_type_policy,
            DataType::F32,
            absolute_max_shape,
        );
        let checkpoint = self.mutation_checkpoint();
        let output = self.inferred_tensor(output, config.name(), "C");
        let absolute_max_output =
            self.inferred_tensor(absolute_max_output, config.name(), "Amax_C");
        let outputs = MatmulFp8QuantizeOutputs::new(output, absolute_max_output);
        if let Err(error) = self.matmul_fp8_quantize_into(
            MatmulFp8QuantizeTensors::new(inputs, scale_output, outputs),
            config,
        ) {
            self.rollback_to(checkpoint);
            return Err(error);
        }
        Ok(outputs)
    }

    /// Dequantizes both operands and infers the matrix product with independently selected output encoding and placement.
    /// A rejected composition removes its newly created tensors and operations.
    ///
    /// # Errors
    ///
    /// Returns an error if either operand or block-scale tensor is unknown, block-scale dequantization shapes or configurations are invalid, or matmul inference or validation fails.
    pub fn matmul_block_scale(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        config: MatmulConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        let result =
            self.matmul_block_scale_infer_inner(inputs, dequantize_a, dequantize_b, config, output);
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    fn matmul_block_scale_infer_inner(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        config: MatmulConfig,
        output: TensorOutputConfig,
    ) -> Result<TensorId> {
        let BlockScaleMatmulInputs {
            a,
            b,
            scale_a,
            scale_b,
        } = inputs;
        let a_dequant =
            self.block_scale_dequantize(BlockScaleDequantizeInputs::new(a, scale_a), dequantize_a)?;
        let b_dequant =
            self.block_scale_dequantize(BlockScaleDequantizeInputs::new(b, scale_b), dequantize_b)?;
        self.matmul(a_dequant, b_dequant, config, output)
    }

    /// Dequantizes the operands, multiplies them and block-quantizes the product.
    /// `intermediate_data_type` overrides the virtual product's encoding before quantization.
    /// An omitted intermediate encoding follows graph intermediate policy and then matmul precision.
    /// Quantized values and block scales select their placement and encoding independently.
    /// A rejected composition removes every newly created intermediate and operation.
    ///
    /// # Errors
    ///
    /// Returns an error if block-scale dequantization or matmul inference fails, the output quantization configuration or shapes are invalid, or required quantized-output or scale encodings are unspecified.
    pub fn matmul_block_scale_quantize(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        quantize_output: BlockScaleQuantizeConfig,
        outputs: BlockScaleQuantizeOutputs<TensorOutputConfig>,
        config: MatmulConfig,
        intermediate_data_type: Option<DataType>,
    ) -> Result<BlockScaleQuantizeOutputs> {
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let output = self.matmul_block_scale_infer_inner(
                inputs,
                dequantize_a,
                dequantize_b,
                config,
                TensorOutputConfig::new().with_data_type(intermediate_data_type),
            )?;
            self.block_scale_quantize(output, outputs, quantize_output)
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    fn matmul_default_result_type(&self, config: &MatmulConfig) -> DataType {
        match self.resolve_compute_data_type(config.compute_type()) {
            DataType::FastFloatForFp8 => DataType::F32,
            data_type => data_type,
        }
    }

    pub(crate) fn validate_matmul_output_shape(
        &self,
        a: TensorId,
        b: TensorId,
        output: TensorId,
        config: &MatmulConfig,
        allow_head_reduction: bool,
        error_name: &'static str,
    ) -> Result<&Shape> {
        let output_tensor = self.tensor_spec(output)?;
        let actual = &output_tensor.shape;
        let mut dimensions =
            self.infer_matmul_output_dimensions(a, b, Some(output_tensor), config, error_name)?;
        if actual.rank() == dimensions.len() {
            if allow_head_reduction && dimensions.len() == 4 {
                let a_shape = self
                    .tensor_spec(a)?
                    .shape_with_scalar_rank(dimensions.len())?;
                let b_shape = self
                    .tensor_spec(b)?
                    .shape_with_scalar_rank(dimensions.len())?;
                let a_heads = a_shape.dimensions()[1];
                let b_heads = b_shape.dimensions()[1];
                let output_heads = actual.dimensions()[1];
                if a_heads == b_heads && attention_heads_compatible(a_heads, output_heads) {
                    dimensions[1] = output_heads;
                }
            }
            let matrix_axis = dimensions.len() - 2;
            if config.m_override().is_some() {
                dimensions[matrix_axis] = actual.dimensions()[matrix_axis];
            }
            if config.n_override().is_some() {
                dimensions[matrix_axis + 1] = actual.dimensions()[matrix_axis + 1];
            }
        }
        self.validate_tensor_dimensions(output, &dimensions, error_name)?;
        Ok(actual)
    }

    fn infer_matmul_output_shape(
        &self,
        a: TensorId,
        b: TensorId,
        config: &MatmulConfig,
        error_name: &'static str,
    ) -> Result<Shape> {
        Shape::contiguous(self.infer_matmul_output_dimensions(a, b, None, config, error_name)?)
    }

    fn infer_matmul_output_dimensions(
        &self,
        a: TensorId,
        b: TensorId,
        output: Option<&TensorSpec>,
        config: &MatmulConfig,
        error_name: &'static str,
    ) -> Result<Vec<i64>> {
        let a_tensor = self.tensor_spec(a)?;
        let b_tensor = self.tensor_spec(b)?;
        let mut rank = [a_tensor, b_tensor]
            .into_iter()
            .filter(|tensor| !tensor.is_by_value())
            .map(|tensor| tensor.shape.rank())
            .chain(
                output
                    .filter(|tensor| !tensor.is_by_value())
                    .map(|tensor| tensor.shape.rank()),
            )
            .max()
            .unwrap_or(2);
        for tensor in [
            config.m_override(),
            config.n_override(),
            config.k_override(),
        ]
        .into_iter()
        .flatten()
        {
            let tensor = self.tensor_spec(tensor)?;
            if !tensor.is_by_value() {
                rank = rank.max(tensor.shape.rank());
            }
        }
        let a_shape = a_tensor.shape_with_scalar_rank(rank)?;
        let b_shape = b_tensor.shape_with_scalar_rank(rank)?;
        let a_dims = a_shape.dimensions();
        let b_dims = b_shape.dimensions();
        let shape_mismatch = || {
            FrontendError::MatmulShapeMismatch {
                lhs_id: a,
                rhs_id: b,
                operation: error_name,
                lhs: a_dims.to_vec(),
                rhs: b_dims.to_vec(),
            }
            .into()
        };
        if a_dims.len() < 2 || a_dims.len() != b_dims.len() {
            return Err(shape_mismatch());
        }
        // Match C++ inference; native engine support decides which batch ratios can execute.
        let matrix_axis = a_dims.len() - 2;
        let mut dimensions = a_dims[..matrix_axis]
            .iter()
            .zip(&b_dims[..matrix_axis])
            .map(|(&a, &b)| a.max(b))
            .collect::<Vec<_>>();
        dimensions.extend([a_dims[matrix_axis], b_dims[matrix_axis + 1]]);
        Ok(dimensions)
    }
}
