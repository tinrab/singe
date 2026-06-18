use crate::{
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::{BlockScaleQuantizeOutputs, Graph},
        operation::{
            BlockScaleDequantizeConfig, BlockScaleQuantizeConfig, MatmulConfig, MatmulFp8Config,
            Operation, PointwiseOperation, ReductionOperation,
        },
    },
    math::NanPropagation,
    pointwise::PointwiseMode,
    reduction::ReduceTensorOperator,
    tensor::{Shape, TensorId, TensorSpec},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8Inputs {
    pub a: TensorId,
    pub b: TensorId,
    pub descale_a: TensorId,
    pub descale_b: TensorId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8Tensors {
    pub inputs: MatmulFp8Inputs,
    pub output: TensorId,
}

impl MatmulFp8Tensors {
    pub fn new(inputs: MatmulFp8Inputs, output: TensorId) -> Self {
        Self { inputs, output }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8QuantizeTensors {
    pub inputs: MatmulFp8Inputs,
    pub scale_output: TensorId,
    pub output: TensorId,
    pub absolute_max_output: TensorId,
}

impl MatmulFp8QuantizeTensors {
    pub fn new(
        inputs: MatmulFp8Inputs,
        scale_output: TensorId,
        output: TensorId,
        absolute_max_output: TensorId,
    ) -> Self {
        Self {
            inputs,
            scale_output,
            output,
            absolute_max_output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8QuantizeOutputs {
    pub output: TensorId,
    pub absolute_max_output: TensorId,
}

impl MatmulFp8QuantizeOutputs {
    pub fn new(output: TensorId, absolute_max_output: TensorId) -> Self {
        Self {
            output,
            absolute_max_output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatmulFp8OperationTensors {
    pub inputs: MatmulFp8Inputs,
    pub scale_output: TensorId,
    pub output: TensorId,
    pub absolute_max_output: TensorId,
}

impl MatmulFp8OperationTensors {
    pub fn new(
        inputs: MatmulFp8Inputs,
        scale_output: TensorId,
        output: TensorId,
        absolute_max_output: TensorId,
    ) -> Self {
        Self {
            inputs,
            scale_output,
            output,
            absolute_max_output,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockScaleMatmulInputs {
    pub a: TensorId,
    pub b: TensorId,
    pub scale_a: TensorId,
    pub scale_b: TensorId,
}

impl Graph {
    /// Adds a matmul operation with an explicit output tensor.
    ///
    /// The last two dimensions of `a` and `b` are interpreted as matrix
    /// dimensions and earlier dimensions as batch dimensions.
    pub fn matmul(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        compute_type: DataType,
    ) -> Result<()> {
        self.matmul_with_config(a, b, c, MatmulConfig::new(compute_type))
    }

    /// Adds a matmul operation with explicit frontend matmul attributes.
    pub fn matmul_with_config(
        &mut self,
        a: TensorId,
        b: TensorId,
        c: TensorId,
        config: MatmulConfig,
    ) -> Result<()> {
        let expected_output = self.infer_matmul_output_shape(a, b, "matmul shapes")?;
        self.validate_tensor_dimensions(c, expected_output.dimensions(), "matmul output shape")?;
        self.operations.push(Operation::Matmul { a, b, c, config });
        Ok(())
    }

    /// Adds a matmul operation and creates the output tensor.
    pub fn matmul_infer_with_config(
        &mut self,
        a: TensorId,
        b: TensorId,
        config: MatmulConfig,
    ) -> Result<TensorId> {
        let output_shape = self.infer_matmul_output_shape(a, b, "matmul shapes")?;
        let output_data_type = self.effective_intermediate_data_type(config.compute_type());
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.matmul_with_config(a, b, output, config)?;
        Ok(output)
    }

    /// Adds an FP8 matmul followed by descale pointwise operations.
    ///
    /// FP8 matmul support in cuDNN frontend uses explicit scale/descale tensors
    /// in supported fusion patterns.
    pub fn matmul_fp8(&mut self, tensors: MatmulFp8Tensors, compute_type: DataType) -> Result<()> {
        self.matmul_fp8_with_config(tensors, MatmulConfig::new(compute_type))
    }

    /// Adds an FP8 matmul with explicit frontend matmul attributes.
    pub fn matmul_fp8_with_config(
        &mut self,
        tensors: MatmulFp8Tensors,
        config: MatmulConfig,
    ) -> Result<()> {
        self.matmul_fp8_with_config_inner(tensors, config, true)
    }

    fn matmul_fp8_with_config_inner(
        &mut self,
        tensors: MatmulFp8Tensors,
        config: MatmulConfig,
        validate_output_data_type: bool,
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
        let compute_type = config.compute_type();
        let matmul_shape = self.infer_matmul_output_shape(a, b, "matmul fp8 shapes")?;
        if validate_output_data_type {
            self.validate_tensor_data_type(
                output,
                self.effective_io_data_type(compute_type),
                "matmul fp8 output",
            )?;
        }
        self.validate_tensor_dimensions(output, matmul_shape.dimensions(), "matmul fp8 output")?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let matmul_output =
                self.tensor(TensorSpec::new(compute_type, matmul_shape).virtual_tensor());
            self.matmul_with_config(a, b, matmul_output, config)?;
            let scaled_a = self.tensor(
                TensorSpec::new(
                    compute_type,
                    self.tensor_config(matmul_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: matmul_output,
                rhs: descale_a,
                output: scaled_a,
                compute_type,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            let scaled_b = self.tensor(
                TensorSpec::new(compute_type, self.tensor_config(scaled_a)?.shape.clone())
                    .virtual_tensor(),
            );
            let output_tensor = self.tensor_config(output)?.clone();
            let use_direct_output = output_tensor.shape.dimensions()
                == self.tensor_config(scaled_b)?.shape.dimensions()
                && output_tensor.shape.strides() == self.tensor_config(scaled_b)?.shape.strides();
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: scaled_a,
                rhs: descale_b,
                output: if use_direct_output { output } else { scaled_b },
                compute_type,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            if !use_direct_output {
                self.reshape(scaled_b, output)?;
            }
            Ok(())
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    /// Adds an FP8 matmul and creates the output tensor.
    pub fn matmul_fp8_infer(
        &mut self,
        inputs: MatmulFp8Inputs,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let MatmulFp8Inputs { a, b, .. } = inputs;
        let output_shape = self.infer_matmul_output_shape(a, b, "matmul fp8 shapes")?;
        let output_data_type = self.effective_io_data_type(compute_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.matmul_fp8_with_config(
            MatmulFp8Tensors { inputs, output },
            MatmulConfig::new(compute_type),
        )?;
        Ok(output)
    }

    /// Adds an FP8 matmul, descales the inputs, and quantizes the output.
    pub fn matmul_fp8_quantize(
        &mut self,
        tensors: MatmulFp8QuantizeTensors,
        compute_type: DataType,
    ) -> Result<()> {
        self.matmul_fp8_quantize_with_config(tensors, MatmulConfig::new(compute_type))
    }

    pub fn matmul_fp8_quantize_with_config(
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
            output,
            absolute_max_output,
        } = tensors;
        let compute_type = config.compute_type();
        let pre_scale_shape = self.infer_matmul_output_shape(a, b, "matmul fp8 shapes")?;
        let output_data_type = self.tensor_config(output)?.data_type;
        let supported_output_data_types = [
            DataType::F32,
            DataType::F16,
            DataType::BF16,
            DataType::F8E4M3,
            DataType::F8E5M2,
        ];
        if !supported_output_data_types.contains(&output_data_type) {
            return Err(Error::FrontendTensorDataTypeUnsupported {
                tensor_id: output,
                operation: "matmul fp8 quantize output".into(),
                supported: supported_output_data_types.to_vec(),
                actual: output_data_type,
            });
        }
        self.validate_tensor_dimensions(
            output,
            pre_scale_shape.dimensions(),
            "matmul fp8 quantize output",
        )?;
        let absolute_max_shape = Shape::contiguous(vec![1; pre_scale_shape.dimensions().len()])?;
        self.validate_tensor_data_type(
            absolute_max_output,
            compute_type,
            "matmul fp8 quantize absolute_max output",
        )?;
        self.validate_tensor_dimensions(
            absolute_max_output,
            absolute_max_shape.dimensions(),
            "matmul fp8 quantize absolute_max output",
        )?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let pre_scale =
                self.tensor(TensorSpec::new(compute_type, pre_scale_shape).virtual_tensor());
            self.matmul_fp8_with_config_inner(
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
                false,
            )?;
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: pre_scale,
                rhs: scale_output,
                output,
                compute_type,
                nan_propagation: NanPropagation::NotPropagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::AbsoluteMax,
                input: pre_scale,
                output: absolute_max_output,
                compute_type,
                is_deterministic: false,
            });
            Ok(())
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    pub fn matmul_fp8_op(
        &mut self,
        tensors: MatmulFp8OperationTensors,
        config: MatmulFp8Config,
    ) -> Result<()> {
        let MatmulFp8OperationTensors {
            inputs:
                MatmulFp8Inputs {
                    a,
                    b,
                    descale_a,
                    descale_b,
                },
            scale_output: scale_c,
            output: c,
            absolute_max_output: absolute_max_c,
        } = tensors;
        let compute_type = config.compute_type();
        let output_data_type = self.effective_io_data_type(compute_type);
        let matmul_shape = self.infer_matmul_output_shape(a, b, "matmul fp8 shapes")?;
        self.validate_tensor_data_type(c, output_data_type, "matmul fp8 output")?;
        self.validate_tensor_dimensions(c, matmul_shape.dimensions(), "matmul fp8 output")?;
        let absolute_max_shape = Shape::contiguous(vec![1; matmul_shape.dimensions().len()])?;
        self.validate_tensor_data_type(
            absolute_max_c,
            compute_type,
            "matmul fp8 absolute_max output",
        )?;
        self.validate_tensor_dimensions(
            absolute_max_c,
            absolute_max_shape.dimensions(),
            "matmul fp8 absolute_max output",
        )?;
        let checkpoint = self.mutation_checkpoint();
        let result = (|| {
            let matmul_output =
                self.tensor(TensorSpec::new(compute_type, matmul_shape).virtual_tensor());
            let scaled_a = self.tensor(
                TensorSpec::new(
                    compute_type,
                    self.tensor_config(matmul_output)?.shape.clone(),
                )
                .virtual_tensor(),
            );
            let mut matmul_config = MatmulConfig::new(compute_type);
            if let Some(m_override) = config.m_override() {
                matmul_config = matmul_config.with_m_override(m_override);
            }
            if let Some(k_override) = config.k_override() {
                matmul_config = matmul_config.with_k_override(k_override);
            }
            self.matmul_with_config(a, b, matmul_output, matmul_config)?;
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: matmul_output,
                rhs: descale_a,
                output: scaled_a,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            let scaled_b = self.tensor(
                TensorSpec::new(compute_type, self.tensor_config(scaled_a)?.shape.clone())
                    .virtual_tensor(),
            );
            self.pointwise(PointwiseOperation::Binary {
                mode: PointwiseMode::Mul,
                lhs: scaled_a,
                rhs: descale_b,
                output: scaled_b,
                compute_type,
                nan_propagation: NanPropagation::Propagate,
                alpha1: 1.0,
                alpha2: 1.0,
            });
            let output_tensor = self.tensor_config(c)?.clone();
            if output_tensor.shape.dimensions() == self.tensor_config(scaled_b)?.shape.dimensions()
                && output_tensor.shape.strides() == self.tensor_config(scaled_b)?.shape.strides()
            {
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Mul,
                    lhs: scaled_b,
                    rhs: scale_c,
                    output: c,
                    compute_type,
                    nan_propagation: NanPropagation::Propagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
            } else {
                let scaled_c = self.tensor(
                    TensorSpec::new(compute_type, self.tensor_config(scaled_b)?.shape.clone())
                        .virtual_tensor(),
                );
                self.pointwise(PointwiseOperation::Binary {
                    mode: PointwiseMode::Mul,
                    lhs: scaled_b,
                    rhs: scale_c,
                    output: scaled_c,
                    compute_type,
                    nan_propagation: NanPropagation::Propagate,
                    alpha1: 1.0,
                    alpha2: 1.0,
                });
                self.reshape(scaled_c, c)?;
            }
            self.reduction(ReductionOperation::Reduce {
                op: ReduceTensorOperator::AbsoluteMax,
                input: scaled_b,
                output: absolute_max_c,
                compute_type,
                is_deterministic: false,
            });
            Ok(())
        })();
        if result.is_err() {
            self.rollback_to(checkpoint);
        }
        result
    }

    pub fn matmul_fp8_quantize_infer(
        &mut self,
        inputs: MatmulFp8Inputs,
        scale_output: TensorId,
        compute_type: DataType,
    ) -> Result<MatmulFp8QuantizeOutputs> {
        let MatmulFp8Inputs { a, b, .. } = inputs;
        let output_shape = self.infer_matmul_output_shape(a, b, "matmul fp8 shapes")?;
        let output_data_type = self.effective_io_data_type(compute_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape.clone()));
        let absolute_max_output = self.tensor(TensorSpec::new(
            compute_type,
            Shape::contiguous(vec![1; output_shape.dimensions().len()])?
                .with_strides(vec![1; output_shape.strides().len()])?,
        ));
        self.matmul_fp8_quantize(
            MatmulFp8QuantizeTensors {
                inputs,
                scale_output,
                output,
                absolute_max_output,
            },
            compute_type,
        )?;
        Ok(MatmulFp8QuantizeOutputs::new(output, absolute_max_output))
    }

    pub fn matmul_block_scale_infer(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let checkpoint = self.mutation_checkpoint();
        match self.matmul_block_scale_infer_inner(inputs, dequantize_a, dequantize_b, compute_type)
        {
            Ok(output) => Ok(output),
            Err(error) => {
                self.rollback_to(checkpoint);
                Err(error)
            }
        }
    }

    fn matmul_block_scale_infer_inner(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        compute_type: DataType,
    ) -> Result<TensorId> {
        let BlockScaleMatmulInputs {
            a,
            b,
            scale_a,
            scale_b,
        } = inputs;
        let a_dequant = self.block_scale_dequantize_infer(a, scale_a, dequantize_a)?;
        let b_dequant = self.block_scale_dequantize_infer(b, scale_b, dequantize_b)?;
        let output_shape =
            self.infer_matmul_output_shape(a_dequant, b_dequant, "block scale matmul shapes")?;
        let output_data_type = self.effective_io_data_type(compute_type);
        let output = self.tensor(TensorSpec::new(output_data_type, output_shape));
        self.matmul(a_dequant, b_dequant, output, compute_type)?;
        Ok(output)
    }

    pub fn matmul_block_scale_quantize_infer(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        quantize_output: BlockScaleQuantizeConfig,
        compute_type: DataType,
    ) -> Result<BlockScaleQuantizeOutputs> {
        let checkpoint = self.mutation_checkpoint();
        match self.matmul_block_scale_quantize_infer_inner(
            inputs,
            dequantize_a,
            dequantize_b,
            quantize_output,
            compute_type,
        ) {
            Ok(outputs) => Ok(outputs),
            Err(error) => {
                self.rollback_to(checkpoint);
                Err(error)
            }
        }
    }

    fn matmul_block_scale_quantize_infer_inner(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        dequantize_a: BlockScaleDequantizeConfig,
        dequantize_b: BlockScaleDequantizeConfig,
        quantize_output: BlockScaleQuantizeConfig,
        compute_type: DataType,
    ) -> Result<BlockScaleQuantizeOutputs> {
        let output =
            self.matmul_block_scale_infer_inner(inputs, dequantize_a, dequantize_b, compute_type)?;
        self.block_scale_quantize_infer(output, quantize_output)
    }

    pub fn matmul_nvfp4_infer(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        block_size: i64,
    ) -> Result<TensorId> {
        self.matmul_block_scale_infer(
            inputs,
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, block_size as i32]),
            BlockScaleDequantizeConfig::new(DataType::F32, vec![block_size as i32, 1]),
            DataType::F32,
        )
    }

    pub fn matmul_nvfp4_quantize_infer(
        &mut self,
        inputs: BlockScaleMatmulInputs,
        block_size: i64,
    ) -> Result<BlockScaleQuantizeOutputs> {
        self.matmul_block_scale_quantize_infer(
            inputs,
            BlockScaleDequantizeConfig::new(DataType::F32, vec![1, block_size as i32]),
            BlockScaleDequantizeConfig::new(DataType::F32, vec![block_size as i32, 1]),
            BlockScaleQuantizeConfig::new(DataType::F32, block_size).with_axis(2),
            DataType::F32,
        )
    }

    fn infer_matmul_output_shape(
        &self,
        a: TensorId,
        b: TensorId,
        error_name: &str,
    ) -> Result<Shape> {
        let a_tensor = self.tensor_config(a)?.clone();
        let b_tensor = self.tensor_config(b)?.clone();
        let a_dims = a_tensor.shape.dimensions();
        let b_dims = b_tensor.shape.dimensions();
        let shape_mismatch = || Error::FrontendMatmulShapeMismatch {
            lhs_id: a,
            rhs_id: b,
            operation: error_name.into(),
            lhs: a_dims.to_vec(),
            rhs: b_dims.to_vec(),
        };
        match (a_dims.len(), b_dims.len()) {
            (3, 3) if a_dims[0] == b_dims[0] => {
                match (a_dims[2] == b_dims[1], a_dims[2] == b_dims[2]) {
                    (true, _) => Shape::contiguous([a_dims[0], a_dims[1], b_dims[2]]),
                    (_, true) => Shape::contiguous([a_dims[0], a_dims[1], b_dims[1]]),
                    _ => Err(shape_mismatch()),
                }
            }
            (4, 4) if a_dims[0] == b_dims[0] && a_dims[1] == b_dims[1] => {
                match (a_dims[3] == b_dims[2], a_dims[3] == b_dims[3]) {
                    (true, _) => Shape::contiguous([a_dims[0], a_dims[1], a_dims[2], b_dims[3]]),
                    (_, true) => Shape::contiguous([a_dims[0], a_dims[1], a_dims[2], b_dims[2]]),
                    _ => Err(shape_mismatch()),
                }
            }
            _ => Err(shape_mismatch()),
        }
    }
}
