use crate::{
    backend::{
        registry::{
            BatchNormalizationAttributes, Conv2dAttributes, FlattenAttributes,
            LayerNormalizationAttributes, Operation, OperationAttributes, Pool2dAttributes,
            ReductionAttributes, SliceAttributes, StridedAttributes, TransposeAttributes,
        },
        spec::reduction_problem,
    },
    data_type::DataType,
    graph::{
        DimensionExpr,
        ir::{OperationNode, ValueId},
    },
    onnx::{
        error::Result,
        exporter::{
            Exporter,
            proto::{dtype, float, integer, integers, scalar_bytes, string, unsupported},
        },
    },
    scalar::Scalar,
    shape::{
        Shape,
        transform::{normalize_dim_usize, transpose_order},
    },
};

// What exporting one graph operation starts from.
struct OperationExport<'a> {
    value: ValueId,
    node: &'a OperationNode,
    inputs: Vec<String>,
    output: String,
    data_type: DataType,
    operation: Operation,
}

impl Exporter<'_> {
    pub(crate) fn operation(
        &mut self,
        value: ValueId,
        node: &OperationNode,
        inputs: Vec<String>,
        output: String,
    ) -> Result<()> {
        use Operation as Op;
        use OperationAttributes as Attr;
        let data_type = self.source.required_value_data_type(value)?;
        let operation = node
            .opcode()
            .ok_or_else(|| unsupported("node without a canonical operation"))?;
        let attributes = node.attributes().clone();
        let mut op = OperationExport {
            value,
            node,
            inputs,
            output,
            data_type,
            operation,
        };
        match (operation, attributes) {
            (_, Attr::None) if let Some(name) = operation.onnx_operator() => {
                self.matching_floats(&op)?;
                self.node(name, op.inputs, op.output, vec![]);
            }
            (Op::Gelu | Op::GeluExact, Attr::None) => {
                self.matching_floats(&op)?;
                let approximate = if op.operation == Op::Gelu {
                    "tanh"
                } else {
                    "none"
                };
                self.node(
                    "Gelu",
                    op.inputs,
                    op.output,
                    vec![string("approximate", approximate)],
                );
            }
            (Op::Silu, Attr::None) => self.silu(op)?,
            (
                Op::Scale | Op::AddScalar | Op::SubScalar | Op::RSubScalar | Op::RDivScalar,
                Attr::Scalar(value),
            ) => self.scalar_arithmetic(op, value)?,
            (Op::Contiguous, Attr::None) => self.node("Identity", op.inputs, op.output, vec![]),
            (Op::CopyTo, _) => {
                return Err(unsupported(
                    "explicit device-copy semantics are not part of ONNX",
                ));
            }
            (Op::Cast, Attr::Cast(data_type)) => self.node(
                "Cast",
                op.inputs,
                op.output,
                vec![integer("to", i64::from(dtype(data_type)?))],
            ),
            (Op::Transpose, Attr::Transpose(TransposeAttributes { dim0, dim1 })) => {
                let rank = self
                    .source
                    .required_value_shape(op.node.inputs()[0])?
                    .rank();
                let order = transpose_order(rank, dim0, dim1)?;
                if rank == 0 {
                    self.node("Identity", op.inputs, op.output, vec![]);
                } else {
                    self.node(
                        "Transpose",
                        op.inputs,
                        op.output,
                        vec![integers("perm", order)],
                    );
                }
            }
            (Op::Permute, Attr::Permute(order)) => {
                let rank = self
                    .source
                    .required_value_shape(op.node.inputs()[0])?
                    .rank();
                let order = order
                    .into_iter()
                    .map(|axis| if axis < 0 { rank + axis } else { axis })
                    .collect();
                self.node(
                    "Transpose",
                    op.inputs,
                    op.output,
                    vec![integers("perm", order)],
                );
            }
            (Op::Reshape, Attr::Shape(shape)) => self.reshape(op, &shape)?,
            (Op::Flatten, Attr::Flatten(FlattenAttributes { start_dim, end_dim })) => {
                self.flatten(op, start_dim, end_dim)?;
            }
            (Op::Squeeze, Attr::Dimension(dim)) => {
                self.squeeze(node, op.inputs, op.output, dim)?;
            }
            (Op::Unsqueeze, Attr::Dimension(dim)) => {
                op.inputs.push(self.integers(vec![1], &[dim])?);
                self.node("Unsqueeze", op.inputs, op.output, vec![]);
            }
            (Op::Softmax | Op::LogSoftmax, Attr::Dimension(dim)) => self.softmax(op, dim)?,
            (
                Op::Mean | Op::Sum | Op::Max | Op::Min,
                Attr::Reduction(ReductionAttributes {
                    dimensions,
                    keepdim,
                }),
            ) => self.reduction(op, &dimensions, keepdim)?,
            (Op::Take | Op::IndexSelect, Attr::IndexSelect(dim)) => {
                self.node("Gather", op.inputs, op.output, vec![integer("axis", dim)]);
            }
            (Op::Embedding, Attr::None) => {
                self.node("Gather", op.inputs, op.output, vec![integer("axis", 0)]);
            }
            (Op::Concat, Attr::Concat(dim)) => {
                self.node("Concat", op.inputs, op.output, vec![integer("axis", dim)]);
            }
            (
                Op::Conv2d,
                Attr::Conv2d(Conv2dAttributes {
                    stride,
                    padding,
                    dilation,
                    groups,
                    ..
                }),
            ) => {
                self.spatial_floats(&op)?;
                self.node(
                    "Conv",
                    op.inputs,
                    op.output,
                    vec![
                        integers("strides", vec![stride.0, stride.1]),
                        integers("pads", vec![padding.0, padding.2, padding.1, padding.3]),
                        integers("dilations", vec![dilation.0, dilation.1]),
                        integer("group", groups),
                    ],
                );
            }
            (Op::AvgPool2d | Op::MaxPool2d, Attr::Pool2d(pool)) => self.pool2d(op, pool)?,
            (Op::LayerNorm, Attr::LayerNormalization(attributes)) => {
                self.layer_norm(op, attributes)?;
            }
            (
                Op::BatchNorm,
                Attr::BatchNormalization(BatchNormalizationAttributes {
                    has_weight,
                    has_bias,
                    has_running_mean: true,
                    has_running_var: true,
                    training: false,
                    eps,
                    ..
                }),
            ) => self.batch_norm(op, has_weight, has_bias, eps)?,
            (Op::Slice, Attr::Slice(slice)) => self.slice(op, slice)?,
            // ONNX random operators draw from the same distributions with their own seeding, so exported values differ from Singe's Philox streams; the generator input stays unused.
            (Op::Uniform | Op::Normal, Attr::Random(random)) => {
                let (name, first, second) = if op.operation == Op::Uniform {
                    ("RandomUniform", "low", "high")
                } else {
                    ("RandomNormal", "mean", "scale")
                };
                self.node(
                    name,
                    vec![],
                    op.output,
                    vec![
                        integer("dtype", i64::from(dtype(random.data_type)?)),
                        float(first, random.first),
                        float(second, random.second),
                        integers("shape", random.dimensions),
                    ],
                );
            }
            (Op::Bernoulli, Attr::Random(_)) => self.node(
                "Bernoulli",
                vec![op.inputs[0].clone()],
                op.output,
                vec![integer("dtype", i64::from(dtype(DataType::Bool)?))],
            ),
            (Op::Multinomial, Attr::Random(random)) => self.multinomial(op, random.samples)?,
            // Strided operands are flat, so storage positions index them directly.
            (Op::AsStrided, Attr::Strided(strided)) => {
                let positions = strided_positions(&strided)?;
                let indices = self.integers(strided.dimensions, &positions)?;
                op.inputs.push(indices);
                self.node("Gather", op.inputs, op.output, vec![integer("axis", 0)]);
            }
            (Op::StridedScatter, Attr::Strided(strided)) => self.strided_scatter(op, &strided)?,
            (Op::Custom(custom), _) => {
                return Err(unsupported(format!(
                    "custom operation {} has no onnx mapping",
                    custom.name()
                )));
            }
            _ => {
                return Err(unsupported(format!(
                    "canonical operation {operation:?} and its attributes are outside the opset-20 export subset"
                )));
            }
        }
        Ok(())
    }

    // ONNX binary operators do not apply Singe/PyTorch dtype promotion, so mixed operands are refused rather than narrowed.
    fn matching_floats(&self, op: &OperationExport<'_>) -> Result<()> {
        let dtype = op.data_type;
        if !matches!(
            dtype,
            DataType::F16 | DataType::Bf16 | DataType::F32 | DataType::F64
        ) || op.node.inputs().iter().any(|&input| {
            self.source
                .required_value_data_type(input)
                .is_ok_and(|input| input != dtype)
        }) {
            return Err(unsupported(format!(
                "{:?} requires matching floating storage types for export",
                op.operation
            )));
        }
        Ok(())
    }

    fn spatial_floats(&self, op: &OperationExport<'_>) -> Result<()> {
        self.matching_floats(op)?;
        if op.data_type == DataType::Bf16 {
            return Err(unsupported(format!(
                "opset-20 {:?} does not support bfloat16",
                op.operation
            )));
        }
        Ok(())
    }

    fn silu(&mut self, op: OperationExport<'_>) -> Result<()> {
        self.matching_floats(&op)?;
        let OperationExport {
            mut inputs,
            output,
            data_type,
            ..
        } = op;
        let storage_type = data_type;
        let widened = matches!(storage_type, DataType::F16 | DataType::Bf16);
        if widened {
            inputs[0] = self.temporary(
                "Cast",
                inputs.clone(),
                vec![integer("to", i64::from(dtype(DataType::F32)?))],
            );
        }

        let sigmoid = self.temporary("Sigmoid", inputs.clone(), vec![]);
        inputs.push(sigmoid);
        if widened {
            let computed = self.temporary("Mul", inputs, vec![]);
            self.node(
                "Cast",
                vec![computed],
                output,
                vec![integer("to", i64::from(dtype(storage_type)?))],
            );
        } else {
            self.node("Mul", inputs, output, vec![]);
        }
        Ok(())
    }

    fn scalar_arithmetic(&mut self, op: OperationExport<'_>, value: Scalar) -> Result<()> {
        self.matching_floats(&op)?;
        let OperationExport {
            mut inputs,
            output,
            data_type,
            operation,
            ..
        } = op;
        let storage_type = data_type;
        let compute_type = if matches!(storage_type, DataType::F16 | DataType::Bf16) {
            DataType::F32
        } else {
            storage_type
        };
        if compute_type != storage_type {
            inputs[0] = self.temporary(
                "Cast",
                inputs.clone(),
                vec![integer("to", i64::from(dtype(compute_type)?))],
            );
        }

        let scalar = self.scalar(value, compute_type)?;
        if matches!(operation, Operation::RSubScalar | Operation::RDivScalar) {
            inputs.insert(0, scalar);
        } else {
            inputs.push(scalar);
        }

        let name = match operation {
            Operation::Scale => "Mul",
            Operation::AddScalar => "Add",
            Operation::SubScalar | Operation::RSubScalar => "Sub",
            _ => "Div",
        };
        if compute_type == storage_type {
            self.node(name, inputs, output, vec![]);
        } else {
            let computed = self.temporary(name, inputs, vec![]);
            self.node(
                "Cast",
                vec![computed],
                output,
                vec![integer("to", i64::from(dtype(storage_type)?))],
            );
        }
        Ok(())
    }

    fn reshape(&mut self, op: OperationExport<'_>, shape: &Shape) -> Result<()> {
        let OperationExport {
            value,
            mut inputs,
            output,
            ..
        } = op;
        let (dimensions, allow_zero) = self
            .source
            .shape_program()
            .reshapes
            .get(&value)
            .map_or_else(
                || {
                    (
                        shape
                            .dimensions()
                            .iter()
                            .copied()
                            .map(DimensionExpr::constant)
                            .collect(),
                        true,
                    )
                },
                |expression| (expression.dimensions.clone(), expression.allow_zero),
            );
        inputs.push(self.shape_vector(&dimensions)?);
        self.node(
            "Reshape",
            inputs,
            output,
            vec![integer("allowzero", i64::from(allow_zero))],
        );
        Ok(())
    }

    fn flatten(&mut self, op: OperationExport<'_>, start_dim: i64, end_dim: i64) -> Result<()> {
        let OperationExport {
            node,
            mut inputs,
            output,
            ..
        } = op;
        let source = node.inputs()[0];
        let rank = self.source.required_value_shape(source)?.rank();
        let start = normalize_dim_usize(start_dim, rank.max(1))?;
        let end = normalize_dim_usize(end_dim, rank.max(1))?;
        let mut dimensions = Vec::new();
        let mut combined = DimensionExpr::constant(1);
        if rank == 0 {
            dimensions.push(combined.clone());
        }

        for axis in 0..usize::try_from(rank).map_err(|_| unsupported("flatten rank"))? {
            let dim = DimensionExpr::axis(None, source.index(), axis);
            if axis < start || axis > end {
                dimensions.push(dim);
            } else {
                combined = combined * dim;
            }
            if axis == end {
                dimensions.push(combined.clone());
            }
        }

        inputs.push(self.shape_vector(&dimensions)?);
        self.node("Reshape", inputs, output, vec![integer("allowzero", 1)]);
        Ok(())
    }

    fn softmax(&mut self, op: OperationExport<'_>, dim: i64) -> Result<()> {
        self.matching_floats(&op)?;
        let OperationExport {
            value,
            mut inputs,
            output,
            operation,
            ..
        } = op;
        let name = if operation == Operation::Softmax {
            "Softmax"
        } else {
            "LogSoftmax"
        };
        if self.source.required_value_shape(value)?.rank() == 0 {
            // ONNX softmax needs an axis, so a scalar runs as a one-element vector.
            let axes = self.integers(vec![1], &[0])?;
            inputs.push(axes.clone());
            let vector = self.temporary("Unsqueeze", inputs, vec![]);
            let result = self.temporary(name, vec![vector], vec![integer("axis", 0)]);
            self.node("Squeeze", vec![result, axes], output, vec![]);
        } else {
            self.node(name, inputs, output, vec![integer("axis", dim)]);
        }
        Ok(())
    }

    fn reduction(
        &mut self,
        op: OperationExport<'_>,
        dimensions: &[i64],
        keepdim: bool,
    ) -> Result<()> {
        self.matching_floats(&op)?;
        let OperationExport {
            node,
            mut inputs,
            output,
            operation,
            ..
        } = op;
        let shape = self.source.required_value_shape(node.inputs()[0])?;
        let dimensions = reduction_problem(shape, dimensions, keepdim)?
            .dimensions
            .into_iter()
            .map(|axis| i64::try_from(axis).map_err(|_| unsupported("reduction axis")))
            .collect::<Result<Vec<_>>>()?;
        inputs.push(self.integers(
            vec![i64::try_from(dimensions.len()).map_err(|_| unsupported("reduction rank"))?],
            &dimensions,
        )?);
        let name = match operation {
            Operation::Mean => "ReduceMean",
            Operation::Sum => "ReduceSum",
            Operation::Max => "ReduceMax",
            _ => "ReduceMin",
        };
        self.node(
            name,
            inputs,
            output,
            vec![integer("keepdims", i64::from(keepdim))],
        );
        Ok(())
    }

    fn pool2d(&mut self, op: OperationExport<'_>, pool: Pool2dAttributes) -> Result<()> {
        self.spatial_floats(&op)?;
        let Pool2dAttributes {
            kernel_size,
            stride,
            padding,
            dilation,
            ceil_mode,
            count_include_pad,
        } = pool;
        let OperationExport {
            inputs,
            output,
            operation,
            ..
        } = op;
        let mut attrs = vec![
            integers("kernel_shape", vec![kernel_size.0, kernel_size.1]),
            integers("strides", vec![stride.0, stride.1]),
            integers("pads", vec![padding.0, padding.2, padding.1, padding.3]),
            integer("ceil_mode", i64::from(ceil_mode)),
        ];
        if operation == Operation::MaxPool2d {
            attrs.push(integers("dilations", vec![dilation.0, dilation.1]));
        } else {
            if dilation != (1, 1) {
                return Err(unsupported("opset-20 AveragePool dilation"));
            }

            attrs.push(integer(
                "count_include_pad",
                i64::from(count_include_pad.unwrap_or(false)),
            ));
        }

        self.node(
            if operation == Operation::MaxPool2d {
                "MaxPool"
            } else {
                "AveragePool"
            },
            inputs,
            output,
            attrs,
        );
        Ok(())
    }

    fn layer_norm(
        &mut self,
        op: OperationExport<'_>,
        attributes: LayerNormalizationAttributes,
    ) -> Result<()> {
        self.matching_floats(&op)?;
        let LayerNormalizationAttributes {
            normalized_shape,
            has_weight,
            has_bias,
            eps,
        } = attributes;
        let OperationExport {
            inputs,
            output,
            data_type,
            ..
        } = op;
        if data_type == DataType::F64 {
            return Err(unsupported(
                "FP64 LayerNormalization requires FP64 accumulation",
            ));
        }

        let mut operands = vec![inputs[0].clone()];
        if has_weight {
            operands.push(inputs[1].clone());
        } else {
            operands.push(self.filled(data_type, normalized_shape.clone(), 1.)?);
        }

        if has_bias {
            operands.push(inputs[1 + usize::from(has_weight)].clone());
        }

        self.node(
            "LayerNormalization",
            operands,
            output,
            vec![
                integer(
                    "axis",
                    -i64::try_from(normalized_shape.len())
                        .map_err(|_| unsupported("normalization rank"))?,
                ),
                float("epsilon", eps),
                integer("stash_type", 1),
            ],
        );
        Ok(())
    }

    fn batch_norm(
        &mut self,
        op: OperationExport<'_>,
        has_weight: bool,
        has_bias: bool,
        eps: f32,
    ) -> Result<()> {
        self.matching_floats(&op)?;
        let OperationExport {
            node,
            inputs,
            output,
            data_type,
            ..
        } = op;
        let channels = self
            .source
            .required_value_shape(node.inputs()[0])?
            .dimensions()[1];
        let weight = if has_weight {
            inputs[1].clone()
        } else {
            self.filled(data_type, vec![channels], 1.)?
        };
        let bias = if has_bias {
            inputs[1 + usize::from(has_weight)].clone()
        } else {
            self.filled(data_type, vec![channels], 0.)?
        };
        let stats = 1 + usize::from(has_weight) + usize::from(has_bias);
        self.node(
            "BatchNormalization",
            vec![
                inputs[0].clone(),
                weight,
                bias,
                inputs[stats].clone(),
                inputs[stats + 1].clone(),
            ],
            output,
            vec![float("epsilon", eps), integer("training_mode", 0)],
        );
        Ok(())
    }

    fn slice(&mut self, op: OperationExport<'_>, slice: SliceAttributes) -> Result<()> {
        let SliceAttributes {
            dim,
            start,
            length,
            step,
        } = slice;
        let OperationExport {
            mut inputs, output, ..
        } = op;
        let end = start + (length - 1).max(0) * step + i64::from(length > 0);
        let starts = self.integers(vec![1], &[start])?;
        let ends = self.integers(vec![1], &[end])?;
        let axes = self.integers(vec![1], &[dim])?;
        let steps = self.integers(vec![1], &[step])?;
        inputs.extend([starts, ends, axes, steps]);
        self.node("Slice", inputs, output, vec![]);
        Ok(())
    }

    fn multinomial(&mut self, op: OperationExport<'_>, samples: i64) -> Result<()> {
        let OperationExport {
            value,
            inputs,
            output,
            ..
        } = op;
        // ONNX samples from unnormalized log-probabilities of a `[batch, classes]` input.
        let vector = self.source.required_value_shape(value)?.rank() == 1;
        let mut logits = self.temporary("Log", vec![inputs[0].clone()], vec![]);
        let axes = self.integers(vec![1], &[0])?;
        if vector {
            logits = self.temporary("Unsqueeze", vec![logits, axes.clone()], vec![]);
        }

        let attributes = vec![
            integer("sample_size", samples),
            integer("dtype", i64::from(dtype(DataType::I64)?)),
        ];
        if vector {
            let samples = self.temporary("Multinomial", vec![logits], attributes);
            self.node("Squeeze", vec![samples, axes], output, vec![]);
        } else {
            self.node("Multinomial", vec![logits], output, attributes);
        }
        Ok(())
    }

    fn strided_scatter(
        &mut self,
        op: OperationExport<'_>,
        strided: &StridedAttributes,
    ) -> Result<()> {
        let OperationExport { inputs, output, .. } = op;
        let positions = strided_positions(strided)?;
        let count =
            i64::try_from(positions.len()).map_err(|_| unsupported("strided scatter size"))?;
        let indices = self.integers(vec![count], &positions)?;
        let shape = self.integers(vec![1], &[-1])?;
        let updates = self.temporary("Reshape", vec![inputs[1].clone(), shape], vec![]);
        self.node(
            "ScatterElements",
            vec![inputs[0].clone(), indices, updates],
            output,
            vec![integer("axis", 0)],
        );
        Ok(())
    }

    fn filled(&mut self, dtype: DataType, dimensions: Vec<i64>, value: f64) -> Result<String> {
        let count = dimensions.iter().try_fold(1_usize, |count, &dim| {
            let dim =
                usize::try_from(dim).map_err(|_| unsupported("affine parameter dimension"))?;
            count
                .checked_mul(dim)
                .ok_or_else(|| unsupported("affine parameter size overflow"))
        })?;
        let scalar = scalar_bytes(Scalar::Float(value), dtype)?;
        let bytes = count
            .checked_mul(scalar.len())
            .ok_or_else(|| unsupported("affine parameter bytes overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(bytes)
            .map_err(|_| unsupported("affine parameter allocation"))?;
        for _ in 0..count {
            values.extend_from_slice(&scalar);
        }
        self.literal(dtype, dimensions, values)
    }
}

/// Returns the storage position of every addressed element, in row-major order of `strided.dimensions`.
fn strided_positions(strided: &StridedAttributes) -> Result<Vec<i64>> {
    let offset = i64::try_from(strided.offset).map_err(|_| unsupported("strided offset"))?;
    let mut positions = vec![offset];
    for (&dimension, &stride) in strided.dimensions.iter().zip(&strided.strides) {
        positions = positions
            .iter()
            .flat_map(|&start| (0..dimension).map(move |index| start + index * stride))
            .collect();
    }
    Ok(positions)
}
