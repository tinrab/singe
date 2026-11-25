use singe_onnx::proto::onnx::AttributeProto;

use crate::{
    backend::registry::{Operation, OperationRequest, TopKAttributes},
    data_type::DataType,
    graph::DimensionExpr,
    onnx::{
        error::Result,
        exporter::{
            Exporter,
            ops::{OperationExport, topk_attributes},
            proto::{dtype, integer, scalar_bytes, string, unsupported},
        },
    },
    scalar::Scalar,
    shape::{Shape, normalize_dim_usize},
    value::ValueSpec,
};

impl Exporter<'_> {
    pub(crate) fn cast(&mut self, input: String, to: DataType) -> Result<String> {
        Ok(self.temporary(
            "Cast",
            vec![input],
            vec![integer("to", i64::from(dtype(to)?))],
        ))
    }

    pub(crate) fn finish(
        &mut self,
        operation: &str,
        inputs: Vec<String>,
        output: String,
        attributes: Vec<AttributeProto>,
        computed: DataType,
        data_type: DataType,
    ) -> Result<()> {
        if computed == data_type {
            self.node(operation, inputs, output, attributes);
        } else {
            let result = self.temporary(operation, inputs, attributes);
            self.node(
                "Cast",
                vec![result],
                output,
                vec![integer("to", i64::from(dtype(data_type)?))],
            );
        }
        Ok(())
    }

    pub(crate) fn input_types(&self, op: &OperationExport<'_>) -> Result<Vec<DataType>> {
        op.node
            .inputs()
            .iter()
            .map(|&input| Ok(self.source.required_value_data_type(input)?))
            .collect()
    }

    fn computation_type(&self, op: &OperationExport<'_>) -> Result<DataType> {
        let inputs = op
            .node
            .inputs()
            .iter()
            .map(|&input| {
                Ok(ValueSpec::new(
                    self.source.required_value_data_type(input)?,
                    self.source.required_value_shape(input)?.clone(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(
            OperationRequest::new(op.operation, op.node.attributes().clone(), inputs)
                .infer()?
                .computation_data_type(),
        )
    }

    fn operand_type(&self, op: &OperationExport<'_>) -> Result<DataType> {
        let types = self.input_types(op)?;
        let first = *types
            .first()
            .ok_or_else(|| unsupported(format!("{:?} without operands", op.operation)))?;
        if types.iter().all(|&data_type| data_type == first)
            && (op.data_type == DataType::Bool || first == op.data_type)
        {
            return Ok(first);
        }
        self.computation_type(op)
    }

    fn converted(&mut self, op: &OperationExport<'_>, to: DataType) -> Result<Vec<String>> {
        let types = self.input_types(op)?;
        op.inputs
            .iter()
            .zip(types)
            .map(|(input, data_type)| {
                if data_type == to {
                    Ok(input.clone())
                } else {
                    self.cast(input.clone(), to)
                }
            })
            .collect()
    }

    pub(crate) fn elementwise(
        &mut self,
        op: OperationExport<'_>,
        name: &str,
        attributes: Vec<AttributeProto>,
    ) -> Result<()> {
        let operands = self.operand_type(&op)?;
        let inputs = self.converted(&op, operands)?;
        let computed = if op.data_type == DataType::Bool {
            DataType::Bool
        } else {
            operands
        };
        self.finish(name, inputs, op.output, attributes, computed, op.data_type)
    }

    pub(crate) fn not_equal(&mut self, op: OperationExport<'_>) -> Result<()> {
        let operands = self.operand_type(&op)?;
        let inputs = self.converted(&op, operands)?;
        let equal = self.temporary("Equal", inputs, vec![]);
        self.node("Not", vec![equal], op.output, vec![]);
        Ok(())
    }

    pub(crate) fn logical(&mut self, op: OperationExport<'_>) -> Result<()> {
        let inputs = self.converted(&op, DataType::Bool)?;
        let name = match op.operation {
            Operation::LogicalAnd => "And",
            Operation::LogicalOr => "Or",
            _ => "Not",
        };
        self.finish(
            name,
            inputs,
            op.output,
            vec![],
            DataType::Bool,
            op.data_type,
        )
    }

    pub(crate) fn bitwise(&mut self, op: OperationExport<'_>) -> Result<()> {
        let operands = self.operand_type(&op)?;
        let inputs = self.converted(&op, operands)?;
        // Opset 28 shifts signed integers too, right shifts copying the sign bit as Singe's do.
        let shift = |direction: &str| ("BitShift", vec![string("direction", direction)]);
        // Boolean bitwise operations are the logical ones.
        let boolean = operands == DataType::Bool;
        let (name, attributes) = match op.operation {
            Operation::BitwiseAnd if boolean => ("And", vec![]),
            Operation::BitwiseOr if boolean => ("Or", vec![]),
            Operation::BitwiseXor if boolean => ("Xor", vec![]),
            Operation::BitwiseNot if boolean => ("Not", vec![]),
            Operation::BitwiseAnd => ("BitwiseAnd", vec![]),
            Operation::BitwiseOr => ("BitwiseOr", vec![]),
            Operation::BitwiseXor => ("BitwiseXor", vec![]),
            Operation::BitwiseNot => ("BitwiseNot", vec![]),
            Operation::BitwiseLeftShift => shift("LEFT"),
            _ => shift("RIGHT"),
        };
        self.finish(name, inputs, op.output, attributes, operands, op.data_type)
    }

    pub(crate) fn composed(
        &mut self,
        op: OperationExport<'_>,
        compose: impl FnOnce(&mut Self, Vec<String>, DataType, String) -> Result<()>,
    ) -> Result<()> {
        let operands = match self.operand_type(&op)? {
            DataType::F16 | DataType::Bf16 => DataType::F32,
            operands => operands,
        };
        let inputs = self.converted(&op, operands)?;
        if operands == op.data_type {
            return compose(self, inputs, operands, op.output);
        }

        let result = self.fresh();
        compose(self, inputs, operands, result.clone())?;
        self.node(
            "Cast",
            vec![result],
            op.output,
            vec![integer("to", i64::from(dtype(op.data_type)?))],
        );
        Ok(())
    }

    pub(crate) fn resample_linear(&mut self, op: OperationExport<'_>, dim: i64) -> Result<()> {
        let rank = self
            .source
            .required_value_shape(op.node.inputs()[0])?
            .rank();
        let axis = if dim < 0 { dim + rank } else { dim };
        if !(0..rank).contains(&axis) {
            return Err(unsupported("resample axis"));
        }

        let computed = match op.data_type {
            DataType::F16 | DataType::Bf16 => DataType::F32,
            data_type => data_type,
        };
        let [input, low, high, weight] = op.inputs.as_slice() else {
            return Err(unsupported("resample operands"));
        };
        let (input, low, high) = (input.clone(), low.clone(), high.clone());
        let weight = weight.clone();
        let input = if op.data_type == computed {
            input
        } else {
            self.cast(input, computed)?
        };
        let weight = if op.data_type == computed {
            weight
        } else {
            self.cast(weight, computed)?
        };
        let trailing = (1..rank - axis).collect::<Vec<_>>();
        let weight = if trailing.is_empty() {
            weight
        } else {
            let axes = self.integers(
                vec![i64::try_from(trailing.len()).map_err(|_| unsupported("resample rank"))?],
                &trailing,
            )?;
            self.temporary("Unsqueeze", vec![weight, axes], vec![])
        };
        let gather = vec![integer("axis", axis)];
        let lower = self.temporary("Gather", vec![input.clone(), low], gather.clone());
        let upper = self.temporary("Gather", vec![input, high], gather);
        let span = self.temporary("Sub", vec![upper, lower.clone()], vec![]);
        let step = self.temporary("Mul", vec![span, weight], vec![]);
        if computed == op.data_type {
            self.node("Add", vec![lower, step], op.output, vec![]);
        } else {
            let blended = self.temporary("Add", vec![lower, step], vec![]);
            self.node(
                "Cast",
                vec![blended],
                op.output,
                vec![integer("to", i64::from(dtype(op.data_type)?))],
            );
        }
        Ok(())
    }

    pub(crate) fn remainder(&mut self, op: OperationExport<'_>) -> Result<()> {
        self.elementwise(op, "Mod", vec![integer("fmod", 0)])
    }

    pub(crate) fn unary_composition(&mut self, op: OperationExport<'_>) -> Result<()> {
        let operation = op.operation;
        self.composed(op, |exporter, inputs, data_type, output| {
            let input = inputs[0].clone();
            match operation {
                Operation::Square => {
                    exporter.node("Mul", vec![input.clone(), input], output, vec![]);
                }
                Operation::Rsqrt => {
                    let root = exporter.temporary("Sqrt", vec![input], vec![]);
                    exporter.node("Reciprocal", vec![root], output, vec![]);
                }
                Operation::Log1p => {
                    let one = exporter.scalar(Scalar::Float(1.0), data_type)?;
                    let shifted = exporter.temporary("Add", vec![input, one], vec![]);
                    exporter.node("Log", vec![shifted], output, vec![]);
                }
                Operation::Expm1 => {
                    let one = exporter.scalar(Scalar::Float(1.0), data_type)?;
                    let exponential = exporter.temporary("Exp", vec![input], vec![]);
                    exporter.node("Sub", vec![exponential, one], output, vec![]);
                }
                _ if !data_type.is_float() => {
                    exporter.node("Identity", vec![input], output, vec![]);
                }
                _ => {
                    let zero = exporter.scalar(Scalar::Float(0.0), data_type)?;
                    let negative = exporter.temporary("Less", vec![input.clone(), zero], vec![]);
                    let ceiling = exporter.temporary("Ceil", vec![input.clone()], vec![]);
                    let floor = exporter.temporary("Floor", vec![input], vec![]);
                    exporter.node("Where", vec![negative, ceiling, floor], output, vec![]);
                }
            }
            Ok(())
        })
    }

    pub(crate) fn lerp(&mut self, op: OperationExport<'_>, weight: f32) -> Result<()> {
        self.composed(op, |exporter, inputs, data_type, output| {
            let weight = exporter.scalar(Scalar::Float(f64::from(weight)), data_type)?;
            let difference =
                exporter.temporary("Sub", vec![inputs[1].clone(), inputs[0].clone()], vec![]);
            let step = exporter.temporary("Mul", vec![weight, difference], vec![]);
            exporter.node("Add", vec![inputs[0].clone(), step], output, vec![]);
            Ok(())
        })
    }

    pub(crate) fn clamp(&mut self, op: OperationExport<'_>, min: f32, max: f32) -> Result<()> {
        self.matching_floats(&op)?;
        let data_type = op.data_type;
        let low = self.scalar(Scalar::Float(f64::from(min)), data_type)?;
        let high = self.scalar(Scalar::Float(f64::from(max)), data_type)?;
        let mut inputs = op.inputs;
        inputs.extend([low, high]);
        self.node("Clip", inputs, op.output, vec![]);
        Ok(())
    }

    pub(crate) fn repeat_kv(&mut self, op: OperationExport<'_>, repeats: i64) -> Result<()> {
        let source = op.node.inputs()[0];
        if self.source.required_value_shape(source)?.rank() != 4 {
            return Err(unsupported("repeat_kv of a tensor that is not rank 4"));
        }

        let axes = self.integers(vec![1], &[2])?;
        let unsqueezed = self.temporary("Unsqueeze", vec![op.inputs[0].clone(), axes], vec![]);
        let copies = self.integers(vec![5], &[1, 1, repeats, 1, 1])?;
        let expanded = self.temporary("Expand", vec![unsqueezed, copies], vec![]);
        let axis = |axis| DimensionExpr::axis(None, source.index(), axis);
        let shape = self.shape_vector(&[
            axis(0),
            axis(1) * DimensionExpr::constant(repeats),
            axis(2),
            axis(3),
        ])?;
        self.node(
            "Reshape",
            vec![expanded, shape],
            op.output,
            vec![integer("allowzero", 1)],
        );
        Ok(())
    }

    pub(crate) fn expand(&mut self, op: OperationExport<'_>, shape: &Shape) -> Result<()> {
        let input = self
            .source
            .required_value_shape(op.node.inputs()[0])?
            .clone();
        let target = shape.dimensions();
        let offset = target
            .len()
            .checked_sub(input.dimensions().len())
            .ok_or_else(|| unsupported("expand to a rank lower than its input's"))?;
        let dimensions = target
            .iter()
            .enumerate()
            .map(|(axis, &extent)| {
                let kept = axis
                    .checked_sub(offset)
                    .is_some_and(|axis| input.dimensions()[axis] == extent);
                if kept { 1 } else { extent }
            })
            .collect::<Vec<_>>();
        let length = i64::try_from(dimensions.len()).map_err(|_| unsupported("expand rank"))?;
        let dimensions = self.integers(vec![length], &dimensions)?;
        let mut inputs = op.inputs;
        inputs.push(dimensions);
        self.node("Expand", inputs, op.output, vec![]);
        Ok(())
    }

    pub(crate) fn pad(
        &mut self,
        op: OperationExport<'_>,
        padding: &[(i64, i64)],
        value: f32,
    ) -> Result<()> {
        let rank = self
            .source
            .required_value_shape(op.node.inputs()[0])?
            .rank();
        let rank = usize::try_from(rank).map_err(|_| unsupported("pad rank"))?;
        let widths = (0..rank)
            .map(|axis| padding.get(axis).copied().unwrap_or((0, 0)))
            .collect::<Vec<_>>();
        let pads = widths
            .iter()
            .map(|&(before, _)| before)
            .chain(widths.iter().map(|&(_, after)| after))
            .collect::<Vec<_>>();
        let length = i64::try_from(pads.len()).map_err(|_| unsupported("pad rank"))?;
        let pads = self.integers(vec![length], &pads)?;
        let fill = self.scalar(Scalar::Float(f64::from(value)), op.data_type)?;
        let mut inputs = op.inputs;
        inputs.extend([pads, fill]);
        self.node("Pad", inputs, op.output, vec![string("mode", "constant")]);
        Ok(())
    }

    pub(crate) fn arg_reduction(&mut self, op: OperationExport<'_>, dim: i64) -> Result<()> {
        let name = if op.operation == Operation::ArgMax {
            "ArgMax"
        } else {
            "ArgMin"
        };
        self.finish(
            name,
            op.inputs,
            op.output,
            vec![
                integer("axis", dim),
                integer("keepdims", 0),
                integer("select_last_index", 0),
            ],
            DataType::I64,
            op.data_type,
        )
    }

    pub(crate) fn cumulative(&mut self, op: OperationExport<'_>, dim: i64) -> Result<()> {
        let input_type = self.source.required_value_data_type(op.node.inputs()[0])?;
        let mut inputs = op.inputs;
        if input_type != op.data_type {
            inputs[0] = self.cast(inputs[0].clone(), op.data_type)?;
        }

        let rank = self
            .source
            .required_value_shape(op.node.inputs()[0])?
            .rank();
        if rank == 0 {
            // A scalar is one slice of one element.
            self.node("Identity", inputs, op.output, vec![]);
            return Ok(());
        }

        inputs.push(self.integers(vec![], &[dim])?);
        let name = if op.operation == Operation::Cumsum {
            "CumSum"
        } else {
            "CumProd"
        };
        self.node(name, inputs, op.output, vec![]);
        Ok(())
    }

    pub(crate) fn topk_result(
        &mut self,
        op: OperationExport<'_>,
        topk: TopKAttributes,
    ) -> Result<()> {
        let k = self.integers(vec![1], &[topk.k])?;
        let unused = self.fresh();
        let outputs = if op.operation == Operation::TopKValues {
            vec![op.output, unused]
        } else {
            vec![unused, op.output]
        };
        self.outputs_node(
            "TopK",
            vec![op.inputs[0].clone(), k],
            outputs,
            topk_attributes(topk),
        );
        Ok(())
    }

    pub(crate) fn rfft(&mut self, op: OperationExport<'_>) -> Result<()> {
        self.matching_floats(&op)?;
        let rank = self
            .source
            .required_value_shape(op.node.inputs()[0])?
            .rank();
        let last = self.integers(vec![1], &[rank])?;
        let complex = self.temporary("Unsqueeze", vec![op.inputs[0].clone(), last], vec![]);
        let axis = self.integers(vec![], &[rank - 1])?;
        self.node(
            "DFT",
            vec![complex, String::new(), axis],
            op.output,
            vec![integer("onesided", 1), integer("inverse", 0)],
        );
        Ok(())
    }

    pub(crate) fn irfft(&mut self, op: OperationExport<'_>, length: i64) -> Result<()> {
        self.matching_floats(&op)?;
        let input = self.source.required_value_shape(op.node.inputs()[0])?;
        let rank = input.rank();
        let frequencies = input.dimensions()[normalize_dim_usize(-2, rank)?];
        let axis = rank - 2;
        let mut pads = vec![0; usize::try_from(2 * rank).map_err(|_| unsupported("irfft rank"))?];
        pads[usize::try_from(rank + axis).map_err(|_| unsupported("irfft rank"))?] =
            length - frequencies;
        let pads = self.integers(vec![2 * rank], &pads)?;
        let padded = self.temporary(
            "Pad",
            vec![op.inputs[0].clone(), pads],
            vec![string("mode", "constant")],
        );
        let half = length / 2;
        let weights = (0..length)
            .map(|frequency| {
                if frequency == 0 || (length % 2 == 0 && frequency == half) {
                    1.0
                } else if frequency <= half {
                    2.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<f64>>();
        let bytes = weights
            .iter()
            .map(|&weight| scalar_bytes(Scalar::Float(weight), op.data_type))
            .collect::<Result<Vec<_>>>()?
            .concat();
        let weights = self.literal(op.data_type, vec![length, 1], bytes)?;
        let weighted = self.temporary("Mul", vec![padded, weights], vec![]);
        let axis = self.integers(vec![], &[axis])?;
        let signal = self.temporary(
            "DFT",
            vec![weighted, String::new(), axis],
            vec![integer("onesided", 0), integer("inverse", 1)],
        );
        let starts = self.integers(vec![1], &[0])?;
        let ends = self.integers(vec![1], &[1])?;
        let last = self.integers(vec![1], &[rank - 1])?;
        let real = self.temporary("Slice", vec![signal, starts, ends, last.clone()], vec![]);
        self.node("Squeeze", vec![real, last], op.output, vec![]);
        Ok(())
    }
}
