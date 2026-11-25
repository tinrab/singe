use crate::{
    backend::registry::{
        LayerNormalizationAttributes, Operation, ScaledDotProductAttentionAttributes,
    },
    data_type::DataType,
    graph::DimensionExpr,
    onnx::{
        error::Result,
        exporter::{
            Exporter,
            ops::OperationExport,
            proto::{float, integer, unsupported},
        },
    },
    shape::Shape,
};

impl Exporter<'_> {
    pub(crate) fn attention(
        &mut self,
        op: OperationExport<'_>,
        attention: ScaledDotProductAttentionAttributes,
    ) -> Result<()> {
        let ScaledDotProductAttentionAttributes {
            scale,
            causal,
            window,
            softcap,
            masked,
            sinks,
            key_lengths,
        } = attention;
        if sinks || (key_lengths && (causal || masked)) {
            return Err(unsupported(
                "onnx attention has no sinks and aligns key lengths with the causal mask at the bottom right",
            ));
        }

        for &operand in &op.node.inputs()[..3] {
            if self.source.required_value_shape(operand)?.rank() != 4 {
                return Err(unsupported("attention operands that are not rank 4"));
            }
        }

        let computed = match op.data_type {
            DataType::F16 | DataType::Bf16 => DataType::F32,
            data_type => data_type,
        };
        let types = self.input_types(&op)?;
        let mut inputs = Vec::new();
        for (input, data_type) in op.inputs.iter().zip(types).take(3 + usize::from(masked)) {
            inputs.push(if data_type == computed {
                input.clone()
            } else {
                self.cast(input.clone(), computed)?
            });
        }

        // `nonpad_kv_seqlen` counts each sequence's keys, as Singe's key lengths do, with no cache in `past_key` and `past_value`.
        if key_lengths {
            let lengths = self.cast(op.inputs[3].clone(), DataType::I64)?;
            inputs.extend([String::new(), String::new(), String::new(), lengths]);
        }

        let mut attributes = vec![integer("is_causal", i64::from(causal))];
        if let Some(scale) = scale {
            attributes.push(float("scale", scale));
        }

        if let Some(softcap) = softcap {
            attributes.push(float("softcap", softcap));
        }

        // Singe's window hides the keys `window` or more positions before the query, and `left_window_size` counts the earlier keys a query sees.
        if let Some(window) = window {
            if window < 1 {
                return Err(unsupported("attention window shorter than one key"));
            }

            attributes.push(integer("left_window_size", window - 1));
        }
        self.finish(
            "Attention",
            inputs,
            op.output,
            attributes,
            computed,
            op.data_type,
        )
    }

    pub(crate) fn rms_norm(
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
        let axis = -i64::try_from(normalized_shape.len())
            .map_err(|_| unsupported("normalization rank"))?;
        let scale = if has_weight {
            inputs[1].clone()
        } else {
            self.filled(data_type, normalized_shape, 1.)?
        };
        let operands = vec![inputs[0].clone(), scale];
        let attributes = vec![
            integer("axis", axis),
            float("epsilon", eps),
            integer("stash_type", 1),
        ];
        if has_bias {
            let normalized = self.temporary("RMSNormalization", operands, attributes);
            let bias = inputs[1 + usize::from(has_weight)].clone();
            self.node("Add", vec![normalized, bias], output, vec![]);
        } else {
            self.node("RMSNormalization", operands, output, attributes);
        }
        Ok(())
    }

    pub(crate) fn rotary(&mut self, op: OperationExport<'_>) -> Result<()> {
        let interleaved = op.operation == Operation::RotaryEmbedding;
        let input = op.node.inputs()[0];
        let input_shape = self.source.required_value_shape(input)?.clone();
        let cos_shape = self
            .source
            .required_value_shape(op.node.inputs()[1])?
            .clone();
        let fused = match (input_shape.dimensions(), cos_shape.dimensions()) {
            (&[_, _, sequence, width], &[cos_sequence, pairs]) => (interleaved
                || (cos_sequence == sequence && 2 * pairs == width))
                .then_some((width, pairs)),
            _ => None,
        };
        let Some((width, pairs)) = fused else {
            if interleaved {
                return Err(unsupported(
                    "rotary embedding of a tensor that is not rank 4",
                ));
            }
            return self.rotary_halves(op, &input_shape);
        };
        let computed = match op.data_type {
            DataType::F16 | DataType::Bf16 => DataType::F32,
            data_type => data_type,
        };
        let types = self.input_types(&op)?;
        let mut inputs = Vec::new();
        for (name, data_type) in op.inputs.iter().zip(types) {
            inputs.push(if data_type == computed {
                name.clone()
            } else {
                self.cast(name.clone(), computed)?
            });
        }

        let leading = self.integers(vec![1], &[0])?;
        let batch = self.shape_vector(&[
            DimensionExpr::axis(None, input.index(), 0),
            DimensionExpr::constant(1),
            DimensionExpr::constant(1),
        ])?;
        for cache in &mut inputs[1..] {
            let unsqueezed =
                self.temporary("Unsqueeze", vec![cache.clone(), leading.clone()], vec![]);
            *cache = self.temporary("Expand", vec![unsqueezed, batch.clone()], vec![]);
        }

        let mut attributes = vec![integer("interleaved", i64::from(interleaved))];
        if 2 * pairs < width {
            attributes.push(integer("rotary_embedding_dim", 2 * pairs));
        }
        self.finish(
            "RotaryEmbedding",
            inputs,
            op.output,
            attributes,
            computed,
            op.data_type,
        )
    }

    fn rotary_halves(&mut self, op: OperationExport<'_>, input_shape: &Shape) -> Result<()> {
        let width = *input_shape
            .dimensions()
            .last()
            .ok_or_else(|| unsupported("rotary embedding of a scalar"))?;
        let half = width / 2;
        self.composed(op, |exporter, inputs, _, output| {
            let [input, cos, sin] = inputs.as_slice() else {
                return Err(unsupported("rotary embedding operands"));
            };
            let axis = exporter.integers(vec![1], &[-1])?;
            let mut slice = |start: i64, end: i64| -> Result<String> {
                let starts = exporter.integers(vec![1], &[start])?;
                let ends = exporter.integers(vec![1], &[end])?;
                Ok(exporter.temporary(
                    "Slice",
                    vec![input.clone(), starts, ends, axis.clone()],
                    vec![],
                ))
            };
            let first = slice(0, half)?;
            let second = slice(half, width)?;
            let mut product = |a: &String, b: &String| {
                exporter.temporary("Mul", vec![a.clone(), b.clone()], vec![])
            };
            let (first_cos, second_sin) = (product(&first, cos), product(&second, sin));
            let (second_cos, first_sin) = (product(&second, cos), product(&first, sin));
            let rotated_first = exporter.temporary("Sub", vec![first_cos, second_sin], vec![]);
            let rotated_second = exporter.temporary("Add", vec![second_cos, first_sin], vec![]);
            exporter.node(
                "Concat",
                vec![rotated_first, rotated_second],
                output,
                vec![integer("axis", -1)],
            );
            Ok(())
        })
    }

    pub(crate) fn swiglu(&mut self, op: OperationExport<'_>) -> Result<()> {
        self.composed(op, |exporter, inputs, _, output| {
            let (value, gate) = (inputs[0].clone(), inputs[1].clone());
            let sigmoid = exporter.temporary("Sigmoid", vec![gate.clone()], vec![]);
            let silu = exporter.temporary("Mul", vec![gate, sigmoid], vec![]);
            exporter.node("Mul", vec![value, silu], output, vec![]);
            Ok(())
        })
    }
}
