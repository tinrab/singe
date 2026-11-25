use singe_onnx::Node;

use crate::{
    error::{Error as SingeError, Result as SingeResult},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{expect_input_range, expect_outputs, optional_input, register_node_outputs},
    },
    tensor::Tensor,
};

fn refuse(importer: &Importer, node: &Node, reason: &str) -> Error {
    Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        reason,
    )
}

pub fn lower_rotary_embedding(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 3, 4)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let input = importer.required_input(node, 0)?;
    let cos = importer.required_input(node, 1)?;
    let sin = importer.required_input(node, 2)?;
    let positions = optional_input(importer, node, 3)?;
    let interleaved = attrs.int("interleaved", 0)? != 0;
    let &[_, _, _, width] = input.shape.dimensions() else {
        return Err(refuse(
            importer,
            node,
            "an input other than [batch, heads, sequence, width]",
        ));
    };
    let rotated = match attrs.int("rotary_embedding_dim", 0)? {
        0 => width,
        rotated => rotated,
    };
    if rotated <= 0 || rotated % 2 != 0 || rotated > width {
        return Err(refuse(
            importer,
            node,
            "a rotary dimension that is odd or wider than the input",
        ));
    }

    if cos.data_type != input.data_type || sin.data_type != input.data_type {
        return Err(refuse(
            importer,
            node,
            "caches of another type than the input",
        ));
    }

    let mut operands = vec![input.node, cos.node, sin.node];
    operands.extend(positions.as_ref().map(|positions| positions.node));
    if !importer.have_fixed_shapes(&operands)? {
        return Err(refuse(importer, node, "inputs with run-time axes"));
    }

    let device = importer.options().device;
    let [id] = importer
        .graph()
        .append_traced(&operands, device, |inputs| {
            let [input, cos, sin, positions @ ..] = inputs else {
                return Err(SingeError::missing_input("rotary embedding"));
            };
            let cache = |table: &Tensor| -> SingeResult<Tensor> {
                let table = match positions.first() {
                    Some(positions) => {
                        let &[batch, sequence] = positions.shape().dimensions() else {
                            return Err(SingeError::invalid_argument(
                                "rotary embedding positions rank",
                            ));
                        };
                        table
                            .index_select(0, &positions.reshape([-1])?)?
                            .reshape([batch, sequence, -1])?
                    }
                    None => (*table).clone(),
                };
                table.unsqueeze(1)
            };
            Ok(vec![rotate(
                input,
                &cache(cos)?,
                &cache(sin)?,
                rotated,
                interleaved,
            )?])
        })?[..]
    else {
        return Err(refuse(importer, node, "a rotation without one result"));
    };
    let traced = importer.graph().required_node(id)?;
    let value = ImportedValue::new(id, traced.output_data_type(), traced.output_shape().clone());
    register_node_outputs(importer, node, vec![value])
}

fn rotate(
    input: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    rotated: i64,
    interleaved: bool,
) -> SingeResult<Tensor> {
    let dimensions = input.shape().dimensions().to_vec();
    let width = dimensions[3];
    let part = input.narrow(3, 0, rotated)?;
    let result = if interleaved {
        let mut paired = part.shape().dimensions().to_vec();
        paired[3] = rotated / 2;
        paired.push(2);
        let pairs = part.reshape(paired)?;
        let (even, odd) = (pairs.select(4, 0)?, pairs.select(4, 1)?);
        let first = even.mul(cos)?.sub(&odd.mul(sin)?)?;
        let second = even.mul(sin)?.add(&odd.mul(cos)?)?;
        Tensor::concat(&[first.unsqueeze(4)?, second.unsqueeze(4)?], 4)?
            .reshape(part.shape().dimensions().to_vec())?
    } else {
        part.rotary_halves(cos, sin)?
    };
    if rotated == width {
        return Ok(result);
    }
    Tensor::concat(&[result, input.narrow(3, rotated, width - rotated)?], 3)
}
