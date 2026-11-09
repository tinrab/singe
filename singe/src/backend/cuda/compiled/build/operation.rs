use crate::{
    backend::{registry::Operation, spec::ConvolutionWindow},
    error::{Error, Result},
    graph::ir::{Graph, OperationNode, ValueId},
};

#[derive(Clone, Copy)]
pub struct GraphIrConv2d {
    pub(crate) input: ValueId,
    pub(crate) weight: ValueId,
    pub(crate) bias: Option<ValueId>,
    pub(crate) stride: (i64, i64),
    pub(crate) padding: (i64, i64, i64, i64),
    pub(crate) dilation: (i64, i64),
    pub(crate) groups: i64,
}

pub struct GraphIrContraction {
    pub(crate) a: ValueId,
    pub(crate) a_mode: Vec<i64>,
    pub(crate) b: ValueId,
    pub(crate) b_mode: Vec<i64>,
    pub(crate) c: ValueId,
    pub(crate) c_mode: Vec<i64>,
    pub(crate) d_mode: Vec<i64>,
    pub(crate) alpha: f32,
    pub(crate) beta: f32,
}

pub fn conv2d_operation(ir: &Graph, id: ValueId) -> Result<Option<GraphIrConv2d>> {
    let Some(operation) = operation_with_opcode(ir, id, Operation::Conv2d)? else {
        return Ok(None);
    };
    let attributes = operation
        .attributes()
        .conv2d()
        .copied()
        .ok_or_else(|| Error::missing_node("graph ir conv2d"))?;
    let inputs = operation.inputs().to_vec();
    let (input, weight, bias) = match (attributes.has_bias, inputs.as_slice()) {
        (false, [input, weight]) => (*input, *weight, None),
        (true, [input, weight, bias]) => (*input, *weight, Some(*bias)),
        _ => return Err(Error::missing_input("graph ir conv2d")),
    };
    Ok(Some(GraphIrConv2d {
        input,
        weight,
        bias,
        stride: attributes.stride,
        padding: attributes.padding,
        dilation: attributes.dilation,
        groups: attributes.groups,
    }))
}

pub struct GraphIrConvolution {
    pub(crate) input: ValueId,
    pub(crate) weight: ValueId,
    pub(crate) bias: Option<ValueId>,
    pub(crate) window: ConvolutionWindow,
}

// A 2D or 3D convolution, whichever produces `id`.
pub fn convolution_operation(ir: &Graph, id: ValueId) -> Result<Option<GraphIrConvolution>> {
    if let Some(conv2d) = conv2d_operation(ir, id)? {
        return Ok(Some(GraphIrConvolution {
            input: conv2d.input,
            weight: conv2d.weight,
            bias: conv2d.bias,
            window: ConvolutionWindow::conv2d(
                conv2d.stride,
                conv2d.padding,
                conv2d.dilation,
                conv2d.groups,
            ),
        }));
    }

    let Some(operation) = operation_with_opcode(ir, id, Operation::Conv3d)? else {
        return Ok(None);
    };
    let attributes = operation
        .attributes()
        .conv3d()
        .ok_or_else(|| Error::missing_node("graph ir convolution"))?;
    let (input, weight, bias) = match (attributes.has_bias, operation.inputs()) {
        (false, &[input, weight]) => (input, weight, None),
        (true, &[input, weight, bias]) => (input, weight, Some(bias)),
        _ => return Err(Error::missing_input("graph ir convolution")),
    };
    Ok(Some(GraphIrConvolution {
        input,
        weight,
        bias,
        window: ConvolutionWindow::conv3d(attributes),
    }))
}

pub fn contraction_operation(ir: &Graph, id: ValueId) -> Result<Option<GraphIrContraction>> {
    let Some(operation) = operation_with_opcode(ir, id, Operation::Contraction)? else {
        return Ok(None);
    };
    let attributes = operation
        .attributes()
        .contraction()
        .ok_or_else(|| Error::missing_node("graph ir contraction"))?;
    let inputs = operation.inputs().to_vec();
    let [a, b, c] = inputs.as_slice() else {
        return Err(Error::missing_input("graph ir contraction"));
    };
    Ok(Some(GraphIrContraction {
        a: *a,
        a_mode: attributes.a_mode.clone(),
        b: *b,
        b_mode: attributes.b_mode.clone(),
        c: *c,
        c_mode: attributes.c_mode.clone(),
        d_mode: attributes.d_mode.clone(),
        alpha: attributes.alpha,
        beta: attributes.beta,
    }))
}

pub fn contiguous_input(ir: &Graph, id: ValueId) -> Result<Option<ValueId>> {
    let Some(operation) = operation_with_opcode(ir, id, Operation::Contiguous)? else {
        return Ok(None);
    };
    let inputs = operation.inputs().to_vec();
    let [input] = inputs.as_slice() else {
        return Err(Error::missing_input("graph ir contiguous input"));
    };
    Ok(Some(*input))
}

pub fn copy_to_input(ir: &Graph, id: ValueId) -> Result<Option<ValueId>> {
    let Some(operation) = operation_with_opcode(ir, id, Operation::CopyTo)? else {
        return Ok(None);
    };
    let inputs = operation.inputs().to_vec();
    let [input] = inputs.as_slice() else {
        return Err(Error::missing_input("graph ir copy to input"));
    };
    Ok(Some(*input))
}

pub fn binary_inputs(ir: &Graph, id: ValueId, opcode: Operation) -> Result<Option<[ValueId; 2]>> {
    let Some(operation) = operation_with_opcode(ir, id, opcode)? else {
        return Ok(None);
    };
    let inputs = operation.inputs().to_vec();
    let [lhs, rhs] = inputs.as_slice() else {
        return Err(Error::missing_input("graph ir binary inputs"));
    };
    Ok(Some([*lhs, *rhs]))
}

pub fn operation_with_opcode(
    ir: &Graph,
    id: ValueId,
    opcode: Operation,
) -> Result<Option<&OperationNode>> {
    let operation = ir.required_operation_for_value(id)?;
    Ok((operation.opcode() == Some(opcode)).then_some(operation))
}
