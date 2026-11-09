use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    data_type::DataType,
    graph::GraphOperationOutput,
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        ops::{
            add_binary_intermediate, add_scale, add_transpose, expect_input_range, expect_outputs,
            optional_input, optional_quantization_zero_point, quantization_scale,
            register_node_outputs, zero_point_initializer,
        },
        shape::gemm_output_shape,
        utility::{node_label, node_name},
    },
    shape::Shape,
};

pub fn lower_gemm(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 2, 3)?;
    expect_outputs(importer, node, 1)?;
    let attrs = Attributes::create(node)?;
    let a = importer.required_input(node, 0)?;
    let b = importer.required_input(node, 1)?;
    let c = optional_input(importer, node, 2)?;
    let trans_a = attrs.int("transA", 0)? != 0;
    let trans_b = attrs.int("transB", 0)? != 0;
    let alpha = attrs.float("alpha", 1.0)?;
    let beta = attrs.float("beta", 1.0)?;
    let current = build_gemm(importer, node, a, b, c, trans_a, trans_b, alpha, beta)?;
    register_node_outputs(importer, node, vec![current])
}

#[expect(
    clippy::float_cmp,
    reason = "exact equality selects an identity, specialization, or deterministic tie; approximate equality changes semantics"
)]
pub fn build_gemm(
    importer: &mut Importer,
    node: &Node,
    mut a: ImportedValue,
    mut b: ImportedValue,
    c: Option<ImportedValue>,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
) -> Result<ImportedValue> {
    if trans_a {
        a = add_transpose(importer, &node_label(node), &a)?;
    }

    if trans_b {
        b = add_transpose(importer, &node_label(node), &b)?;
    }

    let output_shape = gemm_output_shape(&a.shape, &b.shape, false, false)?;
    let matmul = add_binary_intermediate(
        importer,
        None,
        Operation::MatMul,
        [a.node, b.node],
        a.data_type,
        output_shape.clone(),
    )?;
    let mut current = ImportedValue::new(matmul, a.data_type, output_shape);
    if alpha != 1.0 {
        current = add_scale(importer, current, alpha)?;
    }

    if let Some(c) = c {
        current = add_bias(importer, &current, c, beta)?;
    }
    Ok(current)
}

#[expect(
    clippy::float_cmp,
    reason = "exact equality selects an identity, specialization, or deterministic tie; approximate equality changes semantics"
)]
fn add_bias(
    importer: &mut Importer,
    current: &ImportedValue,
    mut bias: ImportedValue,
    beta: f32,
) -> Result<ImportedValue> {
    if beta != 1.0 {
        bias = add_scale(importer, bias, beta)?;
    }

    let shape = current.shape.broadcast_with(&bias.shape)?;
    let add = add_binary_intermediate(
        importer,
        None,
        Operation::Add,
        [current.node, bias.node],
        current.data_type,
        shape.clone(),
    )?;
    Ok(ImportedValue::new(add, current.data_type, shape))
}

pub fn lower_qlinear_matmul(importer: &mut Importer, node: &Node) -> Result<()> {
    expect_input_range(importer, node, 6, 8)?;
    expect_outputs(importer, node, 1)?;

    let plan = qlinear_matmul_plan(importer, node)?;
    let device = importer.options().device;
    let matmul = importer.graph().add_named_quantized_matmul_operation(
        node_name(node).map(|name| format!("{name}/quantized_matmul")),
        [plan.a.node, plan.b.node],
        (plan.a_scale, plan.a_zero_point),
        (plan.b_scale, plan.b_zero_point),
        GraphOperationOutput::new(DataType::F32, plan.output_shape.clone(), device),
    )?;
    let output = importer.graph().add_named_quantization_operation(
        node_name(node),
        Operation::Quantize,
        matmul,
        plan.y_scale,
        plan.y_zero_point,
        plan.y_data_type,
        plan.output_shape.clone(),
        device,
    )?;
    register_node_outputs(
        importer,
        node,
        vec![ImportedValue::new(
            output,
            plan.y_data_type,
            plan.output_shape,
        )],
    )
}

pub struct QLinearMatMulPlan {
    pub(crate) a: ImportedValue,
    pub(crate) b: ImportedValue,
    pub(crate) a_scale: f32,
    pub(crate) a_zero_point: i64,
    pub(crate) b_scale: f32,
    pub(crate) b_zero_point: i64,
    pub(crate) y_scale: f32,
    pub(crate) y_zero_point: i64,
    pub(crate) y_data_type: DataType,
    pub(crate) output_shape: Shape,
}

pub fn qlinear_matmul_plan(importer: &Importer, node: &Node) -> Result<QLinearMatMulPlan> {
    let a = importer.required_input(node, 0)?;
    let b = importer.required_input(node, 3)?;
    validate_input_types(importer, node, &a, &b)?;
    let a_scale = quantization_scale(importer, node, 1, "a_scale")?;
    let a_zero_point =
        optional_quantization_zero_point(importer, node, 2, "a_zero_point", a.data_type, 0)?;
    let b_scale = quantization_scale(importer, node, 4, "b_scale")?;
    let b_zero_point =
        optional_quantization_zero_point(importer, node, 5, "b_zero_point", b.data_type, 0)?;
    let y_scale = quantization_scale(importer, node, 6, "y_scale")?;
    let (y_zero_point, y_data_type) = output_zero_point(importer, node)?;
    let output_shape = gemm_output_shape(&a.shape, &b.shape, false, false)?;
    Ok(QLinearMatMulPlan {
        a,
        b,
        a_scale,
        a_zero_point,
        b_scale,
        b_zero_point,
        y_scale,
        y_zero_point,
        y_data_type,
        output_shape,
    })
}

fn validate_input_types(
    importer: &Importer,
    node: &Node,
    a: &ImportedValue,
    b: &ImportedValue,
) -> Result<()> {
    if matches!(a.data_type, DataType::I8 | DataType::U8)
        && matches!(b.data_type, DataType::I8 | DataType::U8)
    {
        return Ok(());
    }
    Err(Error::unsupported_operator(
        node.domain.as_deref(),
        &node.operator_type,
        importer.opset_version(node.domain.as_deref()),
        "inputs must be int8 or uint8",
    ))
}

fn output_zero_point(importer: &Importer, node: &Node) -> Result<(i64, DataType)> {
    match optional_input(importer, node, 7)? {
        Some(zero_point) => {
            let data_type = zero_point.data_type;
            Ok((
                zero_point_initializer(importer, node, 7, "y_zero_point", data_type)?,
                data_type,
            ))
        }
        None => Ok((0, DataType::U8)),
    }
}
