use singe_onnx::Node;

use crate::{
    backend::registry::Operation,
    onnx::{
        error::{Error, Result},
        importer::Importer,
        ops::{
            constant::{lower_constant, lower_constant_of_shape},
            conv_pool::{lower_avg_pool, lower_conv, lower_global_avg_pool, lower_max_pool},
            conversion::{lower_cast, lower_dequantize_linear, lower_quantize_linear},
            elementwise::{
                lower_binary, lower_clip, lower_comparison, lower_dropout, lower_gelu,
                lower_identity, lower_leaky_relu, lower_pow, lower_softmax, lower_unary,
                lower_variadic_binary, lower_where,
            },
            linear::{lower_gemm, lower_qlinear_matmul},
            neural::{
                lower_attention, lower_batch_normalization, lower_layer_normalization, lower_topk,
            },
            reduction::{
                ComposedSumReductionKind, NormReductionKind, ReductionKind, lower_arg_reduction,
                lower_composed_sum_reduction, lower_norm_reduction, lower_reduction,
            },
            shape_ops::{
                lower_concat, lower_expand, lower_flatten, lower_gather, lower_pad, lower_reshape,
                lower_scatter_elements, lower_shape, lower_slice, lower_squeeze, lower_transpose,
                lower_unsqueeze,
            },
        },
    },
};

pub fn lower_node(importer: &mut Importer, node: &Node) -> Result<()> {
    let version = importer.opset_version(node.domain.as_deref());
    if node
        .domain
        .as_deref()
        .is_some_and(|domain| !domain.is_empty())
    {
        return Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            version,
            "custom domains are not supported",
        ));
    }

    if importer.lower_metadata_arithmetic(node)? {
        return Ok(());
    }

    if let Some(result) = lower_elementwise(importer, node) {
        return result;
    }

    if let Some(result) = lower_reduction_family(importer, node) {
        return result;
    }
    match node.operator_type.as_str() {
        "Conv" => lower_conv(importer, node),
        "Constant" => lower_constant(importer, node),
        "ConstantOfShape" => lower_constant_of_shape(importer, node),
        "MaxPool" => lower_max_pool(importer, node),
        "AveragePool" => lower_avg_pool(importer, node),
        "GlobalAveragePool" => lower_global_avg_pool(importer, node),
        "Shape" => lower_shape(importer, node),
        "Pad" => lower_pad(importer, node),
        "Slice" => lower_slice(importer, node),
        "Expand" => lower_expand(importer, node),
        "Reshape" => lower_reshape(importer, node),
        "Gemm" => lower_gemm(importer, node),
        "QLinearMatMul" => lower_qlinear_matmul(importer, node),
        "Flatten" => lower_flatten(importer, node),
        "Transpose" => lower_transpose(importer, node),
        "Concat" => lower_concat(importer, node),
        "Gather" => lower_gather(importer, node),
        "ScatterElements" => lower_scatter_elements(importer, node),
        "Squeeze" => lower_squeeze(importer, node),
        "Unsqueeze" => lower_unsqueeze(importer, node),
        "Cast" => lower_cast(importer, node),
        "QuantizeLinear" => lower_quantize_linear(importer, node),
        "DequantizeLinear" => lower_dequantize_linear(importer, node),
        "Attention" => lower_attention(importer, node),
        "TopK" => lower_topk(importer, node),
        "BatchNormalization" => lower_batch_normalization(importer, node),
        "LayerNormalization" => lower_layer_normalization(importer, node),
        _ => Err(Error::unsupported_operator(
            node.domain.as_deref(),
            &node.operator_type,
            version,
            "operator lowerer is not implemented",
        )),
    }
}

pub fn lower_elementwise(importer: &mut Importer, node: &Node) -> Option<Result<()>> {
    Some(match node.operator_type.as_str() {
        "Identity" => lower_identity(importer, node),
        "Dropout" => lower_dropout(importer, node),
        "Clip" => lower_clip(importer, node),
        "LeakyRelu" => lower_leaky_relu(importer, node),
        "Gelu" => lower_gelu(importer, node),
        "Silu" | "SiLU" | "Swish" => lower_unary(importer, node, Operation::Silu),
        "SwiGLU" | "SwiGlu" | "Swiglu" => lower_binary(importer, node, Operation::SwiGlu),
        "Softmax" => lower_softmax(importer, node, false),
        "LogSoftmax" => lower_softmax(importer, node, true),
        "Pow" => lower_pow(importer, node),
        "Max" => lower_variadic_binary(importer, node, Operation::Maximum),
        "Min" => lower_variadic_binary(importer, node, Operation::Minimum),
        "Equal" => lower_comparison(importer, node, Operation::Equal),
        "NotEqual" => lower_comparison(importer, node, Operation::NotEqual),
        "Less" => lower_comparison(importer, node, Operation::Less),
        "LessOrEqual" => lower_comparison(importer, node, Operation::LessEqual),
        "Greater" => lower_comparison(importer, node, Operation::Greater),
        "GreaterOrEqual" => lower_comparison(importer, node, Operation::GreaterEqual),
        "Where" => lower_where(importer, node),
        name => {
            let operation = Operation::from_onnx_operator(name)?;
            if operation.is_unary_elementwise() {
                lower_unary(importer, node, operation)
            } else {
                lower_binary(importer, node, operation)
            }
        }
    })
}

pub fn lower_reduction_family(importer: &mut Importer, node: &Node) -> Option<Result<()>> {
    Some(match node.operator_type.as_str() {
        "ReduceMean" => lower_reduction(importer, node, ReductionKind::Mean),
        "ReduceSum" => lower_reduction(importer, node, ReductionKind::Sum),
        "ReduceMax" => lower_reduction(importer, node, ReductionKind::Max),
        "ReduceMin" => lower_reduction(importer, node, ReductionKind::Min),
        "ReduceProd" => lower_reduction(importer, node, ReductionKind::Product),
        "ReduceL1" => lower_norm_reduction(importer, node, NormReductionKind::L1),
        "ReduceL2" => lower_norm_reduction(importer, node, NormReductionKind::L2),
        "ReduceSumSquare" => {
            lower_composed_sum_reduction(importer, node, ComposedSumReductionKind::SumSquare)
        }
        "ReduceLogSum" => {
            lower_composed_sum_reduction(importer, node, ComposedSumReductionKind::LogSum)
        }
        "ReduceLogSumExp" => {
            lower_composed_sum_reduction(importer, node, ComposedSumReductionKind::LogSumExp)
        }
        "ArgMax" => lower_arg_reduction(importer, node, Operation::ArgMax),
        "ArgMin" => lower_arg_reduction(importer, node, Operation::ArgMin),
        _ => return None,
    })
}
