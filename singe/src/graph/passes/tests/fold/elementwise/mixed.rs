use super::*;

use crate::graph::node::NodeKind;

#[test]
fn test_fold_small_constants_pass_folds_cpu_softmax_where_and_quantization() {
    let vector = Shape::create(vec![2]).unwrap();
    let matrix = Shape::create(vec![1, 2]).unwrap();
    let logits = Tensor::from_vec(Device::Cpu, &matrix, vec![0.0_f32, 0.0]).unwrap();
    let softmax_output = logits.softmax(1).unwrap();
    let condition = Tensor::from_vec(Device::Cpu, &vector, vec![true, false]).unwrap();
    let x = Tensor::from_vec(Device::Cpu, &vector, vec![10.0_f32, 20.0]).unwrap();
    let y = Tensor::from_vec(Device::Cpu, &vector, vec![30.0_f32, 40.0]).unwrap();
    let where_output = Tensor::from_vec(Device::Cpu, &vector, vec![10.0_f32, 40.0]).unwrap();
    let quant_input = Tensor::from_vec(Device::Cpu, &vector, vec![-1.0_f32, 2.0]).unwrap();
    let quantized_output = quant_input.quantize(0.5, 0, DataType::I8).unwrap();
    let dequantized_output = quantized_output.dequantize(0.5, 0).unwrap();
    let mut graph = Graph::new();
    let logits = graph.add_constant(logits);
    let softmax = add_dimension_operation(
        &mut graph,
        RegistryOperation::Softmax,
        logits,
        1,
        &softmax_output,
    );
    let condition = graph.add_constant(condition);
    let x = graph.add_constant(x);
    let y = graph.add_constant(y);
    let where_node = graph
        .add_named_operation(
            None,
            NodeKind::where_operation(condition, x, y),
            where_output.data_type(),
            where_output.shape().clone(),
            where_output.device(),
        )
        .unwrap();
    let quant_input = graph.add_constant(quant_input);
    let quantized = add_quantization(
        &mut graph,
        RegistryOperation::Quantize,
        quant_input,
        &quantized_output,
    );
    let dequantized = add_quantization(
        &mut graph,
        RegistryOperation::Dequantize,
        quantized,
        &dequantized_output,
    );
    graph
        .set_outputs(vec![softmax, where_node, dequantized])
        .unwrap();

    let folded = fold_small_constants(&graph, 13);
    assert_folded_constants(
        &folded,
        [
            (softmax, &[0.5, 0.5][..]),
            (where_node, &[10.0, 40.0]),
            (dequantized, &[-1.0, 2.0]),
        ],
    );
}
