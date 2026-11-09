use super::*;

use crate::graph::node::NodeKind;

#[test]
fn test_fold_small_constants_pass_folds_cpu_shape_expressions() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let base =
        Tensor::from_vec(Device::Cpu, &shape, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reshaped_output = base.reshape(vec![3, 2]).unwrap();
    let sliced_output = reshaped_output.narrow(0, 1, 2).unwrap();
    let squeezed_output = sliced_output.unsqueeze(0).unwrap().squeeze(0).unwrap();
    let mut graph = Graph::new();
    let constant = graph.add_constant(base);
    let reshaped = graph
        .add_named_operation(
            None,
            NodeKind::reshape(constant, Shape::create(vec![3, 2]).unwrap()),
            reshaped_output.data_type(),
            Shape::create(vec![3, 2]).unwrap(),
            reshaped_output.device(),
        )
        .unwrap();
    let sliced = graph
        .add_named_slice_operation(
            None,
            reshaped,
            0,
            1,
            2,
            sliced_output.data_type(),
            sliced_output.shape(),
            sliced_output.device(),
        )
        .unwrap();
    let unsqueezed_output = squeezed_output.unsqueeze(0).unwrap();
    let unsqueezed = add_dimension_operation(
        &mut graph,
        RegistryOperation::Unsqueeze,
        sliced,
        0,
        &unsqueezed_output,
    );
    let squeezed = add_dimension_operation(
        &mut graph,
        RegistryOperation::Squeeze,
        unsqueezed,
        0,
        &squeezed_output,
    );
    graph.set_outputs(vec![squeezed]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::FoldSmallConstants,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (folded, results) = pipeline.normalize(&graph).unwrap();

    assert_eq!(results[0].pass_name, "fold_small_constants");
    assert_eq!(results[0].checked_nodes, 5);
    let tensor = folded_constant_tensor(&folded, squeezed);
    assert_eq!(tensor.shape().dimensions(), &[2, 2]);
    assert_close!(&tensor.to_vec::<f32>().unwrap(), &[3.0, 4.0, 5.0, 6.0]);
}
