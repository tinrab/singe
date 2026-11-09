use super::*;
use crate::{backend::registry::ClampAttributes, graph::node::NodeKind};

#[test]
fn test_infer_value_metadata_pass_validates_clamp_bounds() {
    let shape = Shape::create(vec![2, 3]).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape.clone(), Device::Cpu);
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, shape).unwrap();
    let clamp = add_operation_fixture(
        &mut graph,
        Operation::Clamp,
        OperationAttributes::Clamp(ClampAttributes { min: 2., max: 1. }),
        vec![input],
        &output,
    );
    graph.set_outputs(vec![clamp]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![GraphPassKind::InferValueMetadata]);
    let error = pipeline.normalize(&graph).unwrap_err();

    assert!(matches!(
        error,
        Error::InvalidArgument { ref name } if name == "clamp bounds"
    ));
}

#[test]
fn test_infer_value_metadata_pass_infers_broadcast_shape() {
    let lhs_shape = Shape::create(vec![2, 1]).unwrap();
    let rhs_shape = Shape::create(vec![1, 3]).unwrap();
    let output_shape = Shape::create(vec![2, 3]).unwrap();
    let output = Tensor::zeroed_on(Device::Cpu, DataType::F32, &output_shape).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F32, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F32, rhs_shape, Device::Cpu);
    let add = add_binary_operation(&mut graph, RegistryOperation::Add, [lhs, rhs], &output);
    replace_with_stale_shape_metadata(&mut graph, add);
    graph.set_outputs(vec![add]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        add,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}

#[test]
fn test_infer_value_metadata_pass_infers_lerp_broadcast_shape() {
    let lhs_shape = Shape::create(vec![2, 1, 4]).unwrap();
    let rhs_shape = Shape::create(vec![1, 3, 4]).unwrap();
    let output_shape = Shape::create(vec![2, 3, 4]).unwrap();
    let mut graph = Graph::new();
    let lhs = graph.add_input("lhs", DataType::F32, lhs_shape, Device::Cpu);
    let rhs = graph.add_input("rhs", DataType::F32, rhs_shape, Device::Cpu);
    let lerp = graph
        .add_named_operation(
            None,
            NodeKind::lerp(lhs, rhs, 0.25),
            DataType::F32,
            output_shape.clone(),
            Device::Cpu,
        )
        .unwrap();
    replace_with_stale_shape_metadata(&mut graph, lerp);
    graph.set_outputs(vec![lerp]).unwrap();

    let pipeline = GraphPassPipeline::new(vec![
        GraphPassKind::InferValueMetadata,
        GraphPassKind::ValidateValueMetadata,
        GraphPassKind::ValidateNodeContracts,
        GraphPassKind::ValidateOutputs,
    ]);
    let (normalized, _) = pipeline.normalize(&graph).unwrap();

    assert_value_metadata(
        &normalized,
        lerp,
        DataType::F32,
        output_shape.dimensions(),
        output_shape.dimensions(),
        Device::Cpu,
    );
}
