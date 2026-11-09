use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    backend::registry::{
        BatchNormalizationAttributes, Operation, OperationAttributes, OperationRequest,
        ReductionAttributes, ResultAlias, TopKAttributes, TransposeAttributes,
    },
    graph::{
        Graph, GraphOperationInputs, GraphOperationOutput, GraphOperationRequest,
        GraphPrepareOptions,
    },
    scalar::Scalar,
    value::ValueSpec,
};

fn value(data_type: DataType, dimensions: &[i64]) -> ValueSpec {
    ValueSpec::new(data_type, Shape::create(dimensions.to_vec()).unwrap())
}

#[test]
fn test_canonical_builder_payloads_use_pure_inference_and_reject_invalid_insertions_atomically() {
    let mut graph = Graph::new();
    let lhs = graph.add_input(
        "lhs",
        DataType::F32,
        Shape::create([2, 1]).unwrap(),
        Device::Cpu,
    );
    let rhs = graph.add_input("rhs", DataType::F64, Shape::scalar(), Device::Cpu);
    let operands = vec![lhs, rhs];
    let invalid = graph.add_operation_request(GraphOperationRequest::new(
        None,
        GraphOperationInputs::new(
            Operation::Add,
            OperationAttributes::Dimension(0),
            operands.clone(),
        ),
        GraphOperationOutput::new(DataType::F32, Shape::create([2, 1]).unwrap(), Device::Cpu),
    ));
    assert!(invalid.is_err());
    let output_mismatch = graph.add_operation_request(GraphOperationRequest::new(
        None,
        GraphOperationInputs::new(Operation::Add, OperationAttributes::None, operands.clone()),
        GraphOperationOutput::new(DataType::F64, Shape::create([2, 1]).unwrap(), Device::Cpu),
    ));
    assert!(output_mismatch.is_err());
    let output = graph
        .add_operation_request(GraphOperationRequest::new(
            Some("promoted add".into()),
            GraphOperationInputs::new(Operation::Add, OperationAttributes::None, operands),
            GraphOperationOutput::new(DataType::F32, Shape::create([2, 1]).unwrap(), Device::Cpu),
        ))
        .unwrap();
    // Neither rejected insertion consumes a graph-local node identity.
    assert_eq!(output.index(), 2);
    graph.set_outputs(vec![output]).unwrap();
    let mut executor = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let inputs = HashMap::from([
        (
            "lhs".into(),
            Tensor::from_vec(Device::Cpu, Shape::create([2, 1]).unwrap(), vec![1_f32, 2.]).unwrap(),
        ),
        ("rhs".into(), Tensor::scalar(Device::Cpu, 0.5_f64).unwrap()),
    ]);
    let results = executor.run(&inputs).unwrap();
    assert_eq!(results[0].data_type(), DataType::F32);
    assert_eq!(results[0].to_vec::<f32>().unwrap(), [1.5, 2.5]);
    let operation = executor
        .analysis()
        .ir()
        .operations()
        .iter()
        .find(|operation| operation.opcode() == Some(Operation::Add))
        .unwrap();
    assert_eq!(operation.attributes(), &OperationAttributes::None);
}

#[test]
fn test_graph_construction_rejects_state_mutation_even_for_empty_outputs() {
    let mut graph = Graph::new();
    let input = graph.add_input(
        "x",
        DataType::F32,
        Shape::create([0, 2]).unwrap(),
        Device::Cpu,
    );
    let mean = graph.add_input(
        "mean",
        DataType::F32,
        Shape::create([2]).unwrap(),
        Device::Cpu,
    );
    let result = graph.add_operation_request(GraphOperationRequest::batch_normalization(
        None,
        input,
        None,
        None,
        Some(mean),
        None,
        true,
        0.1,
        1e-5,
        GraphOperationOutput::new(DataType::F32, Shape::create([0, 2]).unwrap(), Device::Cpu),
    ));
    assert!(result.is_err());
    graph.set_outputs(vec![input]).unwrap();
    assert_eq!(graph.compile().unwrap().value_order().len(), 1);
}

#[test]
fn test_ordered_topk_specs_need_no_storage_even_for_unallocatable_inputs() {
    let input = value(DataType::F32, &[1_i64 << 40, 4]);
    let checked = OperationRequest::new(
        Operation::TopK,
        OperationAttributes::TopK(TopKAttributes {
            k: 2,
            dim: 1,
            largest: true,
            sorted: true,
        }),
        vec![input],
    )
    .check()
    .unwrap();
    let outputs = checked.inference().outputs();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape().dimensions(), &[1_i64 << 40, 2]);
    assert_eq!(outputs[0].data_type(), DataType::F32);
    assert_eq!(outputs[1].data_type(), DataType::I64);
    assert_eq!(
        checked.inference().aliases(),
        [ResultAlias::Independent, ResultAlias::Independent]
    );
}

#[test]
fn test_typed_attributes_operands_and_index_roles_are_checked_before_execution() {
    assert!(
        OperationRequest::new(
            Operation::Add,
            OperationAttributes::Dimension(0),
            vec![value(DataType::F32, &[2]), value(DataType::F32, &[2])]
        )
        .infer()
        .is_err()
    );
    assert!(
        OperationRequest::new(
            Operation::Add,
            OperationAttributes::None,
            vec![value(DataType::F32, &[2])]
        )
        .infer()
        .is_err()
    );
    assert!(
        OperationRequest::new(
            Operation::IndexSelect,
            OperationAttributes::IndexSelect(0),
            vec![value(DataType::F32, &[2, 3]), value(DataType::F32, &[1])]
        )
        .infer()
        .is_err()
    );
    assert!(
        OperationRequest::new(
            Operation::Where,
            OperationAttributes::None,
            vec![
                value(DataType::I64, &[2]),
                value(DataType::F32, &[2]),
                value(DataType::F32, &[2])
            ]
        )
        .infer()
        .is_err()
    );
}

#[test]
fn test_common_requests_apply_promotion_and_preserve_logical_aliases() {
    let arithmetic = OperationRequest::new(
        Operation::Add,
        OperationAttributes::None,
        vec![value(DataType::F32, &[2, 1]), value(DataType::F64, &[])],
    )
    .infer()
    .unwrap();
    assert_eq!(arithmetic.outputs()[0], value(DataType::F32, &[2, 1]));
    let scalar = OperationRequest::new(
        Operation::Scale,
        OperationAttributes::Scalar(Scalar::Float(1.25)),
        vec![value(DataType::I64, &[2])],
    )
    .infer()
    .unwrap();
    assert_eq!(scalar.outputs()[0].data_type(), DataType::F32);
    let view = OperationRequest::new(
        Operation::Transpose,
        OperationAttributes::Transpose(TransposeAttributes { dim0: 0, dim1: 1 }),
        vec![value(DataType::F32, &[2, 3])],
    )
    .infer()
    .unwrap();
    assert_eq!(view.outputs()[0], value(DataType::F32, &[3, 2]));
    assert_eq!(view.aliases(), [ResultAlias::ViewOf(0)]);
    let copy = OperationRequest::new(
        Operation::Copy,
        OperationAttributes::None,
        vec![value(DataType::I64, &[2])],
    )
    .infer()
    .unwrap();
    assert_eq!(copy.aliases(), [ResultAlias::Independent]);
}

#[test]
fn test_inference_rejects_state_mutation_and_invalid_reduction_axes() {
    let training = OperationRequest::new(
        Operation::BatchNorm,
        OperationAttributes::BatchNormalization(BatchNormalizationAttributes {
            has_weight: false,
            has_bias: false,
            has_running_mean: true,
            has_running_var: false,
            training: true,
            momentum: 0.1,
            eps: 1e-5,
        }),
        vec![value(DataType::F32, &[2, 3]), value(DataType::F32, &[3])],
    );
    assert!(training.infer().is_err());
    let duplicate = OperationRequest::new(
        Operation::Sum,
        OperationAttributes::Reduction(ReductionAttributes {
            dimensions: vec![1, -1],
            keepdim: false,
        }),
        vec![value(DataType::F32, &[2, 3])],
    );
    assert!(duplicate.infer().is_err());
    let empty_max = OperationRequest::new(
        Operation::Max,
        OperationAttributes::Reduction(ReductionAttributes {
            dimensions: vec![1],
            keepdim: false,
        }),
        vec![value(DataType::F32, &[2, 0])],
    );
    assert!(empty_max.infer().is_err());
}
