use std::{
    collections::HashMap,
    ptr,
    sync::{Arc, Barrier},
    thread,
};

use singe::{
    DataType, Device, Error, Shape, Tensor,
    backend::registry::{FlattenAttributes, Operation, OperationAttributes, TransposeAttributes},
    graph::{
        Graph, GraphFingerprintOptions, GraphOperationInputs, GraphOperationOutput,
        GraphOperationRequest, GraphPrepareOptions,
    },
};

fn tensor(dimensions: &[i64], values: &[f32]) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values.to_vec(),
    )
    .unwrap()
}

#[test]
fn test_prepared_runs_share_analysis_and_keep_previous_results() {
    let (captured, _) = Graph::capture(
        [
            ("x", tensor(&[2, 2], &[1., 2., 3., 4.])),
            ("weight", tensor(&[2, 2], &[1., -1., 2., 1.])),
            ("bias", tensor(&[2], &[1., 2.])),
        ],
        |inputs| {
            let hidden = inputs[0].matmul(&inputs[1])?.add(&inputs[2])?.relu()?;
            let output = hidden.transpose(0, 1)?.reshape(vec![4])?;
            Ok(vec![output.clone(), output])
        },
    )
    .unwrap();
    let mut graph = captured;
    graph
        .add_static_initializer("weight", tensor(&[2, 2], &[1., -1., 2., 1.]))
        .unwrap();
    graph
        .add_initializer("bias", tensor(&[2], &[1., 2.]))
        .unwrap();
    let prepared = graph.compile().unwrap();
    let independent = prepared.clone();
    assert!(ptr::eq(prepared.ir(), independent.ir()));

    let first = prepared
        .run(&HashMap::from([(
            "x".into(),
            tensor(&[2, 2], &[1., 2., 3., 4.]),
        )]))
        .unwrap();
    let second = independent
        .run(&HashMap::from([(
            "x".into(),
            tensor(&[2, 2], &[0., 1., -1., 2.]),
        )]))
        .unwrap();
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [6., 12., 3., 3.]);
    assert_eq!(second[0].to_vec::<f32>().unwrap(), [3., 4., 3., 5.]);
    assert_eq!(first[0].to_vec::<f32>().unwrap(), [6., 12., 3., 3.]);
    assert_eq!(first[0].id(), first[1].id());
    assert!(ptr::eq(prepared.ir(), independent.ir()));

    let bad = HashMap::from([("x".into(), tensor(&[4], &[1., 2., 3., 4.]))]);
    assert!(prepared.run(&bad).is_err());
}

#[test]
fn test_optimization_preserves_unused_input_validation_and_defaults() {
    let (captured, _) = Graph::capture(
        [
            ("empty", tensor(&[0, 3], &[])),
            ("unused", tensor(&[1], &[2.])),
        ],
        |inputs| Ok(vec![inputs[0].sum(&[1], false)?]),
    )
    .unwrap();
    let mut graph = captured;
    graph
        .add_initializer("unused", tensor(&[1], &[7.]))
        .unwrap();
    let prepared = graph.compile().unwrap();
    assert_eq!(prepared.signature().inputs().len(), 2);
    let input = HashMap::from([("empty".into(), tensor(&[0, 3], &[]))]);
    assert_eq!(
        prepared.run(&input).unwrap()[0].to_vec::<f32>().unwrap(),
        [] as [f32; 0]
    );
    assert!(matches!(
        prepared.run(&HashMap::new()),
        Err(Error::MissingInput { .. })
    ));
    let bad_unused = HashMap::from([
        ("empty".into(), tensor(&[0, 3], &[])),
        ("unused".into(), tensor(&[2], &[1., 2.])),
    ]);
    assert!(prepared.run(&bad_unused).is_err());
    let bad_empty = HashMap::from([("empty".into(), tensor(&[0, 4], &[]))]);
    assert!(prepared.run(&bad_empty).is_err());
}

#[test]
fn test_prepared_state_is_an_immutable_snapshot() {
    let (captured, _) = Graph::capture([("weight", tensor(&[2], &[1., 2.]))], |inputs| {
        Ok(vec![inputs[0].mul_scalar(2.)?])
    })
    .unwrap();
    let mut graph = captured;
    graph
        .add_static_initializer("weight", tensor(&[2], &[1., 2.]))
        .unwrap();
    let old = graph.compile().unwrap();
    graph
        .add_static_initializer("weight", tensor(&[2], &[4., 5.]))
        .unwrap();
    let new = graph.compile().unwrap();
    assert_eq!(
        old.run(&HashMap::new()).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [2., 4.]
    );
    assert_eq!(
        new.run(&HashMap::new()).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [8., 10.]
    );
    assert_ne!(
        old.fingerprint_with_options(&GraphFingerprintOptions::default())
            .unwrap(),
        new.fingerprint_with_options(&GraphFingerprintOptions::default())
            .unwrap()
    );
    assert!(
        old.run(&HashMap::from([("weight".into(), tensor(&[2], &[4., 5.]))]))
            .is_err()
    );
}

#[test]
fn test_names_and_duplicate_projections_survive_preparation() {
    let mut graph = Graph::new();
    let input = graph.add_input(
        "x",
        DataType::F32,
        Shape::create(vec![2]).unwrap(),
        Device::Cpu,
    );
    graph
        .set_named_output_projections(vec![
            ("second".into(), input, 0),
            ("first".into(), input, 0),
        ])
        .unwrap();
    let prepared = graph.compile().unwrap();
    assert_eq!(
        prepared
            .signature()
            .outputs()
            .iter()
            .map(|output| output.name())
            .collect::<Vec<_>>(),
        [Some("second"), Some("first")]
    );
    let outputs = prepared
        .run(&HashMap::from([("x".into(), tensor(&[2], &[2., 3.]))]))
        .unwrap();
    assert_eq!(outputs[0].id(), outputs[1].id());
    assert_eq!(outputs[1].to_vec::<f32>().unwrap(), [2., 3.]);
    graph
        .set_named_output_projections(vec![
            ("renamed".into(), input, 0),
            ("first".into(), input, 0),
        ])
        .unwrap();
    assert_ne!(
        prepared
            .fingerprint_with_options(&GraphFingerprintOptions::default())
            .unwrap(),
        graph.fingerprint().unwrap()
    );
}

#[test]
fn test_symbolic_guards_are_reused_and_validate_unused_input_equalities() {
    let (captured, _) = Graph::capture(
        [
            ("x", tensor(&[2, 3], &[1.; 6])),
            ("unused", tensor(&[2, 3], &[0.; 6])),
        ],
        |inputs| Ok(vec![inputs[0].mul_scalar(2.)?]),
    )
    .unwrap();
    let mut graph = captured;
    for name in ["x", "unused"] {
        graph
            .add_input_dimension_guard(name, 0, "batch", Some(1), Some(8))
            .unwrap();
    }
    let prepared = graph.compile().unwrap();
    let output = prepared
        .run(&HashMap::from([
            ("x".into(), tensor(&[5, 3], &[3.; 15])),
            ("unused".into(), tensor(&[5, 3], &[0.; 15])),
        ]))
        .unwrap();
    assert_eq!(output[0].shape().dimensions(), &[5, 3]);
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [6.; 15]);
    let mismatched = HashMap::from([
        ("x".into(), tensor(&[5, 3], &[3.; 15])),
        ("unused".into(), tensor(&[4, 3], &[0.; 12])),
    ]);
    assert!(prepared.run(&mismatched).is_err());
}

fn exercise_concurrent_instances(device: Device) {
    let (captured, _) = Graph::capture([("x", tensor(&[2], &[1., 2.]))], |inputs| {
        Ok(vec![inputs[0].mul_scalar(3.)?])
    })
    .unwrap();
    let first = captured.prepare(GraphPrepareOptions::new(device)).unwrap();
    let analysis = first.analysis().clone();
    let plan = first.execution_plan();
    let barrier = Arc::new(Barrier::new(3));
    thread::scope(|scope| {
        let workers = (0..3)
            .map(|index| {
                let barrier = Arc::clone(&barrier);
                let analysis = analysis.clone();
                let plan = &plan;
                scope.spawn(move || {
                    barrier.wait();
                    let mut instance = plan.create_instance().unwrap();
                    assert!(ptr::eq(instance.analysis().ir(), analysis.ir()));
                    let mut retained = None;
                    for step in 0..4 {
                        let value = (index * 10 + step) as f32;
                        let input = tensor(&[2], &[value, value + 1.])
                            .to_device(device)
                            .unwrap();
                        let outputs = instance.run(&HashMap::from([("x".into(), input)])).unwrap();
                        assert_eq!(
                            outputs[0].to_vec::<f32>().unwrap(),
                            [3. * value, 3. * (value + 1.)]
                        );
                        if step == 0 {
                            retained = Some(outputs[0].clone());
                        }
                    }
                    let value = (index * 10) as f32;
                    assert_eq!(
                        retained.unwrap().to_vec::<f32>().unwrap(),
                        [3. * value, 3. * (value + 1.)]
                    );
                    assert_eq!(instance.runs(), 4);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}

#[test]
fn test_cpu_instances_share_analysis_while_running_concurrently() {
    exercise_concurrent_instances(Device::Cpu);
}

fn exercise_dynamic_flatten_materialization(device: Device) {
    let mut graph = Graph::new();
    let input = graph.add_input(
        "x",
        DataType::F32,
        Shape::create([1, 3]).unwrap(),
        Device::Cpu,
    );
    let transposed = graph
        .add_operation_request(GraphOperationRequest::new(
            None,
            GraphOperationInputs::new(
                Operation::Transpose,
                OperationAttributes::Transpose(TransposeAttributes { dim0: 0, dim1: 1 }),
                vec![input],
            ),
            GraphOperationOutput::new(DataType::F32, Shape::create([3, 1]).unwrap(), Device::Cpu),
        ))
        .unwrap();
    let output = graph
        .add_operation_request(GraphOperationRequest::new(
            None,
            GraphOperationInputs::new(
                Operation::Flatten,
                OperationAttributes::Flatten(FlattenAttributes {
                    start_dim: 0,
                    end_dim: 1,
                }),
                vec![transposed],
            ),
            GraphOperationOutput::new(DataType::F32, Shape::create([3]).unwrap(), Device::Cpu),
        ))
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();
    graph
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(2))
        .unwrap();
    let mut executor = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let original = executor.analysis().clone();
    for (batch, expected, copies) in [
        (1, vec![0., 1., 2.], 0),
        (2, vec![0., 3., 1., 4., 2., 5.], 1),
        (1, vec![0., 1., 2.], 0),
    ] {
        let values = (0..batch * 3).map(|value| value as f32).collect::<Vec<_>>();
        let input = tensor(&[batch, 3], &values).to_device(device).unwrap();
        let output = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
        assert_eq!(output[0].to_vec::<f32>().unwrap(), expected);
        assert_eq!(
            executor
                .analysis()
                .ir()
                .operations()
                .iter()
                .filter(|op| { op.opcode() == Some(Operation::Contiguous) })
                .count(),
            copies
        );
    }
    assert!(ptr::eq(original.ir(), executor.analysis().ir()));
    assert_eq!(executor.cache_stats().analysis_builds(), 2);
}

#[test]
fn test_cpu_shape_specialization_materializes_only_incompatible_flatten() {
    exercise_dynamic_flatten_materialization(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_shape_specialization_materializes_only_incompatible_flatten() {
    exercise_dynamic_flatten_materialization(Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_instances_share_analysis_while_running_concurrently() {
    exercise_concurrent_instances(Device::Cuda(0));
}
