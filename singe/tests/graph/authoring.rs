use std::collections::HashMap;

use singe::{
    DataType, Device, Result, Shape, Tensor,
    backend::registry::{Operation, OperationAttributes, TopKAttributes, TransposeAttributes},
    dispatch,
    graph::{CaptureBindings, Graph, GraphPrepareOptions},
    nn::{Layer, Model, ModelState, StateBuilder},
    value::ValueSpec,
};

#[cfg(feature = "testing")]
use singe::testing::TensorActivity;

#[cfg(feature = "testing")]
#[test]
fn test_activity_counters_observe_real_tensor_storage_and_kernel_invocations() {
    let before = TensorActivity::current();
    let input =
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![-1_f32, 2.]).unwrap();
    let created = TensorActivity::current();
    assert!(created.descriptors > before.descriptors);
    assert!(created.storage_requests > before.storage_requests);
    input.relu().unwrap();
    assert!(TensorActivity::current().checked_invocations > created.checked_invocations);
}

#[derive(Debug)]
struct TopIndices;

impl Layer for TopIndices {
    fn forward(&self, input: &Tensor, _state: &ModelState) -> Result<Tensor> {
        let attributes = OperationAttributes::TopK(TopKAttributes {
            k: 1,
            dim: -1,
            largest: true,
            sorted: true,
        });
        // A mistaken single-result invocation must reject before eager dispatch.
        #[cfg(feature = "testing")]
        let before = TensorActivity::current();
        assert!(dispatch::call_one(Operation::TopK, attributes.clone(), &[input]).is_err());
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current(), before);
        Ok(dispatch::call(Operation::TopK, attributes, &[input])?.remove(1))
    }
}

#[test]
fn test_custom_layers_share_ordered_invocations_between_eager_capture_and_metadata() {
    let model = Model::new(TopIndices, StateBuilder::new().finish().unwrap());
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![1_f32, 5., 2., 9., 3., 6.],
    )
    .unwrap();
    assert_eq!(
        model.forward(&input).unwrap().to_vec::<i64>().unwrap(),
        [1, 0]
    );
    for program in [
        model.build("x", input.value_spec()).unwrap(),
        model.capture("x", input.clone()).unwrap(),
    ] {
        let mut executor = program
            .prepare(GraphPrepareOptions::new(Device::Cpu))
            .unwrap();
        assert_eq!(
            executor
                .run(&HashMap::from([("x".into(), input.clone())]))
                .unwrap()[0]
                .to_vec::<i64>()
                .unwrap(),
            [1, 0]
        );
    }
}

#[test]
fn test_ordered_results_and_named_outputs_survive_building_from_metadata() {
    let graph = Graph::build(
        [(
            "x",
            ValueSpec::new(DataType::F32, Shape::create([2, 3]).unwrap()),
        )],
        CaptureBindings::new(),
        |inputs| {
            let transposed = dispatch::call_one(
                Operation::Transpose,
                OperationAttributes::Transpose(TransposeAttributes { dim0: 0, dim1: 1 }),
                &[&inputs[0]],
            )?;
            let results = dispatch::call(
                Operation::TopK,
                OperationAttributes::TopK(TopKAttributes {
                    k: 1,
                    dim: 0,
                    largest: true,
                    sorted: true,
                }),
                &[&transposed],
            )?;
            assert_eq!(results.len(), 2);
            assert_eq!(results[0].data_type(), DataType::F32);
            assert_eq!(results[1].data_type(), DataType::I64);
            let cast = dispatch::call_one(
                Operation::Cast,
                OperationAttributes::Cast(DataType::F32),
                &[&results[1]],
            )?;
            let sum = dispatch::call_one(
                Operation::Add,
                OperationAttributes::None,
                &[&results[0], &cast],
            )?;
            Ok(vec![sum, results[1].clone(), results[0].clone()])
        },
    )
    .unwrap()
    .with_output_names(["sum", "indices", "values"])
    .unwrap();
    let program = graph.program().unwrap();
    let mut executor = program
        .prepare(
            &program.bind_state(&HashMap::new()).unwrap(),
            GraphPrepareOptions::new(Device::Cpu),
        )
        .unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([2, 3]).unwrap(),
        vec![1_f32, 5., 2., 9., 3., 6.],
    )
    .unwrap();
    let output = executor.run(&HashMap::from([("x".into(), input)])).unwrap();
    assert_eq!(output[0].to_vec::<f32>().unwrap(), [6., 9.]);
    assert_eq!(output[1].to_vec::<i64>().unwrap(), [1, 0]);
    assert_eq!(output[2].to_vec::<f32>().unwrap(), [5., 9.]);
    assert_eq!(program.signature().outputs()[0].name(), Some("sum"));
}

#[test]
fn test_invalid_metadata_fails_while_building() {
    let spec = ValueSpec::new(DataType::F16, Shape::create([2, 3]).unwrap());
    let mismatched = Graph::build([("x", spec.clone())], CaptureBindings::new(), |inputs| {
        Ok(vec![inputs[0].matmul(&inputs[0])?])
    });
    assert!(mismatched.is_err());
    let repeated = Graph::build(
        [("x", spec.clone()), ("x", spec.clone())],
        CaptureBindings::new(),
        |inputs| Ok(vec![inputs[0].clone()]),
    );
    assert!(repeated.is_err());
    let empty = Graph::build([("x", spec)], CaptureBindings::new(), |_| Ok(Vec::new()));
    assert!(empty.is_err());
    let names = Graph::build(
        [(
            "x",
            ValueSpec::new(DataType::F16, Shape::create([2, 3]).unwrap()),
        )],
        CaptureBindings::new(),
        |inputs| Ok(vec![inputs[0].clone()]),
    )
    .unwrap();
    assert!(names.clone().with_output_names(["a", "b"]).is_err());
    assert!(names.with_output_names([""]).is_err());
}
