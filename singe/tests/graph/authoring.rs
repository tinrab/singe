use std::collections::HashMap;

use singe::{
    DataType, Device, Result, Shape, Tensor,
    backend::registry::{Operation, OperationAttributes, TopKAttributes},
    dispatch,
    graph::{CaptureBindings, Graph, GraphPrepareOptions},
    nn::{Layer, Model, ModelState, StateBuilder},
    value::ValueSpec,
};

#[cfg(feature = "testing")]
use singe::testing::TensorActivity;

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
