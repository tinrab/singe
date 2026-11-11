use std::collections::HashMap;

use singe::{
    DataType, Device, Error, Shape, Tensor,
    graph::{CaptureBindings, Graph, GraphPrepareOptions, capture},
    nn::{Linear, Model, StateBuilder},
};

fn meta(shape: &[i64]) -> Tensor {
    Tensor::zeroed_on(Device::Meta, DataType::F32, Shape::create(shape).unwrap()).unwrap()
}

fn cpu(shape: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(Device::Cpu, Shape::create(shape).unwrap(), values).unwrap()
}

#[test]
fn test_meta_tensors_carry_metadata_without_values() {
    let tensor = meta(&[2, 3]);
    assert_eq!(tensor.device(), Device::Meta);
    assert_eq!(tensor.shape().dimensions(), &[2, 3]);
    assert_eq!(tensor.storage().byte_size(), 24);
    assert_eq!(tensor.storage().allocation_byte_size(), 0);
    assert!(matches!(tensor.to_vec::<f32>(), Err(Error::MetaTensorData)));
    assert!(matches!(
        tensor.to_device(Device::Cpu),
        Err(Error::MetaTensorData)
    ));

    let moved = cpu(&[2], vec![1., 2.]).to_device(Device::Meta).unwrap();
    assert_eq!(moved.device(), Device::Meta);
    assert_eq!(moved.shape().dimensions(), &[2]);
}

#[test]
fn test_models_built_on_meta_state_trace_programs_that_bind_real_state() {
    let mut state = StateBuilder::new();
    let weight = state.parameter("weight", meta(&[4, 3])).unwrap();
    let bias = state.parameter("bias", meta(&[4])).unwrap();
    let model = Model::new(Linear::new(weight, Some(bias)), state.finish().unwrap());
    let output = model.forward(&meta(&[2, 3])).unwrap();
    assert_eq!(output.device(), Device::Meta);
    assert_eq!(output.shape().dimensions(), &[2, 4]);

    let traced = model.capture("x", meta(&[2, 3])).unwrap();
    let program = traced.program();
    assert_eq!(
        program.signature().outputs()[0].spec().shape().dimensions(),
        &[2, 4]
    );
    let weights = (0..12).map(|value| value as f32).collect();
    let real = program
        .bind_state(&HashMap::from([
            ("weight".to_owned(), cpu(&[4, 3], weights)),
            ("bias".to_owned(), cpu(&[4], vec![1., 1., 1., 1.])),
        ]))
        .unwrap();
    let mut executor = program
        .prepare(&real, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let result = executor
        .run(&HashMap::from([(
            "x".to_owned(),
            cpu(&[2, 3], vec![1., 0., 0., 0., 1., 0.]),
        )]))
        .unwrap();
    assert_eq!(
        result[0].to_vec::<f32>().unwrap(),
        [1., 4., 7., 10., 2., 5., 8., 11.]
    );
}

#[test]
fn test_meta_literals_keep_their_values_through_placement() {
    let (graph, outputs) = Graph::capture([("x", meta(&[2, 2]))], |inputs| {
        Ok(vec![inputs[0].div_scalar(4.)?])
    })
    .unwrap();
    assert_eq!(outputs[0].device(), Device::Meta);
    let divisor = graph
        .values()
        .iter()
        .find_map(|value| graph.constant_tensor(value.id()))
        .unwrap();
    assert_eq!(divisor.device(), Device::Meta);
    assert_eq!(divisor.to_vec::<f32>().unwrap(), [4.]);

    let mut executor = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let result = executor
        .run(&HashMap::from([(
            "x".to_owned(),
            cpu(&[2, 2], vec![4., 8., 12., 16.]),
        )]))
        .unwrap();
    assert_eq!(result[0].to_vec::<f32>().unwrap(), [1., 2., 3., 4.]);
}

#[test]
fn test_meta_constants_without_values_cannot_leave_the_meta_device() {
    let (graph, _) = Graph::capture([("x", meta(&[2]))], |inputs| {
        let offset = capture::constant("offset", &meta(&[2]))?;
        Ok(vec![inputs[0].add(&offset)?])
    })
    .unwrap();
    assert!(matches!(
        graph.prepare(GraphPrepareOptions::new(Device::Cpu)),
        Err(Error::MetaTensorData)
    ));

    let offset = meta(&[2]);
    let bindings = CaptureBindings::new().with_buffer("offset", offset.clone());
    let (graph, _) = Graph::capture_with_bindings([("x", meta(&[2]))], bindings, |inputs| {
        Ok(vec![inputs[0].add(&offset)?])
    })
    .unwrap();
    let program = graph.program().unwrap();
    let state = program
        .bind_state(&HashMap::from([(
            "offset".to_owned(),
            cpu(&[2], vec![10., 20.]),
        )]))
        .unwrap();
    let mut executor = program
        .prepare(&state, GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    let result = executor
        .run(&HashMap::from([("x".to_owned(), cpu(&[2], vec![1., 2.]))]))
        .unwrap();
    assert_eq!(result[0].to_vec::<f32>().unwrap(), [11., 22.]);
}
