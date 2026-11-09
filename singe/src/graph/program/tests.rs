#![cfg(all(feature = "testing", feature = "onnx"))]

use std::collections::HashMap;

use crate::{
    data_type::DataType,
    device::Device,
    graph::{Graph, GraphInputSpec, GraphPrepareOptions},
    nn::StateBuilder,
    shape::Shape,
    storage::cpu::CpuStorage,
    tensor::Tensor,
    value::ValueSpec,
};

#[test]
fn test_runtime_shape_specialization_does_not_reread_validated_state_values() {
    let mut state = StateBuilder::new();
    state
        .buffer(
            "guard",
            Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![1_i64]).unwrap(),
        )
        .unwrap();
    let state = state.finish().unwrap();
    let meta = state.to(Device::Meta, None).unwrap();
    let spec = GraphInputSpec::new(ValueSpec::new(
        DataType::F32,
        Shape::create([2, 0]).unwrap(),
    ))
    .with_dimension(0, "batch", Some(1), Some(8))
    .unwrap();
    let graph = Graph::build([("x", spec)], meta.capture_bindings().unwrap(), |inputs| {
        let batch = inputs[0].dimension(0)?;
        Ok(vec![inputs[0].reshape_with(&[batch, 0.into()])?])
    })
    .unwrap();
    let program = state
        .program(&graph)
        .unwrap()
        .specialize_state(["guard".into()], state.snapshot())
        .unwrap();
    let mut executor = program
        .prepare(
            &state.snapshot().clone(),
            GraphPrepareOptions::new(Device::Cpu),
        )
        .unwrap();
    let input = Tensor::from_vec(
        Device::Cpu,
        Shape::create([3, 0]).unwrap(),
        Vec::<f32>::new(),
    )
    .unwrap();
    let exports = CpuStorage::copy_stats().exports;
    executor.run(&HashMap::from([("x".into(), input)])).unwrap();
    assert_eq!(CpuStorage::copy_stats().exports, exports);
    let changed = state
        .replace(&HashMap::from([(
            "guard".into(),
            Tensor::from_vec(Device::Cpu, Shape::scalar(), vec![2_i64]).unwrap(),
        )]))
        .unwrap();
    assert!(
        program
            .prepare(
                &changed.snapshot().clone(),
                GraphPrepareOptions::new(Device::Cpu)
            )
            .is_err()
    );
}
