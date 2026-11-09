use std::{collections::HashMap, ptr};

use singe::{
    Device, Shape, Tensor,
    graph::{Graph, GraphPrepareOptions},
};

fn tensor(dimensions: &[i64], values: Vec<f32>) -> Tensor {
    Tensor::from_vec(
        Device::Cpu,
        Shape::create(dimensions.to_vec()).unwrap(),
        values,
    )
    .unwrap()
}

fn graph() -> Graph {
    let weight = tensor(&[3, 2], vec![1., -1., 2., 1., 0.5, 2.]);
    let bias = tensor(&[2], vec![1., -2.]);
    let (capture, _) = Graph::capture(
        [
            ("x", tensor(&[2, 3], vec![1.; 6])),
            ("weight", weight.clone()),
            ("bias", bias.clone()),
        ],
        |inputs| {
            Ok(vec![
                inputs[0]
                    .matmul(&inputs[1])?
                    .add(&inputs[2])?
                    .relu()?
                    .transpose(0, 1)?,
            ])
        },
    )
    .unwrap();
    let mut graph = capture;
    graph.add_parameter("weight", weight).unwrap();
    graph.add_buffer("bias", bias).unwrap();
    graph
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph
}

fn inputs(device: Device, batch: usize, value: f32) -> HashMap<String, Tensor> {
    HashMap::from([(
        "x".into(),
        tensor(&[i64::try_from(batch).unwrap(), 3], vec![value; batch * 3])
            .to_device(device)
            .unwrap(),
    )])
}

fn exercise_cache(device: Device) {
    let mut executor = graph()
        .prepare(GraphPrepareOptions::new(device).with_cache_capacity(2))
        .unwrap();
    let original = executor.analysis().clone();
    let mut retained = Vec::new();
    for (step, batch) in [2_usize, 5, 2, 7, 5].into_iter().enumerate() {
        let value = (step + 1) as f32;
        let outputs = executor.run(&inputs(device, batch, value)).unwrap();
        let expected = [
            vec![3.5f32.mul_add(value, 1.); batch],
            vec![2.0f32.mul_add(value, -2.).max(0.); batch],
        ]
        .concat();
        assert_eq!(
            outputs[0].shape().dimensions(),
            &[2, i64::try_from(batch).unwrap()]
        );
        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), expected);
        retained.push((outputs[0].clone(), expected));
        if step == 2 {
            assert!(ptr::eq(original.ir(), executor.analysis().ir()));
        }
    }
    let stats = executor.cache_stats();
    assert_eq!(stats.entries(), 2);
    assert_eq!(stats.analysis_builds(), 4);
    assert_eq!(stats.misses(), 4);
    assert_eq!(stats.hits(), 2);
    assert_eq!(stats.evictions(), 2);
    let active = executor.analysis().clone();
    executor.run(&inputs(device, 5, 9.)).unwrap();
    assert!(ptr::eq(active.ir(), executor.analysis().ir()));
    let before = executor.cache_stats();
    assert!(executor.run(&inputs(device, 9, 1.)).is_err());
    assert_eq!(before, executor.cache_stats());
    for (output, expected) in retained {
        assert_eq!(output.to_vec::<f32>().unwrap(), expected);
    }
}

#[test]
fn test_cpu_repeated_shapes_reuse_analysis_and_evict_only_inactive_plans() {
    exercise_cache(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_repeated_shapes_reuse_analysis_and_evict_only_inactive_plans() {
    exercise_cache(Device::Cuda(0));
}

#[test]
fn test_zero_cache_capacity_is_rejected_during_preparation() {
    assert!(
        graph()
            .prepare(GraphPrepareOptions::new(Device::Cpu).with_cache_capacity(0))
            .is_err()
    );
}

#[test]
fn test_cpu_memory_reports_deduplicate_shared_state_across_cached_plans() {
    let mut executor = graph()
        .prepare(GraphPrepareOptions::new(Device::Cpu).with_cache_capacity(2))
        .unwrap();
    let first = executor.memory_stats().unwrap();
    assert_eq!(first.shared_state_allocation_bytes(), 32);
    assert_eq!(first.bound_allocation_bytes(), 32);
    let retained = executor.run(&inputs(Device::Cpu, 5, 1.)).unwrap();
    assert_eq!(executor.cache_stats().entries(), 2);
    assert_eq!(executor.memory_stats().unwrap(), first);
    let report = executor.report().unwrap();
    assert!(report.contains("cache entries: 2 / 2"));
    assert!(report.contains("inactive replay allocation bytes: 0"));
    assert!(report.contains("external outputs: caller-owned, excluded"));
    assert_eq!(
        retained[0].to_vec::<f32>().unwrap(),
        [vec![4.5; 5], vec![0.; 5]].concat()
    );
}

fn exercise_dynamic_default(device: Device) {
    let (captured, _) = Graph::capture([("x", tensor(&[2], vec![1., 2.]))], |inputs| {
        Ok(vec![inputs[0].add_scalar(1.)?])
    })
    .unwrap();
    let mut graph = captured;
    graph
        .add_initializer("x", tensor(&[2], vec![1., 2.]))
        .unwrap();
    graph
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();
    let mut executor = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let changed = HashMap::from([(
        "x".into(),
        tensor(&[5], vec![3.; 5]).to_device(device).unwrap(),
    )]);
    assert_eq!(
        executor.run(&changed).unwrap()[0].to_vec::<f32>().unwrap(),
        [4.; 5]
    );
    assert_eq!(
        executor.run(&HashMap::new()).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [2., 3.]
    );
    assert_eq!(executor.cache_stats().analysis_builds(), 2);
    assert_eq!(executor.analysis().signature().inputs().len(), 1);
}

#[test]
fn test_cpu_shape_specialization_keeps_overridable_initializer_defaults() {
    exercise_dynamic_default(Device::Cpu);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_shape_specialization_keeps_overridable_initializer_defaults() {
    exercise_dynamic_default(Device::Cuda(0));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_memory_reports_include_inactive_plans_and_follow_eviction() {
    let mut executor = graph()
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_cache_capacity(2))
        .unwrap();
    let first = executor.memory_stats().unwrap();
    assert!(first.active_replay_allocation_bytes() > 0);
    assert_eq!(first.inactive_replay_allocation_bytes(), 0);
    let retained = executor.run(&inputs(Device::Cuda(0), 5, 1.)).unwrap();
    let second = executor.memory_stats().unwrap();
    assert_eq!(
        second.inactive_replay_allocation_bytes(),
        first.active_replay_allocation_bytes()
    );
    assert_eq!(
        second.inactive_workspace_bytes(),
        first.active_workspace_bytes()
    );
    assert!(
        second.retained_allocation_bytes().unwrap() > first.retained_allocation_bytes().unwrap()
    );
    executor.run(&inputs(Device::Cuda(0), 7, 1.)).unwrap();
    let third = executor.memory_stats().unwrap();
    assert_eq!(
        third.inactive_replay_allocation_bytes(),
        second.active_replay_allocation_bytes()
    );
    assert_eq!(executor.cache_stats().evictions(), 1);
    assert_eq!(third.shared_state_allocation_bytes(), 32);
    assert_eq!(third.bound_allocation_bytes(), 32);
    assert_eq!(
        retained[0].to_vec::<f32>().unwrap(),
        [vec![4.5; 5], vec![0.; 5]].concat()
    );
    assert!(
        executor
            .report()
            .unwrap()
            .contains("allocator cached free bytes (context-wide):")
    );
}
