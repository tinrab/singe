#![cfg(feature = "cuda")]

use std::{collections::HashMap, sync::Arc};

use singe::{
    Device, Shape, Tensor,
    backend::cuda::{CudaGraphProvider, CudaStream, DynamicShapePolicy, LoweringPolicy},
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
};
use singe_cuda::context::Context as NativeCudaContext;

fn input(device: Device, batch: i64) -> HashMap<String, Tensor> {
    HashMap::from([(
        "x".into(),
        Tensor::from_vec(
            device,
            Shape::create([batch]).unwrap(),
            vec![1_f32; usize::try_from(batch).unwrap()],
        )
        .unwrap(),
    )])
}

fn source() -> Graph {
    let (captured, _) = Graph::capture(input(Device::Cpu, 2), |inputs| {
        Ok(vec![inputs[0].add_scalar(2.)?])
    })
    .unwrap();
    let mut graph = captured;
    graph
        .add_input_dimension_guard("x", 0, "batch", Some(1), Some(8))
        .unwrap();
    graph
}

#[test]
fn test_common_cuda_provider_controls_change_bound_keys_and_run_on_explicit_streams() {
    let graph = source();
    let default = graph
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let mut builtin = graph
        .prepare(
            GraphPrepareOptions::new(Device::Cuda(0)).with_cuda_options(
                GraphCudaOptions::new()
                    .with_lowering_policy(LoweringPolicy::new(vec![CudaGraphProvider::Kernel])),
            ),
        )
        .unwrap();
    assert_eq!(
        default.specialization_key().program(),
        builtin.specialization_key().program()
    );
    assert_ne!(default.specialization_key(), builtin.specialization_key());
    assert!(builtin.report().unwrap().contains("Kernel"));
    let first_stream = CudaStream::create(Device::Cuda(0)).unwrap();
    let second_stream = CudaStream::create(Device::Cuda(0)).unwrap();
    let retained = builtin
        .run_on(&input(Device::Cuda(0), 2), &first_stream)
        .unwrap();
    assert_eq!(
        builtin
            .run_on(&input(Device::Cuda(0), 4), &second_stream)
            .unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [3.; 4]
    );
    assert_eq!(retained[0].to_vec::<f32>().unwrap(), [3.; 2]);
    assert_eq!(builtin.runs(), 2);
    let mut cpu = graph
        .prepare(GraphPrepareOptions::new(Device::Cpu))
        .unwrap();
    assert!(cpu.run_on(&input(Device::Cpu, 2), &first_stream).is_err());
    assert_eq!(cpu.runs(), 0);
    assert!(
        graph
            .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_cuda_options(
                GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(Vec::new())),
            ))
            .is_err()
    );
}

#[test]
fn test_recording_limits_fail_before_replacing_the_usable_specialization() {
    let config = GraphPrepareOptions::new(Device::Cuda(0))
        .with_cache_capacity(2)
        .with_cuda_options(GraphCudaOptions::new().with_dynamic_shape_policy(
            DynamicShapePolicy::exact_shape().with_max_recordings_per_graph(2),
        ));
    let mut executor = source().prepare(config).unwrap();
    executor.run(&input(Device::Cuda(0), 3)).unwrap();
    let key = executor.specialization_key().clone();
    let stats = executor.cache_stats();
    assert!(executor.run(&input(Device::Cuda(0), 4)).is_err());
    assert_eq!(executor.specialization_key(), &key);
    assert_eq!(executor.cache_stats(), stats);
    assert_eq!(
        executor.run(&input(Device::Cuda(0), 2)).unwrap()[0]
            .to_vec::<f32>()
            .unwrap(),
        [3.; 2]
    );
    assert!(
        source()
            .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_cuda_options(
                GraphCudaOptions::new().with_dynamic_shape_policy(
                    DynamicShapePolicy::exact_shape().with_max_recordings_per_graph(0)
                ),
            ))
            .is_err()
    );
}

#[test]
fn test_a_rejecting_policy_refuses_dynamic_dimensions() {
    assert!(
        source()
            .prepare(GraphPrepareOptions::new(Device::Cuda(0)).with_cuda_options(
                GraphCudaOptions::new().with_dynamic_shape_policy(DynamicShapePolicy::reject()),
            ))
            .is_err()
    );
}

#[test]
fn test_explicit_streams_require_the_actual_context_even_on_the_same_device() {
    let mut executor = source()
        .prepare(GraphPrepareOptions::new(Device::Cuda(0)))
        .unwrap();
    let inputs = input(Device::Cuda(0), 2);
    let foreign = NativeCudaContext::create().unwrap();
    let stream = CudaStream::new(Arc::new(foreign.create_stream().unwrap()));
    assert_eq!(stream.device(), executor.device());
    let key = executor.specialization_key().clone();
    let stats = executor.cache_stats();
    let error = executor.run_on(&inputs, &stream).unwrap_err();
    assert!(error.to_string().contains("stream context"), "{error}");
    assert_eq!(executor.specialization_key(), &key);
    assert_eq!(executor.cache_stats(), stats);
    assert_eq!(
        executor.run(&inputs).unwrap()[0].to_vec::<f32>().unwrap(),
        [3.; 2]
    );
}
