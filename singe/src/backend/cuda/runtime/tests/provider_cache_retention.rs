use std::{collections::HashMap, sync::Arc};

use singe_cublas::context::Context as CublasContext;
use singe_cuda::{stream::Stream, testing::device_lock};
use singe_cudnn::error::{Error as CudnnError, Status as CudnnStatus};

use crate::{
    DataType, Device, Shape, Tensor,
    backend::{
        cuda::{
            self, CudaGraphProvider, CudaStream, DynamicShapePolicy, LoweringPolicy,
            cudnn_backend::{
                CudnnBackendGraphKey, CudnnBackendGraphKind, cached_cudnn_backend_graph_template,
                cudnn_backend_tensor_metadata_key,
            },
            runtime::{CudaRuntime, CudaRuntimeCacheLimits, runtime_for_device},
        },
        registry::Operation,
    },
    error::Error,
    graph::{
        Graph, GraphCudaOptions, GraphOperationOutput, GraphOperationRequest, GraphPrepareOptions,
    },
    layout::Layout,
};

struct ResetCaches(Arc<CudaRuntime>);

impl Drop for ResetCaches {
    fn drop(&mut self) {
        self.0
            .set_provider_cache_limits(CudaRuntimeCacheLimits::default())
            .unwrap();
        self.0.clear_cache().unwrap();
    }
}

fn bounded_runtime() -> ResetCaches {
    let runtime = runtime_for_device(Device::Cuda(0)).unwrap();
    runtime.clear_cache().unwrap();
    let mut limits = CudaRuntimeCacheLimits::default();
    limits.cudnn.entries = 2;
    limits.cutensor.entries = 2;
    limits.generated_kernels.entries = 2;
    runtime.set_provider_cache_limits(limits).unwrap();
    ResetCaches(runtime)
}

#[test]
fn test_rejected_cudnn_preparations_are_bounded_after_instances_drop() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let before = runtime.0.provider_cache_stats().unwrap().cudnn;
    for rows in 2..18 {
        let shape = Shape::create([1, 2, rows, 2]).unwrap();
        let weight_shape = Shape::create([1, 2, 2, 2]).unwrap();
        // A convolution over two input channels, which cuDNN lowers, sums the eight ones of each window.
        let (captured, _) = Graph::capture(
            [
                ("x", Tensor::full(Device::Cpu, &shape, 1_f32).unwrap()),
                (
                    "w",
                    Tensor::full(Device::Cpu, &weight_shape, 1_f32).unwrap(),
                ),
            ],
            |inputs| {
                Ok(vec![inputs[0].convolution2d(
                    &inputs[1],
                    None,
                    (1, 1),
                    (0, 0, 0, 0),
                    (1, 1),
                    1,
                )?])
            },
        )
        .unwrap();
        let config = GraphPrepareOptions::new(Device::Cuda(0)).with_cuda_options(
            GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
                CudaGraphProvider::Alias,
                CudaGraphProvider::CudnnBackend,
            ])),
        );
        match captured.prepare(config) {
            Ok(mut executor) => {
                let input = Tensor::full(Device::Cuda(0), &shape, 1_f32).unwrap();
                let weight = Tensor::full(Device::Cuda(0), &weight_shape, 1_f32).unwrap();
                let output = executor
                    .run(&HashMap::from([("x".into(), input), ("w".into(), weight)]))
                    .unwrap();
                assert!(
                    output[0]
                        .to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|&value| value == 8.)
                );
            }
            Err(Error::Unsupported { .. }) => {}
            Err(error) => panic!("unexpected preparation failure: {error}"),
        }
        let stats = runtime.0.provider_cache_stats().unwrap().cudnn;
        assert!(stats.entries <= 2);
        assert!(stats.accounted_bytes <= CudaRuntimeCacheLimits::default().cudnn.accounted_bytes);
    }
    assert!(runtime.0.provider_cache_stats().unwrap().cudnn.evictions >= before.evictions + 14);
}

#[test]
fn test_transient_cudnn_failures_are_retried_and_stable_diagnostics_survive_hits() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let shape = Shape::create([1]).unwrap();
    let key = CudnnBackendGraphKey::new(
        CudnnBackendGraphKind::ScalarPointwise(Operation::Clamp),
        vec![],
        vec![],
        cudnn_backend_tensor_metadata_key(DataType::F32, &shape, &Layout::from_shape(&shape)),
    );
    for status in [
        CudnnStatus::InternalErrorDeviceAllocationFailed,
        CudnnStatus::InternalErrorHostAllocationFailed,
        CudnnStatus::ExecutionFailedCudaDriver,
        CudnnStatus::NotInitialized,
    ] {
        for _ in 0..2 {
            let result = cached_cudnn_backend_graph_template(&runtime.0, key.clone(), |_| {
                Err(Error::Cudnn(CudnnError::Cudnn {
                    code: status,
                    message: "resource diagnostic".into(),
                }))
            });
            assert!(matches!(result, Err(Error::Cudnn(_))));
            assert_eq!(runtime.0.provider_cache_stats().unwrap().cudnn.entries, 0);
        }
    }
    let result = cached_cudnn_backend_graph_template(&runtime.0, key.clone(), |_| {
        Err(Error::unsupported_operation_with_details(
            "test rejection",
            "stable diagnostic",
        ))
    });
    assert!(format!("{:?}", result.unwrap_err()).contains("stable diagnostic"));
    let result =
        cached_cudnn_backend_graph_template(&runtime.0, key, |_| panic!("negative cache missed"));
    assert!(format!("{:?}", result.unwrap_err()).contains("stable diagnostic"));
}

fn contraction(rows: i64) -> (Graph, HashMap<String, Tensor>) {
    let shapes = [[rows, 3], [3, 2], [rows, 2]];
    let mut graph = Graph::new();
    let mut nodes = Vec::new();
    let mut inputs = HashMap::new();
    for (name, dimensions) in ["a", "b", "c"].into_iter().zip(shapes) {
        let shape = Shape::create(dimensions).unwrap();
        nodes.push(graph.add_input(name, DataType::F32, shape.clone(), Device::Cuda(0)));
        inputs.insert(
            name.into(),
            Tensor::full(Device::Cuda(0), shape, 1_f32).unwrap(),
        );
    }
    let output = graph
        .add_operation_request(GraphOperationRequest::contraction(
            None,
            nodes[0],
            vec![0, 2],
            nodes[1],
            vec![2, 1],
            nodes[2],
            vec![0, 1],
            vec![0, 1],
            1.,
            0.,
            GraphOperationOutput::new(
                DataType::F32,
                Shape::create([rows, 2]).unwrap(),
                Device::Cuda(0),
            ),
        ))
        .unwrap();
    graph.set_outputs(vec![output]).unwrap();
    (graph, inputs)
}

#[test]
fn test_successful_native_plans_and_pending_outputs_survive_provider_eviction() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let (graph, inputs) = contraction(2);
    let mut first = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new().with_dynamic_shape_policy(DynamicShapePolicy::default()),
    )
    .unwrap();
    let weak = {
        let cache = runtime.0.cutensor_contraction_plans().lock().unwrap();
        Arc::downgrade(
            cache
                .values()
                .next()
                .expect("cuTENSOR plan must be cached")
                .plan(),
        )
    };
    let stream = CudaStream::create(Device::Cuda(0)).unwrap();
    let pending = first.enqueue_on(&inputs, &stream).unwrap();
    let mut retained_outputs = Vec::new();
    for rows in 3..11 {
        let (graph, inputs) = contraction(rows);
        let mut executor = cuda::compile_graph(
            &graph,
            &GraphCudaOptions::new().with_dynamic_shape_policy(DynamicShapePolicy::default()),
        )
        .unwrap();
        retained_outputs.extend(executor.run(&inputs).unwrap());
        let stats = runtime.0.provider_cache_stats().unwrap().cutensor;
        assert!(stats.entries <= 2);
        assert!(
            stats.workspace_bytes <= CudaRuntimeCacheLimits::default().cutensor.workspace_bytes
        );
    }
    assert!(weak.upgrade().is_some());
    first.synchronize().unwrap();
    for output in pending {
        // SAFETY: this fresh pending output was fully written by the synchronized executable.
        retained_outputs.push(unsafe { output.assume_init() }.unwrap());
    }
    assert!(runtime.0.provider_cache_stats().unwrap().cutensor.evictions >= 7);
    drop(first);
    assert!(weak.upgrade().is_none());
    runtime.0.clear_cache().unwrap();
    for output in retained_outputs {
        assert!(
            output
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|&value| value == 3.)
        );
    }
}

#[test]
fn test_compiled_modules_survive_eviction_while_their_graph_is_pending() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let before = runtime.0.provider_cache_stats().unwrap().generated_kernels;
    let mut first = None;
    let stream = CudaStream::create(Device::Cuda(0)).unwrap();
    for length in 2..10 {
        // A transposed input's strides are part of its kernel, so each length compiles another kernel.
        let input = Tensor::full(Device::Cuda(0), Shape::create([2, length]).unwrap(), -0_f64)
            .unwrap()
            .transpose(0, 1)
            .unwrap();
        let (captured, _) =
            Graph::capture([("x", input.clone())], |inputs| Ok(vec![inputs[0].relu()?])).unwrap();
        let mut graph = cuda::compile_graph(&captured, &GraphCudaOptions::default()).unwrap();
        let pending = graph
            .enqueue_on(&HashMap::from([("x".into(), input)]), &stream)
            .unwrap();
        if first.is_none() {
            first = Some((graph, pending));
        }
        assert!(
            runtime
                .0
                .provider_cache_stats()
                .unwrap()
                .generated_kernels
                .entries
                <= 2
        );
    }
    let (mut graph, pending) = first.unwrap();
    assert!(
        runtime
            .0
            .provider_cache_stats()
            .unwrap()
            .generated_kernels
            .evictions
            >= before.evictions + 6
    );
    graph.synchronize().unwrap();
    runtime.0.clear_cache().unwrap();
    for output in pending {
        // SAFETY: the owning executable completed all writes to this fresh allocation.
        let output = unsafe { output.assume_init() }.unwrap();
        assert!(
            output
                .to_vec::<f64>()
                .unwrap()
                .iter()
                .all(|value| value.is_sign_negative() && *value == 0.)
        );
    }
}

#[test]
fn test_cublaslt_algorithm_fingerprints_use_native_attribute_widths() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let shape = Shape::create([64, 64]).unwrap();
    let a = Tensor::full(Device::Cuda(0), &shape, 1_f32).unwrap();
    let b = Tensor::full(Device::Cuda(0), &shape, 1_f32).unwrap();
    let (captured, _) = Graph::capture([("a", a.clone()), ("b", b.clone())], |inputs| {
        Ok(vec![inputs[0].matmul(&inputs[1])?])
    })
    .unwrap();
    let mut graph = cuda::compile_graph(
        &captured,
        &GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::CublasLt,
        ])),
    )
    .unwrap();
    assert!(runtime.0.provider_cache_stats().unwrap().cublaslt.entries > 0);
    let outputs = graph
        .run(&HashMap::from([("a".into(), a), ("b".into(), b)]))
        .unwrap();
    assert!(
        outputs[0]
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 64.)
    );
}

#[test]
fn test_legacy_cublas_capture_keeps_only_the_current_stream() {
    let _lock = device_lock(0).expect("provider cache verification requires CUDA device 0");
    let runtime = bounded_runtime();
    let context = CublasContext::create(runtime.0.context()).unwrap();
    for _ in 0..16 {
        let stream = CudaStream::create(Device::Cuda(0)).unwrap();
        context.set_stream(Some(stream.raw())).unwrap();
    }
    let raw = context.into_raw();
    assert_eq!(raw.streams.len(), 1);
    // SAFETY: this record was just transferred from the same unique native owner.
    let context = unsafe { CublasContext::from_raw(raw) }.unwrap();
    context.set_stream(None).unwrap();
    let raw = context.into_raw();
    assert_eq!(raw.streams, [] as [Stream; 0]);
    // SAFETY: this record still carries the unique native handle and its CUDA context.
    drop(unsafe { CublasContext::from_raw(raw) }.unwrap());

    let shape = Shape::create([64, 64]).unwrap();
    let a = Tensor::full(Device::Cuda(0), &shape, 1_f32).unwrap();
    let b = Tensor::full(Device::Cuda(0), &shape, 1_f32).unwrap();
    let (captured, _) = Graph::capture([("a", a.clone()), ("b", b.clone())], |inputs| {
        Ok(vec![inputs[0].matmul(&inputs[1])?])
    })
    .unwrap();
    let mut graph = cuda::compile_graph(
        &captured,
        &GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::Cublas,
        ])),
    )
    .unwrap();
    assert_eq!(graph.stats().cublas_child_graph_nodes(), 1);
    let outputs = graph
        .run(&HashMap::from([("a".into(), a), ("b".into(), b)]))
        .unwrap();
    assert!(
        outputs[0]
            .to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|&value| value == 64.)
    );
}
