use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    graph::{ComputePrecision, Determinism, Graph, GraphPrepareOptions},
};

#[cfg(feature = "cuda")]
use singe::{DataType, tensor::TensorDataType};

fn operands(device: Device) -> (Tensor, Tensor) {
    let a = Tensor::from_vec(
        device,
        Shape::create([16, 32]).unwrap(),
        (0..512)
            .map(|i| if i % 2 == 0 { 1.0001_f32 } else { 1. })
            .collect(),
    )
    .unwrap();
    let b = Tensor::from_vec(
        device,
        Shape::create([32, 16]).unwrap(),
        (0..512)
            .map(|i| if (i / 16) % 2 == 0 { 1_f32 } else { -1. })
            .collect(),
    )
    .unwrap();
    (a, b)
}

fn graph() -> Graph {
    let (a, b) = operands(Device::Cpu);
    Graph::capture([("a", a), ("b", b)], |xs| Ok(vec![xs[0].matmul(&xs[1])?]))
        .unwrap()
        .0
}

fn assert_full_precision(output: &Tensor) {
    let expected = (1.0001_f32 - 1.) * 16.;
    for value in output.to_vec::<f32>().unwrap() {
        assert!(
            (value - expected).abs() < 1e-6,
            "actual {value}, expected {expected}"
        );
    }
}

#[test]
fn test_numerical_configuration_is_part_of_specialization_identity() {
    let source = graph();
    let (a, b) = operands(Device::Cpu);
    let inputs = HashMap::from([("a".into(), a), ("b".into(), b)]);
    let config = GraphPrepareOptions::new(Device::Cpu);
    assert_eq!(config.compute_precision(), ComputePrecision::Full);
    assert_eq!(config.determinism(), Determinism::Allow);
    let mut keys = Vec::new();
    for config in [
        config.clone(),
        config.clone().with_determinism(Determinism::Require),
        config.with_compute_precision(ComputePrecision::AllowTf32),
    ] {
        let mut executor = source.prepare(config.clone()).unwrap();
        assert_eq!(
            executor.specialization_key().compute_precision(),
            config.compute_precision()
        );
        assert_eq!(
            executor.specialization_key().determinism(),
            config.determinism()
        );
        assert_full_precision(&executor.run(&inputs).unwrap()[0]);
        keys.push(executor.specialization_key().clone());
        assert!(executor.report().unwrap().contains("compute precision:"));
    }
    assert_ne!(keys[0], keys[1]);
    assert_ne!(keys[0], keys[2]);
    assert_eq!(keys[0].program(), keys[1].program());
    assert_eq!(keys[0].program(), keys[2].program());
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_precision_and_determinism_apply_to_eager_dispatch_and_native_blas() {
    use singe::{
        backend::cuda::{self, CudaGraphProvider, LoweringPolicy},
        graph::{GraphCudaOptions, GraphExecutionMode},
    };
    let source = graph();
    cuda::clear_cache(Device::Cuda(0)).unwrap();
    let (a, b) = operands(Device::Cuda(0));
    let inputs = HashMap::from([("a".into(), a), ("b".into(), b)]);
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        for provider in [CudaGraphProvider::Cublas, CudaGraphProvider::CublasLt] {
            let config = GraphPrepareOptions::new(Device::Cuda(0))
                .with_execution_mode(mode)
                .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(
                    LoweringPolicy::new(vec![CudaGraphProvider::Alias, provider]),
                ));
            let mut full = source.prepare(config.clone()).unwrap();
            let mut deterministic = source
                .prepare(config.clone().with_determinism(Determinism::Require))
                .unwrap();
            let mut tf32 = source
                .prepare(
                    config
                        .with_compute_precision(ComputePrecision::AllowTf32)
                        .with_determinism(Determinism::Require),
                )
                .unwrap();
            let first = deterministic.run(&inputs).unwrap()[0].to_bytes().unwrap();
            for _ in 0..3 {
                assert_full_precision(&full.run(&inputs).unwrap()[0]);
                assert_eq!(
                    deterministic.run(&inputs).unwrap()[0].to_bytes().unwrap(),
                    first
                );
                let approximate = tf32.run(&inputs).unwrap();
                for value in approximate[0].to_vec::<f32>().unwrap() {
                    assert!(value.abs() < 0.003);
                }
            }
            assert_ne!(
                full.specialization_key(),
                deterministic.specialization_key()
            );
            assert_ne!(full.specialization_key(), tf32.specialization_key());
        }
    }
    assert!(
        cuda::cublaslt_matmul_algorithm_cache_stats(Device::Cuda(0))
            .unwrap()
            .supported_tf32_compute_entries()
            > 0
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cudnn_cache_entries_are_separate_for_each_numerical_policy() {
    use singe::{
        Error,
        backend::cuda::{self, CudaGraphProvider, LoweringPolicy},
        graph::GraphCudaOptions,
    };
    cuda::clear_cache(Device::Cuda(0)).unwrap();
    let (input_shape, weight_shape) = (
        Shape::create([1, 2, 2, 4]).unwrap(),
        Shape::create([1, 2, 1, 2]).unwrap(),
    );
    let (source, _) = Graph::capture(
        [
            ("x", Tensor::full(Device::Cpu, &input_shape, 1_f32).unwrap()),
            (
                "w",
                Tensor::full(Device::Cpu, &weight_shape, 1_f32).unwrap(),
            ),
        ],
        // A convolution over two input channels, which cuDNN lowers, sums the four ones of each window.
        |xs| {
            Ok(vec![xs[0].convolution2d(
                &xs[1],
                None,
                (1, 2),
                (0, 0, 0, 0),
                (1, 1),
                1,
            )?])
        },
    )
    .unwrap();
    let before = cuda::provider_cache_stats(Device::Cuda(0)).unwrap().cudnn;
    for precision in [ComputePrecision::Full, ComputePrecision::AllowTf32] {
        for determinism in [Determinism::Allow, Determinism::Require] {
            for _ in 0..2 {
                let config = GraphPrepareOptions::new(Device::Cuda(0))
                    .with_compute_precision(precision)
                    .with_determinism(determinism)
                    .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(
                        LoweringPolicy::new(vec![
                            CudaGraphProvider::Alias,
                            CudaGraphProvider::CudnnBackend,
                        ]),
                    ));
                match source.prepare(config) {
                    Ok(mut executor) => {
                        let outputs = executor
                            .run(&HashMap::from([
                                (
                                    "x".into(),
                                    Tensor::full(Device::Cuda(0), &input_shape, 1_f32).unwrap(),
                                ),
                                (
                                    "w".into(),
                                    Tensor::full(Device::Cuda(0), &weight_shape, 1_f32).unwrap(),
                                ),
                            ]))
                            .unwrap();
                        assert_eq!(outputs[0].to_vec::<f32>().unwrap(), [4.; 4]);
                    }
                    Err(error) => {
                        let mut root = &error;
                        while let Error::GraphSource { source, .. } = root {
                            root = source;
                        }
                        assert!(matches!(root, Error::Unsupported { .. }), "{error}");
                    }
                }
            }
        }
    }
    let after = cuda::provider_cache_stats(Device::Cuda(0)).unwrap().cudnn;
    assert_eq!(after.entries, 4);
    // Admission also rechecks after building a template outside the cache lock.
    assert!(after.misses - before.misses >= 4);
    assert!(after.hits - before.hits >= 4);
}

#[cfg(feature = "cuda")]
#[test]
fn test_a_provider_without_a_checked_determinism_contract_is_rejected_before_dispatch() {
    #[cfg(feature = "testing")]
    use singe::testing::TensorActivity;
    use singe::{Error, graph::GraphExecutionMode};
    let tensor = || Tensor::full(Device::Cpu, Shape::create([2, 2]).unwrap(), 1_f32).unwrap();
    let (source, _) = Graph::capture([("a", tensor()), ("b", tensor()), ("c", tensor())], |xs| {
        Ok(vec![xs[0].contraction(
            &[0, 2],
            &xs[1],
            &[2, 1],
            &xs[2],
            &[0, 1],
            &[0, 1],
            1.,
            0.,
        )?])
    })
    .unwrap();
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        #[cfg(feature = "testing")]
        let before = TensorActivity::current().checked_invocations;
        let error = source
            .prepare(
                GraphPrepareOptions::new(Device::Cuda(0))
                    .with_execution_mode(mode)
                    .with_determinism(Determinism::Require),
            )
            .unwrap_err();
        #[cfg(feature = "testing")]
        assert_eq!(TensorActivity::current().checked_invocations, before);
        assert!(matches!(error, Error::Unsupported { .. }), "{error}");
    }
}

#[cfg(feature = "cuda")]
fn exercise_half_accumulation<T: TensorDataType>(zero: T, one: T, two: T) {
    use singe::{
        backend::cuda::{CudaGraphProvider, LoweringPolicy},
        graph::{GraphCudaOptions, GraphExecutionMode},
    };
    let device = Device::Cuda(0);
    let a = Tensor::from_vec(
        device,
        Shape::create([16, 4096]).unwrap(),
        vec![one; 16 * 4096],
    )
    .unwrap();
    let b = Tensor::from_vec(
        device,
        Shape::create([4096, 16]).unwrap(),
        vec![one; 16 * 4096],
    )
    .unwrap();
    let x = Tensor::from_vec(
        device,
        Shape::create([1, 4096]).unwrap(),
        (0..4096)
            .map(|i| if i % 2 == 0 { zero } else { two })
            .collect(),
    )
    .unwrap();
    let z = Tensor::from_vec(device, Shape::create([1, 4096]).unwrap(), vec![zero; 4096]).unwrap();
    let (source, _) = Graph::capture(
        [
            ("a", a.clone()),
            ("b", b.clone()),
            ("x", x.clone()),
            ("z", z.clone()),
        ],
        |xs| {
            Ok(vec![
                xs[0].matmul(&xs[1])?,
                xs[0].mean(&[1], false)?,
                xs[2].layer_norm(&[4096], None, None, 1e-5)?,
                xs[2].rms_norm(&[4096], None, None, 1e-5)?,
                xs[3].softmax(1)?,
                xs[3].log_softmax(1)?,
            ])
        },
    )
    .unwrap();
    let inputs = HashMap::from([
        ("a".into(), a.clone()),
        ("b".into(), b.clone()),
        ("x".into(), x),
        ("z".into(), z),
    ]);
    let values = |tensor: &Tensor| {
        tensor
            .to_device(Device::Cpu)
            .unwrap()
            .cast(DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    };
    for mode in [GraphExecutionMode::Eager, GraphExecutionMode::Compiled] {
        let mut executor = source
            .prepare(
                GraphPrepareOptions::new(device)
                    .with_execution_mode(mode)
                    .with_determinism(Determinism::Require)
                    .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(
                        LoweringPolicy::new(vec![
                            CudaGraphProvider::Alias,
                            CudaGraphProvider::Kernel,
                        ]),
                    )),
            )
            .unwrap();
        let outputs = executor.run(&inputs).unwrap();
        assert!(values(&outputs[0]).iter().all(|&value| value == 4096.));
        assert!(values(&outputs[1]).iter().all(|&value| value == 1.));
        for (i, value) in values(&outputs[2]).into_iter().enumerate() {
            let expected = if i % 2 == 0 { -1_f32 } else { 1. };
            assert!((value - expected).abs() < 0.02, "layer norm: {value}");
        }
        for (i, value) in values(&outputs[3]).into_iter().enumerate() {
            let expected = if i % 2 == 0 { 0_f32 } else { 2_f32.sqrt() };
            assert!((value - expected).abs() < 0.02, "rms norm: {value}");
        }
        assert!(values(&outputs[4]).iter().all(|&value| value == 1. / 4096.));
        assert!(
            values(&outputs[5])
                .iter()
                .all(|&value| (value + 4096_f32.ln()).abs() < 0.02)
        );
    }
    let (matmul, _) = Graph::capture([("a", a.clone()), ("b", b.clone())], |xs| {
        Ok(vec![xs[0].matmul(&xs[1])?])
    })
    .unwrap();
    let mut native = matmul
        .prepare(
            GraphPrepareOptions::new(device)
                .with_determinism(Determinism::Require)
                .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(
                    LoweringPolicy::new(vec![
                        CudaGraphProvider::Alias,
                        CudaGraphProvider::CublasLt,
                    ]),
                )),
        )
        .unwrap();
    assert!(
        values(
            &native
                .run(&HashMap::from([("a".into(), a), ("b".into(), b)]))
                .unwrap()[0]
        )
        .iter()
        .all(|&value| value == 4096.)
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_half_matmul_mean_and_normalization_use_fp32_accumulation_in_general_and_native_routes() {
    use half::{bf16, f16};
    exercise_half_accumulation(f16::ZERO, f16::ONE, f16::from_f32(2.));
    exercise_half_accumulation(bf16::ZERO, bf16::ONE, bf16::from_f32(2.));
}
