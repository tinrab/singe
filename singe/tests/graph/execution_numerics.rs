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
