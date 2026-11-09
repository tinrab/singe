use std::{
    panic::{self, AssertUnwindSafe},
    thread,
};

use super::*;
use crate::{backend::registry::Operation, graph::test_support::*};

#[test]
fn test_numerical_scopes_restore_after_nesting_and_unwinding_and_do_not_leak_to_workers() {
    let default = NumericalPolicy::default();
    let requested = NumericalPolicy {
        precision: ComputePrecision::AllowTf32,
        determinism: Determinism::Require,
    };
    assert_eq!(NumericalPolicy::current(), default);
    {
        let _outer = NumericalScope::enter(requested);
        assert_eq!(
            thread::spawn(NumericalPolicy::current).join().unwrap(),
            default
        );
        assert!(
            panic::catch_unwind(AssertUnwindSafe(|| {
                let _inner = NumericalScope::enter(default);
                assert_eq!(NumericalPolicy::current(), default);
                panic!("exercise unwinding");
            }))
            .is_err()
        );
        assert_eq!(NumericalPolicy::current(), requested);
    }
    assert_eq!(NumericalPolicy::current(), default);
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_engine_algorithm_autotune_and_native_keys_include_every_numerical_policy() {
    use std::collections::HashSet;

    use singe_cublas::{lt::types::Epilogue, types::ComputeType};

    use crate::{
        DataType, Device, Shape, Tensor,
        backend::{
            cuda::{
                PlanningContext,
                compiled::plan::{
                    cache_key::{DynamicShapePolicy, GraphCacheKey},
                    kernel::KernelFeasibility,
                    lowering::{CudaGraphProvider, LoweringPlan, LoweringPolicy},
                    pipeline::CompilePipeline,
                },
                cudnn_backend::{
                    CudnnBackendGraphKey, CudnnBackendGraphKind, CudnnBackendTensorKey,
                },
                runtime::{AutotuneCacheKey, CublasLtMatmulAlgorithmKey, CublasLtMatmulKeyParts},
            },
            registry::Operation,
        },
        graph::{Graph, GraphFingerprintOptions},
    };
    let (source, _) = Graph::capture(
        [(
            "x",
            Tensor::full(Device::Cpu, Shape::create([2, 2]).unwrap(), 1_f32).unwrap(),
        )],
        |xs| Ok(vec![xs[0].relu()?]),
    )
    .unwrap();
    let compiled = source.compile().unwrap();
    let mut engines = HashSet::new();
    let mut algorithms = HashSet::new();
    let mut autotune = HashSet::new();
    let mut native = HashSet::new();
    let mut previous: Option<CublasLtMatmulAlgorithmKey> = None;
    for precision in [
        ComputePrecision::Full,
        ComputePrecision::AllowTf32,
        ComputePrecision::Relaxed,
    ] {
        for determinism in [Determinism::Allow, Determinism::Require] {
            let _scope = NumericalScope::enter(NumericalPolicy {
                precision,
                determinism,
            });
            let tensor = || CudnnBackendTensorKey::new(DataType::F32, vec![2, 2], vec![2, 1], 16);
            engines.insert(CudnnBackendGraphKey::new(
                CudnnBackendGraphKind::ScalarPointwise(Operation::Scale),
                vec![],
                vec![tensor()],
                tensor(),
            ));
            let algorithm = CublasLtMatmulAlgorithmKey::new(CublasLtMatmulKeyParts {
                data_type: DataType::F32,
                compute_type: ComputeType::F32,
                transpose_a: false,
                transpose_b: false,
                epilogue: Epilogue::Default,
                rows: 2,
                inner: 2,
                cols: 2,
                batch_count: 1,
                a_strided_batch_offset: 0,
                b_strided_batch_offset: 0,
                workspace_limit_bytes: 0,
                library_version: "test".to_owned(),
            });
            if let Some(previous) = &previous {
                assert!(!algorithm.matches_matmul_algorithm_family(previous));
            }
            previous = Some(algorithm.clone());
            algorithms.insert(algorithm);
            autotune.insert(AutotuneCacheKey::new(
                Operation::MatMul,
                DataType::F32,
                vec![],
                CudaGraphProvider::CublasLt,
                "test".into(),
                Some(0),
            ));
            let policy = LoweringPolicy::default();
            let lowering = LoweringPlan::create(
                compiled.ir(),
                compiled.value_order(),
                &policy,
                &KernelFeasibility::new(&PlanningContext::default()),
            )
            .unwrap();
            native.insert(
                GraphCacheKey::create_from_compiled_graph(
                    &compiled,
                    compiled.ir(),
                    compiled.value_order(),
                    &lowering,
                    &GraphFingerprintOptions::with_compile_options(compiled.compile_options()),
                    CompilePipeline::default(),
                    policy,
                    DynamicShapePolicy::exact_shape(),
                )
                .unwrap(),
            );
        }
    }
    for count in [
        engines.len(),
        algorithms.len(),
        autotune.len(),
        native.len(),
    ] {
        assert_eq!(count, 6);
    }
    // Summaries also identify entries in the graph memory pool.
    // Retain the recorded policy when inspecting a key outside its original scope.
    let summaries = native
        .iter()
        .map(GraphCacheKey::summary)
        .collect::<HashSet<_>>();
    assert_eq!(summaries.len(), 6);
    for key in &native {
        for other in &native {
            if key != other {
                assert!(!key.same_dynamic_recording_group(other));
            }
        }
    }
}

fn exceptional_values() -> Vec<f32> {
    vec![
        1.0,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ]
}

fn assert_same_values(expected: &Tensor, actual: &Tensor) {
    for (expected, actual) in expected
        .to_vec::<f32>()
        .unwrap()
        .into_iter()
        .zip(actual.to_vec::<f32>().unwrap())
    {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(
                expected.to_bits(),
                actual.to_bits(),
                "{expected:?} != {actual:?}"
            );
        }
    }
}

#[test]
fn test_strict_compilation_preserves_rounding_and_exceptional_values() {
    let values = exceptional_values();
    let shape = Shape::create([i64::try_from(values.len()).unwrap()]).unwrap();
    let input_tensor = Tensor::from_vec(Device::Cpu, &shape, values).unwrap();
    let mut graph = Graph::new();
    let input = graph.add_input("x", DataType::F32, shape, Device::Cpu);
    let add = add_scalar_operation(&mut graph, Operation::AddScalar, input, 1e20, &input_tensor);
    let cancellation =
        add_scalar_operation(&mut graph, Operation::SubScalar, add, 1e20, &input_tensor);
    let scale = add_scalar_operation(&mut graph, Operation::Scale, input, 2.0, &input_tensor);
    let overflow = add_scalar_operation(&mut graph, Operation::Scale, scale, 0.5, &input_tensor);
    let reciprocal = add_unary_operation(&mut graph, Operation::Reciprocal, input, &input_tensor);
    let double_reciprocal =
        add_unary_operation(&mut graph, Operation::Reciprocal, reciprocal, &input_tensor);
    let zero_add =
        add_scalar_operation(&mut graph, Operation::AddScalar, input, 0.0, &input_tensor);
    let zero_product =
        add_scalar_operation(&mut graph, Operation::Scale, input, 0.0, &input_tensor);
    let negation = add_unary_operation(&mut graph, Operation::Neg, input, &input_tensor);
    let reverse_sub =
        add_scalar_operation(&mut graph, Operation::RSubScalar, input, 0.0, &input_tensor);
    graph
        .set_outputs(vec![
            cancellation,
            overflow,
            double_reciprocal,
            zero_add,
            zero_product,
            negation,
            reverse_sub,
        ])
        .unwrap();
    let inputs = HashMap::from([("x".to_string(), input_tensor)]);
    let eager = graph.run(&inputs).unwrap();
    let compiled = graph.compile().unwrap();
    let optimized = compiled.run(&inputs).unwrap();
    assert_eq!(eager.len(), optimized.len());
    for (expected, actual) in eager.iter().zip(&optimized) {
        assert_same_values(expected, actual);
    }
}
