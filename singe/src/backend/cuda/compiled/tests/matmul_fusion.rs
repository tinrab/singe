use std::collections::HashMap;

use singe_codegen::schedule::Schedule;
use singe_core::assert_close;

use crate::{
    backend::{
        cuda::{
            self, CudaGraphProvider, KernelRecorder,
            compiled::{executable::GraphCompileProviderDecision, plan::lowering::LoweringPolicy},
        },
        registry::PackedFormat,
    },
    data_type::DataType,
    device::Device,
    graph::{CaptureBindings, Graph, GraphCudaOptions},
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Returns `dimensions` of small varied values of `data_type` on `device`.
// Half-precision sums stay within these tests' absolute tolerances over this narrow range.
fn values(device: Device, dimensions: &[i64], data_type: DataType, seed: usize) -> Tensor {
    let shape = Shape::create(dimensions).unwrap();
    let data = (0..shape.element_count())
        .map(|index| ((index * 37 + seed * 11) % 23) as f32 / 23.0 - 0.5)
        .collect::<Vec<_>>();
    Tensor::from_vec(device, &shape, data)
        .unwrap()
        .cast(data_type)
        .unwrap()
}

#[test]
fn test_generated_matmuls_fuse_dequantized_weights_and_residual_adds() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let (inner, outputs) = (1024_i64, 3072_i64);
    let Some(x) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(x);
    let x = values(device, &[1, inner], DataType::Bf16, 1);
    let residual = values(device, &[1, outputs], DataType::Bf16, 2);
    let quantized = Tensor::from_vec(
        device,
        Shape::create([outputs, inner]).unwrap(),
        (0..outputs * inner)
            .map(|index| i8::try_from(index % 15 - 7).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let scale = values(device, &[outputs], DataType::F32, 3)
        .abs()
        .unwrap()
        .add_scalar(0.01)
        .unwrap();
    let zero_point =
        Tensor::zeroed_on(device, DataType::I64, Shape::create([outputs]).unwrap()).unwrap();

    // With no library allowed, the matmul fuses.
    let bindings = CaptureBindings::new()
        .with_parameter("quantized", quantized.clone())
        .with_parameter("scale", scale.clone())
        .with_parameter("zero_point", zero_point.clone());
    let (graph, eager) = Graph::capture_with_bindings(
        [("x", x.clone()), ("residual", residual.clone())],
        bindings,
        |inputs| {
            let weight = quantized
                .dequantize_with(&scale, &zero_point, Some(0), None)?
                .cast(DataType::Bf16)?
                .transpose(0, 1)?;
            Ok(vec![inputs[0].matmul(&weight)?.add(&inputs[1])?])
        },
    )
    .unwrap();
    let recorder = KernelRecorder::new();
    let compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new()
            .with_lowering_policy(LoweringPolicy::generated_kernels())
            .with_kernel_recorder(recorder.clone()),
    );
    let names = recorder
        .take()
        .unwrap()
        .into_iter()
        .map(|planned| planned.kernel.name)
        .collect::<Vec<_>>();
    let mut compiled = compiled.unwrap();
    assert!(
        names
            .iter()
            .any(|name| name == "singe_dequantize_matmul_add"),
        "{names:?}"
    );
    assert!(
        !names.iter().any(|name| name == "singe_dequantize"),
        "{names:?}"
    );

    let inputs = HashMap::from([("x".to_owned(), x), ("residual".to_owned(), residual)]);
    let output = compiled.run(&inputs).unwrap().remove(0);
    let expected = eager[0]
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let actual = output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 2e-2);
}

/// Returns bytes of `dimensions` that cover every 4-bit code, in an `I8` or `U8` tensor on `device`.
fn packed_bytes(device: Device, dimensions: &[i64], data_type: DataType) -> Tensor {
    let shape = Shape::create(dimensions).unwrap();
    let bytes = (0..shape.element_count())
        .map(|index| u8::try_from((index * 29 + 7) % 256).unwrap())
        .collect::<Vec<_>>();
    match data_type {
        DataType::I8 => Tensor::from_vec(
            device,
            &shape,
            bytes.into_iter().map(u8::cast_signed).collect::<Vec<_>>(),
        ),
        _ => Tensor::from_vec(device, &shape, bytes),
    }
    .unwrap()
}

#[test]
fn test_packed_dequantization_on_cuda_matches_the_cpu() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    for container in [DataType::I8, DataType::U8] {
        for format in [PackedFormat::Int4, PackedFormat::Float4E2M1] {
            // Per tensor, per row, and per unpacked column, the axis the bytes pack.
            for (axis, parameters) in [(None, vec![]), (Some(0), vec![3_i64]), (Some(1), vec![10])]
            {
                let run = |device: Device| {
                    let bytes = packed_bytes(device, &[3, 5], container)
                        .to_device(device)
                        .unwrap();
                    let shape = Shape::create(parameters.clone()).unwrap();
                    let count = shape.element_count();
                    let scale = Tensor::from_vec(
                        device,
                        &shape,
                        (0..count)
                            .map(|index| (index as f32).mul_add(0.125, 0.25))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let zero_point = Tensor::from_vec(
                        device,
                        &shape,
                        (0..count)
                            .map(|index| i64::try_from(index % 3).unwrap() - 1)
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    bytes
                        .dequantize_packed(&scale, &zero_point, axis, format)
                        .unwrap()
                        .to_vec::<f32>()
                        .unwrap()
                };
                assert_eq!(
                    run(device),
                    run(Device::Cpu),
                    "{container:?} {format:?} {axis:?}"
                );
            }
        }
    }
}

#[test]
fn test_generated_matmuls_read_packed_int4_weights() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(probe) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::F32,
        Shape::create([1]).unwrap(),
    )) else {
        return;
    };
    drop(probe);
    let (inner, outputs) = (1024_i64, 3072_i64);
    let x = values(device, &[1, inner], DataType::Bf16, 1);
    let quantized = packed_bytes(device, &[outputs, inner / 2], DataType::I8);
    let scale = values(device, &[outputs], DataType::F32, 3)
        .abs()
        .unwrap()
        .add_scalar(0.01)
        .unwrap();
    let zero_point =
        Tensor::zeroed_on(device, DataType::I64, Shape::create([outputs]).unwrap()).unwrap();
    let bindings = CaptureBindings::new()
        .with_parameter("quantized", quantized.clone())
        .with_parameter("scale", scale.clone())
        .with_parameter("zero_point", zero_point.clone());
    let (graph, eager) = Graph::capture_with_bindings([("x", x.clone())], bindings, |inputs| {
        let weight = quantized
            .dequantize_packed(&scale, &zero_point, Some(0), PackedFormat::Int4)?
            .cast(DataType::Bf16)?
            .transpose(0, 1)?;
        Ok(vec![inputs[0].matmul(&weight)?])
    })
    .unwrap();
    let recorder = KernelRecorder::new();
    let compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new()
            .with_lowering_policy(LoweringPolicy::generated_kernels())
            .with_kernel_recorder(recorder.clone()),
    );
    let names = recorder
        .take()
        .unwrap()
        .into_iter()
        .map(|planned| planned.kernel.name)
        .collect::<Vec<_>>();
    let mut compiled = compiled.unwrap();
    assert!(
        names.iter().any(|name| name == "singe_dequantize_matmul"),
        "{names:?}"
    );
    let output = compiled
        .run(&HashMap::from([("x".to_owned(), x)]))
        .unwrap()
        .remove(0);
    let expected = eager[0]
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let actual = output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
    // The `BF16` products reach 8, where one unit in the last place is 0.03125, and their sums round in another order.
    assert_close!(&actual, &expected, 7e-2);
}

#[test]
fn test_generated_matmuls_compute_swiglu_in_their_loads_and_apply_epilogue_chains() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let (inner, outputs) = (3072_i64, 1024_i64);
    let Some(up) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::Bf16,
        Shape::create([1, inner]).unwrap(),
    )) else {
        return;
    };
    drop(up);
    let up = values(device, &[1, inner], DataType::Bf16, 1);
    let gate = values(device, &[1, inner], DataType::Bf16, 2);
    let residual = values(device, &[1, outputs], DataType::Bf16, 3);
    let weight = values(device, &[outputs, inner], DataType::Bf16, 4);
    // A decoder layer's down projection at batch 1: SwiGLU, the matmul, the residual add, and a scale after it.
    let bindings = CaptureBindings::new().with_parameter("weight", weight.clone());
    let (graph, eager) = Graph::capture_with_bindings(
        [
            ("up", up.clone()),
            ("gate", gate.clone()),
            ("residual", residual.clone()),
        ],
        bindings,
        |inputs| {
            let hidden = inputs[0].swiglu(&inputs[1])?;
            Ok(vec![
                hidden
                    .matmul(&weight.transpose(0, 1)?)?
                    .add(&inputs[2])?
                    .mul_scalar(0.5)?,
            ])
        },
    )
    .unwrap();
    let recorder = KernelRecorder::new();
    let compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new()
            .with_lowering_policy(LoweringPolicy::generated_kernels())
            .with_kernel_recorder(recorder.clone()),
    );
    let names = recorder
        .take()
        .unwrap()
        .into_iter()
        .map(|planned| planned.kernel.name)
        .collect::<Vec<_>>();
    let mut compiled = compiled.unwrap();
    assert!(
        names
            .iter()
            .any(|name| name == "singe_fused_swiglu_matmul_add_scale"),
        "{names:?}\n{}",
        compiled.report().fusion_plan()
    );
    let inputs = HashMap::from([
        ("up".to_owned(), up),
        ("gate".to_owned(), gate),
        ("residual".to_owned(), residual),
    ]);
    let output = compiled.run(&inputs).unwrap().remove(0);
    let expected = eager[0]
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let actual = output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 2e-2);
}

#[test]
fn test_library_matmuls_add_their_residual_in_one_cublaslt_call() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let (rows, inner, outputs) = (16_i64, 1024_i64, 1024_i64);
    let Some(x) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::Bf16,
        Shape::create([rows, inner]).unwrap(),
    )) else {
        return;
    };
    drop(x);
    let x = values(device, &[rows, inner], DataType::Bf16, 1);
    let residual = values(device, &[rows, outputs], DataType::Bf16, 2);
    let weight = values(device, &[outputs, inner], DataType::Bf16, 3);
    let bindings = CaptureBindings::new().with_parameter("weight", weight.clone());
    let (graph, eager) = Graph::capture_with_bindings(
        [("x", x.clone()), ("residual", residual.clone())],
        bindings,
        |inputs| {
            Ok(vec![
                inputs[0]
                    .matmul(&weight.transpose(0, 1)?)?
                    .add(&inputs[1])?,
            ])
        },
    )
    .unwrap();
    // Without cuDNN, which fuses the add itself, the add runs inside the matmul's cuBLASLt call, which reads the residual as `C`.
    let options = GraphCudaOptions::new().with_lowering_policy(LoweringPolicy::new(vec![
        CudaGraphProvider::Alias,
        CudaGraphProvider::Kernel,
        CudaGraphProvider::Fusion,
        CudaGraphProvider::CublasLt,
        CudaGraphProvider::Cublas,
    ]));
    let mut compiled = cuda::compile_graph(&graph, &options).unwrap();
    let providers = compiled
        .report()
        .operations()
        .iter()
        .map(|operation| {
            (
                operation.operation().to_owned(),
                operation
                    .realized_decision()
                    .and_then(GraphCompileProviderDecision::provider),
            )
        })
        .collect::<Vec<_>>();
    assert!(
        providers.contains(&("add".to_owned(), Some(CudaGraphProvider::CublasLt))),
        "{providers:?}"
    );
    let inputs = HashMap::from([("x".to_owned(), x), ("residual".to_owned(), residual)]);
    let output = compiled.run(&inputs).unwrap().remove(0);
    let expected = eager[0]
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let actual = output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
    // cuBLASLt adds the residual before rounding the product to `BF16`, which the unfused graph rounds first, so they differ by up to one unit in the last place, 2^-7 of the value.
    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= expected.abs() / 128.0 + 1e-2,
            "element {index}: {actual} against {expected}"
        );
    }
}

#[test]
fn test_generated_matmuls_of_many_rows_run_their_prologue_and_epilogue_on_tensor_cores() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let (rows, inner, outputs) = (96_i64, 320_i64, 200_i64);
    let Some(up) = skip_if_stub(Tensor::zeroed_on(
        device,
        DataType::Bf16,
        Shape::create([rows, inner]).unwrap(),
    )) else {
        return;
    };
    drop(up);
    let up = values(device, &[rows, inner], DataType::Bf16, 1);
    let gate = values(device, &[rows, inner], DataType::Bf16, 2);
    let residual = values(device, &[rows, outputs], DataType::Bf16, 3);
    // The weight is read through a slice of every other column, which cuBLAS cannot take, so the generated kernel runs it.
    let weight = values(device, &[outputs, 2 * inner], DataType::Bf16, 4);
    let bindings = CaptureBindings::new().with_parameter("weight", weight.clone());
    let (graph, eager) = Graph::capture_with_bindings(
        [
            ("up", up.clone()),
            ("gate", gate.clone()),
            ("residual", residual.clone()),
        ],
        bindings,
        |inputs| {
            let hidden = inputs[0].swiglu(&inputs[1])?;
            Ok(vec![
                hidden
                    .matmul(&weight.slice(1, 0, inner, 2)?.transpose(0, 1)?)?
                    .add(&inputs[2])?
                    .mul_scalar(0.5)?,
            ])
        },
    )
    .unwrap();
    let recorder = KernelRecorder::new();
    let mut compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new()
            .with_kernel_recorder(recorder.clone())
            .with_lowering_policy(LoweringPolicy::new(vec![
                CudaGraphProvider::Alias,
                CudaGraphProvider::Kernel,
                CudaGraphProvider::Fusion,
            ])),
    )
    .unwrap();
    let planned = recorder.take().unwrap();
    assert!(
        planned.iter().any(|planned| {
            planned.kernel.name == "singe_fused_swiglu_matmul_add_scale"
                && matches!(planned.schedule, Schedule::Mma { .. })
        }),
        "{:?}",
        planned
            .iter()
            .map(|planned| (&planned.kernel.name, planned.schedule))
            .collect::<Vec<_>>()
    );
    let inputs = HashMap::from([
        ("up".to_owned(), up),
        ("gate".to_owned(), gate),
        ("residual".to_owned(), residual),
    ]);
    let output = compiled.run(&inputs).unwrap().remove(0);
    let expected = eager[0]
        .cast(DataType::F32)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    let actual = output.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
    assert_close!(&actual, &expected, 2e-2);
}

// With no matmul library in the policy, a matmul fuses without any provider tuning record.
#[test]
fn test_matmul_fuses_under_a_policy_with_only_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let x = values(device, &[8, 16], DataType::F32, 1);
    let weight = values(device, &[32, 16], DataType::F32, 2);
    let residual = values(device, &[8, 32], DataType::F32, 3);
    let (graph, eager) = Graph::capture(
        [
            ("x", x.clone()),
            ("weight", weight.clone()),
            ("residual", residual.clone()),
        ],
        |inputs| {
            Ok(vec![
                inputs[0]
                    .matmul(&inputs[1].transpose(0, 1)?)?
                    .add(&inputs[2])?,
            ])
        },
    )
    .unwrap();
    let recorder = KernelRecorder::new();
    let mut compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new()
            .with_lowering_policy(LoweringPolicy::generated_kernels())
            .with_kernel_recorder(recorder.clone()),
    )
    .unwrap();
    let names = recorder
        .take()
        .unwrap()
        .into_iter()
        .map(|planned| planned.kernel.name)
        .collect::<Vec<_>>();
    assert_eq!(names, ["singe_matmul_add"]);
    let inputs = HashMap::from([
        ("x".to_owned(), x),
        ("weight".to_owned(), weight),
        ("residual".to_owned(), residual),
    ]);
    let output = compiled.run(&inputs).unwrap().remove(0);
    let actual = output.to_vec::<f32>().unwrap();
    let expected = eager[0].to_vec::<f32>().unwrap();
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!((actual - expected).abs() <= 1e-3, "{actual} != {expected}");
    }
}

// Each library's fusions run only under a policy that allows that library.
#[test]
fn test_matmul_fusions_respect_the_lowering_policy() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let x = values(device, &[8, 16], DataType::F32, 1);
    let weight = values(device, &[32, 16], DataType::F32, 2);
    let residual = values(device, &[8, 32], DataType::F32, 3);
    let providers = |library, add: bool| {
        let (graph, _) = Graph::capture(
            [
                ("x", x.clone()),
                ("weight", weight.clone()),
                ("residual", residual.clone()),
            ],
            |inputs| {
                let product = inputs[0].matmul(&inputs[1].transpose(0, 1)?)?;
                Ok(vec![if add {
                    product.add(&inputs[2])?
                } else {
                    product.relu()?
                }])
            },
        )
        .unwrap();
        let policy = LoweringPolicy::new(vec![
            CudaGraphProvider::Alias,
            CudaGraphProvider::Kernel,
            CudaGraphProvider::Fusion,
            library,
        ]);
        let compiled = cuda::compile_graph(
            &graph,
            &GraphCudaOptions::new().with_lowering_policy(policy),
        )
        .unwrap();
        compiled
            .report()
            .operations()
            .iter()
            .filter(|operation| operation.operation() != "transpose")
            .filter_map(|operation| {
                operation
                    .realized_decision()
                    .and_then(GraphCompileProviderDecision::provider)
            })
            .collect::<Vec<_>>()
    };
    for add in [false, true] {
        let cudnn = providers(CudaGraphProvider::CudnnBackend, add);
        assert!(!cudnn.contains(&CudaGraphProvider::CublasLt), "{cudnn:?}");
    }
    // cuBLASLt takes its activation under a policy without cuDNN.
    let cublaslt = providers(CudaGraphProvider::CublasLt, false);
    assert_eq!(
        cublaslt,
        [CudaGraphProvider::CublasLt, CudaGraphProvider::CublasLt]
    );
}
