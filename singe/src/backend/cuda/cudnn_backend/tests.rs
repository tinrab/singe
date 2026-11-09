use std::{collections::HashMap, sync::atomic::Ordering};

use crate::{
    backend::{
        cuda::{
            self, CudaCompiledGraph,
            compiled::plan::lowering::{CudaGraphProvider, LoweringPolicy},
            cudnn_backend::{CAPTURED_GRAPHS, FORCE_CAPTURED_GRAPHS},
            eager::convolution::conv3d,
            runtime::runtime_for_device,
        },
        registry::Conv3dAttributes,
    },
    data_type::DataType,
    device::Device,
    graph::{Graph, GraphCudaOptions},
    shape::Shape,
    tensor::Tensor,
    testing::{maybe_lock_device, skip_if_stub},
};

/// Restores native CUDA graph recording when the test ends, including on panic.
struct ForceCapturedGraphs;

impl ForceCapturedGraphs {
    fn enable() -> Self {
        FORCE_CAPTURED_GRAPHS.store(true, Ordering::Relaxed);
        Self
    }
}

impl Drop for ForceCapturedGraphs {
    fn drop(&mut self) {
        FORCE_CAPTURED_GRAPHS.store(false, Ordering::Relaxed);
    }
}

fn convolution_graph(lhs: Tensor, rhs: Tensor) -> Graph {
    let (captured, _) = Graph::capture([("lhs", lhs), ("rhs", rhs)], |inputs| {
        Ok(vec![inputs[0].convolution2d(
            &inputs[1],
            None,
            (1, 1),
            (0, 0, 0, 0),
            (1, 1),
            1,
        )?])
    })
    .unwrap();
    captured
}

fn convolution_reference(lhs: &[f32], rhs: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; 64];
    for channel in 0..16 {
        for position in 0..4 {
            output[channel * 4 + position] = (0..16)
                .map(|input| lhs[input * 4 + position] * rhs[channel * 16 + input])
                .sum();
        }
    }
    output
}

#[test]
fn test_cudnn_matmul_selects_full_precision_plan_or_rejects_it() {
    use crate::{
        backend::{
            cuda::cudnn_backend::{
                CUDNN_GRAPH_INPUT0_TENSOR_ID, CUDNN_GRAPH_INPUT1_TENSOR_ID, CudnnBackendGraphKind,
                build_cudnn_matmul_template, execute_cudnn_template,
            },
            registry::{Operation, operation_schema},
        },
        data_type::DataType,
        error::Error,
    };

    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let shape = Shape::create([16, 16]).unwrap();
    let lhs = Tensor::from_vec(
        Device::Cuda(0),
        &shape,
        vec![1.0_f32 + 2.0_f32.powi(-12); 256],
    )
    .unwrap();
    let rhs = Tensor::from_vec(Device::Cuda(0), &shape, vec![1.0_f32; 256]).unwrap();
    let result = execute_cudnn_template(
        &operation_schema(Operation::MatMul),
        vec![
            (CUDNN_GRAPH_INPUT0_TENSOR_ID, &lhs),
            (CUDNN_GRAPH_INPUT1_TENSOR_ID, &rhs),
        ],
        DataType::F32,
        &shape,
        CudnnBackendGraphKind::Matmul,
        vec![],
        |context, inputs, output| {
            build_cudnn_matmul_template(context, &inputs[0], &inputs[1], output, None)
        },
    );
    match result {
        Ok(output) => assert_eq!(output.to_vec::<f32>().unwrap(), vec![16.003_906; 256]),
        Err(Error::Unsupported { .. }) => {}
        Err(error) => panic!("unexpected cuDNN preparation failure: {error}"),
    }
}

#[test]
fn test_cuda_compiled_graph_reuses_captured_cudnn_steps_with_changed_inputs() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let Some(_) = skip_if_stub(runtime_for_device(Device::Cuda(0))) else {
        return;
    };
    let input_shape = Shape::create([1, 16, 2, 2]).unwrap();
    let weight_shape = Shape::create([16, 16, 1, 1]).unwrap();
    let first_lhs = [1.0_f32, -2.0, 3.0, 0.5].repeat(16);
    let first_rhs = [0.5_f32, 1.0, -1.0, 2.0].repeat(64);
    let second_lhs = [-1.0_f32, 4.0, 2.0, -0.5].repeat(16);
    let second_rhs = [2.0_f32, -1.0, 1.5, 0.25].repeat(64);
    let input_tensor =
        |values: &[f32]| Tensor::from_vec(Device::Cuda(0), &input_shape, values.to_vec());
    let weight_tensor =
        |values: &[f32]| Tensor::from_vec(Device::Cuda(0), &weight_shape, values.to_vec());
    let Some(lhs) = skip_if_stub(input_tensor(&first_lhs)) else {
        return;
    };
    let rhs = weight_tensor(&first_rhs).unwrap();

    let _force = ForceCapturedGraphs::enable();
    let captured_before = CAPTURED_GRAPHS.load(Ordering::Relaxed);
    let policy = LoweringPolicy::new(vec![CudaGraphProvider::CudnnBackend]);
    let Some(mut compiled) = skip_if_stub(cuda::compile_graph(
        &convolution_graph(lhs.clone(), rhs.clone()),
        &GraphCudaOptions::new().with_lowering_policy(policy),
    )) else {
        return;
    };
    assert!(compiled.stats().cudnn_backend_graph_nodes() > 0);
    let after_compile = CAPTURED_GRAPHS.load(Ordering::Relaxed);
    assert!(
        after_compile > captured_before,
        "compile should capture the cuDNN plan"
    );

    let run = |compiled: &mut CudaCompiledGraph, lhs: Tensor, rhs: Tensor| {
        compiled
            .run(&HashMap::from([
                ("lhs".to_string(), lhs),
                ("rhs".to_string(), rhs),
            ]))
            .unwrap()
            .remove(0)
            .to_vec::<f32>()
            .unwrap()
    };
    assert_eq!(
        run(&mut compiled, lhs, rhs),
        convolution_reference(&first_lhs, &first_rhs)
    );

    // Replay copies new inputs into the retained bindings of the captured convolution.
    let output = run(
        &mut compiled,
        input_tensor(&second_lhs).unwrap(),
        weight_tensor(&second_rhs).unwrap(),
    );
    assert_eq!(output, convolution_reference(&second_lhs, &second_rhs));
    assert_eq!(CAPTURED_GRAPHS.load(Ordering::Relaxed), after_compile);
}

#[test]
fn test_cuda_conv3d_runs_one_cudnn_convolution_matching_its_decomposition() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(input) = skip_if_stub(Tensor::randn(
        device,
        Shape::create([2, 4, 6, 9, 10]).unwrap(),
        DataType::F32,
        None,
    )) else {
        return;
    };
    let weight = Tensor::randn(
        device,
        Shape::create([6, 2, 3, 3, 2]).unwrap(),
        DataType::F32,
        None,
    )
    .unwrap();
    let bias = Tensor::randn(device, Shape::create([6]).unwrap(), DataType::F32, None).unwrap();
    let attributes = Conv3dAttributes {
        has_bias: true,
        stride: (2, 1, 2),
        padding: (1, 1, 0),
        dilation: (1, 2, 1),
        groups: 2,
    };
    let native = conv3d(&input, &weight, Some(&bias), &attributes).unwrap();
    let composite = input
        .convolution3d_by_planes(&weight, Some(&bias), &attributes)
        .unwrap();
    assert_eq!(native.shape(), composite.shape());
    let (native, composite) = (
        native.to_vec::<f32>().unwrap(),
        composite.to_vec::<f32>().unwrap(),
    );
    for (native, composite) in native.iter().zip(&composite) {
        assert!(
            (native - composite).abs() <= 1e-4 * (1.0 + composite.abs()),
            "{native} != {composite}"
        );
    }
}

#[test]
fn test_compiled_conv3d_runs_as_one_cudnn_node() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let Some(input) = skip_if_stub(Tensor::randn(
        device,
        Shape::create([1, 3, 5, 8, 8]).unwrap(),
        DataType::F32,
        None,
    )) else {
        return;
    };
    let weight = Tensor::randn(
        device,
        Shape::create([4, 3, 3, 3, 3]).unwrap(),
        DataType::F32,
        None,
    )
    .unwrap();
    let bias = Tensor::randn(device, Shape::create([4]).unwrap(), DataType::F32, None).unwrap();
    let convolve = |input: &Tensor, weight: &Tensor, bias: &Tensor| {
        input.convolution3d(weight, Some(bias), (1, 2, 2), (1, 1, 1), (1, 1, 1), 1)
    };
    let (graph, _) = Graph::capture(
        [
            ("input", input.clone()),
            ("weight", weight.clone()),
            ("bias", bias.clone()),
        ],
        |inputs| Ok(vec![convolve(&inputs[0], &inputs[1], &inputs[2])?]),
    )
    .unwrap();
    let mut compiled = cuda::compile_graph(&graph, &GraphCudaOptions::new()).unwrap();
    assert_eq!(compiled.stats().cudnn_backend_graph_nodes(), 1);
    let output = compiled
        .run(&HashMap::from([
            ("input".to_string(), input.clone()),
            ("weight".to_string(), weight.clone()),
            ("bias".to_string(), bias.clone()),
        ]))
        .unwrap()
        .remove(0)
        .to_vec::<f32>()
        .unwrap();
    let expected = convolve(&input, &weight, &bias)
        .unwrap()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(output, expected);
}
