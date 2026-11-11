#![cfg(all(feature = "testing", feature = "cuda"))]

// A library can be allowed for only part of what it runs, in graphs and in exported programs.

use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    backend::cuda::{CudnnFeature, LoweringPolicy, export::CudaExport},
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
    tensor::AttentionOptions,
    testing::{CompiledProgram, maybe_lock_device, program_target},
};

const DEVICE: Device = Device::Cuda(0);

fn random(device: Device, dimensions: &[i64], seed: u32) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(device, Shape::create(dimensions.to_vec()).unwrap(), values).unwrap()
}

// One graph of what cuDNN runs: a convolution, a matmul, and half-precision attention.
fn cudnn_graph(device: Device) -> (Graph, Vec<(&'static str, Tensor)>) {
    let half = |dimensions: &[i64], seed| {
        random(device, dimensions, seed)
            .cast(singe::DataType::F16)
            .unwrap()
    };
    let inputs = vec![
        ("image", random(device, &[1, 3, 8, 8], 1)),
        ("weight", random(device, &[4, 3, 3, 3], 2)),
        ("lhs", half(&[6, 5], 3)),
        ("rhs", half(&[5, 4], 4)),
        ("query", half(&[1, 2, 3, 16], 5)),
        ("key", half(&[1, 2, 4, 16], 6)),
        ("value", half(&[1, 2, 4, 16], 7)),
    ];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![
            inputs[0].convolution2d(&inputs[1], None, (1, 1), (0, 0, 0, 0), (1, 1), 1)?,
            inputs[2].matmul(&inputs[3])?,
            inputs[4].attend(
                &inputs[5],
                &inputs[6],
                None,
                None,
                AttentionOptions {
                    causal: true,
                    ..AttentionOptions::default()
                },
            )?,
        ])
    })
    .unwrap();
    (graph, inputs)
}

// What a graph's report says runs `name`.
fn graph_provider(report: &str, name: &str) -> String {
    report
        .lines()
        .find(|line| line.starts_with("operation ") && line.contains(&format!(" {name}: provider")))
        .unwrap_or_else(|| panic!("no {name} in {report}"))
        .split("provider ")
        .nth(1)
        .unwrap()
        .split(", ")
        .next()
        .unwrap()
        .to_owned()
}

#[test]
fn test_graphs_and_programs_match_with_one_feature_allowed() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let (graph, inputs) = cudnn_graph(DEVICE);
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let run = |policy: LoweringPolicy| {
        let mut prepared = graph
            .prepare(
                GraphPrepareOptions::new(DEVICE)
                    .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(policy)),
            )
            .unwrap();
        let outputs = prepared.run(&named).unwrap();
        (outputs, prepared.report().unwrap())
    };
    let values = |tensor: &Tensor| {
        tensor
            .cast(singe::DataType::F32)
            .unwrap()
            .to_vec::<f32>()
            .unwrap()
    };
    let assert_close = |actual: &[Tensor], expected: &[Tensor], label: &str| {
        for (actual, expected) in actual.iter().zip(expected) {
            for (a, e) in values(actual).iter().zip(values(expected)) {
                assert!((a - e).abs() <= 1e-2, "{label}: {a} is not {e}");
            }
        }
    };
    let (expected, _) = run(LoweringPolicy::generated_kernels());
    let cudnn = "Some(CudnnBackend)";
    let kernel = "Some(Kernel)";
    let (convolution, report) =
        run(LoweringPolicy::default().with_libraries([CudnnFeature::Convolution]));
    assert_eq!(graph_provider(&report, "conv2d"), cudnn);
    assert_eq!(graph_provider(&report, "matmul"), kernel);
    assert_eq!(
        graph_provider(&report, "scaled_dot_product_attention"),
        kernel
    );
    assert_close(&convolution, &expected, "convolution");
    let (attention, report) =
        run(LoweringPolicy::default().with_libraries([CudnnFeature::Attention]));
    assert_eq!(graph_provider(&report, "conv2d"), kernel);
    assert_eq!(graph_provider(&report, "matmul"), kernel);
    assert_eq!(
        graph_provider(&report, "scaled_dot_product_attention"),
        cudnn
    );
    assert_close(&attention, &expected, "attention");
    if program_target(DEVICE).is_some() {
        let architecture = program_target(DEVICE).unwrap();
        let program = CudaExport::new(architecture)
            .with_libraries([CudnnFeature::Convolution])
            .export(&graph)
            .unwrap();
        let compiled = CompiledProgram::create(&program, "library_feature_convolution").unwrap();
        let tensors = inputs
            .iter()
            .map(|(_, tensor)| tensor.clone())
            .collect::<Vec<_>>();
        let outputs = compiled.run(&tensors, 1).unwrap();
        assert_close(&outputs, &expected, "program");
    }
}
