#![cfg(all(feature = "testing", feature = "cuda"))]

// A library can be allowed for only part of what it runs, in graphs and in exported programs.

use std::collections::HashMap;

use singe::{
    Device, Shape, Tensor,
    backend::{
        cuda::{
            CublasFeature, CublasLtFeature, CudaGraphProvider, CudnnFeature, CufftFeature,
            CutensorFeature, LibraryFeature, LoweringPolicy,
            export::{CudaExport, CudaProgram, Library, StepProvider},
        },
        registry::{
            BackendExecution, CudaGraphProviderSet, Operation, operation_support,
            registered_operations,
        },
    },
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
    tensor::AttentionOptions,
    testing::{CompiledProgram, maybe_lock_device, program_target},
};

const DEVICE: Device = Device::Cuda(0);

const LIBRARIES: [CudaGraphProvider; 5] = [
    CudaGraphProvider::CudnnBackend,
    CudaGraphProvider::Cublas,
    CudaGraphProvider::CublasLt,
    CudaGraphProvider::Cufft,
    CudaGraphProvider::Cutensor,
];

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

// What a program runs `operation` with.
fn provider_of(program: &CudaProgram, operation: Operation) -> StepProvider {
    program
        .providers
        .iter()
        .find(|provider| provider.operation == Some(operation))
        .unwrap_or_else(|| panic!("no {operation:?} in {:?}", program.providers))
        .provider
}

fn export<L: Into<LibraryFeature>>(
    graph: &Graph,
    libraries: impl IntoIterator<Item = L>,
) -> CudaProgram {
    CudaExport::new("sm_120".parse().unwrap())
        .with_libraries(libraries)
        .export(graph)
        .unwrap()
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
fn test_every_operation_a_library_runs_has_a_feature_of_its_own() {
    for &operation in registered_operations() {
        let support = operation_support(operation);
        if !support.supports(BackendExecution::CudaGraph) {
            continue;
        }
        for &provider in support.cuda_graph_providers() {
            if LIBRARIES.contains(&provider) {
                let feature = LibraryFeature::of(provider, operation)
                    .unwrap_or_else(|| panic!("{provider:?} runs {operation:?} with no feature"));
                assert_eq!(feature.provider(), provider);
            }
        }
    }
}

#[test]
fn test_features_default_to_all() {
    assert_eq!(CudnnFeature::default(), CudnnFeature::All);
    assert_eq!(CublasFeature::default(), CublasFeature::All);
    assert_eq!(CublasLtFeature::default(), CublasLtFeature::All);
    assert_eq!(CufftFeature::default(), CufftFeature::All);
    assert_eq!(CutensorFeature::default(), CutensorFeature::All);
}

#[test]
fn test_a_set_allows_what_its_features_name() {
    let cases: [(LibraryFeature, &[(CudaGraphProvider, Operation)]); 6] = [
        (
            CudnnFeature::Convolution.into(),
            &[
                (CudaGraphProvider::CudnnBackend, Operation::Conv2d),
                (CudaGraphProvider::CudnnBackend, Operation::Conv3d),
            ],
        ),
        (
            CudnnFeature::Attention.into(),
            &[
                (
                    CudaGraphProvider::CudnnBackend,
                    Operation::ScaledDotProductAttention,
                ),
                (CudaGraphProvider::CudnnBackend, Operation::PagedAttention),
                (
                    CudaGraphProvider::CudnnBackend,
                    Operation::PagedAttentionOutput,
                ),
                (
                    CudaGraphProvider::CudnnBackend,
                    Operation::PagedAttentionLse,
                ),
            ],
        ),
        (
            CublasLtFeature::ScaledMatmul.into(),
            &[(CudaGraphProvider::CublasLt, Operation::ScaledMatMul)],
        ),
        (
            CufftFeature::Rfft.into(),
            &[(CudaGraphProvider::Cufft, Operation::Rfft)],
        ),
        (
            CutensorFeature::Contraction.into(),
            &[(CudaGraphProvider::Cutensor, Operation::Contraction)],
        ),
        (
            CublasFeature::Matmul.into(),
            &[(CudaGraphProvider::Cublas, Operation::MatMul)],
        ),
    ];
    let pairs = cases
        .iter()
        .flat_map(|(_, pairs)| pairs.iter().copied())
        .collect::<Vec<_>>();
    for (feature, named) in cases {
        let set = CudaGraphProviderSet::empty().with_feature(feature);
        for &(provider, operation) in &pairs {
            assert_eq!(
                set.allows(provider, operation),
                named.contains(&(provider, operation)),
                "{feature:?} on {provider:?} {operation:?}"
            );
        }
    }
    // A whole library allows everything a feature of it names.
    let whole = CudaGraphProviderSet::empty().with(CudaGraphProvider::CudnnBackend);
    assert!(whole.allows(CudaGraphProvider::CudnnBackend, Operation::Conv3d));
    assert!(whole.allows(CudaGraphProvider::CudnnBackend, Operation::MatMul));
    assert!(!whole.allows(CudaGraphProvider::Cufft, Operation::Rfft));
}

#[test]
fn test_repeating_a_library_adds_to_what_it_may_run() {
    let policy = LoweringPolicy::default()
        .with_libraries([CudnnFeature::Convolution, CudnnFeature::Attention]);
    assert!(policy.allows(CudaGraphProvider::CudnnBackend, Operation::Conv2d));
    assert!(policy.allows(
        CudaGraphProvider::CudnnBackend,
        Operation::ScaledDotProductAttention
    ));
    assert!(!policy.allows(CudaGraphProvider::CudnnBackend, Operation::MatMul));
    assert!(!policy.allows(CudaGraphProvider::Cublas, Operation::MatMul));
    assert!(policy.provider_enabled(CudaGraphProvider::Kernel));
    assert!(policy.provider_enabled(CudaGraphProvider::Fusion));
}

#[test]
fn test_programs_call_a_library_only_for_its_features() {
    let (graph, _) = cudnn_graph(Device::Cpu);
    let cudnn = StepProvider::Library(Library::Cudnn);
    let cases = [
        (CudnnFeature::Convolution, [true, false, false]),
        (CudnnFeature::Matmul, [false, true, false]),
        (CudnnFeature::Attention, [false, false, true]),
        (CudnnFeature::All, [true, true, true]),
    ];
    for (feature, expected) in cases {
        let program = export(&graph, [feature]);
        let actual = [
            Operation::Conv2d,
            Operation::MatMul,
            Operation::ScaledDotProductAttention,
        ]
        .map(|operation| provider_of(&program, operation) == cudnn);
        assert_eq!(actual, expected, "{feature:?}");
    }
    // A plain library name is still the whole library, and several libraries take a list of features.
    let program = export(&graph, [Library::Cudnn]);
    assert_eq!(provider_of(&program, Operation::Conv2d), cudnn);
    let program = export(
        &graph,
        [
            LibraryFeature::Cudnn(CudnnFeature::Attention),
            LibraryFeature::Cublas(CublasFeature::Matmul),
        ],
    );
    assert_eq!(
        provider_of(&program, Operation::ScaledDotProductAttention),
        cudnn
    );
    assert_eq!(
        provider_of(&program, Operation::MatMul),
        StepProvider::Library(Library::Cublas)
    );
    assert_ne!(provider_of(&program, Operation::Conv2d), cudnn);
}

#[test]
fn test_an_operation_only_its_feature_keeps_whole_decomposes_without_it() {
    let inputs = [("signal", random(Device::Cpu, &[2, 1024], 8))];
    let (graph, _) = Graph::capture(inputs, |inputs| {
        let spectrum = inputs[0].rfft()?;
        Ok(vec![spectrum.clone(), spectrum.irfft(1024)?])
    })
    .unwrap();
    let whole = export(&graph, [CufftFeature::All]);
    assert_eq!(whole.decomposed, []);
    let forward = export(&graph, [CufftFeature::Rfft]);
    assert_eq!(forward.decomposed, [Operation::Irfft]);
    assert_eq!(
        provider_of(&forward, Operation::Rfft),
        StepProvider::Library(Library::Cufft)
    );
    let inverse = export(&graph, [CufftFeature::Irfft]);
    assert_eq!(inverse.decomposed, [Operation::Rfft]);
    assert_eq!(
        provider_of(&inverse, Operation::Irfft),
        StepProvider::Library(Library::Cufft)
    );
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
