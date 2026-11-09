#![cfg(all(feature = "testing", feature = "cuda"))]

// Attention runs through cuDNN only when configured, and matches its generated kernels when it does.

use std::collections::HashMap;

use singe::{
    DataType, Device, Shape, Tensor,
    backend::cuda::{
        CudaGraphProvider, LoweringPolicy,
        export::{CudaExport, Library, ProgramStep},
    },
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
    tensor::{AttentionOptions, Paging},
    testing::{CompiledProgram, maybe_lock_device, program_target},
};
use singe_testing::{Tolerance, assert_matches, host};

const DEVICE: Device = Device::Cuda(0);

const OUTPUT_TOLERANCE: Tolerance = Tolerance::absolute(
    1e-2,
    "cuDNN computes in F32, and each result rounds to half precision on its own",
);

const LSE_TOLERANCE: Tolerance = Tolerance::absolute(
    2e-3,
    "cuDNN computes the log-sum-exp in F32 in another order",
);

const SAME: Tolerance = Tolerance::exact("the same generated kernels run both times");

fn random(dimensions: &[i64], data_type: DataType, seed: u32) -> Tensor {
    let count = usize::try_from(dimensions.iter().product::<i64>()).unwrap();
    let mut state = seed;
    let values = (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f32::from(u16::try_from(state >> 16).unwrap()) / 32_768.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_vec(DEVICE, Shape::create(dimensions.to_vec()).unwrap(), values)
        .unwrap()
        .cast(data_type)
        .unwrap()
}

fn indices(values: &[i32], data_type: DataType) -> Tensor {
    Tensor::from_vec(
        DEVICE,
        Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
        values.to_vec(),
    )
    .unwrap()
    .cast(data_type)
    .unwrap()
}

fn with_cudnn() -> LoweringPolicy {
    LoweringPolicy::new(vec![
        CudaGraphProvider::Alias,
        CudaGraphProvider::Kernel,
        CudaGraphProvider::CudnnBackend,
        CudaGraphProvider::Fusion,
    ])
}

// Whether a report says cuDNN ran a node, not only that it was offered one.
fn ran_cudnn(report: &str) -> bool {
    report.contains("provider Some(CudnnBackend)")
}

// Runs `graph` under `policy`, and returns its outputs and whether cuDNN ran any of it.
fn run(graph: &Graph, inputs: &[(&str, Tensor)], policy: LoweringPolicy) -> (Vec<Tensor>, bool) {
    let mut prepared = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE)
                .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(policy)),
        )
        .unwrap();
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let outputs = prepared.run(&named).unwrap();
    let report = prepared.report().unwrap();
    (outputs, ran_cudnn(&report))
}

// Runs `graph` as an exported program of the libraries given, and returns its outputs and whether it calls cuDNN.
fn run_program(
    graph: &Graph,
    inputs: &[(&str, Tensor)],
    libraries: Vec<Library>,
    name: &str,
) -> Option<(Vec<Tensor>, bool)> {
    let architecture = program_target(DEVICE)?;
    let program = CudaExport::new(architecture)
        .with_libraries(libraries)
        .export(graph)
        .unwrap();
    let calls_cudnn = program.steps.iter().any(|step| {
        matches!(step, ProgramStep::Library { call, .. } if call.library() == Library::Cudnn)
    });
    let compiled = CompiledProgram::create(&program, name).unwrap();
    let instance = compiled.instance().unwrap();
    let tensors = inputs
        .iter()
        .map(|(_, tensor)| tensor.clone())
        .collect::<Vec<_>>();
    let outputs = instance.run(&tensors, 1).unwrap();
    Some((outputs, calls_cudnn))
}

#[test]
fn test_attention_through_cudnn_matches_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let option_sets = [
        ("plain", AttentionOptions::default()),
        (
            "causal",
            AttentionOptions {
                causal: true,
                ..AttentionOptions::default()
            },
        ),
        (
            "window",
            AttentionOptions {
                window: Some(2),
                ..AttentionOptions::default()
            },
        ),
        (
            "causal window",
            AttentionOptions {
                causal: true,
                window: Some(2),
                ..AttentionOptions::default()
            },
        ),
        (
            "scale",
            AttentionOptions {
                scale: Some(0.5),
                causal: true,
                ..AttentionOptions::default()
            },
        ),
    ];
    for data_type in [DataType::F16, DataType::Bf16] {
        for (key_heads, groups) in [(4, 1), (2, 2)] {
            let query = random(&[2, 4, 3, 16], data_type, 1);
            let key = random(&[2, key_heads, 5, 16], data_type, 2);
            let value = random(&[2, key_heads, 5, 16], data_type, 3);
            let inputs = [("query", query), ("key", key), ("value", value)];
            for (name, options) in &option_sets {
                let label = format!("{data_type:?} {name} groups {groups}");
                let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
                    let (key, value) = if groups > 1 {
                        (inputs[1].repeat_kv(groups)?, inputs[2].repeat_kv(groups)?)
                    } else {
                        (inputs[1].clone(), inputs[2].clone())
                    };
                    Ok(vec![inputs[0].attend(&key, &value, None, None, *options)?])
                })
                .unwrap();
                let (expected, generated_used_cudnn) =
                    run(&graph, &inputs, LoweringPolicy::generated_kernels());
                assert!(!generated_used_cudnn, "{label}");
                let (actual, used_cudnn) = run(&graph, &inputs, with_cudnn());
                assert!(used_cudnn, "{label} did not reach cuDNN");
                assert_matches(
                    &host(&actual[0]),
                    &host(&expected[0]),
                    OUTPUT_TOLERANCE,
                    &label,
                );

                let program_name = format!(
                    "attention_{}",
                    label.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
                );
                let Some((generated, calls_cudnn)) = run_program(
                    &graph,
                    &inputs,
                    Vec::new(),
                    &format!("{program_name}_generated"),
                ) else {
                    continue;
                };
                assert!(!calls_cudnn, "{label}");
                assert_matches(
                    &host(&generated[0]),
                    &host(&expected[0]),
                    OUTPUT_TOLERANCE,
                    &label,
                );
                let (library, calls_cudnn) =
                    run_program(&graph, &inputs, vec![Library::Cudnn], &program_name).unwrap();
                assert!(calls_cudnn, "{label} program does not call cuDNN");
                assert_matches(
                    &host(&library[0]),
                    &host(&expected[0]),
                    OUTPUT_TOLERANCE,
                    &label,
                );
            }
        }
    }
}

#[test]
fn test_attention_options_cudnn_lacks_stay_with_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let query = random(&[1, 2, 2, 16], DataType::F16, 4);
    let key = random(&[1, 2, 3, 16], DataType::F16, 5);
    let value = random(&[1, 2, 3, 16], DataType::F16, 6);
    let inputs = [("query", query), ("key", key), ("value", value)];
    let (graph, _) = Graph::capture(inputs.clone(), |inputs| {
        Ok(vec![inputs[0].attend(
            &inputs[1],
            &inputs[2],
            None,
            None,
            AttentionOptions {
                softcap: Some(5.0),
                ..AttentionOptions::default()
            },
        )?])
    })
    .unwrap();
    let (expected, _) = run(&graph, &inputs, LoweringPolicy::generated_kernels());
    let (actual, used_cudnn) = run(&graph, &inputs, with_cudnn());
    assert!(!used_cudnn);
    assert_matches(&host(&actual[0]), &host(&expected[0]), SAME, "softcap");
}

// Compares the rows of `tokens` queries that a sequence holds, since generated kernels leave the others unwritten.
fn assert_rows_close(
    actual: &Tensor,
    expected: &Tensor,
    case: &PagedCase,
    tolerance: Tolerance,
    label: &str,
) {
    let first = i64::from(case.query_offsets[0]);
    let last = i64::from(*case.query_offsets.last().unwrap());
    let held = |tensor: &Tensor| tensor.narrow(0, first, last - first).unwrap();
    assert_matches(
        &host(&held(actual)),
        &host(&held(expected)),
        tolerance,
        label,
    );
}

// One way sequences can use a paged cache.
#[derive(Debug, Clone, Copy)]
struct PagedCase {
    name: &'static str,
    tokens: i64,
    query_offsets: &'static [i32],
    page_offsets: &'static [i32],
    pages: &'static [i32],
    last_page_lengths: &'static [i32],
}

const PAGED_CASES: [PagedCase; 7] = [
    PagedCase {
        name: "prefill",
        tokens: 5,
        query_offsets: &[0, 3, 5],
        page_offsets: &[0, 2, 4],
        pages: &[0, 1, 3, 2],
        last_page_lengths: &[2, 1],
    },
    PagedCase {
        name: "decode",
        tokens: 2,
        query_offsets: &[0, 1, 2],
        page_offsets: &[0, 2, 4],
        pages: &[0, 1, 3, 2],
        last_page_lengths: &[2, 4],
    },
    PagedCase {
        name: "one sequence",
        tokens: 3,
        query_offsets: &[0, 3],
        page_offsets: &[0, 2],
        pages: &[2, 0],
        last_page_lengths: &[3],
    },
    PagedCase {
        name: "tokens no sequence holds",
        tokens: 5,
        query_offsets: &[1, 2, 4],
        page_offsets: &[0, 2, 3],
        pages: &[0, 1, 3],
        last_page_lengths: &[2, 2],
    },
    PagedCase {
        name: "a sequence with no keys",
        tokens: 4,
        query_offsets: &[0, 1, 4],
        page_offsets: &[0, 0, 2],
        pages: &[3, 2],
        last_page_lengths: &[0, 2],
    },
    PagedCase {
        name: "more queries than keys",
        tokens: 4,
        query_offsets: &[0, 3, 4],
        page_offsets: &[0, 1, 2],
        pages: &[1, 0],
        last_page_lengths: &[2, 3],
    },
    PagedCase {
        name: "a sequence with no queries",
        tokens: 2,
        query_offsets: &[0, 0, 2],
        page_offsets: &[0, 1, 2],
        pages: &[1, 0],
        last_page_lengths: &[2, 3],
    },
];

fn paged_graph(inputs: &[(&str, Tensor)], options: AttentionOptions) -> Graph {
    Graph::capture(inputs.to_vec(), |inputs| {
        let (output, lse) = inputs[0].paged_attention(
            &inputs[1],
            &inputs[2],
            Paging {
                query_offsets: &inputs[3],
                page_offsets: &inputs[4],
                pages: &inputs[5],
                last_page_lengths: &inputs[6],
            },
            options,
        )?;
        Ok(vec![output, lse])
    })
    .unwrap()
    .0
}

#[test]
fn test_paged_attention_through_cudnn_matches_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let option_sets = [
        ("plain", AttentionOptions::default()),
        (
            "causal",
            AttentionOptions {
                causal: true,
                ..AttentionOptions::default()
            },
        ),
        (
            "causal window",
            AttentionOptions {
                causal: true,
                window: Some(2),
                ..AttentionOptions::default()
            },
        ),
        (
            "window",
            AttentionOptions {
                window: Some(2),
                ..AttentionOptions::default()
            },
        ),
        (
            "scale",
            AttentionOptions {
                scale: Some(0.5),
                causal: true,
                ..AttentionOptions::default()
            },
        ),
    ];
    let (heads, kv_heads, page_size, cache_pages, dim) = (4_i64, 2_i64, 4_i64, 4_i64, 16_i64);
    for data_type in [DataType::F16, DataType::Bf16] {
        for index_type in [DataType::I32, DataType::I64] {
            for case in PAGED_CASES {
                let inputs = [
                    ("query", random(&[case.tokens, heads, dim], data_type, 7)),
                    (
                        "key",
                        random(&[cache_pages, page_size, kv_heads, dim], data_type, 8),
                    ),
                    (
                        "value",
                        random(&[cache_pages, page_size, kv_heads, dim], data_type, 9),
                    ),
                    ("query_offsets", indices(case.query_offsets, index_type)),
                    ("page_offsets", indices(case.page_offsets, index_type)),
                    ("pages", indices(case.pages, index_type)),
                    (
                        "last_page_lengths",
                        indices(case.last_page_lengths, index_type),
                    ),
                ];
                for (name, options) in &option_sets {
                    let label = format!("{data_type:?} {index_type:?} {} {name}", case.name);
                    let graph = paged_graph(&inputs, *options);
                    let (expected, generated_used_cudnn) =
                        run(&graph, &inputs, LoweringPolicy::generated_kernels());
                    assert!(!generated_used_cudnn, "{label}");
                    let (actual, used_cudnn) = run(&graph, &inputs, with_cudnn());
                    assert!(used_cudnn, "{label} did not reach cuDNN");
                    assert_rows_close(&actual[0], &expected[0], &case, OUTPUT_TOLERANCE, &label);
                    assert_rows_close(&actual[1], &expected[1], &case, LSE_TOLERANCE, &label);
                }
            }
        }
    }
}

#[test]
fn test_paged_attention_programs_through_cudnn_match_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let (heads, kv_heads, page_size, cache_pages, dim) = (4_i64, 2_i64, 4_i64, 4_i64, 16_i64);
    let options = AttentionOptions {
        causal: true,
        ..AttentionOptions::default()
    };
    for case in PAGED_CASES {
        let inputs = [
            (
                "query",
                random(&[case.tokens, heads, dim], DataType::Bf16, 7),
            ),
            (
                "key",
                random(&[cache_pages, page_size, kv_heads, dim], DataType::Bf16, 8),
            ),
            (
                "value",
                random(&[cache_pages, page_size, kv_heads, dim], DataType::Bf16, 9),
            ),
            ("query_offsets", indices(case.query_offsets, DataType::I32)),
            ("page_offsets", indices(case.page_offsets, DataType::I32)),
            ("pages", indices(case.pages, DataType::I32)),
            (
                "last_page_lengths",
                indices(case.last_page_lengths, DataType::I32),
            ),
        ];
        let graph = paged_graph(&inputs, options);
        let name = format!(
            "paged_{}",
            case.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_")
        );
        let Some((expected, calls_cudnn)) =
            run_program(&graph, &inputs, Vec::new(), &format!("{name}_generated"))
        else {
            return;
        };
        assert!(!calls_cudnn, "{name}");
        let (actual, calls_cudnn) =
            run_program(&graph, &inputs, vec![Library::Cudnn], &name).unwrap();
        assert!(calls_cudnn, "{name} does not call cuDNN");
        assert_rows_close(&actual[0], &expected[0], &case, OUTPUT_TOLERANCE, &name);
        assert_rows_close(&actual[1], &expected[1], &case, LSE_TOLERANCE, &name);
    }
}

#[test]
fn test_pages_cudnn_refuses_stay_with_generated_kernels() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let inputs = [
        ("query", random(&[2, 2, 16], DataType::F16, 7)),
        ("key", random(&[2, 3, 2, 16], DataType::F16, 8)),
        ("value", random(&[2, 3, 2, 16], DataType::F16, 9)),
        ("query_offsets", indices(&[0, 2], DataType::I32)),
        ("page_offsets", indices(&[0, 2], DataType::I32)),
        ("pages", indices(&[1, 0], DataType::I32)),
        ("last_page_lengths", indices(&[2], DataType::I32)),
    ];
    let graph = paged_graph(
        &inputs,
        AttentionOptions {
            causal: true,
            ..AttentionOptions::default()
        },
    );
    let (expected, _) = run(&graph, &inputs, LoweringPolicy::generated_kernels());
    let (actual, used_cudnn) = run(&graph, &inputs, with_cudnn());
    assert!(!used_cudnn);
    assert_matches(&host(&actual[0]), &host(&expected[0]), SAME, "output");
    assert_matches(&host(&actual[1]), &host(&expected[1]), SAME, "lse");
    if let Some((program, calls_cudnn)) =
        run_program(&graph, &inputs, vec![Library::Cudnn], "paged_odd")
    {
        assert!(!calls_cudnn);
        assert_matches(
            &host(&program[0]),
            &host(&expected[0]),
            OUTPUT_TOLERANCE,
            "program output",
        );
    }
}

#[test]
fn test_cudnn_paging_refuses_what_generated_kernels_refuse() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let (heads, kv_heads, page_size, cache_pages, dim) = (4_i64, 2_i64, 4_i64, 2_i64, 16_i64);
    let valid = [
        ("query", random(&[2, heads, dim], DataType::Bf16, 7)),
        (
            "key",
            random(&[cache_pages, page_size, kv_heads, dim], DataType::Bf16, 8),
        ),
        (
            "value",
            random(&[cache_pages, page_size, kv_heads, dim], DataType::Bf16, 9),
        ),
        ("query_offsets", indices(&[0, 2], DataType::I32)),
        ("page_offsets", indices(&[0, 1], DataType::I32)),
        ("pages", indices(&[0], DataType::I32)),
        ("last_page_lengths", indices(&[2], DataType::I32)),
    ];
    let graph = paged_graph(
        &valid,
        AttentionOptions {
            causal: true,
            ..AttentionOptions::default()
        },
    );
    let mut prepared = graph
        .prepare(
            GraphPrepareOptions::new(DEVICE)
                .with_cuda_options(GraphCudaOptions::new().with_lowering_policy(with_cudnn())),
        )
        .unwrap();
    let named = |changes: &[(&str, Tensor)]| {
        valid
            .iter()
            .map(|(name, tensor)| {
                let tensor = changes
                    .iter()
                    .find(|(changed, _)| changed == name)
                    .map_or(tensor, |(_, tensor)| tensor);
                ((*name).to_owned(), tensor.clone())
            })
            .collect::<HashMap<_, _>>()
    };
    prepared.run(&named(&[])).unwrap();
    assert!(ran_cudnn(&prepared.report().unwrap()));
    let invalid = [
        ("query_offsets", indices(&[0, 3], DataType::I32)),
        ("query_offsets", indices(&[-1, 2], DataType::I32)),
        ("page_offsets", indices(&[0, 2], DataType::I32)),
        ("pages", indices(&[2], DataType::I32)),
        ("last_page_lengths", indices(&[5], DataType::I32)),
    ];
    for change in invalid {
        let name = change.0;
        assert!(
            prepared.run(&named(&[change])).is_err(),
            "{name} was not refused"
        );
        // A refused run leaves the graph usable, with nothing written outside its buffers.
        prepared.run(&named(&[])).unwrap();
    }
}
