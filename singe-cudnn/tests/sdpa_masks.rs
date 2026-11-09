#![cfg(feature = "testing")]

// Each mask form of `sdpa_auto`, against scores computed by hand from what the band documents: `diagonal - left < key <= diagonal + right`.
// cuDNN 9.27 computed wrong scores from scalars that were compile-time constants, which raises no error, so only values can show it.

use half::f16;
use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::Stream};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    context::Context,
    data_type::DataType,
    error::Error,
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand,
            AttentionImplementation, AttentionMaskMode, AttentionScoreConfig,
            AttentionSequenceInputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

const BATCH: usize = 2;
const HEADS: usize = 2;
const QUERIES: usize = 64;
const KEYS: usize = 128;
const DIM: usize = 64;
const SCALE: f32 = 0.125;

// The rows each batch uses, shorter than its tensors, which only bottom-right alignment reads.
const QUERY_LENGTH: usize = 48;
const KEY_LENGTH: usize = 96;

#[derive(Debug, Clone, Copy)]
struct Case {
    name: &'static str,
    alignment: AttentionDiagonalAlignment,
    left: Option<i64>,
    right: Option<i64>,
    padding: bool,
}

const fn top_left(name: &'static str, left: Option<i64>, right: Option<i64>) -> Case {
    Case {
        name,
        alignment: AttentionDiagonalAlignment::TopLeft,
        left,
        right,
        padding: false,
    }
}

const fn bottom_right(
    name: &'static str,
    left: Option<i64>,
    right: Option<i64>,
    padding: bool,
) -> Case {
    Case {
        name,
        alignment: AttentionDiagonalAlignment::BottomRight,
        left,
        right,
        padding,
    }
}

const CASES: [Case; 8] = [
    top_left("top-left causal", None, Some(0)),
    top_left("top-left right", None, Some(8)),
    top_left("top-left left", Some(32), None),
    top_left("top-left window", Some(32), Some(8)),
    bottom_right("bottom-right causal", None, Some(0), false),
    bottom_right("bottom-right right", None, Some(8), false),
    bottom_right("bottom-right left, padded", Some(32), None, true),
    bottom_right("bottom-right window", Some(32), Some(8), false),
];

fn values(count: usize, seed: &mut u32) -> Vec<f16> {
    (0..count)
        .map(|_| {
            *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            f16::from_f32(f32::from(u16::try_from(*seed >> 16).unwrap()) / 65_536.0)
        })
        .collect()
}

// One head's rows of a `[B, H, S, D]` tensor.
fn row(tensor: &[f16], batch: usize, head: usize, position: usize) -> &[f16] {
    let start = (((batch * HEADS + head) * tensor.len() / (BATCH * HEADS)) / DIM + position) * DIM;
    &tensor[start..start + DIM]
}

// The output's rows this mask defines, each as `(batch, head, query, values)`.
fn reference(
    case: Case,
    query: &[f16],
    key: &[f16],
    value: &[f16],
) -> Vec<(usize, usize, usize, Vec<f32>)> {
    let bottom_right = matches!(case.alignment, AttentionDiagonalAlignment::BottomRight);
    let queries = if bottom_right { QUERY_LENGTH } else { QUERIES };
    let shift = if bottom_right {
        i64::try_from(KEY_LENGTH - QUERY_LENGTH).unwrap()
    } else {
        0
    };
    let mut rows = Vec::new();
    for batch in 0..BATCH {
        for head in 0..HEADS {
            for position in 0..queries {
                let diagonal = i64::try_from(position).unwrap() + shift;
                let q = row(query, batch, head, position);
                let scores = (0..KEYS)
                    .map(|index| {
                        let at = i64::try_from(index).unwrap();
                        let visible = case.left.is_none_or(|left| diagonal - left < at)
                            && case.right.is_none_or(|right| at <= diagonal + right)
                            && (!bottom_right || index < KEY_LENGTH);
                        visible.then(|| {
                            SCALE
                                * q.iter()
                                    .zip(row(key, batch, head, index))
                                    .map(|(q, k)| q.to_f32() * k.to_f32())
                                    .sum::<f32>()
                        })
                    })
                    .collect::<Vec<_>>();
                let most = scores
                    .iter()
                    .flatten()
                    .fold(f32::NEG_INFINITY, |most, score| most.max(*score));
                let weights = scores
                    .iter()
                    .map(|score| score.map_or(0.0, |score| (score - most).exp()))
                    .collect::<Vec<_>>();
                let total = weights.iter().sum::<f32>();
                let mut output = vec![0.0_f32; DIM];
                for (index, weight) in weights.iter().enumerate() {
                    for (output, value) in output.iter_mut().zip(row(value, batch, head, index)) {
                        *output += weight / total * value.to_f32();
                    }
                }
                rows.push((batch, head, position, output));
            }
        }
    }
    rows
}

fn run(
    cudnn: &Context,
    stream: &Stream,
    case: Case,
    implementation: AttentionImplementation,
    scale: TensorSpec,
    [query, key, value]: [&[f16]; 3],
) -> Result<Vec<f16>, Error> {
    let context = cudnn.cuda_context();
    let shape = |rows: usize| {
        Shape::contiguous([
            i64::try_from(BATCH).unwrap(),
            i64::try_from(HEADS).unwrap(),
            i64::try_from(rows).unwrap(),
            i64::try_from(DIM).unwrap(),
        ])
        .unwrap()
    };
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let q = graph.tensor(TensorSpec::new(DataType::F16, shape(QUERIES)));
    let k = graph.tensor(TensorSpec::new(DataType::F16, shape(KEYS)));
    let v = graph.tensor(TensorSpec::new(DataType::F16, shape(KEYS)));
    let scale = graph.tensor(scale);
    let lengths = Shape::contiguous([i64::try_from(BATCH).unwrap(), 1, 1, 1]).unwrap();
    let bottom_right = matches!(case.alignment, AttentionDiagonalAlignment::BottomRight);
    let query_length = graph.tensor(TensorSpec::new(DataType::I32, lengths.clone()));
    let key_length = graph.tensor(TensorSpec::new(DataType::I32, lengths));
    let band = AttentionDiagonalBand::new(case.alignment)
        .with_left_window(case.left)
        .with_right_window(case.right)
        .with_padding(case.padding);
    let mut scores =
        AttentionScoreConfig::new().with_mask_mode(AttentionMaskMode::DiagonalBand(band));
    if bottom_right {
        scores = scores
            .with_sequence_inputs(AttentionSequenceInputs::per_batch(query_length, key_length));
    }
    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale,
        },
        AttentionConfig::new()
            .with_implementation(implementation)
            .with_score_config(scores),
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor()),
    )?;
    let output = outputs.output;
    graph.mark_as_output(output)?;
    let compiled = graph.compile(
        Some(cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let q_memory = DeviceMemory::from_slice(context, query).unwrap();
    let k_memory = DeviceMemory::from_slice(context, key).unwrap();
    let v_memory = DeviceMemory::from_slice(context, value).unwrap();
    let query_lengths =
        DeviceMemory::from_slice(context, &[i32::try_from(QUERY_LENGTH).unwrap(); BATCH]).unwrap();
    let key_lengths =
        DeviceMemory::from_slice(context, &[i32::try_from(KEY_LENGTH).unwrap(); BATCH]).unwrap();
    let count = BATCH * HEADS * QUERIES * DIM;
    let mut o_memory = DeviceMemory::<f16>::zeroed(context, count).unwrap();
    let workspace_bytes = compiled.workspace_size();
    let mut workspace = (workspace_bytes > 0).then(|| {
        DeviceMemory::<u8>::zeroed(context, workspace_bytes)
            .unwrap()
            .into_uninit()
    });
    let mut bindings = compiled.bindings().unwrap();
    bindings
        .set_input(q, &q_memory)
        .unwrap()
        .set_input(k, &k_memory)
        .unwrap()
        .set_input(v, &v_memory)
        .unwrap();
    if bottom_right {
        bindings
            .set_input(query_length, &query_lengths)
            .unwrap()
            .set_input(key_length, &key_lengths)
            .unwrap();
    }
    bindings.set(output, &mut o_memory).unwrap();
    let mut arguments = ExecuteArgs::new(&bindings);
    if let Some(workspace) = workspace.as_mut() {
        arguments = arguments.with_workspace(workspace);
    }
    // SAFETY: the inputs are initialized, the output is exclusive, and every buffer outlives the synchronization below.
    unsafe { compiled.execute(cudnn, arguments) }?;
    stream.synchronize()?;
    drop(bindings);
    let mut host = vec![f16::ZERO; count];
    o_memory.copy_to_host(&mut host).unwrap();
    Ok(host)
}

#[test]
fn test_every_diagonal_band_matches_a_reference_or_is_refused() {
    let Ok(cuda) = CudaContext::create() else {
        return;
    };
    let mut cudnn = Context::create(&cuda).unwrap();
    let stream = cuda.create_stream().unwrap();
    cudnn.set_stream(Some(&stream)).unwrap();

    let mut seed = 7;
    let query = values(BATCH * HEADS * QUERIES * DIM, &mut seed);
    let key = values(BATCH * HEADS * KEYS * DIM, &mut seed);
    let value = values(BATCH * HEADS * KEYS * DIM, &mut seed);
    let mut wrong = Vec::new();
    for case in CASES {
        for implementation in [
            AttentionImplementation::Composite,
            AttentionImplementation::Unified,
        ] {
            let output = run(
                &cudnn,
                &stream,
                case,
                implementation,
                TensorSpec::scalar_f32(SCALE),
                [&query, &key, &value],
            );
            let output = match output {
                Ok(output) => output,
                Err(error) => {
                    eprintln!("{} {implementation:?}: refused: {error}", case.name);
                    continue;
                }
            };
            let mut worst = 0.0_f32;
            for (batch, head, position, expected) in reference(case, &query, &key, &value) {
                for (actual, expected) in row(&output, batch, head, position).iter().zip(&expected)
                {
                    worst = worst.max((actual.to_f32() - expected).abs());
                }
            }
            eprintln!("{} {implementation:?}: worst difference {worst}", case.name);
            if worst > 2e-3 {
                wrong.push((case.name, implementation, worst));
            }
        }
    }
    assert!(wrong.is_empty(), "{wrong:?}");
}

// A caller's constant scale runs as a run-time scalar too, or attention would misread it as it does any constant.
#[test]
fn test_a_constant_scale_runs_as_a_scalar() {
    let Ok(cuda) = CudaContext::create() else {
        return;
    };
    let mut cudnn = Context::create(&cuda).unwrap();
    let stream = cuda.create_stream().unwrap();
    cudnn.set_stream(Some(&stream)).unwrap();
    let mut seed = 11;
    let query = values(BATCH * HEADS * QUERIES * DIM, &mut seed);
    let key = values(BATCH * HEADS * KEYS * DIM, &mut seed);
    let value = values(BATCH * HEADS * KEYS * DIM, &mut seed);
    let case = CASES[0];
    let output = run(
        &cudnn,
        &stream,
        case,
        AttentionImplementation::Unified,
        TensorSpec::constant(SCALE),
        [&query, &key, &value],
    )
    .unwrap();
    for (batch, head, position, expected) in reference(case, &query, &key, &value) {
        for (actual, expected) in row(&output, batch, head, position).iter().zip(&expected) {
            assert!((actual.to_f32() - expected).abs() <= 2e-3);
        }
    }
}
