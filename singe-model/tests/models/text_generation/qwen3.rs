#[cfg(feature = "cuda_13_4")]
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{self, Command},
    time::Duration,
};

#[cfg(feature = "cuda_13_4")]
use serde_json::Value;
#[cfg(feature = "cuda_13_4")]
use singe::generation::{Decoder, DecoderStepGraph};
#[cfg(feature = "cuda_13_4")]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions, generate},
};
#[cfg(feature = "cuda_13_4")]
use singe::{
    backend::cuda::{
        export::CudaExport,
        tuning::{CandidateOutcome, GraphTuneOptions, Measurement, TuningReport, tune_graph},
    },
    graph::GraphCudaOptions,
    testing::{CompiledProgram, build_generation_loop, program_target},
};
use singe_model::text::text_generation::qwen3::{self, Qwen3Layers};

use singe_testing::{
    Tolerance,
    decoder::{Reference, Suite, decoder},
};
#[cfg(feature = "cuda_13_4")]
use singe_testing::{
    decoder::{
        ProgramDecoder, assert_one_specialization, best_token, host_logits, length, tune_step,
    },
    integers,
};

const SUITE: Suite<Qwen3Layers> = Suite {
    references: singe_testing::references!(),
    task: "text_generation",
    family: "qwen3",
    models: &["tiny"],
    load: |directory, device, data_type| qwen3::load(directory, device, data_type).unwrap(),
    f32_tolerance: Tolerance::absolute(0.001, "F32 leaves room for arithmetic alone"),
    bf16_tolerance: Tolerance::absolute(
        0.05,
        "BF16 keeps eight mantissa bits, and Singe accumulates attention in F32 where PyTorch rounds each step",
    ),
    programs_skipped: None,
};

#[test]
fn test_tiny_qwen3_matches_pytorch_on_cpu() {
    SUITE.check_logits(Device::Cpu);
}

#[test]
fn test_tiny_qwen3_decodes_on_cpu() {
    SUITE.check_decoding(Device::Cpu);
}

#[test]
fn test_generation_stops_at_a_stop_token_and_returns_it() {
    let directory = SUITE.directory("tiny");
    let reference = Reference::read(&directory, DataType::F32);
    let model = qwen3::load(directory, Device::Cpu, DataType::F32).unwrap();
    let first = reference.tokens()[0];
    let mut decoder = decoder(&model, DecoderOptions::new(1, 32), false);
    let options = GenerationOptions {
        max_new_tokens: 8,
        stop_tokens: vec![first],
        sampling: None,
        ..GenerationOptions::default()
    };
    assert_eq!(
        generate(&mut decoder, &reference.prompt, &options, None, |_| Ok(())).unwrap(),
        [first]
    );
}

// A compiled step is bound to the weights it was prepared with, so another model's decoder must not take it.
#[test]
fn test_decoders_refuse_a_step_compiled_for_other_weights() {
    let directory = SUITE.directory("tiny");
    let model = qwen3::load(&directory, Device::Cpu, DataType::F32).unwrap();
    let other = qwen3::load(&directory, Device::Cpu, DataType::F32).unwrap();
    let options = DecoderOptions::new(1, 8);
    let compiled = decoder(&model, options, true);
    let mut same = decoder(&model, options, false);
    assert!(same.share_compiled(&compiled).is_ok());
    let mut foreign = CachedDecoder::create(&other, options).unwrap();
    assert!(foreign.share_compiled(&compiled).is_err());
}

#[cfg(feature = "cuda_13_4")]
fn index(token: i64) -> usize {
    usize::try_from(token).unwrap()
}

#[cfg(feature = "cuda_13_4")]
fn real_snapshot() -> PathBuf {
    singe_testing::snapshot("QWEN3")
}

#[cfg(feature = "cuda_13_4")]
fn real_reference() -> Value {
    // The checks below read each step's logits beside its tokens, so the tensors join the metadata they belong to.
    let reference = singe_testing::reference!("text_generation/qwen3/real");
    let mut metadata = reference.metadata().clone();
    for (index, case) in metadata["cases"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        let steps = case["steps"].as_array_mut().unwrap();
        for key in ["top_logits", "top_logits_f32"] {
            let values = reference.values(&format!("case{index}.{key}"));
            let width = values.len() / steps.len();
            for (step, row) in steps.iter_mut().zip(values.chunks(width)) {
                step[key] = Value::from(
                    row.iter()
                        .map(|&value| f64::from(value))
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
    metadata
}

#[cfg(feature = "cuda_13_4")]
#[expect(
    clippy::cast_possible_truncation,
    reason = "the reference's logits are F32 values"
)]
fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_f64().unwrap() as f32)
        .collect()
}

// How a comparison runs its step.
#[cfg(feature = "cuda_13_4")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compiled {
    No,
    Yes,
    Tuned,
    // Through the exported C++ of a step, with every library allowed.
    Program,
}

// How far logits sit from the reference, and how many steps' best token differs from the reference's on a tie.
#[cfg(feature = "cuda_13_4")]
#[derive(Debug, Default)]
struct RealComparison {
    worst: f32,
    tied_flips: usize,
}

// Follows the reference's tokens from one case's prompt on `decoder`, noting how far its logits sit.
#[cfg(feature = "cuda_13_4")]
fn follow_case(
    decoder: &mut impl Decoder,
    case: &Value,
    logits_key: &str,
    steps_per_case: usize,
    tie: f32,
    comparison: &mut RealComparison,
) {
    let prompt = integers(&case["input_ids"]);
    let steps = case["steps"].as_array().unwrap();
    let steps = &steps[..steps_per_case.min(steps.len())];
    let mut logits = decoder.step(&[&prompt]).unwrap();
    for (number, step) in steps.iter().enumerate() {
        let host = host_logits(&logits);
        let top_tokens = integers(&step["top_tokens"]);
        let here = top_tokens
            .iter()
            .map(|&token| host[index(token)])
            .collect::<Vec<_>>();
        let there = floats(&step[logits_key]);
        for (here, there) in here.iter().zip(&there) {
            comparison.worst = comparison.worst.max((here - there).abs());
        }
        let token = step["token"].as_i64().unwrap();
        let best = best_token(&host);
        if best != token {
            // Rounding can only flip a token whose logit the reference has within `tie` of its own choice.
            let chosen = floats(&step["top_logits"])[top_tokens
                .iter()
                .position(|&candidate| candidate == token)
                .unwrap()];
            let rival = top_tokens
                .iter()
                .position(|&candidate| candidate == best)
                .map(|rival| floats(&step["top_logits"])[rival]);
            assert!(
                rival.is_some_and(|rival| (chosen - rival).abs() <= tie),
                "{} at step {number}: chose {best}, reference {token}, reference logits {:?} for {top_tokens:?}, here {here:?}",
                case["prompt"],
                floats(&step["top_logits"]),
            );
            comparison.tied_flips += 1;
        }
        logits = decoder.step(&[&[token]]).unwrap();
    }
}

// Follows the reference's tokens from each prompt, comparing the logits of its top tokens under `logits_key`.
#[cfg(feature = "cuda_13_4")]
fn compare_real(
    snapshot: &Path,
    device: Device,
    data_type: DataType,
    compiled: Compiled,
    logits_key: &str,
    steps_per_case: usize,
    tie: f32,
) -> RealComparison {
    let reference = real_reference();
    let model = qwen3::load(snapshot, device, data_type).unwrap();
    let mut comparison = RealComparison::default();
    let cases = reference["cases"].as_array().unwrap();
    let records = (compiled == Compiled::Tuned).then(|| {
        // Records hold for the cache they were tuned on, so every case shares one size.
        let capacity = cases
            .iter()
            .map(|case| {
                length(&integers(&case["input_ids"])) + length(case["steps"].as_array().unwrap())
            })
            .max()
            .unwrap();
        let decoder = CachedDecoder::create(&model, DecoderOptions::new(1, capacity)).unwrap();
        let step = decoder.step_graph().unwrap();
        (
            capacity,
            tune_step(&step.graph, &step.inputs, device, Measurement::GraphReplay),
        )
    });
    // One program serves every case, and each case starts a new instance of it.
    let program = (compiled == Compiled::Program).then(|| {
        let capacity = cases
            .iter()
            .map(|case| {
                length(&integers(&case["input_ids"])) + length(case["steps"].as_array().unwrap())
            })
            .max()
            .unwrap();
        let options = DecoderOptions::new(1, capacity);
        let decoder = CachedDecoder::create(&model, options).unwrap();
        let step = decoder.step_graph().unwrap();
        // A step that reads four tokens as well, as a joining prompt would through `RunningBatch`.
        let chunk = decoder.chunk_step_graph(3).unwrap();
        let program = CudaExport::new(program_target(device).unwrap())
            .export_steps(&[("t1", &step.graph), ("t4", &chunk.graph)])
            .unwrap();
        let binary = CompiledProgram::create_set(&program, "qwen3_generation").unwrap();
        // The two steps hold the model's weights once.
        assert_eq!(
            binary.weights_bytes(),
            binary.program().render("model").unwrap().weights.len()
        );
        (options, step, binary)
    });
    for case in cases {
        if let Some((options, step, binary)) = &program {
            let mut decoder = ProgramDecoder::create(binary, step, *options);
            follow_case(
                &mut decoder,
                case,
                logits_key,
                steps_per_case,
                tie,
                &mut comparison,
            );
            continue;
        }
        let prompt = integers(&case["input_ids"]);
        let steps = case["steps"].as_array().unwrap();
        let steps = &steps[..steps_per_case.min(steps.len())];
        let capacity = records.as_ref().map_or_else(
            || length(&prompt) + length(steps),
            |(capacity, _)| *capacity,
        );
        let mut decoder = decoder(
            &model,
            DecoderOptions::new(1, capacity),
            compiled != Compiled::No,
        );
        if let Some((_, records)) = &records {
            decoder
                .compile_with(GraphCudaOptions::new().with_tuning_records(records.clone()))
                .unwrap();
        }
        follow_case(
            &mut decoder,
            case,
            logits_key,
            steps_per_case,
            tie,
            &mut comparison,
        );
        if compiled != Compiled::No {
            assert_one_specialization(&decoder);
        }
    }

    if let Some((options, step, _)) = &program {
        check_generation_loop(step, *options, &integers(&cases[0]["input_ids"]), device);
    }
    eprintln!("{data_type:?} on {device}, {compiled:?}: {comparison:?}");
    comparison
}

// The loop an export writes beside real Qwen3's step generates, with nothing of Singe, what Singe's decoder generates through the same program.
#[cfg(feature = "cuda_13_4")]
fn check_generation_loop(
    step: &DecoderStepGraph,
    options: DecoderOptions,
    prompt: &[i64],
    device: Device,
) {
    const NEW_TOKENS: usize = 8;

    let program = CudaExport::new(program_target(device).unwrap())
        .with_decoding(step.layout)
        .export(&step.graph)
        .unwrap();
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../target/programs/qwen3-generate-{}",
        process::id()
    ));
    let _ = fs::remove_dir_all(&directory);
    let executable = build_generation_loop(&program, &directory).unwrap();
    let output = Command::new(&executable)
        .arg(NEW_TOKENS.to_string())
        .args(prompt.iter().map(i64::to_string))
        .current_dir(&directory)
        .output()
        .unwrap();
    let _ = fs::remove_dir_all(&directory);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let generated = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| line.parse::<i64>().unwrap())
        .collect::<Vec<_>>();
    let compiled = CompiledProgram::create(&program, "qwen3_generate").unwrap();
    let mut decoder = ProgramDecoder::create(&compiled, step, options);
    let expected = generate(
        &mut decoder,
        prompt,
        &GenerationOptions {
            max_new_tokens: NEW_TOKENS,
            ..GenerationOptions::default()
        },
        None,
        |_| Ok(()),
    )
    .unwrap();
    eprintln!("real qwen3 through its generation loop: {generated:?}");
    assert_eq!(generated, expected);
}

#[cfg(feature = "cuda_13_4")]
#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_qwen3_f32_logits_match_pytorch_f32_on_cpu() {
    let snapshot = real_snapshot();
    let comparison = compare_real(
        &snapshot,
        Device::Cpu,
        DataType::F32,
        Compiled::No,
        "top_logits_f32",
        4,
        0.25,
    );
    assert!(comparison.worst <= 0.001, "{comparison:?}");
    assert_eq!(comparison.tied_flips, 0);
}

#[cfg(feature = "cuda_13_4")]
mod cuda {
    use std::collections::BTreeMap;

    use super::*;

    // Prints each record the tunes did not agree on, with what timing every value it took against the default showed in each tune.
    // A value with no line in a tune did not reach that tune's comparisons.
    fn print_differing_records(
        measurement: Measurement,
        kept: &[BTreeMap<String, String>],
        reports: &[TuningReport],
    ) {
        let mut keys = kept.iter().flat_map(BTreeMap::keys).collect::<Vec<_>>();
        keys.sort_unstable();
        keys.dedup();
        for key in keys {
            let values = kept
                .iter()
                .map(|records| records.get(key))
                .collect::<Vec<_>>();
            if values.iter().all(|value| *value == values[0]) {
                continue;
            }

            eprintln!("{measurement:?}: tunes differ in {key}");
            for (tune, value) in values.iter().enumerate() {
                eprintln!("  tune {tune} kept {value:?}");
            }

            let mut distinct = values.iter().flatten().collect::<Vec<_>>();
            distinct.sort_unstable();
            distinct.dedup();
            for value in distinct {
                eprintln!("  {value}:");
                for (tune, report) in reports.iter().enumerate() {
                    let compared = report.choices.iter().flat_map(|choice| {
                        choice
                            .comparisons
                            .iter()
                            .chain(&choice.between_winners)
                            .map(move |compared| (&choice.key, compared))
                    });
                    for (point, compared) in compared {
                        if compared.choice.contains(value.as_str()) {
                            eprintln!(
                                "    tune {tune}: {:?} {:?} as {} at {point}",
                                compared.comparison, compared.failure, compared.choice
                            );
                        }
                    }
                }
            }
        }
    }

    // One BF16 unit at the logits' magnitude, between 16 and 32, is an eighth, and each side rounds.
    const BF16_TIE: f32 = 0.25;

    // PyTorch's own BF16 logits differ from its F32 logits by 0.44, and its F16 logits by 0.055.
    const BF16_TOLERANCE: f32 = 0.75;
    const F16_TOLERANCE: f32 = 0.1;

    #[test]
    fn test_tiny_qwen3_matches_pytorch_on_cuda() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        SUITE.check_logits(Device::Cuda(0));
    }

    // A graph change must not lose the first module without anyone noticing: every layer of a decode step runs it.
    #[test]
    fn test_a_compiled_qwen3_step_runs_its_cache_write_module() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        for data_type in [DataType::F32, DataType::Bf16] {
            let directory = SUITE.directory(SUITE.models[0]);
            let model = qwen3::load(&directory, Device::Cuda(0), data_type).unwrap();
            let reference = Reference::read(&directory, data_type);
            let mut decoder = decoder(&model, DecoderOptions::new(1, 32), true);
            decoder.step(&[&reference.prompt]).unwrap();
            let report = decoder.executor().unwrap().report().unwrap();
            assert!(
                report.contains("module: rotary cache write"),
                "{data_type:?}\n{report}"
            );
        }
    }

    #[test]
    fn test_tiny_qwen3_decodes_on_cuda() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        SUITE.check_decoding(Device::Cuda(0));
    }

    // Fused kernels differ in where they round: a separate kernel rounds values to `BF16` that a fused one keeps in `F32`, so the logits of a tuned winner, or of a compiled step against an eager one, sit further apart than on one path, about as far as with fusion off, and a token can flip across two such errors.
    const FUSED_TOLERANCE: f32 = 1.0;
    const FUSED_TIE: f32 = 1.0;

    fn check_real_bf16(compiled: Compiled) {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let snapshot = real_snapshot();
        let (tie, tolerance) = if compiled == Compiled::Tuned {
            (FUSED_TIE, FUSED_TOLERANCE)
        } else {
            (BF16_TIE, BF16_TOLERANCE)
        };
        let comparison = compare_real(
            &snapshot,
            Device::Cuda(0),
            DataType::Bf16,
            compiled,
            "top_logits",
            usize::MAX,
            tie,
        );
        assert!(comparison.worst <= tolerance, "{comparison:?}");
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_eager_bf16_follows_pytorch() {
        check_real_bf16(Compiled::No);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_compiled_bf16_follows_pytorch() {
        check_real_bf16(Compiled::Yes);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_generates_through_its_program() {
        check_real_bf16(Compiled::Program);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_tuned_bf16_follows_pytorch() {
        check_real_bf16(Compiled::Tuned);
    }

    // Tuning under the default budgets, with a cache big enough that holding it several times over would not fit on an 8 GiB GPU.
    // A token costs 115 KB of cache here, so this cache is about 2 GiB.
    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_tunes_with_a_large_cache() {
        // A second tune shows whether the first lists the same choice points, each a full tune of the model.
        const TUNES: usize = 2;

        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let device = Device::Cuda(0);
        let model = qwen3::load(real_snapshot(), device, DataType::Bf16).unwrap();
        let step = CachedDecoder::create(&model, DecoderOptions::new(1, 18_725))
            .unwrap()
            .step_graph()
            .unwrap();
        let build_directory =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../target/programs/tuning");
        let mut self_comparisons = 0;
        let mut self_wins = 0;
        // Programs tune through the same choice points, which the tiny models' tuned tests cover.
        let measurement = Measurement::GraphReplay;
        let mut listed = Vec::new();
        let mut kept = Vec::new();
        let mut reports = Vec::new();
        for tune in 0..TUNES {
            let records = tune_graph(
                &step.graph,
                &step.inputs,
                device,
                &GraphTuneOptions {
                    measurement,
                    build_directory: build_directory.clone(),
                    plan_budget: if tune == 0 {
                        Duration::from_secs(60)
                    } else {
                        Duration::ZERO
                    },
                    self_comparisons: true,
                    ..GraphTuneOptions::default()
                },
            )
            .unwrap();
            let report = records.report().unwrap();
            eprintln!(
                "{measurement:?}, tune {tune}, {} records:\n{report}",
                records.len()
            );
            let library_candidates = |name: &str| {
                report
                    .choices
                    .iter()
                    .flat_map(|choice| &choice.candidates)
                    .filter(|candidate| candidate.choice.contains(name))
                    .fold((0, 0, 0), |(listed, timed, refused), candidate| {
                        (
                            listed + 1,
                            timed
                                + usize::from(matches!(
                                    candidate.outcome,
                                    CandidateOutcome::Timed { .. }
                                )),
                            refused
                                + usize::from(matches!(
                                    candidate.outcome,
                                    CandidateOutcome::Refused
                                )),
                        )
                    })
            };
            eprintln!(
                "listed, timed, refused: cuBLASLt {:?}, cuDNN {:?}",
                library_candidates("CublasLt("),
                library_candidates("Cudnn(")
            );
            for compared in &report.self_comparisons {
                eprintln!(
                    "self: {:?} in turn {} {}",
                    compared.in_rounds,
                    compared.in_turn_ppm,
                    &compared.key[..compared.key.len().min(60)]
                );
            }

            for choice in &report.choices {
                for compared in &choice.comparisons {
                    eprintln!(
                        "compared: {:?} {:?} {} of {}",
                        compared.comparison,
                        compared.failure,
                        &compared.choice[..compared.choice.len().min(60)],
                        &choice.key[..choice.key.len().min(60)]
                    );
                }
                for compared in &choice.between_winners {
                    eprintln!(
                        "between: {:?} {:?} {} at {}",
                        compared.comparison, compared.failure, compared.choice, choice.key
                    );
                }
            }

            assert!(!records.is_empty(), "{report}");
            assert_eq!(report.problems().count(), 0, "{report}");
            for check in &report.checks {
                assert!(check.default_failure.is_none(), "{report}");
                assert!(!check.default_disagreed, "{report}");
            }

            self_comparisons += report.self_comparisons.len();
            self_wins += report
                .self_comparisons
                .iter()
                .filter(|compared| {
                    compared.in_rounds.wins() || compared.in_rounds.inverted().wins()
                })
                .count();

            // Schedules follow the plan chosen, so only the other choice points must repeat.
            listed.push(
                report
                    .choices
                    .iter()
                    .map(|choice| choice.key.clone())
                    .filter(|key| !key.starts_with("schedule"))
                    .collect::<Vec<_>>(),
            );
            kept.push(
                records
                    .entries()
                    .map(|(key, record)| (key.to_owned(), format!("{:?}", record.choice)))
                    .collect::<BTreeMap<_, _>>(),
            );
            reports.push(report.clone());
        }

        for later in &listed[1..] {
            assert_eq!(&listed[0], later, "{measurement:?}");
        }

        print_differing_records(measurement, &kept, &reports);

        // Timing a plan against itself shows what timing alone can do, and that must not look like a win.
        assert_eq!(self_wins, 0, "of {self_comparisons} self comparisons");
    }

    #[test]
    fn test_a_tuned_tiny_qwen3_compiles_and_exports_from_its_records() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        SUITE.check_tuned(Device::Cuda(0));
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_f16_logits_match_pytorch_f32() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let snapshot = real_snapshot();
        for compiled in [Compiled::No, Compiled::Yes] {
            let comparison = compare_real(
                &snapshot,
                Device::Cuda(0),
                DataType::F16,
                compiled,
                "top_logits_f32",
                usize::MAX,
                BF16_TIE,
            );
            assert!(comparison.worst <= F16_TOLERANCE, "{comparison:?}");
        }
    }

    // A fused kernel rounds once where eager operations round at each step, so half-precision logits differ by rounding, and best tokens may differ at a tie.
    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_compiled_logits_differ_from_eager_by_rounding() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let snapshot = real_snapshot();
        let device = Device::Cuda(0);
        let model = qwen3::load(snapshot, device, DataType::Bf16).unwrap();
        let mut worst = 0.0_f32;
        for case in real_reference()["cases"].as_array().unwrap() {
            let prompt = integers(&case["input_ids"]);
            let steps = case["steps"].as_array().unwrap();
            let capacity = length(&prompt) + length(steps);
            let options = DecoderOptions::new(1, capacity);
            let mut eager = decoder(&model, options, false);
            let mut compiled = decoder(&model, options, true);
            let mut eager_logits = eager.step(&[&prompt]).unwrap();
            let mut compiled_logits = compiled.step(&[&prompt]).unwrap();
            for step in steps {
                let eager_host = host_logits(&eager_logits);
                let compiled_host = host_logits(&compiled_logits);
                for (left, right) in eager_host.iter().zip(&compiled_host) {
                    worst = worst.max((left - right).abs());
                }
                let eager_best = best_token(&eager_host);
                let compiled_best = best_token(&compiled_host);
                assert!(
                    (eager_host[index(eager_best)] - eager_host[index(compiled_best)]).abs()
                        <= BF16_TIE,
                    "{}: eager chose {eager_best}, compiled {compiled_best}",
                    case["prompt"],
                );
                let token = step["token"].as_i64().unwrap();
                eager_logits = eager.step(&[&[token]]).unwrap();
                compiled_logits = compiled.step(&[&[token]]).unwrap();
            }
            assert_one_specialization(&compiled);
        }
        eprintln!("worst eager and compiled logit difference: {worst}");
        // A compiled prompt step fuses where the eager one does not, as every compiled step does.
        assert!(worst <= FUSED_TOLERANCE, "differs by {worst}");
    }

    // Free-running tokens feed each step back, so they match PyTorch until a tie, which rounding may break either way.
    fn check_free_running(compiled: bool) {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let snapshot = real_snapshot();
        let device = Device::Cuda(0);
        let model = qwen3::load(snapshot, device, DataType::Bf16).unwrap();
        for case in real_reference()["cases"].as_array().unwrap() {
            let prompt = integers(&case["input_ids"]);
            let steps = case["steps"].as_array().unwrap();
            let capacity = length(&prompt) + length(steps);
            let mut decoder = decoder(&model, DecoderOptions::new(1, capacity), compiled);
            let options = GenerationOptions {
                max_new_tokens: steps.len(),
                ..GenerationOptions::default()
            };
            let tokens = generate(&mut decoder, &prompt, &options, None, |_| Ok(())).unwrap();
            assert_eq!(tokens.len(), steps.len());
            let difference = steps
                .iter()
                .zip(&tokens)
                .position(|(step, &token)| step["token"].as_i64().unwrap() != token);
            eprintln!(
                "{}, compiled: {compiled}: first difference from pytorch {difference:?}",
                case["prompt"]
            );
            if let Some(number) = difference {
                let step = &steps[number];
                let logits = floats(&step["top_logits"]);
                let rank = integers(&step["top_tokens"])
                    .iter()
                    .position(|&token| token == tokens[number])
                    .unwrap();
                assert!(
                    logits[0] - logits[rank] <= BF16_TIE,
                    "{} at step {number}: {} against {logits:?}",
                    case["prompt"],
                    tokens[number],
                );
            }
        }
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_eager_generation_matches_pytorch_until_a_tie() {
        check_free_running(false);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_compiled_generation_matches_pytorch_until_a_tie() {
        check_free_running(true);
    }
}
