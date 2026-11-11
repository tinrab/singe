use singe::Device;
#[cfg(feature = "cuda_13_4")]
use singe::testing::maybe_lock_device;
use singe_model::text::text_generation::qwen3_5::{self, Qwen35Layers};

use singe_testing::{Tolerance, decoder::Suite};

const SUITE: Suite<Qwen35Layers> = Suite {
    references: singe_testing::references!(),
    task: "text_generation",
    family: "qwen3_5",
    models: &["tiny"],
    load: |directory, device, data_type| qwen3_5::load(directory, device, data_type).unwrap(),
    f32_tolerance: Tolerance::absolute(0.001, "F32 leaves room for arithmetic alone"),
    bf16_tolerance: Tolerance::absolute(
        0.05,
        "BF16 keeps eight mantissa bits, and Singe accumulates attention in F32 where PyTorch rounds each step",
    ),
};

#[test]
fn test_tiny_qwen3_5_matches_transformers_on_cpu() {
    SUITE.check_logits(Device::Cpu);
}

#[test]
fn test_tiny_qwen3_5_decodes_on_cpu() {
    SUITE.check_decoding(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_qwen3_5_matches_transformers_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_logits(Device::Cuda(0));
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_qwen3_5_decodes_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_decoding(Device::Cuda(0));
}

// The real model follows each of PyTorch's greedy tokens from a chat prompt, so every step's logits compare with its own, and a token may differ only where PyTorch's top two nearly tie.
#[cfg(feature = "cuda_13_4")]
mod real {
    use std::{
        fs,
        path::PathBuf,
        process::{self, Command},
    };

    use singe::{
        DataType, Device,
        backend::cuda::{self, export::CudaExport},
        generation::{CachedDecoder, Decoder, DecoderOptions},
        testing::{build_set_generation_loop, maybe_lock_device, program_target},
    };
    use singe_model::text::text_generation::qwen3_5;
    use singe_testing::{
        decoder::{assert_one_specialization, best_token, decoder, host_logits, length},
        integers,
    };

    // One BF16 unit at the logits' magnitude, between 16 and 32, is an eighth, and each side rounds.
    const BF16_TIE: f32 = 0.25;

    // How far Singe's BF16 logits may sit from PyTorch's, which round at other points.
    const BF16_TOLERANCE: f32 = 0.75;

    fn index(token: i64) -> usize {
        usize::try_from(token).unwrap()
    }

    fn check_real_bf16(compiled: bool) {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let reference = singe_testing::reference!("text_generation/qwen3_5/real");
        let model = qwen3_5::load(
            singe_testing::snapshot("QWEN35"),
            Device::Cuda(0),
            DataType::Bf16,
        )
        .unwrap();
        let (mut worst, mut flips) = (0.0_f32, 0);
        for (number, case) in reference.metadata()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let prompt = integers(&case["input_ids"]);
            let steps = case["steps"].as_array().unwrap();
            let top_logits = reference.values(&format!("case{number}.top_logits"));
            let width = top_logits.len() / steps.len();
            let capacity = length(&prompt) + length(steps);
            let mut decoder = decoder(&model, DecoderOptions::new(1, capacity), compiled);
            let mut logits = decoder.step(&[&prompt]).unwrap();
            for (step, expected) in steps.iter().zip(top_logits.chunks(width)) {
                let host = host_logits(&logits);
                let top_tokens = integers(&step["top_tokens"]);
                for (&token, expected) in top_tokens.iter().zip(expected) {
                    worst = worst.max((host[index(token)] - expected).abs());
                }
                let token = step["token"].as_i64().unwrap();
                let best = best_token(&host);
                if best != token {
                    let rival = top_tokens.iter().position(|&candidate| candidate == best);
                    assert!(
                        rival.is_some_and(|rival| expected[0] - expected[rival] <= BF16_TIE),
                        "{}: chose {best} where PyTorch chose {token}, its logits {expected:?} for {top_tokens:?}",
                        case["prompt"],
                    );
                    flips += 1;
                }
                logits = decoder.step(&[&[token]]).unwrap();
            }
            if compiled {
                assert_one_specialization(&decoder);
            }
        }
        eprintln!("compiled: {compiled}, worst logit difference {worst}, tied flips {flips}");
        assert!(worst <= BF16_TOLERANCE, "logits differ by {worst}");
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_5_eager_bf16_follows_pytorch() {
        check_real_bf16(false);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_5_compiled_bf16_follows_pytorch() {
        check_real_bf16(true);
    }

    // The loop an export writes beside the one-token step and a prompt step of 64 rows generates PyTorch's greedy tokens from each chat prompt until a near tie, with nothing of Singe.
    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_5_generates_through_its_exported_loop() {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let Some(architecture) = program_target(Device::Cuda(0)) else {
            return;
        };
        let reference = singe_testing::reference!("text_generation/qwen3_5/real");
        let model = qwen3_5::load(
            singe_testing::snapshot("QWEN35"),
            Device::Cuda(0),
            DataType::Bf16,
        )
        .unwrap();
        let cases = reference.metadata()["cases"].as_array().unwrap();
        let capacity = cases
            .iter()
            .map(|case| {
                length(&integers(&case["input_ids"])) + length(case["steps"].as_array().unwrap())
            })
            .max()
            .unwrap();
        let decoder = CachedDecoder::create(&model, DecoderOptions::new(1, capacity)).unwrap();
        let (step, prompt) = (
            decoder.step_graph().unwrap(),
            decoder.chunk_step_graph(63).unwrap(),
        );
        let program = CudaExport::new(architecture)
            .with_decoding(step.layout)
            .export_steps(&[("decode", &step.graph), ("prompt", &prompt.graph)])
            .unwrap();
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../target/programs/qwen3_5-generate-{}",
            process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        let executable = build_set_generation_loop(&program, &directory).unwrap();
        // The loop loads the weights again, so this process's copies and the memory it caches go first.
        drop((program, step, prompt, decoder, model));
        cuda::clear_cache(Device::Cuda(0)).unwrap();
        for (number, case) in cases.iter().enumerate() {
            let steps = case["steps"].as_array().unwrap();
            let output = Command::new(&executable)
                .arg(steps.len().to_string())
                .args(integers(&case["input_ids"]).iter().map(i64::to_string))
                .current_dir(&directory)
                .output()
                .unwrap();
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
            assert_eq!(generated.len(), steps.len());
            let top_logits = reference.values(&format!("case{number}.top_logits"));
            let width = top_logits.len() / steps.len();
            let difference = steps
                .iter()
                .zip(&generated)
                .position(|(step, &token)| step["token"].as_i64().unwrap() != token);
            eprintln!(
                "{}: first difference from PyTorch {difference:?}",
                case["prompt"]
            );
            if let Some(at) = difference {
                let expected = &top_logits[at * width..][..width];
                let rival = integers(&steps[at]["top_tokens"])
                    .iter()
                    .position(|&token| token == generated[at]);
                assert!(
                    rival.is_some_and(|rival| expected[0] - expected[rival] <= BF16_TIE),
                    "{} at step {at}: {} against {expected:?}",
                    case["prompt"],
                    generated[at],
                );
            }
        }
        let _ = fs::remove_dir_all(&directory);
    }
}
