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
    programs_skipped: Some("exported programs do not carry state caches yet"),
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
    use singe::{
        DataType, Device,
        generation::{Decoder, DecoderOptions},
        testing::maybe_lock_device,
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
}
