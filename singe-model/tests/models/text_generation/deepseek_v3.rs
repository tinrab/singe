use singe::Device;
#[cfg(feature = "cuda_13_4")]
use singe::{
    DataType,
    generation::{Decoder, DecoderOptions},
    testing::maybe_lock_device,
};
use singe_model::text::text_generation::deepseek_v3::{self, DeepseekV3Layers};

#[cfg(feature = "cuda_13_4")]
use singe_testing::decoder::{Reference, decoder};
use singe_testing::{Tolerance, decoder::Suite};

const SUITE: Suite<DeepseekV3Layers> = Suite {
    references: singe_testing::references!(),
    task: "text_generation",
    family: "deepseek_v3",
    models: &[
        "base",
        "direct_query",
        "half_split",
        "yarn",
        "tied",
        "narrow",
    ],
    load: |directory, device, data_type| deepseek_v3::load(directory, device, data_type).unwrap(),
    // The absorbed and expanded forms are equal only if the algebra is right, so F32 leaves room for arithmetic alone.
    f32_tolerance: Tolerance::absolute(0.001, "F32 leaves room for arithmetic alone"),
    bf16_tolerance: Tolerance::absolute(
        0.02,
        "half precision rounds at different points in each form, and these logits are small",
    ),
    programs_skipped: None,
};

// Every layer after the first routes each token to experts, a choice that half-precision rounding can tip when two groups' scores nearly tie, which moves logits by more than rounding does.
// `F32` leaves no such room, so it keeps the dense models' bar.
const MOE_SUITE: Suite<DeepseekV3Layers> = Suite {
    models: &["moe", "moe_per_expert"],
    bf16_tolerance: Tolerance::absolute(0.05, "BF16 keeps eight mantissa bits"),
    ..SUITE
};

#[test]
fn test_tiny_moe_deepseek_v3_matches_transformers_on_cpu() {
    MOE_SUITE.check_logits(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_moe_deepseek_v3_matches_transformers_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    MOE_SUITE.check_logits(Device::Cuda(0));
}

#[test]
fn test_tiny_deepseek_v3_matches_transformers_on_cpu() {
    SUITE.check_logits(Device::Cpu);
}

#[test]
fn test_tiny_deepseek_v3_decodes_on_cpu() {
    SUITE.check_decoding(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_deepseek_v3_matches_transformers_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_logits(Device::Cuda(0));
}

// A step's experts read each one's weights once for all its tokens, which a graph change must not lose without anyone noticing.
#[cfg(feature = "cuda_13_4")]
#[test]
fn test_a_compiled_moe_step_runs_its_experts_as_grouped_matmuls() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let directory = SUITE.directory("moe");
    let model = (SUITE.load)(&directory, Device::Cuda(0), DataType::Bf16);
    let reference = Reference::read(&directory, DataType::Bf16);
    let mut decoder = decoder(&model, DecoderOptions::new(1, 32), true);
    decoder.step(&[&reference.prompt]).unwrap();
    let report = decoder.executor().unwrap().report().unwrap();
    assert!(report.contains("grouped_matmul"), "{report}");
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_a_tuned_moe_model_compiles_and_generates_through_its_program() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    MOE_SUITE.check_tuned_model("moe", Device::Cuda(0));
}

// Latent attention writes its cache through the same module as other attention.
#[cfg(feature = "cuda_13_4")]
#[test]
fn test_a_compiled_deepseek_step_runs_its_cache_write_module() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let directory = SUITE.directory(SUITE.models[0]);
    let model = (SUITE.load)(&directory, Device::Cuda(0), DataType::Bf16);
    let reference = Reference::read(&directory, DataType::Bf16);
    let mut decoder = decoder(&model, DecoderOptions::new(1, 32), true);
    decoder.step(&[&reference.prompt]).unwrap();
    let report = decoder.executor().unwrap().report().unwrap();
    assert!(report.contains("module: rotary cache write"), "{report}");
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_deepseek_v3_decodes_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_decoding(Device::Cuda(0));
}
