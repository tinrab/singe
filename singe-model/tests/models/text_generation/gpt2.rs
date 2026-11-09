use singe::Device;
#[cfg(feature = "cuda_13_4")]
use singe::testing::maybe_lock_device;
use singe_model::text::text_generation::gpt2::{self, Gpt2Layers};

use singe_testing::{Tolerance, decoder::Suite};

const SUITE: Suite<Gpt2Layers> = Suite {
    references: singe_testing::references!(),
    task: "text_generation",
    family: "gpt2",
    models: &["base", "exact_gelu", "default_inner"],
    load: |directory, device, data_type| gpt2::load(directory, device, data_type).unwrap(),
    f32_tolerance: Tolerance::absolute(0.001, "F32 leaves room for arithmetic alone"),
    bf16_tolerance: Tolerance::absolute(
        0.05,
        "BF16 keeps eight mantissa bits, and Singe accumulates attention in F32 where PyTorch rounds each step",
    ),
};

#[test]
fn test_tiny_gpt2_matches_transformers_on_cpu() {
    SUITE.check_logits(Device::Cpu);
}

#[test]
fn test_tiny_gpt2_decodes_on_cpu() {
    SUITE.check_decoding(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_gpt2_matches_transformers_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_logits(Device::Cuda(0));
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_a_tuned_tiny_model_compiles_and_exports_from_its_records() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_tuned(Device::Cuda(0));
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_gpt2_decodes_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    SUITE.check_decoding(Device::Cuda(0));
}
