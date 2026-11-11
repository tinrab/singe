// Everything an author needs is public: this file is outside the crate, and defines a model through that API alone.

use std::path::Path;

use singe::{
    DataType, Device,
    generation::{DecoderOptions, GenerationOptions, generate},
    nn::Model,
};
use singe_model::{
    assembly::{GroupedDecoder, causal, grouped_decoder, load_weights},
    config::{DecoderConfig, Snapshot},
};

use singe_testing::{
    Tolerance,
    decoder::{Reference, Suite, decoder},
};

fn load(snapshot: &Path, device: Device, data_type: DataType) -> Model<GroupedDecoder> {
    let config: DecoderConfig = Snapshot::new(snapshot).config().unwrap();
    let spec = config.spec(data_type, false).unwrap();
    let schema = grouped_decoder(&spec, causal()).unwrap();
    load_weights(&schema, snapshot, device, None).unwrap()
}

const SUITE: Suite<GroupedDecoder> = Suite {
    references: singe_testing::references!(),
    task: "text_generation",
    family: "llama",
    models: &["base"],
    load,
    f32_tolerance: Tolerance::absolute(0.001, "F32 leaves room for arithmetic alone"),
    bf16_tolerance: Tolerance::absolute(0.05, "BF16 keeps eight mantissa bits"),
    programs_skipped: None,
};

#[test]
fn test_a_model_defined_outside_the_crate_matches_transformers() {
    SUITE.check_logits(Device::Cpu);
}

#[test]
fn test_a_model_defined_outside_the_crate_generates_through_the_shared_loop() {
    let directory = Path::new(singe_testing::references!()).join("text_generation/llama/base");
    let reference = Reference::read(&directory, DataType::F32);
    let model = load(&directory, Device::Cpu, DataType::F32);
    let mut decoder = decoder(&model, DecoderOptions::new(1, 32), false);
    let options = GenerationOptions {
        max_new_tokens: reference.tokens().len(),
        ..GenerationOptions::default()
    };
    assert_eq!(
        generate(&mut decoder, &reference.prompt, &options, None, |_| Ok(())).unwrap(),
        reference.tokens()
    );
}
