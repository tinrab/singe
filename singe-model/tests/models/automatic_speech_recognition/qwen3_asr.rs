use std::{path::Path, slice};

#[cfg(feature = "cuda_13_4")]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device,
    generation::{CachedDecoder, Decoder, DecoderOptions, GenerationOptions},
};
use singe_model::audio::automatic_speech_recognition::{
    qwen3_asr::{self, Qwen3Asr, Qwen3AsrDecoderLayers},
    whisper::TranscribeOptions,
};
use singe_testing::{Reference, Tolerance, assert_matches, decoder::host_logits, host, reference};

fn tiny() -> Reference {
    reference!("automatic_speech_recognition/qwen3_asr/tiny")
}

fn load(device: Device) -> Qwen3Asr {
    qwen3_asr::load(
        Path::new(singe_testing::references!()).join("automatic_speech_recognition/qwen3_asr/tiny"),
        device,
        DataType::F32,
    )
    .unwrap()
}

// The front end computes in F32, where NumPy's extractor computes in F64.
const FRONT_END: Tolerance = Tolerance::absolute(2e-3, "the feature extractor computes in F64");

const F32: Tolerance = Tolerance::absolute(1e-4, "F32 leaves room for arithmetic alone");

// The tiny model reads a synthetic signal through its front end and encoder as `transformers` does, and its prompt, whose audio tokens read the encoder's rows, gives PyTorch's logits and greedy tokens, eagerly and compiled.
fn check_tiny(device: Device) {
    let reference = tiny();
    let model = load(device);
    let samples = reference.tensor("samples", device, DataType::F32);
    let (features, frames) = model.features(&samples).unwrap();
    assert_eq!(frames, reference.metadata()["frames"].as_i64().unwrap());
    assert_matches(
        &host(&features),
        &reference.values("features"),
        FRONT_END,
        &format!("features on {device:?}"),
    );
    let features = reference
        .tensor("features", device, DataType::F32)
        .unsqueeze(0)
        .unwrap();
    let encoded = model.encode(&features, frames).unwrap();
    assert_matches(
        &host(&encoded),
        &reference.values("encoded"),
        F32,
        &format!("encoder on {device:?}"),
    );

    let prompt = reference.integers("prompt");
    let first = prompt
        .iter()
        .position(|&token| token == model.config().audio_token_id)
        .unwrap();
    let expected = reference.integers("tokens");
    for compiled in [false, true] {
        let capacity = i64::try_from(prompt.len() + expected.len()).unwrap();
        let mut decoder =
            CachedDecoder::create(model.decoder(), DecoderOptions::new(1, capacity)).unwrap();
        if compiled {
            decoder.compile().unwrap();
        }
        Qwen3AsrDecoderLayers::write_sources(
            decoder.caches(),
            0,
            i64::try_from(first).unwrap(),
            &encoded,
        )
        .unwrap();
        let logits = decoder.step(&[&prompt]).unwrap();
        assert_matches(
            &host_logits(&logits),
            &reference.values("prompt_logits"),
            F32,
            &format!("prompt logits on {device:?}, compiled {compiled}"),
        );
        let options = TranscribeOptions::new(GenerationOptions {
            max_new_tokens: expected.len(),
            ..GenerationOptions::default()
        })
        .with_compile(compiled);
        let generated = model
            .generate(&[(prompt.clone(), encoded.clone())], &options)
            .unwrap();
        assert_eq!(generated, slice::from_ref(&expected), "compiled {compiled}");
    }
}

#[test]
fn test_tiny_qwen3_asr_matches_transformers_on_cpu() {
    check_tiny(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
#[test]
fn test_tiny_qwen3_asr_matches_transformers_on_cuda() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check_tiny(Device::Cuda(0));
}

// Qwen3-ASR-0.6B transcribes the LibriSpeech clip to `transformers`' greedy tokens in F32, from samples through its own front end, naming the language as `transformers` parses it.
#[cfg(feature = "cuda_13_4")]
mod real {
    use std::{fs, path::Path};

    use singe::{DataType, Device, Shape, Tensor, testing::maybe_lock_device};
    use singe_model::{
        audio::automatic_speech_recognition::{TranscriberModel, load_transcriber},
        tokenization::TextTokenizer,
    };
    use singe_testing::{Reference, integers, reference};

    fn clip() -> Tensor {
        let bytes = fs::read(
            Path::new(singe_testing::references!())
                .join("automatic_speech_recognition/qwen3_asr/real/clip.i16"),
        )
        .unwrap();
        let samples = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
            .collect::<Vec<_>>();
        let count = i64::try_from(samples.len()).unwrap();
        Tensor::from_vec(Device::Cuda(0), Shape::create([count]).unwrap(), samples).unwrap()
    }

    fn check_real(compiled: bool) {
        let Some(_lock) = maybe_lock_device() else {
            return;
        };
        let reference: Reference = reference!("automatic_speech_recognition/qwen3_asr/real");
        let reference = reference.metadata();
        let snapshot = singe_testing::snapshot("QWEN3_ASR");
        let transcriber = load_transcriber(&snapshot, Device::Cuda(0), DataType::F32).unwrap();
        let TranscriberModel::Qwen3Asr(model) = transcriber.model() else {
            panic!("the snapshot is Qwen3-ASR");
        };
        let tokenizer = TextTokenizer::load(&snapshot).unwrap();
        let samples = clip();
        let (features, frames) = model.features(&samples).unwrap();
        let encoded = model.encode(&features, frames).unwrap();
        let prompt = model
            .prompt(&tokenizer, encoded.shape().dimensions()[0], None)
            .unwrap();
        assert_eq!(prompt, integers(&reference["prompt"]));
        let expected = integers(&reference["tokens"]);
        let transcription = transcriber
            .transcribe(&samples, None, expected.len(), compiled)
            .unwrap();
        eprintln!("compiled {compiled}: {transcription:?}");
        assert_eq!(
            transcription.language.as_deref(),
            reference["language"].as_str()
        );
        assert_eq!(
            transcription.text,
            reference["transcript"].as_str().unwrap()
        );
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_qwen3_asr_transcribes_like_transformers() {
        check_real(false);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_compiled_qwen3_asr_transcribes_like_transformers() {
        check_real(true);
    }
}
