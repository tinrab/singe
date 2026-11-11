use std::{path::Path, slice};

#[cfg(feature = "cuda_13_4")]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device, Shape, Tensor,
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

// The tiny model reads a synthetic signal through its front end and encoder as `transformers` does, and its prompt, whose audio tokens read the encoder's rows, gives PyTorch's logits and greedy tokens, eagerly and compiled, and on CUDA through its exported loop.
fn check_tiny(device: Device) {
    let reference = tiny();
    let model = load(device);
    let samples = reference.values("samples");
    let length = i64::try_from(samples.len()).unwrap();
    let bucket = model.bucket_samples(model.chunks_of(length));
    let mut padded = samples.clone();
    padded.resize(usize::try_from(bucket).unwrap(), 0.0);
    let padded = Tensor::from_vec(device, Shape::create([bucket]).unwrap(), padded).unwrap();
    let count = Tensor::from_vec(device, Shape::create([1]).unwrap(), vec![length]).unwrap();
    let (features, frames) = model.features(&padded, &count).unwrap();
    assert_eq!(
        frames.to_vec::<i64>().unwrap(),
        [reference.metadata()["frames"].as_i64().unwrap()]
    );
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
    let (rows, count) = model.encode(&features, &frames).unwrap();
    let count = count.to_vec::<i64>().unwrap()[0];
    let encoded = rows.narrow(0, 0, count).unwrap();
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

    #[cfg(feature = "cuda_13_4")]
    if device.is_cuda() {
        let generated = program::transcribe_through_loop(
            model,
            &samples,
            &prompt,
            expected.len(),
            2,
            "qwen3_asr_tiny",
        );
        assert_eq!(generated, expected);
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

// Qwen3-ASR as an exported program set, the encode step reading samples and the decoder's steps reading its rows, through the loop an export writes beside it.
#[cfg(feature = "cuda_13_4")]
mod program {
    use std::{
        fs,
        path::PathBuf,
        process::{self, Command},
    };

    use singe::{
        Device,
        backend::cuda::{self, export::CudaExport},
        generation::{CachedDecoder, DecoderOptions},
        testing::{build_set_generation_loop, program_target},
    };
    use singe_model::audio::automatic_speech_recognition::qwen3_asr::Qwen3Asr;

    // Transcribes `samples` through the exported loop after `prompt`, whose run of audio tokens the loop writes as one starred token, returning the tokens it prints.
    // The encode step reads `spare` chunks more than the clip takes, as an export of a fixed length does for a shorter clip.
    // It drops `model` and frees the device's cached memory before the loop runs, since the loop holds a copy of the weights of its own.
    pub fn transcribe_through_loop(
        model: Qwen3Asr,
        samples: &[f32],
        prompt: &[i64],
        new_tokens: usize,
        spare: i64,
        name: &str,
    ) -> Vec<i64> {
        let architecture = program_target(Device::Cuda(0)).unwrap();
        let audio = model.config().audio_token_id;
        let first = prompt.iter().position(|&token| token == audio).unwrap();
        let chunks = model.chunks_of(i64::try_from(samples.len()).unwrap()) + spare;
        // The encode step writes a row for every position of its chunks, so the cache holds them all after the prompt's start.
        let rows =
            usize::try_from(chunks * model.config().audio_config.max_position_embeddings).unwrap();
        let capacity = (prompt.len() + new_tokens).max(first + rows);
        let decoder = CachedDecoder::create(
            model.decoder(),
            DecoderOptions::new(1, i64::try_from(capacity).unwrap()),
        )
        .unwrap();
        let (encode, _) = model.encode_step_graph(&decoder, chunks).unwrap();
        let (step, chunk) = (
            decoder.step_graph().unwrap(),
            decoder.chunk_step_graph(15).unwrap(),
        );
        let program = CudaExport::new(architecture)
            .with_decoding(step.layout)
            .export_steps(&[
                ("encode", &encode),
                ("decode", &step.graph),
                ("prompt", &chunk.graph),
            ])
            .unwrap();
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../target/programs/{name}-{}", process::id()));
        let _ = fs::remove_dir_all(&directory);
        let executable = build_set_generation_loop(&program, &directory).unwrap();
        drop((program, encode, step, chunk, decoder, model));
        cuda::clear_cache(Device::Cuda(0)).unwrap();
        fs::write(
            directory.join("samples.f32"),
            samples
                .iter()
                .flat_map(|sample| sample.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mut arguments = prompt[..first]
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>();
        arguments.push(format!("{audio}*"));
        arguments.extend(
            prompt[first..]
                .iter()
                .skip_while(|&&token| token == audio)
                .map(i64::to_string),
        );
        let output = Command::new(&executable)
            .arg(new_tokens.to_string())
            .arg("samples.f32")
            .args(&arguments)
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
            .collect();
        let _ = fs::remove_dir_all(&directory);
        generated
    }
}

// Qwen3-ASR-0.6B transcribes the LibriSpeech clip to `transformers`' greedy tokens in F32, from samples through its own front end, naming the language as `transformers` parses it, eagerly, compiled, and through its exported loop.
#[cfg(feature = "cuda_13_4")]
mod real {
    use std::{fs, path::Path};

    use singe::{DataType, Device, Shape, Tensor, testing::maybe_lock_device};
    use singe_model::{
        audio::automatic_speech_recognition::{TranscriberModel, load_transcriber, qwen3_asr},
        tokenization::TextTokenizer,
    };
    use singe_testing::{Reference, integers, reference};

    fn clip() -> Vec<f32> {
        let bytes = fs::read(
            Path::new(singe_testing::references!())
                .join("automatic_speech_recognition/qwen3_asr/real/clip.i16"),
        )
        .unwrap();
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
            .collect()
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
        let clip = clip();
        let count = i64::try_from(clip.len()).unwrap();
        let samples = Tensor::from_vec(
            Device::Cuda(0),
            Shape::create([count]).unwrap(),
            clip.clone(),
        )
        .unwrap();
        let encoded = model.encode_clip(&samples).unwrap();
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
        if compiled {
            drop(transcriber);
            let model = qwen3_asr::load(&snapshot, Device::Cuda(0), DataType::F32).unwrap();
            let generated = super::program::transcribe_through_loop(
                model,
                &clip,
                &prompt,
                expected.len(),
                0,
                "qwen3_asr_real",
            );
            assert_eq!(generated, expected);
        }
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
