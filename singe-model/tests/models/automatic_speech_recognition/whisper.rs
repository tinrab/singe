use std::path::Path;
#[cfg(feature = "cuda_13_4")]
use std::{
    fs,
    path::PathBuf,
    process::{self, Command},
};

#[cfg(feature = "cuda_13_4")]
use serde_json::Value;
use singe::{
    DataType, Device, Tensor,
    generation::{CachedDecoder, Decoder, DecoderOptions, GenerationOptions, RunningBatch},
    nn::{EncoderDecoderCache, Layer},
    signal::{LogMelOptions, LogMelSpectrogram},
};
#[cfg(feature = "cuda_13_4")]
use singe::{
    Shape,
    backend::cuda::{export::CudaExport, tuning::Measurement},
    graph::GraphCudaOptions,
    nn::caches_from_inputs,
    testing::{CompiledProgram, build_set_generation_loop, program_target},
};
use singe_model::audio::automatic_speech_recognition::{
    load_transcriber,
    whisper::{Whisper, WhisperDecoderLayers},
};
#[cfg(feature = "cuda_13_4")]
use singe_model::{
    audio::automatic_speech_recognition::whisper::{
        TranscribeOptions, WhisperTask, generation_options, prompt,
    },
    tokenization::TextTokenizer,
};

#[cfg(feature = "cuda_13_4")]
use singe_testing::{
    EVERY_PATH, check_paths,
    decoder::{ProgramDecoder, tune_step},
    lock_device,
};
use singe_testing::{
    Reference, Tolerance, assert_matches,
    decoder::{best_token, host_rows},
    host, integers, reference, tone_and_chirp,
};

fn tiny() -> Reference {
    reference!("automatic_speech_recognition/whisper/tiny")
}

// What the real reference holds: the clip's prefix, tokens, and transcript.
#[cfg(feature = "cuda_13_4")]
fn real_reference() -> Value {
    let reference: Reference = reference!("automatic_speech_recognition/whisper/real");
    reference.metadata().clone()
}

// Each sequence's greedy steps: the logits PyTorch gave, and the token it chose.
fn steps(reference: &Reference, data_type: DataType) -> Vec<Vec<(Vec<f32>, i64)>> {
    let key = type_key(data_type);
    let tokens = &reference.metadata()["tokens"][key];
    let logits = reference.values(&format!("{key}.step_logits"));
    let sequences = tokens.as_array().unwrap();
    let per_sequence = logits.len() / sequences.len();
    sequences
        .iter()
        .zip(logits.chunks(per_sequence))
        .map(|(tokens, logits)| {
            let tokens = integers(tokens);
            logits
                .chunks(logits.len() / tokens.len())
                .map(<[f32]>::to_vec)
                .zip(tokens)
                .collect()
        })
        .collect()
}

fn type_key(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F32 => "f32",
        _ => "bf16",
    }
}

fn tolerance(data_type: DataType) -> Tolerance {
    match data_type {
        DataType::F32 => Tolerance::absolute(1e-4, "F32 leaves room for arithmetic alone"),
        _ => Tolerance::absolute(
            0.05,
            "BF16 keeps eight mantissa bits, and Singe accumulates attention and norms in F32 where PyTorch rounds some steps",
        ),
    }
}

// The front end computes in F32, where NumPy's extractor computes in F64.
const FRONT_END: Tolerance = Tolerance::absolute(2e-3, "the feature extractor computes in F64");

fn load(device: Device, data_type: DataType) -> Whisper {
    load_transcriber(
        Path::new(singe_testing::references!()).join("automatic_speech_recognition/whisper/tiny"),
        device,
        data_type,
    )
    .unwrap()
}

fn features(reference: &Reference, device: Device, data_type: DataType) -> Tensor {
    reference.tensor("inputs.features", device, data_type)
}

// The log-mel front end computes `WhisperFeatureExtractor`'s features for a fixed signal.
fn check_front_end(device: Device) {
    let reference: Reference = reference!("automatic_speech_recognition/whisper/front_end");
    let samples = usize::try_from(reference.metadata()["samples"].as_i64().unwrap()).unwrap();
    let frames = reference.metadata()["frames"].as_i64().unwrap();
    let front = LogMelSpectrogram::create(LogMelOptions::whisper(80)).unwrap();
    let state = load(Device::Cpu, DataType::F32).encoder().state().clone();
    let features = front
        .forward(&tone_and_chirp(samples, device), &state)
        .unwrap();
    assert_eq!(features.shape().dimensions(), [1, 80, 3000]);
    assert_matches(
        &host(&features.narrow(2, 0, frames).unwrap()),
        &reference.values("leading"),
        FRONT_END,
        &format!("leading frames on {device:?}"),
    );
    assert_matches(
        &host(
            &features
                .sum(&[2], false)
                .unwrap()
                .div_scalar(3000.0)
                .unwrap(),
        ),
        &reference.values("means"),
        FRONT_END,
        &format!("each mel's mean on {device:?}"),
    );
}

// The encoder's outputs, `[B, F, D]`, as `transformers` computes them.
fn check_encoder(
    model: &Whisper,
    reference: &Reference,
    device: Device,
    data_type: DataType,
) -> Tensor {
    let encoded = model
        .encode(&features(reference, device, data_type))
        .unwrap();
    assert_matches(
        &host(&encoded),
        &reference.values(&format!("{}.encoded", type_key(data_type))),
        tolerance(data_type),
        &format!("encoder {data_type:?} on {device:?}"),
    );
    encoded
}

fn write_sources(model: &Whisper, encoded: &Tensor, caches: &[EncoderDecoderCache]) {
    let sequences = usize::try_from(encoded.shape().dimensions()[0]).unwrap();
    for sequence in 0..sequences {
        let row = encoded
            .narrow(0, i64::try_from(sequence).unwrap(), 1)
            .unwrap()
            .squeeze(0)
            .unwrap();
        model
            .decoder()
            .layer()
            .write_sources(&row, sequence, caches, model.decoder().state())
            .unwrap();
    }
}

// A batch of both sequences follows each greedy step's logits, reading the reference's tokens, so each row's cross-attention must read its own encoding; in F32 each row also chooses the reference's token.
fn check_decoding(
    decoder: &mut dyn Decoder,
    reference: &Reference,
    data_type: DataType,
    what: &str,
) {
    let prefix = reference.integers("prefix");
    let sequences = steps(reference, data_type);
    let mut tokens = vec![prefix; sequences.len()];
    for step in 0..sequences[0].len() {
        let rows = tokens.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let logits = host_rows(&decoder.step(&rows).unwrap());
        for (row, sequence) in sequences.iter().enumerate() {
            let (expected, token) = &sequence[step];
            assert_matches(
                &logits[row],
                expected,
                tolerance(data_type),
                &format!("{what} {data_type:?} row {row} step {step}"),
            );
            let token = *token;
            if data_type == DataType::F32 {
                assert_eq!(
                    best_token(&logits[row]),
                    token,
                    "{what} row {row} step {step}"
                );
            }
            tokens[row] = vec![token];
        }
    }
}

fn eager_decoder(model: &Whisper, encoded: &Tensor) -> CachedDecoder<WhisperDecoderLayers> {
    let decoder = model
        .create_decoder(DecoderOptions::new(2, 32).with_page_size(8))
        .unwrap();
    write_sources(model, encoded, decoder.caches());
    decoder
}

fn check_references(device: Device) {
    let reference = tiny();
    for data_type in [DataType::F32, DataType::Bf16] {
        let model = load(device, data_type);
        let encoded = check_encoder(&model, &reference, device, data_type);
        let mut decoder = eager_decoder(&model, &encoded);
        check_decoding(
            &mut decoder,
            &reference,
            data_type,
            &format!("eager on {device:?}"),
        );
    }
}

// The tiny model's two sequences join a running batch two steps apart, each with its encoding written to its row first, and generate the reference's tokens each, as each would alone.
fn check_running_batch(device: Device, compile: bool) {
    let reference = tiny();
    let model = load(device, DataType::F32);
    let encoded = model
        .encode(&features(&reference, device, DataType::F32))
        .unwrap();
    let prefix = reference.integers("prefix");
    let expected = reference.metadata()["tokens"]["f32"]
        .as_array()
        .unwrap()
        .iter()
        .map(integers)
        .collect::<Vec<_>>();
    let mut decoder = model
        .create_decoder(DecoderOptions::new(2, 32).with_page_size(8))
        .unwrap();
    // The prefix pads into the smallest prompt step, as does a prefix joining a running row.
    if compile {
        decoder.compile().unwrap();
    }
    let options = GenerationOptions {
        max_new_tokens: expected[0].len(),
        stop_tokens: Vec::new(),
        sampling: None,
        ..GenerationOptions::default()
    };
    let mut batch = RunningBatch::create(&mut decoder, options).unwrap();
    let mut tokens = vec![Vec::new(); 2];
    for (sequence, wait) in [(0_i64, 2), (1, 0)] {
        let encoding = encoded.narrow(0, sequence, 1).unwrap().squeeze(0).unwrap();
        batch
            .join_with(&prefix, |decoder, row| {
                model.write_sources(decoder, row, &encoding)
            })
            .unwrap()
            .unwrap();
        for _ in 0..wait {
            for generated in batch.advance(None).unwrap() {
                tokens[generated.sequence].push(generated.token);
            }
        }
    }
    while !batch.is_empty() {
        for generated in batch.advance(None).unwrap() {
            tokens[generated.sequence].push(generated.token);
        }
    }
    drop(batch);
    assert_eq!(tokens, expected, "on {device:?}, compiled {compile}");
    if compile {
        assert_eq!(decoder.executors().count(), 3, "on {device:?}");
    }
}

#[test]
fn test_tiny_whisper_sequences_join_a_running_batch_on_cpu() {
    check_running_batch(Device::Cpu, false);
}

#[test]
fn test_the_log_mel_front_end_matches_the_feature_extractor_on_cpu() {
    check_front_end(Device::Cpu);
}

#[test]
fn test_tiny_whisper_matches_transformers_on_cpu() {
    check_references(Device::Cpu);
}

#[cfg(feature = "cuda_13_4")]
fn real_snapshot() -> PathBuf {
    singe_testing::snapshot("WHISPER")
}

// The LibriSpeech clip the real reference transcribes, as samples in [-1, 1).
#[cfg(feature = "cuda_13_4")]
fn clip(device: Device) -> Tensor {
    let bytes = fs::read(
        Path::new(singe_testing::references!())
            .join("automatic_speech_recognition/whisper/real/clip.i16"),
    )
    .unwrap();
    let samples = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
        .collect::<Vec<_>>();
    let count = i64::try_from(samples.len()).unwrap();
    Tensor::from_vec(device, Shape::create([count]).unwrap(), samples).unwrap()
}

// whisper-tiny transcribes the clip to `transformers`' greedy tokens, from samples through its own front end.
#[cfg(feature = "cuda_13_4")]
fn check_real(device: Device, data_type: DataType, compiled: bool) {
    let reference = real_reference();
    let snapshot = real_snapshot();
    let model = load_transcriber(&snapshot, device, data_type).unwrap();
    let tokenizer = TextTokenizer::load(&snapshot).unwrap();
    let prefix = prompt(&tokenizer, "en", WhisperTask::Transcribe).unwrap();
    assert_eq!(prefix, integers(&reference["prefix"]));
    let expected = integers(&reference["tokens"]);
    let options = generation_options(&snapshot, expected.len()).unwrap();
    let samples = clip(device);
    let options = TranscribeOptions::new(options).with_compile(compiled);
    let tokens = model.transcribe(&samples, &prefix, &options).unwrap();
    eprintln!("whisper-tiny {data_type:?} on {device:?}, compiled {compiled}: {tokens:?}");
    assert_eq!(tokens, expected, "{}", reference["transcript"]);
    assert_eq!(
        tokenizer.decode(&tokens, true).unwrap(),
        reference["transcript"].as_str().unwrap()
    );
}

// Two copies of the clip transcribe in one running batch of two rows, compiled, the second joining after the first, to `transformers`' tokens each.
#[cfg(feature = "cuda_13_4")]
fn check_real_batch(device: Device, data_type: DataType) {
    let reference = real_reference();
    let snapshot = real_snapshot();
    let model = load_transcriber(&snapshot, device, data_type).unwrap();
    let expected = integers(&reference["tokens"]);
    let options = TranscribeOptions::new(generation_options(&snapshot, expected.len()).unwrap())
        .with_rows(2)
        .with_compile(true);
    let clips = [clip(device), clip(device)];
    let tokens = model
        .transcribe_batch(&clips, &integers(&reference["prefix"]), &options)
        .unwrap();
    assert_eq!(
        tokens,
        [expected.clone(), expected],
        "{}",
        reference["transcript"]
    );
}

#[cfg(feature = "cuda_13_4")]
#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_whisper_tiny_transcribes_like_transformers_on_cpu() {
    check_real(Device::Cpu, DataType::F32, false);
}

#[cfg(feature = "cuda_13_4")]
mod cuda {
    use super::*;

    const DEVICE: Device = Device::Cuda(0);

    #[test]
    fn test_the_log_mel_front_end_matches_the_feature_extractor_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_front_end(DEVICE);
    }

    #[test]
    fn test_tiny_whisper_sequences_join_a_running_batch_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_running_batch(DEVICE, false);
        check_running_batch(DEVICE, true);
    }

    #[test]
    fn test_tiny_whisper_matches_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_references(DEVICE);
    }

    // The encoder runs on every path, and the decoder step tunes under both measurements and exports as a program, each following the references.
    #[test]
    fn test_a_tuned_tiny_whisper_compiles_and_exports_from_its_records() {
        let Some(_lock) = lock_device() else {
            return;
        };
        let reference = tiny();
        for data_type in [DataType::F32, DataType::Bf16] {
            let model = load(DEVICE, data_type);
            let input = features(&reference, DEVICE, data_type);
            let (graph, encoded) = model
                .encoder()
                .capture_graph("features", input.clone())
                .unwrap();
            check_paths(
                &graph,
                &[("features", input)],
                &reference.values(&format!("{}.encoded", type_key(data_type))),
                tolerance(data_type),
                &format!("whisper encoder {data_type:?}"),
                &EVERY_PATH,
                DEVICE,
            );

            let mut decoder = eager_decoder(&model, &encoded);
            let step = decoder.step_graph().unwrap();
            let records = tune_step(&step.graph, &step.inputs, DEVICE, Measurement::GraphReplay);
            decoder
                .compile_with(GraphCudaOptions::new().with_tuning_records(records))
                .unwrap();
            check_decoding(&mut decoder, &reference, data_type, "tuned");

            let Some(architecture) = program_target(DEVICE) else {
                continue;
            };
            let label = type_key(data_type);
            let records = tune_step(&step.graph, &step.inputs, DEVICE, Measurement::Program);
            let program = CudaExport::new(architecture)
                .with_tuning_records(records)
                .export(&step.graph)
                .unwrap();
            let compiled =
                CompiledProgram::create(&program, &format!("whisper_decoder_{label}")).unwrap();
            let mut names = step
                .inputs
                .keys()
                .filter(|name| name.starts_with("cache."))
                .cloned()
                .collect::<Vec<_>>();
            names.sort_by_key(|name| {
                let mut parts = name.split('.');
                let layer = parts.nth(1).unwrap().parse::<usize>().unwrap();
                let field = parts.next().unwrap();
                let field = ["keys", "values", "cross_keys", "cross_values"]
                    .iter()
                    .position(|known| *known == field)
                    .unwrap();
                (layer, field)
            });
            let tensors = names
                .iter()
                .map(|name| step.inputs[name].clone())
                .collect::<Vec<_>>();
            let caches = caches_from_inputs::<EncoderDecoderCache>(&tensors).unwrap();
            write_sources(&model, &encoded, &caches);
            let mut generating = ProgramDecoder::create(
                &compiled,
                &step,
                DecoderOptions::new(2, 32).with_page_size(8),
            );
            check_decoding(&mut generating, &reference, data_type, "program");
        }
    }

    // Exports the encode step and the decoder's step as one program with its generation loop, builds the loop, and runs it on `features`, `[1, mels, frames]`, returning the tokens it prints.
    fn transcribe_through_loop(
        model: &Whisper,
        features: &Tensor,
        prefix: &[i64],
        new_tokens: usize,
        label: &str,
    ) -> Vec<i64> {
        let decoder = model
            .create_decoder(DecoderOptions::new(1, model.config().max_target_positions))
            .unwrap();
        let step = decoder.step_graph().unwrap();
        let (encode, _) = model.encode_step_graph(&decoder, features).unwrap();
        let set = CudaExport::new(program_target(DEVICE).unwrap())
            .with_decoding(step.layout)
            .export_steps(&[("encode", &encode), ("decode", &step.graph)])
            .unwrap();
        let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target/programs")
            .join(format!("{label}-generate-{}", process::id()));
        let _ = fs::remove_dir_all(&directory);
        let executable = build_set_generation_loop(&set, &directory).unwrap();
        fs::write(directory.join("features.bin"), features.to_bytes().unwrap()).unwrap();
        let output = Command::new(&executable)
            .arg(new_tokens.to_string())
            .arg("features.bin")
            .args(prefix.iter().map(i64::to_string))
            .current_dir(&directory)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{label}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let _ = fs::remove_dir_all(&directory);
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .map(|line| line.parse::<i64>().unwrap())
            .collect()
    }

    // The tiny model's first sequence generates its reference tokens through an exported program of an encode and a decode step, run by its generation loop with nothing of Singe.
    #[test]
    fn test_tiny_whisper_generates_through_its_exported_loop() {
        let Some(_lock) = lock_device() else {
            return;
        };
        if program_target(DEVICE).is_none() {
            return;
        }
        let reference = tiny();
        let model = load(DEVICE, DataType::F32);
        let features = features(&reference, DEVICE, DataType::F32)
            .narrow(0, 0, 1)
            .unwrap()
            .contiguous()
            .unwrap();
        let expected = integers(&reference.metadata()["tokens"]["f32"][0]);
        let generated = transcribe_through_loop(
            &model,
            &features,
            &reference.integers("prefix"),
            expected.len(),
            "whisper_tiny_loop",
        );
        assert_eq!(generated, expected);
    }

    // whisper-tiny transcribes the clip through its exported loop to `transformers`' tokens.
    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_whisper_tiny_transcribes_through_its_exported_loop() {
        let Some(_lock) = lock_device() else {
            return;
        };
        if program_target(DEVICE).is_none() {
            return;
        }
        let reference = real_reference();
        let model = load_transcriber(real_snapshot(), DEVICE, DataType::F32).unwrap();
        let features = model.features(&clip(DEVICE)).unwrap();
        let expected = integers(&reference["tokens"]);
        let generated = transcribe_through_loop(
            &model,
            &features,
            &integers(&reference["prefix"]),
            expected.len(),
            "whisper_real_loop",
        );
        assert_eq!(generated, expected, "{}", reference["transcript"]);
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_whisper_tiny_transcribes_like_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_real(DEVICE, DataType::F32, false);
        check_real(DEVICE, DataType::F32, true);
        check_real(DEVICE, DataType::Bf16, true);
        check_real_batch(DEVICE, DataType::F32);
    }
}
