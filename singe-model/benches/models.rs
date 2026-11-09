// Times the real models a step, a batch, or an image at a time, each where its `SINGE_SNAPSHOT_<MODEL>` is set: `just bench`, `just bench whisper --compare main`.

use std::collections::HashMap;

use singe::{
    DataType, Device, Tensor,
    generation::{CachedDecoder, Decoder, DecoderOptions},
    graph::GraphPrepareOptions,
    nn::TensorGroup,
};
use singe_model::{
    audio::automatic_speech_recognition::{
        load_transcriber,
        whisper::{WhisperTask, prompt},
    },
    text::{sentence_similarity::load_sentence_embedder, text_generation::qwen3},
    tokenization::TextTokenizer,
    vision::{image_feature_extraction::load_image_encoder, image_to_image::load_autoencoder},
};
use singe_testing::{Benches, lock_device, pattern_image, snapshot_if_set, tone_and_chirp};

const DEVICE: Device = Device::Cuda(0);

const SENTENCES: [&str; 4] = [
    "A man is playing a guitar on a stage.",
    "Someone performs music with a guitar in front of an audience.",
    "The stock market fell sharply on Tuesday after the announcement.",
    "Cats sleep for most of the day.",
];

// Reads a result back, which waits for the work behind it.
fn wait(tensor: &Tensor) {
    tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap();
}

// One token per sequence, which is what a decoding step takes.
fn step(decoder: &mut dyn Decoder) {
    let rows = vec![[1_i64]; decoder.sequences()];
    let tokens = rows.iter().map(<[i64; 1]>::as_slice).collect::<Vec<_>>();
    wait(&decoder.step(&tokens).unwrap());
}

fn minilm(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("MINILM") else {
        return;
    };
    let tokens = TextTokenizer::load(&snapshot)
        .unwrap()
        .encode_batch(&SENTENCES, None, DEVICE)
        .unwrap();
    let inputs = tokens.named_tensors().into_iter().collect();
    for data_type in [DataType::F32, DataType::Bf16] {
        let name = format!("minilm/batch/{data_type:?}");
        if !benches.wants(&name) {
            continue;
        }

        let model = load_sentence_embedder(&snapshot, DEVICE, data_type).unwrap();
        let (graph, _) = model.capture_group(&tokens).unwrap();
        let mut executor = graph.prepare(GraphPrepareOptions::new(DEVICE)).unwrap();
        benches.run(&name, || {
            wait(&executor.run(&inputs).unwrap()[0]);
        });
    }
}

fn whisper(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("WHISPER") else {
        return;
    };
    let tokenizer = TextTokenizer::load(&snapshot).unwrap();
    let prefix = prompt(&tokenizer, "en", WhisperTask::Transcribe).unwrap();
    for data_type in [DataType::F32, DataType::Bf16] {
        let encode = format!("whisper-tiny/encode/{data_type:?}");
        let decode = format!("whisper-tiny/step/{data_type:?}");
        if !benches.wants(&encode) && !benches.wants(&decode) {
            continue;
        }

        let model = load_transcriber(&snapshot, DEVICE, data_type).unwrap();
        let features = model.features(&tone_and_chirp(160_000, DEVICE)).unwrap();
        let mut encoded = model.encode(&features).unwrap();
        benches.run(&encode, || {
            encoded = model.encode(&features).unwrap();
            wait(&encoded);
        });
        let mut decoder = model
            .create_decoder(DecoderOptions::new(1, model.config().max_target_positions))
            .unwrap();
        let sources = encoded.squeeze(0).unwrap();
        model.write_sources(&decoder, 0, &sources).unwrap();
        decoder.compile().unwrap();
        decoder.step(&[&prefix]).unwrap();
        // A step past the model's positions fails, so the decoder starts over when it is full.
        let mut held = prefix.len();
        benches.run(&decode, || {
            if held + 1 >= usize::try_from(model.config().max_target_positions).unwrap() {
                decoder.reset();
                model.write_sources(&decoder, 0, &sources).unwrap();
                decoder.step(&[&prefix]).unwrap();
                held = prefix.len();
            }

            step(&mut decoder);
            held += 1;
        });
    }
}

fn qwen3(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("QWEN3") else {
        return;
    };
    let capacity = 512;
    let chunk = (0..128).collect::<Vec<i64>>();
    for compiled in [false, true] {
        let how = if compiled { "compiled" } else { "eager" };
        let step_name = format!("qwen3-0.6b/step/Bf16/{how}");
        let chunk_name = format!("qwen3-0.6b/prompt-128/Bf16/{how}");
        if !benches.wants(&step_name) && !benches.wants(&chunk_name) {
            continue;
        }

        let model = qwen3::load(&snapshot, DEVICE, DataType::Bf16).unwrap();
        let mut decoder = CachedDecoder::create(&model, DecoderOptions::new(1, capacity)).unwrap();
        if compiled {
            decoder.compile().unwrap();
        }

        let prompt = [1, 2, 3, 4];
        decoder.step(&[&prompt]).unwrap();
        let mut held = prompt.len();
        benches.run(&step_name, || {
            if held + 1 >= usize::try_from(capacity).unwrap() {
                decoder.reset();
                decoder.step(&[&prompt]).unwrap();
                held = prompt.len();
            }

            step(&mut decoder);
            held += 1;
        });
        benches.run(&chunk_name, || {
            decoder.reset();
            wait(&decoder.step(&[&chunk]).unwrap());
        });
    }
}

fn dinov2(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("DINOV2") else {
        return;
    };
    for data_type in [DataType::F32, DataType::Bf16] {
        let name = format!("dinov2-small/518/{data_type:?}");
        if !benches.wants(&name) {
            continue;
        }

        let model = load_image_encoder(&snapshot, DEVICE, data_type).unwrap();
        let image = pattern_image(518, DEVICE).cast(data_type).unwrap();
        let (graph, _) = model.capture_graph("images", image.clone()).unwrap();
        let inputs = HashMap::from([("images".to_owned(), image)]);
        let mut executor = graph.prepare(GraphPrepareOptions::new(DEVICE)).unwrap();
        benches.run(&name, || {
            wait(&executor.run(&inputs).unwrap()[0]);
        });
    }
}

fn taesd(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("TAESD") else {
        return;
    };
    for data_type in [DataType::F32, DataType::F16] {
        let name = format!("taesd/256/{data_type:?}");
        if !benches.wants(&name) {
            continue;
        }

        let model = load_autoencoder(&snapshot, DEVICE, data_type).unwrap();
        let image = pattern_image(256, DEVICE).cast(data_type).unwrap();
        benches.run(&name, || {
            let latents = model.encode(&image).unwrap();
            wait(&model.decode(&latents).unwrap());
        });
    }
}

fn main() {
    let Some(_lock) = lock_device() else {
        eprintln!("the benchmarks need a GPU");
        return;
    };
    let mut benches = Benches::from_args();
    minilm(&mut benches);
    whisper(&mut benches);
    qwen3(&mut benches);
    dinov2(&mut benches);
    taesd(&mut benches);
    benches.finish();
}
