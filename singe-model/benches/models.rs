//! Times the real models a step, a batch, or an image at a time, each where its `SINGE_SNAPSHOT_<MODEL>` is set: `just bench`, `just bench whisper --compare main`.

use std::{collections::HashMap, path::Path};

use singe::{
    DataType, Device,
    generation::{CachedDecoder, Decoder, DecoderOptions, PromptChunks},
    graph::GraphPrepareOptions,
    nn::{DecoderModel, Model, TensorGroup},
};
use singe_model::{
    LoadOptions, Source,
    format::Loader,
    head::{Autoencoder, ImageEncoder},
    model::{
        self,
        audio::whisper::{self, WhisperTask, prompt},
        text::{qwen3, qwen3_5},
    },
    processor::text::TextTokenizer,
    task::{Pipeline, text::sentence_similarity::SentenceSimilarityPipeline},
};
use singe_testing::{
    Benches, decode_step_or_restart, lock_device, pattern_image, snapshot_if_set, tone_and_chirp,
    wait,
};

const DEVICE: Device = Device::Cuda(0);

const SENTENCES: &[&str] = &[
    "An engineer is placing transport belts beside the smelters.",
    "Someone builds a conveyor line next to the furnaces.",
    "The biters attacked the northern wall after the pollution spread.",
    "The train delivers iron ore to the outpost every few minutes.",
];

fn source(snapshot: &Path) -> Source {
    Source::open(snapshot).unwrap()
}

fn minilm(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("MINILM") else {
        return;
    };
    let tokens = TextTokenizer::load(&source(&snapshot))
        .unwrap()
        .encode_batch(SENTENCES, None, DEVICE)
        .unwrap();
    let inputs = tokens.named_tensors().into_iter().collect();
    for data_type in [DataType::F32, DataType::Bf16] {
        let name = format!("minilm/batch/{data_type:?}");
        if !benches.wants(&name) {
            continue;
        }

        let pipeline = SentenceSimilarityPipeline::load(
            &source(&snapshot),
            LoadOptions::new(DEVICE, data_type),
        )
        .unwrap();
        let (graph, _) = pipeline
            .get_model()
            .unwrap()
            .capture_group(&tokens)
            .unwrap();
        let mut executor = graph.prepare(GraphPrepareOptions::new(DEVICE)).unwrap();
        benches.run(&name, || {
            wait(&executor.run(&inputs).unwrap()[0]).unwrap();
        });
    }
}

fn whisper(benches: &mut Benches) {
    let Some(snapshot) = snapshot_if_set("WHISPER") else {
        return;
    };
    let tokenizer = TextTokenizer::load(&source(&snapshot)).unwrap();
    let prefix = prompt(&tokenizer, "en", WhisperTask::Transcribe).unwrap();
    for data_type in [DataType::F32, DataType::Bf16] {
        let encode = format!("whisper-tiny/encode/{data_type:?}");
        let decode = format!("whisper-tiny/step/{data_type:?}");
        if !benches.wants(&encode) && !benches.wants(&decode) {
            continue;
        }

        let model = whisper::load(&mut Loader::new(
            &source(&snapshot),
            LoadOptions::new(DEVICE, data_type),
        ))
        .unwrap();
        let features = model.features(&tone_and_chirp(160_000, DEVICE)).unwrap();
        let mut encoded = model.encode(&features).unwrap();
        benches.run(&encode, || {
            encoded = model.encode(&features).unwrap();
            wait(&encoded).unwrap();
        });
        // The prefix is the only prompt, so it is the only prompt step compiled.
        let extra = prefix.len() - 1;
        let mut decoder = model
            .create_decoder(
                DecoderOptions::new(1, model.config().target_positions).with_prompt_chunks(
                    PromptChunks {
                        smallest: extra,
                        largest: extra,
                    },
                ),
            )
            .unwrap();
        let sources = encoded.squeeze(0).unwrap();
        model.write_sources(&decoder, 0, &sources).unwrap();
        decoder.compile().unwrap();
        decoder.step(&[&prefix], None).unwrap();
        let capacity = usize::try_from(model.config().target_positions).unwrap();
        let mut held = prefix.len();
        benches.run(&decode, || {
            decode_step_or_restart(&mut decoder, &mut held, capacity, |decoder| {
                model.write_sources(decoder, 0, &sources)?;
                decoder.step(&[&prefix], None)?;
                Ok(prefix.len())
            })
            .unwrap();
        });
    }
}

/// Times one decode step and a 128-token prompt of the decoder `load` makes from the snapshot `SINGE_SNAPSHOT_<name>` names, eagerly and compiled, under `label`.
fn text_decoder<L: DecoderModel>(
    benches: &mut Benches,
    name: &str,
    label: &str,
    load: impl Fn(&Path) -> Model<L>,
) {
    let Some(snapshot) = snapshot_if_set(name) else {
        return;
    };
    let capacity = 512;
    let chunk = (0..128).collect::<Vec<i64>>();
    for compiled in [false, true] {
        let how = if compiled { "compiled" } else { "eager" };
        let step_name = format!("{label}/step/Bf16/{how}");
        let chunk_name = format!("{label}/prompt-128/Bf16/{how}");
        if !benches.wants(&step_name) && !benches.wants(&chunk_name) {
            continue;
        }

        let model = load(&snapshot);
        let mut decoder = CachedDecoder::create(
            &model,
            DecoderOptions::new(1, i64::try_from(capacity).unwrap()),
        )
        .unwrap();
        if compiled {
            decoder.compile().unwrap();
        }

        let prompt = [1, 2, 3, 4];
        decoder.step(&[&prompt], None).unwrap();
        let mut held = prompt.len();
        benches.run(&step_name, || {
            decode_step_or_restart(&mut decoder, &mut held, capacity, |decoder| {
                decoder.step(&[&prompt], None)?;
                Ok(prompt.len())
            })
            .unwrap();
        });
        benches.run(&chunk_name, || {
            decoder.reset();
            wait(&decoder.step(&[&chunk], None).unwrap()).unwrap();
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

        let model: ImageEncoder =
            model::load(&source(&snapshot), LoadOptions::new(DEVICE, data_type)).unwrap();
        let image = pattern_image(518, DEVICE).cast(data_type).unwrap();
        let (graph, _) = model.capture_graph("images", image.clone()).unwrap();
        let inputs = HashMap::from([("images".to_owned(), image)]);
        let mut executor = graph.prepare(GraphPrepareOptions::new(DEVICE)).unwrap();
        benches.run(&name, || {
            wait(&executor.run(&inputs).unwrap()[0]).unwrap();
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

        let model: Autoencoder =
            model::load(&source(&snapshot), LoadOptions::new(DEVICE, data_type)).unwrap();
        let image = pattern_image(256, DEVICE).cast(data_type).unwrap();
        benches.run(&name, || {
            let latents = model.encode(&image).unwrap();
            wait(&model.decode(&latents).unwrap()).unwrap();
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
    text_decoder(&mut benches, "QWEN3", "qwen3-0.6b", |snapshot| {
        qwen3::load(&mut Loader::new(
            &source(snapshot),
            LoadOptions::new(DEVICE, DataType::Bf16),
        ))
        .unwrap()
    });
    text_decoder(&mut benches, "QWEN35", "qwen3.5-0.8b", |snapshot| {
        qwen3_5::load(&mut Loader::new(
            &source(snapshot),
            LoadOptions::new(DEVICE, DataType::Bf16),
        ))
        .unwrap()
    });
    dinov2(&mut benches);
    taesd(&mut benches);
    benches.finish();
}
