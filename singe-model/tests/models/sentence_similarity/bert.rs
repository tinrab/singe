use std::path::{Path, PathBuf};

#[cfg(feature = "cuda_13_4")]
use singe::nn::TensorGroup;
use singe::{DataType, Device, Shape, Tensor, nn::TokenBatch};
use singe_model::text::sentence_similarity::{SentenceEmbedder, bert, load_sentence_embedder};
#[cfg(feature = "cuda_13_4")]
use singe_model::tokenization::TextTokenizer;

#[cfg(feature = "cuda_13_4")]
use singe_testing::{EVERY_PATH, check_paths, lock_device};
use singe_testing::{Reference, Tolerance, assert_matches, integers, reference};

const MODELS: [&str; 2] = ["base", "unnormalized"];

const F32_TOLERANCE: Tolerance = Tolerance::absolute(1e-4, "F32 leaves room for arithmetic alone");

// BF16 keeps eight mantissa bits, and Singe accumulates attention and norms in F32 where PyTorch rounds some steps.
const BF16_TOLERANCE: Tolerance = Tolerance::absolute(
    0.05,
    "BF16 keeps eight mantissa bits, and Singe accumulates attention and norms in F32 where PyTorch rounds some steps",
);

fn directory(name: &str) -> PathBuf {
    Path::new(singe_testing::references!())
        .join("sentence_similarity/bert")
        .join(name)
}

fn tiny(name: &str) -> Reference {
    reference!(format!("sentence_similarity/bert/{name}"))
}

fn type_key(data_type: DataType) -> &'static str {
    match data_type {
        DataType::F32 => "f32",
        _ => "bf16",
    }
}

fn tolerance(data_type: DataType) -> Tolerance {
    match data_type {
        DataType::F32 => F32_TOLERANCE,
        _ => BF16_TOLERANCE,
    }
}

// The reference's sequences, padded into one batch, with their token types.
fn batch(reference: &Reference, device: Device) -> TokenBatch {
    let sequences = reference.metadata()["sequences"]
        .as_array()
        .unwrap()
        .iter()
        .map(integers)
        .collect::<Vec<_>>();
    let token_types = reference.metadata()["token_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(integers)
        .collect::<Vec<_>>();
    sequences_batch(&sequences, &token_types, device)
}

fn sequences_batch(sequences: &[Vec<i64>], token_types: &[Vec<i64>], device: Device) -> TokenBatch {
    let rows = sequences.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let batch = TokenBatch::from_sequences(device, &rows).unwrap();
    let length = batch.tokens().shape().dimensions()[1];
    let width = usize::try_from(length).unwrap();
    let padded = token_types
        .iter()
        .flat_map(|types| {
            let mut types = types.clone();
            types.resize(width, 0);
            types
        })
        .collect::<Vec<_>>();
    let count = i64::try_from(sequences.len()).unwrap();
    let types = Tensor::from_vec(device, Shape::create([count, length]).unwrap(), padded).unwrap();
    batch.with_token_types(types).unwrap()
}

fn host(tensor: &Tensor) -> Vec<f32> {
    tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap()
}

fn embedder(directory: &Path, device: Device, data_type: DataType) -> SentenceEmbedder {
    load_sentence_embedder(directory, device, data_type).unwrap()
}

// Each tiny model's hidden states and embeddings for a padded batch follow `transformers`, in both types.
fn check_references(device: Device) {
    for name in MODELS {
        let reference = tiny(name);
        for data_type in [DataType::F32, DataType::Bf16] {
            let key = type_key(data_type);
            let what = format!("{name} {data_type:?} on {device:?}");
            let batch = batch(&reference, device);
            let encoder = bert::load(directory(name), device, data_type).unwrap();
            let hidden = encoder.forward_group(&batch).unwrap();
            assert_matches(
                &host(&hidden),
                &reference.values(&format!("{key}.hidden")),
                tolerance(data_type),
                &format!("{what} hidden states"),
            );
            let embeddings = embedder(&directory(name), device, data_type)
                .forward_group(&batch)
                .unwrap();
            assert_matches(
                &host(&embeddings),
                &reference.values(&format!("{key}.embeddings")),
                tolerance(data_type),
                &format!("{what} embeddings"),
            );
        }
    }
}

// A sequence padded into a batch embeds as it does alone, so padding changes nothing a real token sees.
fn check_padding(device: Device) {
    let reference = tiny("base");
    let model = embedder(&directory("base"), device, DataType::F32);
    let together = host(&model.forward_group(&batch(&reference, device)).unwrap());
    let sequences = reference.metadata()["sequences"].as_array().unwrap();
    let token_types = reference.metadata()["token_types"].as_array().unwrap();
    let width = together.len() / sequences.len();
    for (index, (sequence, types)) in sequences.iter().zip(token_types).enumerate() {
        let alone = sequences_batch(&[integers(sequence)], &[integers(types)], device);
        let alone = host(&model.forward_group(&alone).unwrap());
        assert_matches(
            &together[index * width..(index + 1) * width],
            &alone,
            Tolerance::absolute(
                1e-5,
                "a sequence padded into a batch computes what it computes alone",
            ),
            &format!("sequence {index} on {device:?}"),
        );
    }
}

#[test]
fn test_tiny_bert_matches_transformers_on_cpu() {
    check_references(Device::Cpu);
}

#[test]
fn test_a_padded_sequence_embeds_as_it_does_alone_on_cpu() {
    check_padding(Device::Cpu);
}

#[test]
fn test_a_snapshot_without_sentence_transformers_modules_is_refused() {
    let error = load_sentence_embedder(
        Path::new(singe_testing::references!()).join("text_generation/gpt2/base"),
        Device::Cpu,
        DataType::F32,
    )
    .unwrap_err();
    assert!(error.to_string().contains("model config"), "{error}");
}

#[cfg(feature = "cuda_13_4")]
fn real_snapshot() -> PathBuf {
    singe_testing::snapshot("MINILM")
}

// all-MiniLM-L6-v2's embeddings of a few sentences follow `transformers`, and similar sentences are nearer than others.
#[cfg(feature = "cuda_13_4")]
fn check_real(device: Device, data_type: DataType, tolerance: Tolerance) {
    let reference: Reference = reference!("sentence_similarity/bert/real");
    let model = embedder(&real_snapshot(), device, data_type);
    // The sentences start as text, and their ids must be the ones `transformers` gave.
    let sentences = reference.metadata()["sentences"]
        .as_array()
        .unwrap()
        .iter()
        .map(|sentence| sentence.as_str().unwrap())
        .collect::<Vec<_>>();
    let tokenized = TextTokenizer::load(real_snapshot())
        .unwrap()
        .encode_batch(&sentences, None, device)
        .unwrap();
    assert_eq!(
        tokenized.tokens().to_vec::<i64>().unwrap(),
        batch(&reference, device).tokens().to_vec::<i64>().unwrap()
    );
    let embeddings = host(&model.forward_group(&tokenized).unwrap());
    let expected = reference.values(&format!("{}.embeddings", type_key(data_type)));
    assert_matches(
        &embeddings,
        &expected,
        tolerance,
        &format!("all-MiniLM-L6-v2 {data_type:?} on {device:?}"),
    );
    let width = embeddings.len() / sentences.len();
    let row = |index: usize| &embeddings[index * width..(index + 1) * width];
    let similarity = |left: usize, right: usize| {
        row(left)
            .iter()
            .zip(row(right))
            .map(|(left, right)| left * right)
            .sum::<f32>()
    };
    assert!(similarity(0, 1) > similarity(0, 2));
}

#[cfg(feature = "cuda_13_4")]
#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_minilm_f32_embeddings_match_transformers_on_cpu() {
    check_real(
        Device::Cpu,
        DataType::F32,
        Tolerance::absolute(1e-5, "F32 leaves room for arithmetic alone"),
    );
}

#[cfg(feature = "cuda_13_4")]
mod cuda {
    use super::*;

    const DEVICE: Device = Device::Cuda(0);

    #[test]
    fn test_tiny_bert_matches_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_references(DEVICE);
    }

    #[test]
    fn test_a_padded_sequence_embeds_as_it_does_alone_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_padding(DEVICE);
    }

    // A captured embedder computes what it computes eagerly on every path.
    #[test]
    fn test_a_tuned_tiny_bert_compiles_and_exports_from_its_records() {
        let Some(_lock) = lock_device() else {
            return;
        };
        let reference = tiny("base");
        let batch = batch(&reference, DEVICE);
        for data_type in [DataType::F32, DataType::Bf16] {
            let model = embedder(&directory("base"), DEVICE, data_type);
            let (graph, _) = model.capture_group(&batch).unwrap();
            let inputs = batch.named_tensors();
            let named = inputs
                .iter()
                .map(|(name, tensor)| (name.as_str(), tensor.clone()))
                .collect::<Vec<_>>();
            check_paths(
                &graph,
                &named,
                &reference.values(&format!("{}.embeddings", type_key(data_type))),
                tolerance(data_type),
                &format!("bert {data_type:?}"),
                &EVERY_PATH,
                DEVICE,
            );
        }
    }

    #[test]
    #[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
    fn test_minilm_embeddings_match_transformers_on_cuda() {
        let Some(_lock) = lock_device() else {
            return;
        };
        check_real(
            DEVICE,
            DataType::F32,
            Tolerance::absolute(1e-4, "F32 on CUDA sums in another order"),
        );
        check_real(
            DEVICE,
            DataType::Bf16,
            Tolerance::absolute(0.02, "BF16 rounds each layer's activations"),
        );
    }
}
