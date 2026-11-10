// Singe's tokenizers against `transformers`' for each real model.

use singe::Device;
use singe_model::tokenization::TextTokenizer;
use singe_testing::{Reference, reference, snapshot};

// Each text encodes with and without special tokens, and decodes, as `transformers` does it, and a batch an encoder reads pads, truncates, and types its tokens as `transformers` does.
fn check(model: &str) {
    let reference: Reference = reference!(format!("tokenization/{}", model.to_lowercase()));
    let tokenizer = TextTokenizer::load(snapshot(model)).unwrap();
    for case in reference.metadata()["cases"].as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let ids = |key: &str| singe_testing::integers(&case[key]);
        assert_eq!(
            tokenizer.encode(text, false).unwrap(),
            ids("plain"),
            "{model}: {text:?}"
        );
        assert_eq!(
            tokenizer.encode(text, true).unwrap(),
            ids("special"),
            "{model}: {text:?} with special tokens"
        );
        assert_eq!(
            tokenizer.decode(&ids("special"), true).unwrap(),
            case["decoded"].as_str().unwrap(),
            "{model}: decoding {text:?}"
        );
        // Streamed one id at a time, the pieces are whole text and join to the same.
        let mut stream = tokenizer.stream(true);
        let mut streamed = String::new();
        for id in ids("special") {
            if let Some(piece) = stream.push(id).unwrap() {
                streamed.push_str(&piece);
            }
        }
        assert_eq!(
            streamed,
            case["decoded"].as_str().unwrap(),
            "{model}: streaming {text:?}"
        );
    }

    let batch = &reference.metadata()["batch"];
    let texts = batch["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|text| text.as_str().unwrap())
        .collect::<Vec<_>>();
    let max_length = usize::try_from(batch["max_length"].as_u64().unwrap()).unwrap();
    let encoded = tokenizer
        .encode_batch(&texts, Some(max_length), Device::Cpu)
        .unwrap();
    let rows = |tensor: &singe::Tensor| {
        let values = tensor.to_vec::<i64>().unwrap();
        let width = values.len() / texts.len();
        values
            .chunks(width)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>()
    };
    let padding = encoded
        .padding()
        .to_vec::<bool>()
        .unwrap()
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    let expected = |key: &str| {
        batch[key]
            .as_array()
            .unwrap()
            .iter()
            .map(singe_testing::integers)
            .collect::<Vec<_>>()
    };
    let width = padding.len() / texts.len();
    assert_eq!(
        padding
            .chunks(width)
            .map(<[i64]>::to_vec)
            .collect::<Vec<_>>(),
        expected("padding"),
        "{model}: batch padding"
    );
    // Singe pads with id zero where some tokenizers pad with their own token, so only real tokens compare.
    for ((actual, wanted), mask) in rows(encoded.tokens())
        .iter()
        .zip(expected("ids"))
        .zip(expected("padding"))
    {
        let real = |row: &[i64]| {
            row.iter()
                .zip(&mask)
                .filter(|(_, real)| **real == 1)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>()
        };
        assert_eq!(real(actual), real(&wanted), "{model}: batch ids");
    }
    assert_eq!(
        rows(encoded.token_types()),
        expected("token_types"),
        "{model}: batch token types"
    );
}

#[test]
#[ignore = "needs real snapshots, through SINGE_SNAPSHOT_<MODEL>"]
fn test_tokenizers_match_transformers() {
    for model in ["QWEN3", "MINILM", "WHISPER"] {
        check(model);
    }
}
