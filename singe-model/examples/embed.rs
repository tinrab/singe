// Embeds sentences with a sentence-transformers snapshot, such as all-MiniLM-L6-v2, and prints how similar each pair is.

mod support;

use anyhow::Result;
use singe::DataType;
use singe_model::{text::sentence_similarity::load_sentence_embedder, tokenization::TextTokenizer};

use crate::support::Arguments;

fn main() -> Result<()> {
    let arguments = Arguments::parse(&[])?;
    let sentences = if arguments.positional.is_empty() {
        vec![
            "A man is eating food.".to_owned(),
            "A man is eating a piece of bread.".to_owned(),
            "The new movie is awesome.".to_owned(),
        ]
    } else {
        arguments.positional.clone()
    };
    let texts = sentences.iter().map(String::as_str).collect::<Vec<_>>();
    let tokenizer = TextTokenizer::load(&arguments.snapshot)?;
    let batch = tokenizer.encode_batch(&texts, Some(256), arguments.device)?;
    let model = load_sentence_embedder(&arguments.snapshot, arguments.device, DataType::F32)?;
    let embeddings = model
        .forward_group(&batch)?
        .cast(DataType::F32)?
        .to_vec::<f32>()?;
    let width = embeddings.len() / texts.len();
    let row = |index: usize| &embeddings[index * width..(index + 1) * width];
    for left in 0..texts.len() {
        for right in left + 1..texts.len() {
            let similarity = row(left)
                .iter()
                .zip(row(right))
                .map(|(left, right)| left * right)
                .sum::<f32>();
            println!("{similarity:.3}  {:?} ~ {:?}", texts[left], texts[right]);
        }
    }
    Ok(())
}
