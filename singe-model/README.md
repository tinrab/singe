# singe-model

Model configs and architectures built on `singe`, loaded from Hugging Face snapshot directories.
Models are grouped by Hugging Face task, and each lives in a module named for its `model_type`.

This crate provides:

- **Text generation** in `text::text_generation`: DeepSeek-V3, Gemma 2, GPT-2, Llama, Qwen3, and Qwen3.5, with `load_decoder` choosing the model from the snapshot's `model_type`.
- **Sentence similarity** in `text::sentence_similarity`: BERT.
- **Speech recognition** in `audio::automatic_speech_recognition`: Whisper and Qwen3-ASR.
- **Vision** in `vision`: DINOv2 image features, Qwen3.5 image-text-to-text, and TAESD image-to-image.
- **Tokenization and export:** `tokenization::TextTokenizer` reads a snapshot's tokenizer, and `export`, with the `cuda_13_4` feature, writes a model as CUDA C++ programs that run without Singe.

## Example

Load a text-generation snapshot, generate a reply to one prompt, and print it:

```rust,ignore
use singe::{
    DataType, Device,
    generation::{DecoderOptions, GenerationOptions, generate_batch},
};
use singe_model::{
    config::Snapshot, text::text_generation::load_decoder, tokenization::TextTokenizer,
};

fn main() -> anyhow::Result<()> {
    let snapshot = std::env::var("SNAPSHOT_PATH")?;
    let tokenizer = TextTokenizer::load(&snapshot)?;
    let prompt = tokenizer.encode("What is the capital of France?", false)?;
    let max_new_tokens = 32;
    let capacity = i64::try_from(prompt.len() + max_new_tokens)?;

    let mut decoder = load_decoder(
        &snapshot,
        Device::Cuda(0),
        DataType::Bf16,
        DecoderOptions::new(1, capacity),
        true,
    )?;
    let options = GenerationOptions {
        max_new_tokens,
        stop_tokens: Snapshot::new(&snapshot).stop_tokens()?,
        ..GenerationOptions::default()
    };
    let tokens = generate_batch(
        decoder.as_mut(),
        &[prompt.as_slice()],
        &options,
        None,
        |_, _, _| Ok(()),
        |_, _| Ok(()),
    )?;
    println!("{}", tokenizer.decode(&tokens[0], true)?);
    Ok(())
}
```

The last argument of `load_decoder` compiles the decoding step; pass `false` to run it eagerly.
`GenerationOptions::sampling` is `None` by default, which picks the most likely token at each step.

## Features

- `cuda_13_4`: builds `singe` with CUDA, and enables `export`.

## More examples

Each example reads the snapshot directory from `--snapshot DIR` or `SNAPSHOT_PATH`, and runs on CUDA device 0 unless given `--cpu`.

- [Generate](./examples/generate.rs): generate text after one or more prompts, printing it as it arrives.
- [Image chat](./examples/image_chat.rs): describe an image with a Qwen3.5 snapshot.
- [Transcribe](./examples/transcribe.rs): transcribe a 16 kHz mono WAV file with Whisper or Qwen3-ASR.
- [Embed](./examples/embed.rs): embed sentences with a sentence-transformers snapshot and print how similar each pair is.
- [Image features](./examples/image_features.rs): embed an image with DINOv2.
- [Autoencode](./examples/autoencode.rs): encode an image to Stable Diffusion latents with TAESD and decode it again.
- [Export](./examples/export.rs): export a snapshot as CUDA C++ programs.
- [Profile](./examples/profile.rs): profile a text-generation step with CUPTI and print each kernel's share of GPU time.

Run one with `cargo run --release -p singe-model --features cuda_13_4 --example generate -- --snapshot DIR`.
