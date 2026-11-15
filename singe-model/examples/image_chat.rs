// Describes an image with a Qwen3.5 snapshot: `--image FILE`, `--prompt TEXT` for other than a request for one sentence, `--max-new-tokens N`, `--type f32|f16` for other than BF16, `--eager` for eager decoder steps, `--tokens` to print the token ids too, and `--source FILE` to write the image's grid of patches, as two little-endian `I64`s, and its patches where an exported loop reads them, and print the prompt that loop takes.
//
// The image is resized as `transformers`' processor sizes it, bicubically when that changes its size, so an image already at such a size reads exactly as `transformers` reads it.

mod support;

use std::{fs, iter};

use anyhow::{Result, bail};
use singe::{DataType, generation::GenerationOptions};
use singe_model::{
    config::Snapshot,
    tokenization::TextTokenizer,
    vision::image_text_to_text::qwen3_5::{ImagePrompt, load, smart_resize},
};

use crate::support::{Arguments, read_image_sized};

fn main() -> Result<()> {
    let arguments = Arguments::parse(&["eager", "tokens"])?;
    let request = arguments
        .value("prompt")
        .unwrap_or("Describe this image in one sentence.");
    let max_new_tokens = arguments
        .value("max-new-tokens")
        .map_or(Ok(64), str::parse::<usize>)?;
    let data_type = match arguments.value("type") {
        None | Some("bf16") => DataType::Bf16,
        Some("f16") => DataType::F16,
        Some("f32") => DataType::F32,
        Some(other) => bail!("unknown --type {other}"),
    };
    let model = load(&arguments.snapshot, arguments.device, data_type)?;
    let (pixels, width, height) =
        read_image_sized(arguments.required("image")?, |width, height| {
            let (height, width) = smart_resize(
                height.into(),
                width.into(),
                model.size_factor(),
                model.processor().size,
            );
            Ok((u32::try_from(width)?, u32::try_from(height)?))
        })?;
    let image = model.patches(&pixels, height.into(), width.into(), arguments.device)?;
    let (rows, grid) = model.encode(&image)?;

    let tokenizer = TextTokenizer::load(&arguments.snapshot)?;
    let text = format!(
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{request}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    );
    let image_token = model.config().image_token_id;
    let mut tokens = Vec::new();
    for token in tokenizer.encode(&text, false)? {
        if token == image_token {
            tokens.extend(iter::repeat_n(token, usize::try_from(grid[0] * grid[1])?));
        } else {
            tokens.push(token);
        }
    }

    if let Some(path) = arguments.value("source") {
        let mut source = image
            .grid
            .iter()
            .flat_map(|side| side.to_le_bytes())
            .collect::<Vec<_>>();
        source.extend(image.patches.to_bytes()?);
        fs::write(path, source)?;
        let first = tokens.iter().position(|&token| token == image_token);
        let prompt = tokens
            .iter()
            .enumerate()
            .filter(|&(index, &token)| token != image_token || Some(index) == first)
            .map(|(_, &token)| {
                if token == image_token {
                    format!("{token}*")
                } else {
                    token.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join(" ");
        println!("{prompt}");
    }

    let generated = model.generate(
        &[ImagePrompt { tokens, rows, grid }],
        &singe_model::BatchOptions {
            generation: GenerationOptions {
                max_new_tokens,
                stop_tokens: Snapshot::new(&arguments.snapshot).stop_tokens()?,
                ..GenerationOptions::default()
            },
            rows: 1,
            compile: !arguments.flag("eager"),
        },
    )?;
    let generated = &generated[0];
    if arguments.flag("tokens") {
        println!("{generated:?}");
    }

    println!("{}", tokenizer.decode(generated, true)?);
    Ok(())
}
