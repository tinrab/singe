use std::{env, path::PathBuf, time::Instant};

use anyhow::{Context as _, Result, anyhow, bail};
use singe::{
    DataType, Device,
    generation::{DecoderOptions, GenerationOptions, generate},
    random::Generator,
    tensor::SamplingOptions,
};
use singe_model::{config::Snapshot, text::text_generation::load_decoder};
use tokenizers::Tokenizer;

struct Arguments {
    snapshot: Option<PathBuf>,
    prompt: String,
    max_new_tokens: usize,
    device: Device,
    data_type: DataType,
    compiled: bool,
    sample: bool,
    report: bool,
}

fn parse() -> Result<Arguments> {
    let mut parsed = Arguments {
        snapshot: env::var_os("SNAPSHOT_PATH").map(PathBuf::from),
        prompt: "What is the capital of France?".to_owned(),
        max_new_tokens: 64,
        device: Device::Cuda(0),
        data_type: DataType::Bf16,
        compiled: true,
        sample: false,
        report: false,
    };
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| anyhow!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--snapshot" => parsed.snapshot = Some(value()?.into()),
            "--prompt" => parsed.prompt = value()?,
            "--max-new-tokens" => parsed.max_new_tokens = value()?.parse()?,
            "--cpu" => parsed.device = Device::Cpu,
            "--f16" => parsed.data_type = DataType::F16,
            "--f32" => parsed.data_type = DataType::F32,
            "--eager" => parsed.compiled = false,
            "--sample" => parsed.sample = true,
            "--report" => parsed.report = true,
            _ => bail!("unknown argument {argument}"),
        }
    }
    Ok(parsed)
}

// Runs any model `singe_model::text::text_generation::load_decoder` knows on a plain text prompt; chat templates are left to the caller, who can write one into `--prompt`.
fn main() -> Result<()> {
    let arguments = parse()?;
    let snapshot = arguments
        .snapshot
        .as_deref()
        .context("set SNAPSHOT_PATH or pass --snapshot")?;
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_err(|error| anyhow!("{error}"))?;
    let prompt = tokenizer
        .encode(arguments.prompt, false)
        .map_err(|error| anyhow!("{error}"))?
        .get_ids()
        .iter()
        .map(|&id| i64::from(id))
        .collect::<Vec<_>>();

    let started = Instant::now();
    let capacity = i64::try_from(prompt.len() + arguments.max_new_tokens)?;
    let mut decoder = load_decoder(
        snapshot,
        arguments.device,
        arguments.data_type,
        DecoderOptions::new(1, capacity),
        arguments.compiled,
    )?;
    eprintln!("load and prepare: {:?}", started.elapsed());

    let options = GenerationOptions {
        max_new_tokens: arguments.max_new_tokens,
        stop_tokens: Snapshot::new(snapshot).stop_tokens()?,
        sampling: arguments.sample.then(|| {
            SamplingOptions::new()
                .with_temperature(0.6)
                .with_top_k(Some(20))
                .with_top_p(0.95)
        }),
        ..GenerationOptions::default()
    };
    let generator = Generator::new(0);
    let started = Instant::now();
    let mut first_token = None;
    let tokens = generate(
        decoder.as_mut(),
        &prompt,
        &options,
        Some(&generator),
        |_| {
            first_token.get_or_insert_with(|| started.elapsed());
            Ok(())
        },
    )?;
    let total = started.elapsed();
    let ids = tokens
        .iter()
        .map(|&token| u32::try_from(token))
        .collect::<Result<Vec<_>, _>>()?;
    println!(
        "{}",
        tokenizer
            .decode(&ids, true)
            .map_err(|error| anyhow!("{error}"))?
    );
    if arguments.report
        && let Some(report) = decoder.report()?
    {
        eprintln!("{report}");
    }

    if let Some(first) = first_token {
        let later = u32::try_from(tokens.len().saturating_sub(1).max(1))?;
        eprintln!(
            "first token: {first:?}; per later token: {:?}; tokens: {}",
            total.saturating_sub(first) / later,
            tokens.len()
        );
    }
    Ok(())
}
