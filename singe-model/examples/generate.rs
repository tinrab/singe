// Generates text after each `--prompt` in one batch, printing it as it is generated.
//
// Runs any model `singe_model::text::text_generation::load_decoder` knows on plain text prompts; chat templates are left to the caller, who can write one into `--prompt`.
// `--max-new-tokens N`, `--type f16|f32` for other than BF16, `--eager`, `--sample`, and `--report` for what the compiled steps prepared.

mod support;

use std::{
    io::{self, Write as _},
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use singe::{
    DataType,
    generation::{DecoderOptions, GenerationOptions, generate_batch},
    random::Generator,
    tensor::SamplingOptions,
};
use singe_model::{
    config::Snapshot,
    text::text_generation::load_decoder,
    tokenization::{TextStream, TextTokenizer},
};

use crate::support::Arguments;

// Where each sequence's text goes as it arrives: straight to the terminal for one prompt, and a line at a time, tagged with its prompt's number, for several.
struct Printer<'a> {
    streams: Vec<TextStream<'a>>,
    lines: Vec<String>,
    tagged: bool,
}

impl Printer<'_> {
    fn push(&mut self, sequence: usize, token: i64) -> Result<()> {
        let Some(text) = self.streams[sequence].push(token)? else {
            return Ok(());
        };
        let mut out = io::stdout().lock();
        if !self.tagged {
            write!(out, "{text}")?;
            out.flush()?;
            return Ok(());
        }

        let line = &mut self.lines[sequence];
        line.push_str(&text);
        while let Some(end) = line.find('\n') {
            writeln!(out, "[{sequence}] {}", &line[..end])?;
            line.drain(..=end);
        }
        Ok(())
    }

    fn finish(&mut self) -> Result<()> {
        let mut out = io::stdout().lock();
        if !self.tagged {
            writeln!(out)?;
            return Ok(());
        }

        for (sequence, line) in self.lines.iter_mut().enumerate() {
            if !line.is_empty() {
                writeln!(out, "[{sequence}] {line}")?;
                line.clear();
            }
        }
        Ok(())
    }
}

fn main() -> Result<()> {
    let arguments = Arguments::parse(&["eager", "sample", "report"])?;
    let texts = match arguments.values("prompt") {
        texts if texts.is_empty() => vec!["What is the capital of France?"],
        texts => texts,
    };
    let max_new_tokens = arguments
        .value("max-new-tokens")
        .map_or(Ok(64), str::parse::<usize>)?;
    let data_type = match arguments.value("type") {
        None | Some("bf16") => DataType::Bf16,
        Some("f16") => DataType::F16,
        Some("f32") => DataType::F32,
        Some(other) => bail!("unknown --type {other}"),
    };

    let tokenizer = TextTokenizer::load(&arguments.snapshot)?;
    let prompts = texts
        .iter()
        .map(|text| tokenizer.encode(text, false))
        .collect::<singe_model::Result<Vec<_>>>()?;
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);

    let started = Instant::now();
    let capacity = i64::try_from(longest + max_new_tokens)?;
    let mut decoder = load_decoder(
        &arguments.snapshot,
        arguments.device,
        data_type,
        DecoderOptions::new(prompts.len(), capacity),
        !arguments.flag("eager"),
    )?;
    eprintln!("load and prepare: {:?}", started.elapsed());

    let options = GenerationOptions {
        max_new_tokens,
        stop_tokens: Snapshot::new(&arguments.snapshot).stop_tokens()?,
        sampling: arguments.flag("sample").then(|| {
            SamplingOptions::new()
                .with_temperature(0.6)
                .with_top_k(Some(20))
                .with_top_p(0.95)
        }),
        ..GenerationOptions::default()
    };
    let mut printer = Printer {
        streams: prompts.iter().map(|_| tokenizer.stream(true)).collect(),
        lines: vec![String::new(); prompts.len()],
        tagged: prompts.len() > 1,
    };
    let generator = Generator::new(0);
    let rows = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let started = Instant::now();
    let mut first_token = None;
    // Generation takes Singe's errors alone, so a failure to print waits for it to finish.
    let mut printing = Ok(());
    let tokens = generate_batch(
        decoder.as_mut(),
        &rows,
        &options,
        Some(&generator),
        |sequence, token| {
            first_token.get_or_insert_with(|| started.elapsed());
            if printing.is_ok() {
                printing = printer.push(sequence, token);
            }
            Ok(())
        },
    )?;
    let total = started.elapsed();
    printing?;
    printer.finish()?;

    if arguments.flag("report")
        && let Some(report) = decoder.report()?
    {
        eprintln!("{report}");
    }

    let generated = tokens.iter().map(Vec::len).sum::<usize>();
    if let Some(first) = first_token {
        let longest_run = tokens.iter().map(Vec::len).max().unwrap_or(1);
        let later = u32::try_from(longest_run.saturating_sub(1).max(1))?;
        eprintln!(
            "first token: {first:?}; per later step: {:?}; tokens: {generated}, {:.1} per second",
            total.saturating_sub(first) / later,
            rate(generated, total)
        );
    }
    Ok(())
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }

    f64::from(u32::try_from(tokens).unwrap_or(u32::MAX)) / seconds
}
