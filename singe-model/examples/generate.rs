// Generates text after each `--prompt` in one batch, printing it as it is generated.
//
// Runs any model the registry loads as a causal language model on plain text prompts; `--chat` reads each prompt as a user turn through the snapshot's chat template, without thinking.
// `--max-new-tokens N`, `--type f16|f32` for other than BF16, `--eager`, `--sample`, `--report` for what the compiled steps prepared, and `--tokens` to print each prompt's token ids after its text.
// `--drafts N` decodes one prompt of a Qwen3.5 snapshot greedily with its multi-token prediction layer drafting `N` tokens before each step, and says how many the decoder kept.

mod support;

use std::{
    io::{self, Write as _},
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use serde_json::{Map, Value};
use singe::{
    DataType,
    generation::{DecoderOptions, GenerationOptions, SpeculativeDecoder, generate_batch},
    random::Generator,
    tensor::SamplingOptions,
};
use singe_model::{
    LoadOptions, Source,
    format::Loader,
    head::CausalLm,
    model::{self, text::qwen3_5},
    processor::{
        chat::{ChatTemplate, Message},
        text::{TextStream, TextTokenizer},
    },
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
    let arguments = Arguments::parse(&["eager", "sample", "report", "tokens", "chat"])?;
    let source = Source::open(&arguments.snapshot)?;
    let texts = match arguments.values("prompt") {
        texts if texts.is_empty() => vec!["What is the most important resource in Factorio?"],
        texts => texts,
    };
    let max_new_tokens = arguments
        .value("max-new-tokens")
        .map_or(Ok(64), str::parse::<usize>)?;
    let data_type = arguments.data_type(DataType::Bf16)?;

    let tokenizer = TextTokenizer::load(&arguments.snapshot)?;
    let template = if arguments.flag("chat") {
        Some(
            ChatTemplate::load(&source)?
                .ok_or_else(|| anyhow::anyhow!("the snapshot has no chat template"))?,
        )
    } else {
        None
    };
    let thinking = Map::from_iter([("enable_thinking".to_owned(), Value::Bool(false))]);
    let prompts = texts
        .iter()
        .map(|text| {
            let text = match &template {
                Some(template) => template.render(&[Message::user(text)], true, &thinking)?,
                None => (*text).to_owned(),
            };
            tokenizer.encode(&text, false)
        })
        .collect::<singe_model::Result<Vec<_>>>()?;
    let longest = prompts.iter().map(Vec::len).max().unwrap_or(0);
    if let Some(drafts) = arguments.value("drafts") {
        let [prompt] = prompts.as_slice() else {
            bail!("--drafts decodes one prompt");
        };
        return speculate(
            &arguments,
            &tokenizer,
            prompt,
            max_new_tokens,
            data_type,
            drafts.parse()?,
        );
    }

    let started = Instant::now();
    let capacity = i64::try_from(longest + max_new_tokens)?;
    let mut decoder =
        model::load::<Box<dyn CausalLm>>(&source, LoadOptions::new(arguments.device, data_type))?
            .decoder(
            DecoderOptions::new(prompts.len(), capacity),
            !arguments.flag("eager"),
        )?;
    eprintln!("load and prepare: {:?}", started.elapsed());

    let options = GenerationOptions {
        max_new_tokens,
        stop_tokens: source.stop_tokens()?,
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
        |_, _, _| Ok(()),
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
    if arguments.flag("tokens") {
        for sequence in &tokens {
            println!("{sequence:?}");
        }
    }

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

fn speculate(
    arguments: &Arguments,
    tokenizer: &TextTokenizer,
    prompt: &[i64],
    max_new_tokens: usize,
    data_type: DataType,
    drafts: usize,
) -> Result<()> {
    let started = Instant::now();
    let source = Source::open(&arguments.snapshot)?;
    let options = LoadOptions::new(arguments.device, data_type);
    let model = qwen3_5::load_drafting(&mut Loader::new(&source, options))?;
    // A check writes its drafts past the last token it keeps.
    let capacity = i64::try_from(prompt.len() + max_new_tokens + drafts)?;
    let mut decoder = SpeculativeDecoder::create(&model, DecoderOptions::new(1, capacity), drafts)?;
    if !arguments.flag("eager") {
        decoder.compile()?;
    }

    eprintln!("load and prepare: {:?}", started.elapsed());
    let options = GenerationOptions {
        max_new_tokens,
        stop_tokens: source.stop_tokens()?,
        ..GenerationOptions::default()
    };
    let started = Instant::now();
    let speculation = decoder.generate(prompt, &options)?;
    let elapsed = started.elapsed();
    println!("{}", tokenizer.decode(&speculation.tokens, true)?);
    let later = speculation.tokens.len().saturating_sub(1);
    eprintln!(
        "tokens: {}; steps after the prompt: {}, drafts kept: {}, {:.2} tokens per step; {:.1} tokens per second",
        speculation.tokens.len(),
        speculation.steps,
        speculation.accepted,
        f64::from(u32::try_from(later)?) / f64::from(u32::try_from(speculation.steps.max(1))?),
        rate(speculation.tokens.len(), elapsed)
    );
    Ok(())
}

fn rate(tokens: usize, elapsed: Duration) -> f64 {
    let seconds = elapsed.as_secs_f64();
    if seconds == 0.0 {
        return 0.0;
    }

    f64::from(u32::try_from(tokens).unwrap_or(u32::MAX)) / seconds
}
