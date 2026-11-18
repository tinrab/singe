// Profiles a text-generation snapshot's decoding step on the GPU through CUPTI, and prints each kernel's share of the GPU's time, the largest first: `--snapshot DIR`, `--steps N`, `--type f16|f32` in place of BF16, `--eager` for the step the model runs without compiling, `--tune` to compile it with the records tuning that step gives, `--library NAME` for each library it may call, every one if none is named, both for Qwen3 and Qwen3.5, and `--prompt N` to profile a prompt of `N` tokens from an empty cache in place of a one-token step.

mod support;

use std::{
    cmp::Reverse,
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::Result;
use singe::{
    DataType, Tensor,
    backend::cuda::{
        LoweringPolicy,
        export::Library,
        tuning::{GraphTuneOptions, tune_graph},
    },
    generation::{CachedDecoder, Decoder, DecoderOptions},
    graph::GraphCudaOptions,
    nn::{DecoderModel, Model},
};
use singe_cupti::{
    activity::{ActivityBufferOptions, ActivityCollector},
    types::{ActivityFlushFlag, ActivityKind},
};
use singe_model::{
    LoadOptions, Source,
    format::Loader,
    head::CausalLm,
    model::{
        self,
        text::{qwen3, qwen3_5},
    },
};

use crate::support::Arguments;

// The kernels a profile lists, of all it saw.
const SHOWN: usize = 25;

fn wait(logits: &Tensor) -> Result<()> {
    logits.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok(())
}

fn step(decoder: &mut dyn Decoder, prompt: Option<&[i64]>) -> Result<()> {
    match prompt {
        Some(prompt) => {
            decoder.reset();
            wait(&decoder.step(&[prompt], None)?)
        }
        None => wait(&decoder.step(&[&[1]], None)?),
    }
}

fn compiled<L: DecoderModel>(
    model: &Model<L>,
    tokens: usize,
    libraries: &[Library],
    tune: bool,
) -> Result<Box<dyn Decoder>> {
    let options = DecoderOptions::new(1, 512);
    let mut decoder = CachedDecoder::create(model, options)?;
    // The step a run of `tokens` tokens pads to.
    let chunk = options
        .chunk_sizes()
        .into_iter()
        .find(|&chunk| chunk + 1 >= tokens)
        .filter(|_| tokens > 1)
        .unwrap_or(0);
    let policy = LoweringPolicy::default().with_libraries(libraries.iter().copied());
    let mut cuda = GraphCudaOptions::new().with_lowering_policy(policy.clone());
    if tune {
        let step = decoder.chunk_step_graph(chunk)?;
        let device = model.state().device()?;
        let options = GraphTuneOptions {
            policy,
            ..GraphTuneOptions::default()
        };
        let records = tune_graph(&step.graph, &step.inputs, device, &options)?;
        eprintln!("tuned {} choices", records.len());
        cuda = cuda.with_tuning_records(records);
    }

    decoder.compile_with(cuda)?;
    Ok(Box::new(decoder))
}

fn main() -> Result<()> {
    let arguments = Arguments::parse(&["eager", "tune"])?;
    let steps = arguments
        .value("steps")
        .map_or(Ok(100), str::parse::<u32>)?;
    let data_type = arguments.data_type(DataType::Bf16)?;
    let tokens = arguments
        .value("prompt")
        .map_or(Ok(1), str::parse::<usize>)?;
    let source = Source::open(&arguments.snapshot)?;
    let options = LoadOptions::new(arguments.device, data_type);
    let libraries = arguments.values("library");
    let mut decoder = if arguments.flag("tune") || !libraries.is_empty() {
        let libraries = if libraries.is_empty() {
            Library::ALL.to_vec()
        } else {
            libraries
                .into_iter()
                .map(str::parse)
                .collect::<Result<Vec<Library>, _>>()?
        };
        let tune = arguments.flag("tune");
        match model::find(&source)?.name {
            "qwen3" => compiled(
                &qwen3::load(&mut Loader::new(&source, options))?,
                tokens,
                &libraries,
                tune,
            )?,
            "qwen3_5" => compiled(
                &qwen3_5::load(&mut Loader::new(&source, options))?,
                tokens,
                &libraries,
                tune,
            )?,
            other => anyhow::bail!("--tune and --library take Qwen3 or Qwen3.5, not {other}"),
        }
    } else {
        model::load::<Box<dyn CausalLm>>(&source, options)?
            .decoder(DecoderOptions::new(1, 512), !arguments.flag("eager"))?
    };
    let profiled = arguments
        .value("prompt")
        .map(str::parse::<i64>)
        .transpose()?
        .map(|length| (1..=length).collect::<Vec<i64>>());
    let profiled = profiled.as_deref();
    let prompt = (1..=16).collect::<Vec<i64>>();
    wait(&decoder.step(&[&prompt], None)?)?;
    for _ in 0..10 {
        step(decoder.as_mut(), profiled)?;
    }

    let collector = ActivityCollector::create(&ActivityBufferOptions::default())?;
    collector.enable(ActivityKind::ConcurrentKernel)?;
    let started = Instant::now();
    for _ in 0..steps {
        step(decoder.as_mut(), profiled)?;
    }

    let wall = started.elapsed();
    collector.flush_all(ActivityFlushFlag::Forced)?;
    collector.disable(ActivityKind::ConcurrentKernel)?;
    let mut kernels = HashMap::<String, (u64, u64)>::new();
    for buffer in collector.take_buffers()? {
        for record in buffer.records() {
            let Some(kernel) = record?.kernel()? else {
                continue;
            };
            let name = kernel.name.map_or_else(
                || "?".to_owned(),
                |name| name.to_string_lossy().into_owned(),
            );
            let entry = kernels.entry(name).or_default();
            entry.0 += kernel.end.saturating_sub(kernel.start);
            entry.1 += 1;
        }
    }

    let busy = kernels.values().map(|(time, _)| time).sum::<u64>();
    let launches = kernels.values().map(|(_, count)| count).sum::<u64>();
    let per_step = |value: u64| value / u64::from(steps);
    println!(
        "{steps} steps of {data_type:?}: {:.1} µs each, the GPU busy {:.1} µs of it in {} kernels",
        wall.as_secs_f64() * 1e6 / f64::from(steps),
        Duration::from_nanos(per_step(busy)).as_secs_f64() * 1e6,
        per_step(launches)
    );
    let mut listed = kernels.into_iter().collect::<Vec<_>>();
    listed.sort_by_key(|(_, (time, _))| Reverse(*time));
    for (name, (time, count)) in listed.into_iter().take(SHOWN) {
        #[expect(
            clippy::cast_precision_loss,
            reason = "a share needs no exact nanoseconds"
        )]
        let share = time as f64 / busy.max(1) as f64 * 100.0;
        let name = name.chars().take(100).collect::<String>();
        println!(
            "{share:5.1}%  {:8.1} µs  {:3} per step  {name}",
            Duration::from_nanos(per_step(time)).as_secs_f64() * 1e6,
            per_step(count)
        );
    }
    Ok(())
}
