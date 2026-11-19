// Profiles a text-generation snapshot's decoding step on the GPU through CUPTI, and prints each kernel's share of the GPU's time, the largest first: `--snapshot DIR`, `--steps N`, `--type f16|f32` in place of BF16, `--eager` for the step the model runs without compiling, `--tune` to compile it with the records tuning that step gives, `--library NAME` for each library it may call, every one if none is named, both for Llama, Qwen3, and Qwen3.5, and `--prompt N` to profile a prompt of `N` tokens from an empty cache in place of a one-token step.

mod support;

use std::{cmp::Reverse, time::Duration};

use anyhow::{Result, bail};
use singe::{
    DataType,
    backend::cuda::{
        LoweringPolicy,
        export::Library,
        tuning::{GraphTuneOptions, tune_graph},
    },
    generation::{CachedDecoder, Decoder, DecoderOptions},
    graph::GraphCudaOptions,
    nn::{DecoderModel, Model},
};
use singe_model::{
    LoadOptions, Source,
    format::Loader,
    head::CausalLm,
    model::{
        self,
        text::{llama, qwen3, qwen3_5},
    },
};

use singe_testing::{decode_step, profiled, wait};

use crate::support::Arguments;

// The kernels a profile lists, of all it saw.
const SHOWN: usize = 25;

fn step(decoder: &mut dyn Decoder, prompt: Option<&[i64]>) -> Result<()> {
    match prompt {
        Some(prompt) => {
            decoder.reset();
            wait(&decoder.step(&[prompt], None)?)?;
        }
        None => decode_step(decoder)?,
    }
    Ok(())
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
    let steps = arguments.parsed::<u32>("steps")?.unwrap_or(100);
    let data_type = arguments.data_type(DataType::Bf16)?;
    let prompt_length = arguments.parsed::<usize>("prompt")?;
    let tokens = prompt_length.unwrap_or(1);
    let source = Source::open(arguments.snapshot())?;
    let options = LoadOptions::new(arguments.device(), data_type);
    let mut decoder = if arguments.flag("tune") || arguments.value("library").is_some() {
        let libraries = arguments.libraries()?;
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
            "llama" => compiled(
                &llama::load(&mut Loader::new(&source, options))?,
                tokens,
                &libraries,
                tune,
            )?,
            other => bail!("--tune and --library take Llama, Qwen3, or Qwen3.5, not {other}"),
        }
    } else {
        model::load::<Box<dyn CausalLm>>(&source, options)?
            .decoder(DecoderOptions::new(1, 512), !arguments.flag("eager"))?
    };
    let profiled_prompt = prompt_length
        .map(|length| -> Result<Vec<i64>> { Ok((1..=i64::try_from(length)?).collect()) })
        .transpose()?;
    let profiled_prompt = profiled_prompt.as_deref();
    let prompt = (1..=16).collect::<Vec<i64>>();
    wait(&decoder.step(&[&prompt], None)?)?;
    for _ in 0..10 {
        step(decoder.as_mut(), profiled_prompt)?;
    }

    let (ran, wall, gpu) = profiled(|| -> Result<()> {
        for _ in 0..steps {
            step(decoder.as_mut(), profiled_prompt)?;
        }
        Ok(())
    })?;
    ran?;
    let busy = gpu.kernel_time();
    let per_step = |value: Duration| value / steps;
    println!(
        "{steps} steps of {data_type:?}: {:.1} µs each, the GPU busy {:.1} µs of it in {} kernels",
        wall.as_secs_f64() * 1e6 / f64::from(steps),
        per_step(busy).as_secs_f64() * 1e6,
        gpu.launches() / u64::from(steps)
    );
    let mut listed = gpu.kernels.into_iter().collect::<Vec<_>>();
    listed.sort_by_key(|(_, kernel)| Reverse(kernel.time));
    for (name, kernel) in listed.into_iter().take(SHOWN) {
        let share = kernel.time.as_secs_f64() / busy.as_secs_f64().max(f64::MIN_POSITIVE) * 100.0;
        let name = name.chars().take(100).collect::<String>();
        println!(
            "{share:5.1}%  {:8.1} µs  {:3} per step  {name}",
            per_step(kernel.time).as_secs_f64() * 1e6,
            kernel.launches / u64::from(steps)
        );
    }
    Ok(())
}
