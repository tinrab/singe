// Profiles a text-generation snapshot's compiled decoding step on the GPU through CUPTI, and prints each kernel's share of the GPU's time, the largest first: `--snapshot DIR`, `--steps N`, and `--type f16|f32` in place of BF16.

mod support;

use std::{
    cmp::Reverse,
    collections::HashMap,
    time::{Duration, Instant},
};

use anyhow::Result;
use singe::{
    DataType, Tensor,
    generation::{Decoder, DecoderOptions},
};
use singe_cupti::{
    activity::{ActivityBufferCallbackConfig, ActivityCollector},
    types::{ActivityFlushFlag, ActivityKind},
};
use singe_model::text::text_generation::load_decoder;

use crate::support::Arguments;

// The kernels a profile lists, of all it saw.
const SHOWN: usize = 25;

fn wait(logits: &Tensor) -> Result<()> {
    logits.cast(DataType::F32)?.to_vec::<f32>()?;
    Ok(())
}

fn step(decoder: &mut dyn Decoder) -> Result<()> {
    wait(&decoder.step(&[&[1]])?)
}

fn main() -> Result<()> {
    let arguments = Arguments::parse()?;
    let steps = arguments
        .value("steps")
        .map_or(Ok(100), str::parse::<u32>)?;
    let data_type = match arguments.value("type") {
        Some("f16") => DataType::F16,
        Some("f32") => DataType::F32,
        _ => DataType::Bf16,
    };
    let mut decoder = load_decoder(
        &arguments.snapshot,
        arguments.device,
        data_type,
        DecoderOptions::new(1, 512),
        true,
    )?;
    let prompt = (1..=16).collect::<Vec<i64>>();
    wait(&decoder.step(&[&prompt])?)?;
    for _ in 0..10 {
        step(decoder.as_mut())?;
    }

    let collector = ActivityCollector::create(ActivityBufferCallbackConfig::new(8 << 20))?;
    collector.enable(ActivityKind::ConcurrentKernel)?;
    let started = Instant::now();
    for _ in 0..steps {
        step(decoder.as_mut())?;
    }

    let wall = started.elapsed();
    collector.flush_all(ActivityFlushFlag::Forced)?;
    collector.disable(ActivityKind::ConcurrentKernel)?;
    let mut kernels = HashMap::<String, (u64, u64)>::new();
    for buffer in collector.take_buffers()? {
        for record in buffer.records() {
            let Some(kernel) = record?.kernel() else {
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
