//! Prints the CUDA C++ that a captured graph's generated kernels compile to, under the schedules the autotuner picks.
//!
//! `cargo run --release -p singe --features cuda_13_4 --example generated_kernels`
//!
//! The graph is a gated residual block (`x + residual`, an RMS norm, a `SiLU` gate) and a scaled softmax.
//! The tuning step measures the graph's choices and every generated kernel's candidate schedules, and compiling keeps the fastest in memory; each kernel is then printed with its default and tuned schedules and its source.
//! The fusion groups come first, the operations each generated kernel computes, and the helper prelude every kernel shares is printed once, at the end.
//! Set `SINGE_KERNEL_TUNING=0` to print the default schedules' sources without measuring.
//! Set `SINGE_KERNEL_PROGRAMS` to a directory to also write each kernel there as a standalone `.cu` program, with a host function that launches it, for use in a CUDA C++ project.
//! Set `SINGE_KERNEL_DOCS=1` to document each kernel's arguments: the graph values its buffers hold, with their shapes and strides, and the values it was planned with.

use std::{collections::HashMap, env, fs, path::PathBuf, time::Duration};

use singe::{
    backend::{
        cuda::{
            KernelOutput, KernelRecorder,
            tuning::{GraphTuneOptions, TunedChoice, schedule_key, tune_graph},
        },
        registry::{LayerNormalizationAttributes, Operation, OperationAttributes},
    },
    data_type::DataType,
    device::Device,
    dispatch::call_one,
    error::Result,
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
    shape::Shape,
    tensor::Tensor,
};

const TOKENS: i64 = 256;
const HIDDEN: i64 = 1024;
const HEADS: i64 = 16;

fn main() -> Result<()> {
    let device = Device::Cuda(0);
    let tune = env::var("SINGE_KERNEL_TUNING").map_or(true, |value| value != "0");
    let programs = env::var_os("SINGE_KERNEL_PROGRAMS").map(PathBuf::from);
    let documentation = env::var("SINGE_KERNEL_DOCS").is_ok_and(|value| value == "1");
    if let Some(directory) = &programs {
        fs::create_dir_all(directory)?;
    }

    // CUDA samples F32, so BF16 inputs are converted samples.
    let tensor = |dimensions: &[i64]| {
        Tensor::randn(device, Shape::create(dimensions)?, DataType::F32, None)?.cast(DataType::Bf16)
    };
    let inputs = [
        ("x", tensor(&[TOKENS, HIDDEN])?),
        ("residual", tensor(&[TOKENS, HIDDEN])?),
        ("gate", tensor(&[TOKENS, HIDDEN])?),
        ("weight", tensor(&[HIDDEN])?),
        ("scores", tensor(&[HEADS, TOKENS, TOKENS])?),
    ];
    let tuning_inputs = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let (graph, _) = Graph::capture(inputs, |inputs| {
        let [x, residual, gate, weight, scores] = inputs else {
            unreachable!("the graph has five inputs");
        };
        let hidden = x.add(residual)?;
        let normed = rms_norm(&hidden, weight)?;
        let gated = normed.mul(&gate.silu()?)?;
        let probabilities = scores.mul_scalar(0.125)?.softmax(-1)?;
        Ok(vec![hidden, gated, probabilities])
    })?;

    let recorder = KernelRecorder::new();
    let mut cuda = GraphCudaOptions::new().with_kernel_recorder(recorder.clone());
    let records = if tune {
        // `TuningRecords::write` keeps these for later processes.
        let records = tune_graph(
            &graph,
            &tuning_inputs,
            device,
            &GraphTuneOptions {
                choice_budget: Duration::from_secs(10),
                plan_budget: Duration::from_secs(10),
                ..GraphTuneOptions::default()
            },
        )?;
        cuda = cuda.with_tuning_records(records.clone());
        Some(records)
    } else {
        None
    };
    let prepared = graph.prepare(GraphPrepareOptions::new(device).with_cuda_options(cuda));
    // The report ends with the fusion groups, the operations each generated kernel computes.
    let report = prepared?.report()?;
    if let Some(start) = report.find("fusion groups:") {
        println!("// {}", report[start..].trim_end().replace('\n', "\n// "));
    }

    let mut prelude = None;
    let mut written = HashMap::<String, usize>::new();
    for planned in recorder.take()? {
        let schedule = records
            .as_ref()
            .and_then(|records| {
                records
                    .entries()
                    .find(|(key, _)| *key == schedule_key(&planned.kernel, &planned.bindings))
                    .map(|(_, record)| record.choice.clone())
            })
            .and_then(|choice| match choice {
                TunedChoice::Schedule(schedule) => Some(schedule),
                _ => None,
            })
            .unwrap_or(planned.schedule);
        let artifact = planned.emit(schedule)?;
        println!(
            "// {} with sizes {:?}, {} blocks of {} threads",
            artifact.name, planned.bindings.sizes, artifact.launch.blocks, artifact.launch.threads
        );
        println!("// default schedule: {:?}", planned.schedule);
        if tune {
            println!("// tuned schedule:   {schedule:?}");
        }

        let Some(source) = artifact.code.source() else {
            println!(
                "// {} is tile ir bytecode, which has no source",
                artifact.name
            );
            continue;
        };
        // Every source kernel of one language starts with the same prelude, printed once at the end.
        prelude.get_or_insert_with(|| source.prelude());
        println!(
            "{}",
            artifact.render(KernelOutput::Listing { documentation })?
        );
        if let Some(directory) = &programs {
            // Kernels of one name differ by the sizes they bake in, so later ones take a number.
            let count = written.entry(artifact.name.clone()).or_default();
            let file = if *count == 0 {
                format!("{}.cu", artifact.name)
            } else {
                format!("{}_{count}.cu", artifact.name)
            };
            *count += 1;
            let path = directory.join(file);
            fs::write(
                &path,
                artifact.render(KernelOutput::File { documentation })?,
            )?;
            println!(
                "// wrote {}, which defines `launch_{}`",
                path.display(),
                artifact.name
            );
        }
    }

    if let Some(prelude) = prelude {
        println!("// The prelude every kernel above starts with:\n{prelude}");
    }
    Ok(())
}

fn rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    call_one(
        Operation::RmsNorm,
        OperationAttributes::LayerNormalization(LayerNormalizationAttributes {
            normalized_shape: vec![HIDDEN],
            has_weight: true,
            has_bias: false,
            eps: 1e-6,
        }),
        &[x, weight],
    )
}
