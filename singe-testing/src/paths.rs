use std::{collections::HashMap, fmt::Write as _, path::PathBuf, process};

use singe::{
    Device, Tensor,
    backend::cuda::{LoweringPolicy, export::CudaExport, tuning::Measurement},
    graph::{Graph, GraphCudaOptions, GraphPrepareOptions},
    testing::{CompiledProgram, program_target},
};

use crate::{Tolerance, assert_matches, decoder::tune_step, host};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunPath {
    Compiled,
    GeneratedKernels,
    TunedReplay,
    TunedProgram,
}

pub const EVERY_PATH: [RunPath; 4] = [
    RunPath::Compiled,
    RunPath::GeneratedKernels,
    RunPath::TunedReplay,
    RunPath::TunedProgram,
];

pub fn check_paths(
    graph: &Graph,
    inputs: &[(&str, Tensor)],
    expected: &[f32],
    tolerance: Tolerance,
    label: &str,
    paths: &[RunPath],
    device: Device,
) {
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let run = |options: GraphCudaOptions| {
        let mut executor = graph
            .prepare(GraphPrepareOptions::new(device).with_cuda_options(options))
            .unwrap();
        host(&executor.run(&named).unwrap()[0])
    };
    for &path in paths {
        let what = format!("{label} {path:?}");
        let actual =
            match path {
                RunPath::Compiled => run(GraphCudaOptions::new()),
                RunPath::GeneratedKernels => run(GraphCudaOptions::new()
                    .with_lowering_policy(LoweringPolicy::generated_kernels())),
                RunPath::TunedReplay => {
                    let records = tune_step(graph, &named, device, Measurement::GraphReplay);
                    run(GraphCudaOptions::new().with_tuning_records(records))
                }
                RunPath::TunedProgram => {
                    let Some(architecture) = program_target(device) else {
                        continue;
                    };
                    let records = tune_step(graph, &named, device, Measurement::Program);
                    let program = CudaExport::new(architecture)
                        .with_tuning_records(records)
                        .export(graph)
                        .unwrap();
                    let mut name = String::new();
                    for character in label.chars() {
                        name.push(if character.is_ascii_alphanumeric() {
                            character
                        } else {
                            '_'
                        });
                    }

                    let _ = write!(name, "_{}", process::id());
                    let compiled = CompiledProgram::create(&program, &name).unwrap();
                    let ordered = inputs
                        .iter()
                        .map(|(_, tensor)| tensor.clone())
                        .collect::<Vec<_>>();
                    host(&compiled.run(&ordered, 1).unwrap()[0])
                }
            };
        assert_matches(&actual, expected, tolerance, &what);
    }
}

#[must_use]
pub fn target_directory(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../target")
        .join(name)
}
