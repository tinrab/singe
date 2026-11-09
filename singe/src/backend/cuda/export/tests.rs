use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    slice,
    time::Duration,
};

use serde_json::{Value, json};
use singe_cuda::architecture::GpuArchitecture;

use crate::{
    backend::{
        cuda::{
            self, CudaGraphProvider, LoweringPolicy,
            compiled::executable::GraphCompileProviderDecision,
            export::{
                CudaExport, CudaProgram, Library, ProgramFiles, ProgramStep, StateKind,
                StepProvider,
                program::BufferKind,
                writer::{LIBRARY_VERSIONS, fill_template},
            },
            tuning::{
                GraphTuneOptions, Measurement, TunedChoice, TunedRecord, TuningRecords,
                dependent_launch_key, tune_graph,
            },
        },
        registry::Operation,
    },
    data_type::DataType,
    device::Device,
    error::Result,
    graph::{
        CaptureBindings, ComputePrecision, Determinism, Graph, GraphCudaOptions, NumericalPolicy,
        NumericalScope, capture, test_support::patterned_tensor,
    },
    index::TensorIndex,
    random::Generator,
    shape::Shape,
    tensor::{AttentionOptions, Tensor},
    testing::{CompiledProgram, maybe_lock_device, program_target},
};

fn values(dimensions: &[i64], seed: usize) -> Tensor {
    patterned_tensor(Device::Cpu, dimensions, seed)
}

// The architecture of the GPU the test runs programs on, or `None` where there is no GPU or compiler.
fn target() -> Option<(GpuArchitecture, impl Sized)> {
    let lock = maybe_lock_device()?;
    Some((program_target(Device::Cuda(0))?, lock))
}

fn programs_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/programs/tuning")
}

fn compile_program(program: &CudaProgram, name: &str) -> CompiledProgram {
    CompiledProgram::create(program, name).unwrap()
}

fn floats(tensor: &Tensor) -> Vec<f32> {
    tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap()
}

fn assert_close(actual: &Tensor, expected: &Tensor, tolerance: f32, what: &str) {
    assert_eq!(actual.shape(), expected.shape(), "{what}");
    for (index, (actual, expected)) in floats(actual).iter().zip(floats(expected)).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0)
                || (actual.is_nan() && expected.is_nan()),
            "{what}[{index}]: {actual} != {expected}"
        );
    }
}

// Captures `function` over `inputs` on the GPU, exports it under `export`, runs the program, and compares it with Singe's eager result.
fn check(
    export: &CudaExport,
    name: &str,
    inputs: &[(&str, Tensor)],
    tolerance: f32,
    function: impl FnOnce(&[Tensor]) -> Result<Vec<Tensor>>,
) -> CudaProgram {
    let on_device = inputs
        .iter()
        .map(|(name, tensor)| (*name, tensor.to_device(Device::Cuda(0)).unwrap()))
        .collect::<Vec<_>>();
    let (graph, eager) = Graph::capture(on_device.clone(), function).unwrap();
    let program = export.export(&graph).unwrap();
    let compiled = compile_program(&program, name);
    let names = program
        .inputs
        .iter()
        .map(|input| {
            on_device
                .iter()
                .find(|(name, _)| *name == input.name)
                .unwrap()
                .1
                .clone()
        })
        .collect::<Vec<_>>();
    // A second run reuses everything the first set up.
    let instance = compiled.instance().unwrap();
    let outputs = instance.run(&names, 2).unwrap();
    assert_eq!(outputs.len(), eager.len(), "{name}");
    for (actual, expected) in outputs.iter().zip(&eager) {
        assert_close(actual, expected, tolerance, name);
    }
    drop(instance);
    drop(compiled);
    program
}

fn manifest_of(files: &ProgramFiles) -> Value {
    serde_json::from_str(&files.manifest).unwrap()
}

fn compile_command(files: &ProgramFiles) -> String {
    manifest_of(files)["compile"].as_str().unwrap().to_owned()
}

// A decode step writes its cache on every run, so a program must not copy the cache to do so.
#[test]
fn test_programs_write_a_written_input_where_it_is() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    let on_device = |tensor: Tensor| tensor.to_device(device).unwrap();
    let cache = || on_device(values(&[6, 4], 1));
    let source = on_device(values(&[2, 4], 2));
    let index = on_device(
        Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 4]).unwrap(),
    );
    let capture_cache = cache();
    let (graph, eager) = Graph::capture(
        [
            ("cache", capture_cache.clone()),
            ("source", source.clone()),
            ("index", index.clone()),
        ],
        |inputs| {
            inputs[0].index_put_(&[TensorIndex::Indices(inputs[2].clone())], &inputs[1])?;
            Ok(vec![inputs[0].add_scalar(1.0)?])
        },
    )
    .unwrap();
    let program = CudaExport::new(architecture)
        .without_libraries()
        .export(&graph)
        .unwrap();
    let position = program
        .inputs
        .iter()
        .position(|input| input.name == "cache")
        .unwrap();
    assert!(program.inputs[position].written);
    assert!(
        !program.steps.iter().any(|step| matches!(
            step,
            ProgramStep::Copy { destination, .. }
                if matches!(program.buffers[destination.buffer.0].kind, BufferKind::Input(_))
        )),
        "{:?}",
        program.steps
    );
    let compiled = compile_program(&program, "written_in_place");
    let caller_cache = cache();
    let operands = program
        .inputs
        .iter()
        .map(|input| match input.name.as_str() {
            "cache" => caller_cache.clone(),
            "source" => source.clone(),
            _ => index.clone(),
        })
        .collect::<Vec<_>>();
    // The same slots written again leave the same values, so a second run still matches.
    let outputs = compiled.instance().unwrap().run(&operands, 2).unwrap();
    assert_close(&outputs[0], &eager[0], 1e-6, "output");
    assert_close(&caller_cache, &capture_cache, 1e-6, "cache");
}

// `nvcc` compiles `--gpu-architecture=sm_100a` for plain `sm_100`, which rejects what only the suffixed target assembles.
#[test]
fn test_programs_compile_for_their_targets_suffix() {
    let x = values(&[4], 1);
    let (graph, _) = Graph::capture([("x", x)], |inputs| Ok(vec![inputs[0].relu()?])).unwrap();
    let command = |target: &str| {
        let program = CudaExport::new(target.parse().unwrap())
            .without_libraries()
            .export(&graph)
            .unwrap();
        compile_command(&program.render("model").unwrap())
    };
    assert!(command("sm_100a").contains("-gencode=arch=compute_100a,code=sm_100a"));
    assert!(command("sm_120").contains("--gpu-architecture=sm_120"));
}

fn library_steps(program: &CudaProgram) -> Vec<&'static str> {
    program
        .steps
        .iter()
        .filter_map(|step| match step {
            ProgramStep::Library { call, .. } => Some(call.library().name()),
            _ => None,
        })
        .collect()
}

#[test]
fn test_programs_run_generated_kernels_as_singe_does() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let pure = CudaExport::new(architecture).without_libraries();
    // Fusion, branching outputs, views, reductions, and rows.
    let program = check(
        &pure,
        "fused",
        &[("x", values(&[4, 8], 1)), ("y", values(&[4, 8], 2))],
        1e-5,
        |inputs| {
            let shared = inputs[0].add(&inputs[1])?.relu()?;
            Ok(vec![
                shared.mul_scalar(2.0)?,
                shared.tanh()?.softmax(-1)?,
                inputs[0].transpose(0, 1)?,
                inputs[1].sum(&[1], false)?,
            ])
        },
    );
    assert!(library_steps(&program).is_empty());
    // Two results of one kernel, a copy of each row into a concatenation, and a contiguous copy of a strided view.
    check(
        &pure,
        "shapes",
        &[("x", values(&[3, 64], 3)), ("index", values(&[2, 64], 4))],
        1e-5,
        |inputs| {
            let (values, indices) = inputs[0].topk(4, 1, true, true)?;
            Ok(vec![
                values,
                indices,
                Tensor::concat(&[inputs[0].clone(), inputs[1].clone()], 0)?,
                inputs[0].transpose(0, 1)?.contiguous()?.relu()?,
            ])
        },
    );
    // A scatter, which writes its base in place when nothing else reads it.
    let index = Tensor::from_vec(Device::Cpu, Shape::create([2]).unwrap(), vec![1_i64, 3]).unwrap();
    check(
        &pure,
        "scatter",
        &[
            ("x", values(&[6], 5)),
            ("source", values(&[2], 6)),
            ("index", index),
        ],
        1e-5,
        |inputs| {
            Ok(vec![
                inputs[0]
                    .add_scalar(1.0)?
                    .index_copy(0, &inputs[2], &inputs[1])?,
            ])
        },
    );
}

#[test]
fn test_programs_use_tensor_cores_and_keep_half_precision() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let pure = CudaExport::new(architecture).without_libraries();
    let half = |dimensions: &[i64], seed| values(dimensions, seed).cast(DataType::Bf16).unwrap();
    // A linear layer on tensor cores, whose scratch and dynamic shared memory the launch has to carry.
    check(
        &pure,
        "matmul",
        &[("x", half(&[64, 128], 1)), ("w", half(&[64, 128], 2))],
        3e-2,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?]),
    );
    // Fused attention with a cache, as a decoder layer runs it.
    check(
        &pure,
        "attention",
        &[
            ("q", half(&[1, 4, 3, 64], 3)),
            ("k", half(&[1, 2, 32, 64], 4)),
            ("v", half(&[1, 2, 32, 64], 5)),
        ],
        5e-2,
        |inputs| {
            Ok(vec![inputs[0].attend(
                &inputs[1],
                &inputs[2],
                None,
                None,
                AttentionOptions::default(),
            )?])
        },
    );
}

// A convolution as a generated kernel matches cuDNN's, under every window and grouping a library has to handle.
#[test]
fn test_generated_convolutions_match_cudnn() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let pure = CudaExport::new(architecture).without_libraries();
    for (index, (stride, padding, dilation, groups)) in [
        ((1, 1), (0, 0, 0, 0), (1, 1), 1),
        ((2, 1), (1, 1, 2, 2), (1, 2), 1),
        ((1, 1), (1, 1, 1, 1), (1, 1), 2),
    ]
    .into_iter()
    .enumerate()
    {
        let weight = values(&[8, 4 / groups, 3, 3], 2);
        let program = check(
            &pure,
            &format!("convolution{index}"),
            &[
                ("x", values(&[2, 4, 16, 16], 1)),
                ("w", weight),
                ("b", values(&[8], 3)),
            ],
            1e-3,
            |inputs| {
                Ok(vec![inputs[0].convolution2d(
                    &inputs[1],
                    Some(&inputs[2]),
                    stride,
                    padding,
                    dilation,
                    groups,
                )?])
            },
        );
        assert!(library_steps(&program).is_empty());
    }
}

// A program can call each library, and mixes those calls with generated kernels.
#[test]
fn test_programs_call_libraries_and_kernels_together() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let export = |library| CudaExport::new(architecture).with_libraries([library]);
    let program = check(
        &export(Library::Cublas),
        "cublas",
        &[("x", values(&[32, 64], 1)), ("w", values(&[48, 64], 2))],
        1e-3,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?.relu()?]),
    );
    assert_eq!(library_steps(&program), ["cublas"]);
    let program = check(
        &export(Library::Cufft),
        "cufft",
        &[("x", values(&[2, 1024], 3))],
        1e-3,
        |inputs| {
            let spectrum = inputs[0].rfft()?;
            Ok(vec![spectrum.clone(), spectrum.irfft(1024)?])
        },
    );
    assert_eq!(library_steps(&program), ["cufft", "cufft"]);
    let program = check(
        &export(Library::Cutensor),
        "cutensor",
        &[("a", values(&[3, 4], 4)), ("b", values(&[4, 5], 5))],
        1e-3,
        |inputs| {
            Ok(vec![Tensor::einsum(
                "ij,jk->ik",
                &[&inputs[0], &inputs[1]],
            )?])
        },
    );
    assert_eq!(library_steps(&program), ["cutensor"]);
    let program = check(
        &export(Library::Cudnn),
        "cudnn",
        &[
            ("x", values(&[2, 3, 16, 16], 6)),
            ("w", values(&[8, 3, 3, 3], 7)),
            ("b", values(&[8], 8)),
            ("second", values(&[4, 8, 3, 3], 9)),
        ],
        1e-4,
        |inputs| {
            // A chain writes channels-last strides between its convolutions, and the second has no bias.
            let hidden = inputs[0].convolution2d(
                &inputs[1],
                Some(&inputs[2]),
                (1, 1),
                (1, 1, 1, 1),
                (1, 1),
                1,
            )?;
            Ok(vec![hidden.convolution2d(
                &inputs[3],
                None,
                (1, 1),
                (0, 0, 0, 0),
                (1, 1),
                1,
            )?])
        },
    );
    assert_eq!(library_steps(&program), ["cudnn", "cudnn"]);
    let scale = |value: f32| {
        Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![value]).unwrap()
    };
    let fp8 = |dimensions: &[i64], seed| values(dimensions, seed).cast(DataType::F8E4M3).unwrap();
    let program = check(
        &export(Library::CublasLt),
        "cublaslt",
        &[
            ("x", fp8(&[32, 64], 10)),
            ("w", fp8(&[48, 64], 11)),
            ("a", scale(0.5)),
            ("b", scale(2.0)),
        ],
        2e-2,
        |inputs| {
            Ok(vec![inputs[0].scaled_matmul(
                &inputs[1].transpose(0, 1)?,
                &inputs[2],
                &inputs[3],
                DataType::Bf16,
            )?])
        },
    );
    assert_eq!(library_steps(&program), ["cublaslt"]);
}

// Written inputs persist across runs, and constants large enough for the weights file load from it.
#[test]
fn test_programs_write_inputs_and_load_large_constants() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    let cache = Tensor::zeroed_on(device, DataType::F32, Shape::create([2, 3]).unwrap()).unwrap();
    let row = values(&[1, 3], 1).to_device(device).unwrap();
    let (graph, _) = Graph::capture([("cache", cache), ("row", row.clone())], |inputs| {
        inputs[0].narrow(0, 1, 1)?.assign(&inputs[1])?;
        Ok(vec![inputs[0].add(&inputs[0])?])
    })
    .unwrap();
    let export = CudaExport::new(architecture).without_libraries();
    let program = export.export(&graph).unwrap();
    assert!(program.inputs.iter().any(|input| input.written));
    let compiled = compile_program(&program, "written");
    let target = Tensor::zeroed_on(device, DataType::F32, Shape::create([2, 3]).unwrap()).unwrap();
    let inputs = program
        .inputs
        .iter()
        .map(|input| {
            if input.name == "cache" {
                target.clone()
            } else {
                row.clone()
            }
        })
        .collect::<Vec<_>>();
    compiled.run(&inputs, 2).unwrap();
    assert_eq!(floats(&target)[3..], floats(&row)[..]);

    // A parameter too large to embed goes to the weights file.
    let weight = values(&[64, 128], 2).to_device(device).unwrap();
    let x = values(&[4, 64], 3).to_device(device).unwrap();
    let (graph, eager) = Graph::capture([("x", x.clone())], {
        move |inputs| {
            let constant = capture::constant("weight", &weight)?;
            Ok(vec![inputs[0].matmul(&constant)?])
        }
    })
    .unwrap();
    let program = export.export(&graph).unwrap();
    assert!(!program.render("model").unwrap().weights.is_empty());
    let compiled = compile_program(&program, "weights");
    let outputs = compiled.run(&[x], 1).unwrap();
    assert_close(&outputs[0], &eager[0], 1e-3, "weights");
}

// An export needs neither a GPU nor a library: what the policy leaves out decomposes, and what it allows is written as a call for the user's compile to link.
#[test]
fn test_exports_need_no_library_and_decompose_what_is_not_allowed() {
    let architecture = "sm_120".parse().unwrap();
    let signal = values(&[2, 1024], 1);
    let (graph, _) = Graph::capture([("x", signal)], |inputs| Ok(vec![inputs[0].rfft()?])).unwrap();
    let pure = CudaExport::new(architecture).without_libraries();
    let decomposed = pure.export(&graph).unwrap();
    assert!(decomposed.decomposed.contains(&Operation::Rfft));
    assert!(library_steps(&decomposed).is_empty());
    let manifest = manifest_of(&decomposed.render("model").unwrap());
    assert_eq!(manifest["decomposed"], json!(["rfft"]), "{manifest}");
    let library = CudaExport::new(architecture).export(&graph).unwrap();
    assert!(library.decomposed.is_empty());
    assert_eq!(library_steps(&library), ["cufft"]);
    let files = library.render("model").unwrap();
    assert!(files.source.contains("#include <cufft.h>"));
    assert!(files.source.contains("cufftExecR2C"));
    assert_eq!(manifest_of(&files)["link"], "-lcufft");
}

// Two sums of different shapes are two kernels, and two of one shape are one.
#[test]
fn test_programs_name_each_distinct_kernel_once() {
    let architecture = "sm_120".parse().unwrap();
    let (graph, _) = Graph::capture(
        [
            ("a", values(&[4, 8], 1)),
            ("b", values(&[4, 16], 2)),
            ("c", values(&[4, 8], 3)),
        ],
        |inputs| {
            Ok(vec![
                inputs[0].sum(&[1], false)?,
                inputs[1].sum(&[1], false)?,
                inputs[2].sum(&[1], false)?,
            ])
        },
    )
    .unwrap();
    let program = CudaExport::new(architecture)
        .without_libraries()
        .export(&graph)
        .unwrap();
    let source = program.render("model").unwrap().source;
    assert_eq!(source.matches(" void singe_sum(").count(), 2, "{source}");
    assert_eq!(source.matches("launch_singe_sum(").count(), 5, "{source}");
}

// An exported program places each operation where a compiled graph under the same policy does.
#[test]
fn test_programs_agree_with_compiled_graphs_on_providers() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    let (graph, _) = Graph::capture(
        [
            ("x", values(&[32, 64], 1).to_device(device).unwrap()),
            ("w", values(&[48, 64], 2).to_device(device).unwrap()),
        ],
        |inputs| {
            let product = inputs[0].matmul(&inputs[1].transpose(0, 1)?)?;
            Ok(vec![product.softmax(-1)?, inputs[0].sum(&[1], false)?])
        },
    )
    .unwrap();
    let policy = LoweringPolicy::new(vec![
        CudaGraphProvider::Alias,
        CudaGraphProvider::Kernel,
        CudaGraphProvider::Fusion,
        CudaGraphProvider::Cublas,
    ]);
    let program = CudaExport::new(architecture)
        .with_libraries([Library::Cublas])
        .export(&graph)
        .unwrap();
    let compiled = cuda::compile_graph(
        &graph,
        &GraphCudaOptions::new().with_lowering_policy(policy),
    )
    .unwrap();
    let exported = program
        .providers
        .iter()
        .filter_map(|provider| {
            provider
                .operation
                .map(|operation| (operation.name(), provider.provider))
        })
        .collect::<Vec<_>>();
    for operation in compiled.report().operations() {
        let realized = operation
            .realized_decision()
            .and_then(GraphCompileProviderDecision::provider);
        let library = match realized {
            Some(CudaGraphProvider::Cublas) => Library::Cublas,
            Some(CudaGraphProvider::CublasLt) => Library::CublasLt,
            _ => continue,
        };
        assert!(
            exported.contains(&(operation.operation(), StepProvider::Library(library))),
            "{} on {library} in {exported:?}",
            operation.operation()
        );
    }
}

// A program holds its generators' state and advances it, drawing what a generator would across calls.
#[test]
fn test_programs_advance_generators_across_runs() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    let x = values(&[64], 1).to_device(device).unwrap();
    let generator = Generator::new(7);
    let draw = |x: &Tensor, generator: &Generator| {
        Tensor::rand(x.device(), x.shape(), DataType::F32, Some(generator))?.add(x)
    };
    let (graph, _) = Graph::capture([("x", x.clone())], |inputs| {
        Ok(vec![draw(&inputs[0], &generator)?])
    })
    .unwrap();
    let program = CudaExport::new(architecture)
        .without_libraries()
        .export(&graph)
        .unwrap();
    let (name, blocks) = program
        .state
        .iter()
        .find_map(|state| match state.kind {
            StateKind::Generator { blocks } => Some((state.name.clone(), blocks)),
            _ => None,
        })
        .unwrap();
    assert!(program.inputs.iter().all(|input| input.name != name));
    let compiled = compile_program(&program, "random");
    let instance = compiled.instance().unwrap();
    instance.set_generator(&name, 11, 0).unwrap();
    let expected = Generator::new(11);
    for run in 0..3 {
        let outputs = instance.run(slice::from_ref(&x), 1).unwrap();
        assert_close(&outputs[0], &draw(&x, &expected).unwrap(), 1e-6, "random");
        let next = i64::try_from(u64::try_from(run + 1).unwrap() * blocks).unwrap();
        assert_eq!(instance.generator(&name).unwrap(), (11, next));
    }
}

// A result only a run can size comes with its count, and the caller's buffer holds as many rows as could be written.
#[test]
fn test_programs_report_how_many_rows_counted_results_hold() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let pure = CudaExport::new(architecture).without_libraries();
    let mask = (0..16).map(|index| index % 3 == 0).collect::<Vec<_>>();
    let mask = Tensor::from_vec(Device::Cpu, Shape::create([16]).unwrap(), mask).unwrap();
    let program = check(
        &pure,
        "counted",
        &[("x", values(&[16], 1)), ("mask", mask)],
        1e-6,
        |inputs| {
            Ok(vec![
                inputs[0].masked_select(&inputs[1])?,
                inputs[1].nonzero()?,
                inputs[0].relu()?,
            ])
        },
    );
    let counted = program
        .outputs
        .iter()
        .filter(|output| output.count.is_some());
    assert_eq!(counted.count(), 2);
    assert_eq!(program.outputs.len(), 5);
    let header = program.render("model").unwrap().header;
    assert!(
        header.contains("as many leading rows as `output0_count` holds"),
        "{header}"
    );
}

// Tensors have no functions unless asked for, since a model holds hundreds of them.
#[test]
fn test_programs_have_no_tensor_functions_by_default() {
    let shape = Shape::create([4]).unwrap();
    let total = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let x = Tensor::zeroed_on(Device::Cpu, DataType::F32, &shape).unwrap();
    let bindings = CaptureBindings::new().with_buffer("total", total.clone());
    let (graph, _) = Graph::capture_with_bindings([("x", x)], bindings, |inputs| {
        Ok(vec![total.add(&inputs[0])?])
    })
    .unwrap();
    let export = CudaExport::new("sm_120".parse().unwrap()).without_libraries();
    let render = |export: &CudaExport| export.export(&graph).unwrap().render("model").unwrap();
    let plain = render(&export);
    assert!(
        !plain.header.contains("model_get_total"),
        "{}",
        plain.header
    );
    assert!(!plain.source.contains("model_set_total"));
    let accessible = render(&export.with_state_functions(true));
    assert!(
        accessible.header.contains("model_get_total"),
        "{}",
        accessible.header
    );
    assert!(accessible.source.contains("model_set_total"));
}

// A program with state functions has a pair for every tensor of the graph, and a run reads what they set.
#[test]
fn test_programs_read_and_replace_their_tensors() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    // Too large to embed, so it starts from the weights file.
    let weight = values(&[64, 128], 2).to_device(device).unwrap();
    let scale = values(&[64], 3).to_device(device).unwrap();
    let x = values(&[4, 64], 4).to_device(device).unwrap();
    let bindings = CaptureBindings::new().with_parameter("linear.weight", weight.clone());
    let compute = |x: &Tensor, weight: &Tensor, scale: &Tensor| x.mul(scale)?.matmul(weight);
    let (graph, eager) = Graph::capture_with_bindings([("x", x.clone())], bindings, |inputs| {
        let scale = capture::constant("scale", &scale)?;
        Ok(vec![compute(&inputs[0], &weight, &scale)?])
    })
    .unwrap();
    let program = CudaExport::new(architecture)
        .without_libraries()
        .with_state_functions(true)
        .export(&graph)
        .unwrap();
    let kind = |name: &str| {
        program
            .state
            .iter()
            .find(|state| state.name == name)
            .map(|state| state.kind)
    };
    assert_eq!(kind("linear_weight"), Some(StateKind::Initializer));
    assert_eq!(kind("scale"), Some(StateKind::Constant));
    let compiled = compile_program(&program, "state");
    let instance = compiled.instance().unwrap();
    assert_close(
        &instance.get("linear_weight").unwrap(),
        &weight,
        0.0,
        "weight",
    );
    assert_close(&instance.get("scale").unwrap(), &scale, 0.0, "scale");
    let inputs = slice::from_ref(&x);
    assert_close(
        &instance.run(inputs, 1).unwrap()[0],
        &eager[0],
        1e-3,
        "exported",
    );
    // Host and device memory both work.
    let other_weight = values(&[64, 128], 5);
    let other_scale = values(&[64], 6).to_device(device).unwrap();
    instance
        .set_from_host("linear_weight", &other_weight)
        .unwrap();
    instance.set("scale", &other_scale).unwrap();
    assert_close(
        &instance.get("linear_weight").unwrap(),
        &other_weight,
        0.0,
        "replaced",
    );
    let expected = compute(&x, &other_weight.to_device(device).unwrap(), &other_scale).unwrap();
    assert_close(
        &instance.run(inputs, 1).unwrap()[0],
        &expected,
        1e-3,
        "replaced",
    );
}

// Graph names become distinct C++ identifiers, whatever characters they hold.
#[test]
fn test_programs_keep_colliding_names_apart() {
    let (graph, _) = Graph::capture(
        [
            ("a.b", values(&[4], 1)),
            ("a-b", values(&[4], 2)),
            ("class", values(&[4], 3)),
        ],
        |inputs| Ok(vec![inputs[0].add(&inputs[1])?.add(&inputs[2])?]),
    )
    .unwrap();
    let program = CudaExport::new("sm_120".parse().unwrap())
        .without_libraries()
        .export(&graph)
        .unwrap();
    let header = program.render("model").unwrap().header;
    assert!(header.contains("a_b;"), "{header}");
    assert!(header.contains("a_b_1;"), "{header}");
    assert!(header.contains("class_;"), "{header}");
}

// A mix of every kind of step in one program, as an image and text encoder might run.
#[test]
fn test_programs_mix_library_calls_and_fused_kernels() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let export = CudaExport::new(architecture).with_libraries([
        Library::Cublas,
        Library::Cudnn,
        Library::Cufft,
    ]);
    let program = check(
        &export,
        "mixed",
        &[
            ("image", values(&[2, 3, 8, 8], 1)),
            ("filter", values(&[4, 3, 3, 3], 2)),
            ("projection", values(&[64, 1024], 3)),
        ],
        1e-2,
        |inputs| {
            let features = inputs[0]
                .convolution2d(&inputs[1], None, (2, 2), (1, 1, 1, 1), (1, 1), 1)?
                .relu()?;
            let flat = features.flatten(1, -1)?;
            let hidden = flat.matmul(&inputs[2])?.tanh()?;
            let spectrum = hidden.rfft()?;
            Ok(vec![hidden.softmax(-1)?, spectrum.irfft(1024)?])
        },
    );
    let libraries = library_steps(&program);
    for library in ["cudnn", "cublas", "cufft"] {
        assert!(libraries.contains(&library), "{libraries:?}");
    }
    assert!(
        program
            .steps
            .iter()
            .any(|step| matches!(step, ProgramStep::Kernel { .. }))
    );
    let manifest = manifest_of(&program.render("model").unwrap());
    assert!(manifest["libraries"]["cudnn"].is_string(), "{manifest}");
}

// Tuning measures on the device, so a program for its architecture reads the records, and one for another ignores them.
#[test]
fn test_programs_read_records_measured_for_their_architecture() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let half = |dimensions: &[i64], seed| values(dimensions, seed).cast(DataType::Bf16).unwrap();
    let inputs = [("x", half(&[64, 128], 1)), ("w", half(&[64, 128], 2))];
    let function = |inputs: &[Tensor]| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?]);
    let (graph, _) = Graph::capture(inputs.clone(), function).unwrap();
    let named = inputs
        .iter()
        .map(|(name, tensor)| ((*name).to_owned(), tensor.clone()))
        .collect::<HashMap<_, _>>();
    let records = tune_graph(
        &graph,
        &named,
        Device::Cuda(0),
        &GraphTuneOptions {
            policy: LoweringPolicy::generated_kernels(),
            choice_budget: Duration::from_secs(4),
            runs: 3,
            schedules: true,
            measurement: Measurement::Program,
            build_directory: programs_directory(),
            plan_budget: Duration::ZERO,
            plan_speed: false,
            ..GraphTuneOptions::default()
        },
    )
    .unwrap();
    assert!(!records.is_empty());
    let export = CudaExport::new(architecture)
        .without_libraries()
        .with_tuning_records(records.clone());
    let program = check(&export, "tuned", &inputs, 3e-2, function);
    assert!(
        program.record_coverage.used > 0,
        "{:?}",
        program.record_coverage
    );
    let manifest = manifest_of(&program.render("model").unwrap());
    assert_eq!(
        manifest["tuned_kernels"].as_array().map(Vec::len),
        Some(program.tuned_kernels.len()),
        "{manifest}"
    );
    assert!(
        manifest["tuning_records"]["used"].as_u64().unwrap() > 0,
        "{manifest}"
    );
    let other = if architecture.to_string() == "sm_80" {
        "sm_90"
    } else {
        "sm_80"
    };
    // Records made for another architecture are an error that says so, not something a program quietly ignores.
    let error = CudaExport::new(other.parse().unwrap())
        .without_libraries()
        .with_tuning_records(records)
        .export(&graph)
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains(&format!("where this has \"{other}\"")),
        "{error}"
    );
}

// A policy reaches the libraries a program calls and the flags it compiles with.
// Kernels that overlap compute what they do in turn, and what makes that safe is seen to matter: the same program, told to let its successor start at once, reads too early only without its wait.
#[test]
fn test_dependent_launch_is_safe_because_a_kernel_waits() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    if architecture.compute_capability() < (9, 0) {
        return;
    }
    let mut records = TuningRecords::new(None);
    records.insert(
        dependent_launch_key(),
        TunedRecord {
            choice: TunedChoice::DependentLaunch,
            default_ns: None,
            best_ns: 0,
            bytes: 0,
        },
    );
    let export = CudaExport::new(architecture)
        .without_libraries()
        .with_tuning_records(records);
    // A large product a softmax and another product then read.
    let inputs = [
        ("x", values(&[1, 1024, 512], 1)),
        ("w", values(&[1, 512, 1024], 2)),
        ("y", values(&[1, 1024, 64], 3)),
    ];
    let chain = |inputs: &[Tensor]| {
        Ok(vec![
            inputs[0]
                .matmul(&inputs[1])?
                .softmax(-1)?
                .matmul(&inputs[2])?,
        ])
    };
    let program = check(&export, "dependent", &inputs, 1e-2, chain);
    assert!(program.dependent_launch);
    let files = program.render("model").unwrap();
    assert!(files.source.contains("griddepcontrol.wait"));
    assert!(files.source.contains("cudaLaunchKernelEx"));

    let on_device = inputs
        .iter()
        .map(|(name, tensor)| (*name, tensor.to_device(Device::Cuda(0)).unwrap()))
        .collect::<Vec<_>>();
    let (graph, eager) = Graph::capture(on_device.clone(), chain).unwrap();
    let program = export.export(&graph).unwrap();
    let ordered = |of: &[(&str, Tensor)]| {
        program
            .inputs
            .iter()
            .map(|input| {
                of.iter()
                    .find(|(name, _)| *name == input.name)
                    .unwrap()
                    .1
                    .to_device(Device::Cuda(0))
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    // Runs alternate between two inputs, since a kernel reading too early would otherwise find the same values the last run left.
    let other = [
        ("x", values(&[1, 1024, 512], 4)),
        ("w", values(&[1, 512, 1024], 5)),
        ("y", values(&[1, 1024, 64], 6)),
    ];
    let other_expected = chain(
        &other
            .iter()
            .map(|(_, tensor)| tensor.to_device(Device::Cuda(0)).unwrap())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let runs = [
        (ordered(&inputs), floats(&eager[0])),
        (ordered(&other), floats(&other_expected[0])),
    ];
    // How many of many runs compute something else, with each kernel signalling its successor as it starts.
    let early_reads = |name: &str, keep_wait: bool| {
        let directory = programs_directory().join(name);
        program.write_to(&directory, "model").unwrap();
        let path = directory.join("model.cu");
        let trigger = "asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");";
        let wait = "asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");";
        let source = fs::read_to_string(&path).unwrap().replace(trigger, "");
        let source = source.replace(
            wait,
            if keep_wait {
                "asm volatile(\"griddepcontrol.wait;\" ::: \"memory\"); asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");"
            } else {
                "asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");"
            },
        );
        fs::write(&path, source).unwrap();
        let compiled = CompiledProgram::create_from_written(&program, directory).unwrap();
        let instance = compiled.instance().unwrap();
        (0..40)
            .filter(|run| {
                let (inputs, expected) = &runs[run % 2];
                let outputs = instance.run(inputs, 1).unwrap();
                floats(&outputs[0])
                    .iter()
                    .zip(expected)
                    .any(|(actual, expected)| {
                        (actual - expected).abs() > 1e-2 * expected.abs().max(1.0)
                    })
            })
            .count()
    };
    let (with_wait, without_wait) = (
        early_reads("dependent-wait", true),
        early_reads("dependent-no-wait", false),
    );
    eprintln!("wrong runs of 40: {with_wait} with its wait, {without_wait} without");
    assert_eq!(with_wait, 0);
    assert!(without_wait > 0);
}

#[test]
fn test_programs_follow_the_numerical_policy() {
    let (graph, _) = Graph::capture(
        [("x", values(&[32, 64], 1)), ("w", values(&[48, 64], 2))],
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?]),
    )
    .unwrap();
    let export = CudaExport::new("sm_120".parse().unwrap()).with_libraries([Library::Cublas]);
    let files = export.export(&graph).unwrap().render("model").unwrap();
    assert!(files.source.contains("CUBLAS_PEDANTIC_MATH"));
    assert!(!files.source.contains("CUBLAS_ATOMICS_NOT_ALLOWED"));
    assert!(compile_command(&files).contains("--fmad=false"));
    let files = {
        let _scope = NumericalScope::enter(NumericalPolicy {
            precision: ComputePrecision::Relaxed,
            determinism: Determinism::Require,
        });
        export.export(&graph).unwrap().render("model").unwrap()
    };
    assert!(files.source.contains("CUBLAS_TF32_TENSOR_OP_MATH"));
    assert!(files.source.contains("CUBLAS_ATOMICS_NOT_ALLOWED"));
    assert!(compile_command(&files).contains("--use_fast_math"));
}

// A buffer a graph writes keeps its final value in the program, which later runs read and the caller can read back.
#[test]
fn test_programs_keep_what_a_graph_writes_to_its_buffers() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let device = Device::Cuda(0);
    let total = Tensor::zeroed_on(device, DataType::F32, Shape::create([4]).unwrap()).unwrap();
    let x = values(&[4], 1).to_device(device).unwrap();
    let bindings = CaptureBindings::new().with_buffer("total", total.clone());
    let (graph, _) = Graph::capture_with_bindings([("x", x.clone())], bindings, |inputs| {
        let next = total.add(&inputs[0])?;
        total.assign(&next)?;
        Ok(vec![next])
    })
    .unwrap();
    let program = CudaExport::new(architecture)
        .without_libraries()
        .with_state_functions(true)
        .export(&graph)
        .unwrap();
    let compiled = compile_program(&program, "buffers");
    let instance = compiled.instance().unwrap();
    // Capture ran the write once, so the exported value already holds it.
    let zeros = Tensor::zeroed_on(device, DataType::F32, Shape::create([4]).unwrap()).unwrap();
    instance.set("total", &zeros).unwrap();
    let inputs = slice::from_ref(&x);
    let outputs = instance.run(inputs, 3).unwrap();
    let expected = x.mul_scalar(3.0).unwrap();
    assert_close(&outputs[0], &expected, 1e-5, "output");
    assert_close(&instance.get("total").unwrap(), &expected, 1e-5, "buffer");
}

// A generated matmul fuses with what follows it, and a matmul a library runs leaves its neighbours to fuse among themselves.
#[test]
fn test_libraries_decide_what_fuses() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let inputs = [
        ("x", values(&[8, 16], 1)),
        ("w", values(&[32, 16], 2)),
        ("r", values(&[8, 32], 3)),
    ];
    let compute = |inputs: &[Tensor]| {
        let product = inputs[0].matmul(&inputs[1].transpose(0, 1)?)?;
        Ok(vec![product.add(&inputs[2])?.relu()?])
    };
    let export = CudaExport::new(architecture);
    let fused = check(
        &export.clone().without_libraries(),
        "fused",
        &inputs,
        1e-3,
        compute,
    );
    assert_eq!((kernel_steps(&fused), library_steps(&fused).len()), (1, 0));
    let unfused = check(
        &export.clone().without_libraries().with_fusion(false),
        "unfused",
        &inputs,
        1e-3,
        compute,
    );
    assert_eq!(
        (kernel_steps(&unfused), library_steps(&unfused).len()),
        (3, 0)
    );
    let library = check(
        &export.with_libraries([Library::Cublas]),
        "library",
        &inputs,
        1e-3,
        compute,
    );
    assert_eq!(
        (kernel_steps(&library), library_steps(&library)),
        (1, vec!["cublas"])
    );
}

fn kernel_steps(program: &CudaProgram) -> usize {
    program
        .steps
        .iter()
        .filter(|step| matches!(step, ProgramStep::Kernel { .. }))
        .count()
}

// An allowed library runs every matmul its checks accept, and computes what follows inside its call.
#[test]
fn test_allowed_libraries_run_every_matmul_they_accept() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32, DataType::F64] {
        let half = matches!(data_type, DataType::F16 | DataType::Bf16);
        // Tight enough to catch an engine rounding `F32` inputs to `TF32`.
        let tolerance = if half { 5e-2 } else { 1e-4 };
        let cast = |dimensions: &[i64], seed| values(dimensions, seed).cast(data_type).unwrap();
        let inputs = [
            ("x", cast(&[2, 8, 16], 1)),
            ("w", cast(&[2, 32, 16], 2)),
            ("r", cast(&[2, 8, 32], 3)),
        ];
        let product = |inputs: &[Tensor]| inputs[0].matmul(&inputs[1].transpose(1, 2)?);
        for library in [Library::Cublas, Library::CublasLt, Library::Cudnn] {
            // cuDNN's checks leave full-precision `F32` and `F64` matmuls to generated kernels.
            let expected = if library == Library::Cudnn
                && matches!(data_type, DataType::F32 | DataType::F64)
            {
                Vec::new()
            } else {
                vec![library.name()]
            };
            let export = CudaExport::new(architecture).with_libraries([library]);
            let name = format!("{library}-{data_type}");
            let plain = check(
                &export,
                &format!("{name}-plain"),
                &inputs,
                tolerance,
                |inputs| Ok(vec![product(inputs)?]),
            );
            assert_eq!(library_steps(&plain), expected, "{name}");
            if library == Library::Cublas || expected.is_empty() {
                continue;
            }
            // The add and the activation join the call, leaving nothing for a kernel.
            let fused = check(
                &export,
                &format!("{name}-fused"),
                &inputs,
                tolerance,
                |inputs| Ok(vec![product(inputs)?.add(&inputs[2])?.relu()?]),
            );
            assert_eq!(library_steps(&fused), expected, "{name}");
            assert_eq!(kernel_steps(&fused), 0, "{name}");
            let activated = check(
                &export,
                &format!("{name}-gelu"),
                &inputs,
                tolerance,
                |inputs| Ok(vec![product(inputs)?.gelu()?]),
            );
            assert_eq!(library_steps(&activated), expected, "{name}");
            assert_eq!(kernel_steps(&activated), 0, "{name}");
        }
    }
}

// Of the allowed libraries, the first whose checks accept a matmul runs it, with nothing behind it to fall back to.
#[test]
fn test_the_first_library_that_accepts_a_matmul_runs_it_alone() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let inputs = |data_type| {
        [
            ("x", values(&[8, 16], 1).cast(data_type).unwrap()),
            ("w", values(&[32, 16], 2).cast(data_type).unwrap()),
            ("r", values(&[8, 32], 3).cast(data_type).unwrap()),
        ]
    };
    let fused = |inputs: &[Tensor]| {
        Ok(vec![
            inputs[0]
                .matmul(&inputs[1].transpose(0, 1)?)?
                .add(&inputs[2])?
                .relu()?,
        ])
    };
    // A matmul with no batch and one that repeats a batch dimension.
    let lt = CudaExport::new(architecture).with_libraries([Library::CublasLt]);
    let unbatched = check(
        &lt,
        "cublaslt-unbatched",
        &inputs(DataType::F32)[..2],
        1e-4,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?]),
    );
    assert_eq!(library_steps(&unbatched), ["cublaslt"]);
    check(
        &lt,
        "cublaslt-repeated",
        &[
            ("x", values(&[2, 8, 16], 1)),
            ("w", values(&[1, 32, 16], 2)),
        ],
        1e-4,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(1, 2)?)?]),
    );
    let both = CudaExport::new(architecture).with_libraries([Library::Cudnn, Library::CublasLt]);
    let half = check(&both, "first-half", &inputs(DataType::F16), 5e-2, fused);
    assert_eq!(library_steps(&half), ["cudnn"]);
    assert_eq!(kernel_steps(&half), 0);
    // cuDNN's checks leave a full-precision `F32` matmul to the next library.
    let full = check(&both, "first-full", &inputs(DataType::F32), 1e-4, fused);
    assert_eq!(library_steps(&full), ["cublaslt"]);
    let manifest = manifest_of(&full.render("model").unwrap());
    assert!(manifest.get("choices").is_none(), "{manifest}");
    // cuBLAS runs a matmul alone.
    let alone = CudaExport::new(architecture);
    let (graph, _) = Graph::capture(
        [("x", values(&[8, 16], 1)), ("w", values(&[32, 16], 2))],
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?]),
    )
    .unwrap();
    let program = alone.export(&graph).unwrap();
    assert!(
        matches!(program.steps[0], ProgramStep::Library { .. }),
        "{:?}",
        program.steps
    );
}

// cuDNN runs every convolution when it is allowed, depthwise ones too, with a bias and its activation inside its call.
#[test]
fn test_cudnn_runs_every_convolution_when_allowed() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let export = CudaExport::new(architecture).with_libraries([Library::Cudnn]);
    for data_type in [DataType::F16, DataType::Bf16, DataType::F32, DataType::F64] {
        let half = matches!(data_type, DataType::F16 | DataType::Bf16);
        let tolerance = if half { 5e-2 } else { 1e-3 };
        let cast = |dimensions: &[i64], seed| values(dimensions, seed).cast(data_type).unwrap();
        for (name, weight, groups) in [
            ("ordinary", vec![8, 4, 3, 3], 1),
            ("grouped", vec![8, 2, 3, 3], 2),
            ("depthwise", vec![4, 1, 3, 3], 4),
        ] {
            for (bias, relu) in [(false, false), (true, false), (false, true), (true, true)] {
                let inputs = [
                    ("x", cast(&[2, 4, 16, 16], 1)),
                    ("w", cast(&weight, 2)),
                    ("b", cast(&[weight[0]], 3)),
                ];
                let label = format!("{data_type}-{name}-{bias}-{relu}");
                let program = check(&export, &label, &inputs, tolerance, |inputs| {
                    let result = inputs[0].convolution2d(
                        &inputs[1],
                        bias.then_some(&inputs[2]),
                        (1, 1),
                        (1, 1, 1, 1),
                        (1, 1),
                        groups,
                    )?;
                    Ok(vec![if relu { result.relu()? } else { result }])
                });
                assert_eq!(library_steps(&program), ["cudnn"], "{label}");
                // cuDNN's engines fuse an activation only after a bias.
                let separate = relu && !bias;
                assert_eq!(kernel_steps(&program), usize::from(separate), "{label}");
            }
        }
    }
}

// With no library allowed, the same graphs are generated kernels alone.
#[test]
fn test_without_libraries_the_same_work_is_generated_kernels() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let export = CudaExport::new(architecture).without_libraries();
    let matmul = check(
        &export,
        "generated-matmul",
        &[("x", values(&[8, 16], 1)), ("w", values(&[32, 16], 2))],
        1e-3,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(0, 1)?)?.relu()?]),
    );
    assert!(library_steps(&matmul).is_empty());
    assert_eq!(kernel_steps(&matmul), 1);
    let depthwise = check(
        &export,
        "generated-depthwise",
        &[
            ("x", values(&[2, 4, 16, 16], 1)),
            ("w", values(&[4, 1, 3, 3], 2)),
        ],
        1e-3,
        |inputs| {
            Ok(vec![inputs[0].convolution2d(
                &inputs[1],
                None,
                (1, 1),
                (1, 1, 1, 1),
                (1, 1),
                4,
            )?])
        },
    );
    assert!(library_steps(&depthwise).is_empty());
}

// A policy that allows `TF32` lets cuDNN run the `F32` matmuls full precision keeps from it.
#[test]
fn test_cudnn_runs_f32_matmuls_when_the_policy_allows_tf32() {
    let Some((architecture, _lock)) = target() else {
        return;
    };
    let _scope = NumericalScope::enter(NumericalPolicy {
        precision: ComputePrecision::AllowTf32,
        determinism: Determinism::Allow,
    });
    let program = check(
        &CudaExport::new(architecture).with_libraries([Library::Cudnn]),
        "cudnn-tf32",
        &[
            ("x", values(&[2, 8, 16], 1)),
            ("w", values(&[2, 32, 16], 2)),
        ],
        1e-2,
        |inputs| Ok(vec![inputs[0].matmul(&inputs[1].transpose(1, 2)?)?]),
    );
    assert_eq!(library_steps(&program), ["cudnn"]);
}

#[test]
fn test_library_versions_match_the_pinned_headers() {
    let macro_value = |text: &str, name: &str| {
        text.lines()
            .find_map(|line| {
                let mut words = line.split_whitespace();
                (words.next() == Some("#define") && words.next() == Some(name))
                    .then(|| words.next().map(str::to_owned))
                    .flatten()
            })
            .unwrap_or_else(|| panic!("no `{name}` in the header"))
    };
    for (library, version) in LIBRARY_VERSIONS {
        let (root, header, major, minor) = match library {
            Library::Cublas | Library::CublasLt => (
                "CUDA_PATH",
                "include/cublas_api.h",
                "CUBLAS_VER_MAJOR",
                "CUBLAS_VER_MINOR",
            ),
            Library::Cudnn => (
                "CUDNN_INCLUDE_PATH",
                "cudnn_version.h",
                "CUDNN_MAJOR",
                "CUDNN_MINOR",
            ),
            Library::Cufft => (
                "CUFFT_INCLUDE_PATH",
                "cufft.h",
                "CUFFT_VER_MAJOR",
                "CUFFT_VER_MINOR",
            ),
            Library::Cutensor => (
                "CUTENSOR_INCLUDE_PATH",
                "cutensor.h",
                "CUTENSOR_MAJOR",
                "CUTENSOR_MINOR",
            ),
        };
        let Ok(root) = env::var(root) else {
            continue;
        };
        let text = fs::read_to_string(Path::new(&root).join(header)).unwrap();
        assert_eq!(
            version,
            format!(
                "{}.{}",
                macro_value(&text, major),
                macro_value(&text, minor)
            ),
            "{library}"
        );
    }
}

#[test]
fn test_templates_fill_each_placeholder_once() {
    assert_eq!(
        fill_template("a @x@ b @y@ @x@", &[("x", "@y@"), ("y", "2")]),
        "a @y@ b 2 @y@"
    );
}
