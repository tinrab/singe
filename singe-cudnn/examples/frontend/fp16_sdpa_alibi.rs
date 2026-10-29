mod common;

use std::{
    env::args,
    sync::{Arc, Mutex},
};

use singe_cuda::{memory::DeviceMemory, submission::Resources, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        bindings::{BindingAccess, RuntimeOverrides},
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs, alibi_slopes},
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDiagonalAlignment, AttentionDiagonalBand, AttentionMaskMode,
            AttentionScoreConfig,
        },
        plan::{CompileConfig, HeuristicMode},
    },
    operation::FrontendExecution,
    scalar::ScalarValue,
};

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let arguments = args().collect::<Vec<_>>();
    let with_alibi = !arguments
        .iter()
        .any(|argument| argument == "--without-alibi");
    let unfused_fma = arguments.iter().any(|argument| argument == "--unfused-fma");
    let query_heads = 6;
    let key_value_heads = 2;
    let sequence_length = 128;
    let head_dimension = 64;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let query_spec = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, query_heads, sequence_length, head_dimension])?,
    );
    let key_value_spec = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, key_value_heads, sequence_length, head_dimension])?,
    );
    let query = graph.tensor(query_spec.clone());
    let key = graph.tensor(key_value_spec.clone());
    let value = graph.tensor(key_value_spec.clone());
    let scale = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?)
            .with_constant_value(ScalarValue::F32(1.0 / (head_dimension as f32).sqrt()))?,
    );
    let output = graph
        .sdpa_auto(
            SdpaInputs {
                query,
                key,
                value,
                scale,
            },
            AttentionConfig::new()
                .with_unfuse_fma(unfused_fma)
                .with_score_config(
                    AttentionScoreConfig::new()
                        .with_alibi(with_alibi)
                        .with_mask_mode(AttentionMaskMode::DiagonalBand(
                            AttentionDiagonalBand::causal(AttentionDiagonalAlignment::TopLeft),
                        )),
                ),
            SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor()),
        )?
        .output;
    let compiled = Arc::new(graph.compile(
        Some(&context.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?);
    let cuda = Arc::clone(context.cudnn.cuda_context());
    let query_device = DeviceMemory::from_slice(
        &cuda,
        &vec![bf16::from_f32(0.5); query_spec.logical_element_count()?],
    )?;
    let key_device = DeviceMemory::from_slice(
        &cuda,
        &vec![bf16::from_f32(0.25); key_value_spec.logical_element_count()?],
    )?;
    let values = (0..key_value_spec.logical_element_count()?)
        .map(|index| bf16::from_f32((index % 17) as f32 / 17.0))
        .collect::<Vec<_>>();
    let value_device = DeviceMemory::from_slice(&cuda, &values)?;
    if with_alibi {
        // The standard C++ schedule includes the two extra heads after its four-head prefix.
        println!("standard alibi slopes: {:?}", alibi_slopes(query_heads)?);
    }
    let output_device =
        DeviceMemory::<bf16>::zeroed(&cuda, graph.tensor_spec(output)?.logical_element_count()?)?;
    let workspace = common::workspace(&cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    for (id, memory, access) in [
        (query, &query_device, BindingAccess::ReadOnly),
        (key, &key_device, BindingAccess::ReadOnly),
        (value, &value_device, BindingAccess::ReadOnly),
        (output, &output_device, BindingAccess::ReadWrite),
    ] {
        bindings.set_owned_region(id, OwnedDeviceRegion::from_memory(memory), access)?;
    }
    drop((query_device, key_device, value_device));
    // SAFETY: every binding has matching initialized storage and permitted aliases.
    // Retained bindings supply allocation owners and native access declarations automatically.
    // When ALiBi is enabled, the frontend retains its slope tensor and initialization copy.
    let operation = unsafe {
        FrontendExecution::create(
            Arc::new(Mutex::new(context.cudnn)),
            Arc::clone(&compiled),
            bindings,
            workspace,
            &RuntimeOverrides::new(),
            Resources::new(),
            output_device,
        )
    }?;
    drop(compiled);
    let mut captured = context.stream.capture_owned(operation)?;
    let executable = captured.instantiate()?;
    executable.launch(&context.stream)?.wait()?;
    let output = captured
        .take_output()
        .ok_or(FrontendError::InternalStateMissing {
            name: "captured alibi output",
        })?;
    let output = output.copy_to_host_vec()?;
    println!(
        "alibi output: {:?}",
        output
            .iter()
            .take(4)
            .map(|value| value.to_f32())
            .collect::<Vec<_>>()
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_alibi", run())
}
