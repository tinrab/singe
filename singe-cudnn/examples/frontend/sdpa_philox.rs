mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result},
    frontend::{
        composite::sdpa::{SdpaForwardOutputTensors, SdpaInputs},
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            AttentionConfig, AttentionDropoutConfig, AttentionDropoutMode, AttentionImplementation,
            AttentionScoreConfig, RandomNumberSeedSource,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    scalar::ScalarValue,
};

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let shape = Shape::contiguous([2, 4, 64, 64])?;
    let input_elements = shape.element_count()?;
    let query = graph.tensor(TensorSpec::new(DataType::BF16, shape).with_name("Q"));
    let key = graph.tensor_like(query, Some("K"))?;
    let value = graph.tensor_like(query, Some("V"))?;
    let scale = graph.tensor(TensorSpec::scalar_f32(0.125));
    let seed_value = 1234_i64;
    let int32_seed = args().any(|argument| argument == "--int32-seed");
    let by_value_seed = args().any(|argument| argument == "--by-value-seed");
    let tensor_seed =
        int32_seed || by_value_seed || args().any(|argument| argument == "--device-seed");
    let (seed_scalar, seed_bytes) = if int32_seed {
        let seed = i32::try_from(seed_value).map_err(|_| Error::OutOfRange {
            name: "philox seed".into(),
        })?;
        (ScalarValue::I32(seed), seed.to_ne_bytes().to_vec())
    } else {
        (
            ScalarValue::I64(seed_value),
            seed_value.to_ne_bytes().to_vec(),
        )
    };
    let offset_scalar = if int32_seed {
        ScalarValue::I32(0)
    } else {
        ScalarValue::I64(0)
    };
    let tensor_rng = tensor_seed
        .then(|| -> Result<_> {
            let (seed_spec, offset_spec) = if by_value_seed {
                // Preparation expands the default seed and resets the same-rank offset's strides.
                (
                    TensorSpec::scalar(seed_scalar),
                    TensorSpec::scalar(offset_scalar)
                        .with_shape(Shape::with_strides([1; 4], [8, 4, 2, 1])?),
                )
            } else {
                let shape = Shape::contiguous([1; 4])?;
                (
                    TensorSpec::new(seed_scalar.data_type(), shape.clone()),
                    TensorSpec::new(offset_scalar.data_type(), shape),
                )
            };
            let seed = graph.tensor(seed_spec.with_name("Seed"));
            let offset = graph.tensor(offset_spec.with_name("Offset"));
            Ok((seed, offset))
        })
        .transpose()?;
    let seed = match tensor_rng {
        Some((tensor, offset)) => RandomNumberSeedSource::Device { tensor, offset },
        None => RandomNumberSeedSource::Host {
            value: seed_value,
            offset: None,
        },
    };
    let implementation = if args().any(|argument| argument == "--composite") {
        AttentionImplementation::Composite
    } else {
        AttentionImplementation::Auto
    };
    let config = AttentionConfig::new()
        .with_implementation(implementation)
        .with_score_config(AttentionScoreConfig::new().with_dropout_mode(
            AttentionDropoutMode::Philox(AttentionDropoutConfig::new(0.1, seed)),
        ));
    let outputs = graph.sdpa_auto(
        SdpaInputs {
            query,
            key,
            value,
            scale,
        },
        config,
        SdpaForwardOutputTensors::new(TensorOutputConfig::new().output_tensor()),
    )?;
    let output = outputs.output;
    graph.mark_as_output(output)?;
    let output_elements = graph.shape(output)?.storage_span()?;
    let compiled = graph.compile(
        Some(&ctx.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let cuda = ctx.cudnn.cuda_context();
    // Q, K and V can share initialized read-only storage in this example.
    let input_device = DeviceMemory::from_slice(cuda, &vec![bf16::from_f32(0.25); input_elements])?;
    let rng_device = tensor_rng
        .filter(|_| !by_value_seed)
        .map(|(seed, offset)| -> Result<_> {
            Ok((
                seed,
                offset,
                DeviceMemory::from_slice(cuda, &seed_bytes)?,
                DeviceMemory::from_slice(cuda, &vec![0_u8; seed_bytes.len()])?,
            ))
        })
        .transpose()?;
    let mut output_device = DeviceMemory::<bf16>::zeroed(cuda, output_elements)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    for input in [query, key, value] {
        bindings.set_input(input, &input_device.view())?;
    }
    if let Some((seed, offset, seed_memory, offset_memory)) = &rng_device {
        bindings.set_encoded_input_bytes(*seed, &seed_memory.view())?;
        bindings.set_encoded_input_bytes(*offset, &offset_memory.view())?;
    }
    if by_value_seed && let Some((seed, offset)) = tensor_rng {
        bindings.set_scalar(seed, seed_scalar)?;
        bindings.set_scalar(offset, offset_scalar)?;
    }
    bindings.set(output, &mut output_device)?;
    // SAFETY: all inputs are initialized; output and workspace are writable and disjoint from inputs.
    // Every allocation and host scalar remains live through stream completion.
    ctx.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("Philox attention with seed {seed:?} and implementation {implementation:?}");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_sdpa_philox", run())
}
