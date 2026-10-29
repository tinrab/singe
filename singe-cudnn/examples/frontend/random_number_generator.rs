mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            PointwiseOperation, RandomNumberDistribution, RandomNumberGeneratorConfig,
            RandomNumberSeedSource,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let mut config = RandomNumberGeneratorConfig::bernoulli(0.5, 42).with_name("rng");
    let mut device_seed = false;
    for argument in args().skip(1) {
        match argument.as_str() {
            "--device-seed" => device_seed = true,
            "--distribution=bernoulli" => {
                config = config
                    .with_distribution(RandomNumberDistribution::Bernoulli { probability: 0.5 });
            }
            "--distribution=uniform" => {
                config = config.with_distribution(RandomNumberDistribution::Uniform {
                    minimum: -1.0,
                    maximum: 1.0,
                });
            }
            "--distribution=normal" => {
                config = config.with_distribution(RandomNumberDistribution::Normal {
                    mean: 0.0,
                    standard_deviation: 1.0,
                });
            }
            _ => {
                return Err(FrontendError::Compile(format!(
                    "unknown rng example argument: {argument}"
                ))
                .into());
            }
        }
    }

    let ctx = common::ExampleContext::create()?;
    let cuda = ctx.cudnn.cuda_context();
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F32)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let shape = Shape::contiguous([1, 1, 32, 32])?;
    let scalar_shape = Shape::contiguous([1; 4])?;
    let input = graph.tensor(TensorSpec::new(DataType::F32, shape.clone()).with_name("input"));
    let offset =
        graph.tensor(TensorSpec::new(DataType::I64, scalar_shape.clone()).with_name("offset"));
    let seed_tensor = device_seed
        .then(|| graph.tensor(TensorSpec::new(DataType::I64, scalar_shape).with_name("seed")));
    config = config.with_offset(offset);
    if let Some(tensor) = seed_tensor {
        config = config.with_seed_source(RandomNumberSeedSource::Device { tensor, offset });
    }
    let distribution = config.distribution();
    let random =
        graph.random_number_generator(shape, config, TensorOutputConfig::new().virtual_tensor())?;
    let output = graph.pointwise(PointwiseOperation::binary(
        PointwiseMode::Mul,
        input,
        random,
        TensorOutputConfig::new().output_tensor(),
    ))?;
    let compiled = graph.compile(
        Some(&ctx.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;

    let mut image_seed = 123_456_789;
    let values = common::init_f32_image(graph.shape(input)?.storage_span()?, &mut image_seed);
    let input_device = DeviceMemory::<f32>::from_slice(cuda, &values)?;
    let offset_device = DeviceMemory::<i64>::from_slice(cuda, &[128])?;
    let seed_device = seed_tensor
        .map(|_| DeviceMemory::<i64>::from_slice(cuda, &[42]))
        .transpose()?;
    let mut output_device =
        DeviceMemory::<f32>::zeroed(cuda, graph.shape(output)?.storage_span()?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_device)?;
    bindings.set_input(offset, &offset_device)?;
    if let Some((tensor, memory)) = seed_tensor.zip(seed_device.as_ref()) {
        bindings.set_input(tensor, memory)?;
    }
    bindings.set(output, &mut output_device)?;
    // SAFETY: inputs, RNG state, writable output and workspace remain live through stream completion.
    ctx.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed {distribution:?} with device seed: {device_seed}");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_random_number_generator", run())
}
