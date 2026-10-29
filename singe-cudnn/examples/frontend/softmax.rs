mod common;

use std::env::args;

use singe_cuda::{memory::DeviceMemory, view::OwnedDeviceRegion};
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        bindings::BindingAccess,
        graph::{DataTypePolicy, Graph, GraphConfig, SoftmaxOutputs, TensorOutputConfig},
        operation::{DiagonalBandMaskConfig, SoftmaxConfig, SoftmaxStatsBase},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let arguments = args().collect::<Vec<_>>();
    let with_stats = !arguments
        .iter()
        .any(|argument| argument == "--without-stats");
    let save_summaries = arguments
        .iter()
        .any(|argument| argument == "--save-summaries");
    let with_max = save_summaries || arguments.iter().any(|argument| argument == "--max");
    let with_sum_exp = save_summaries || arguments.iter().any(|argument| argument == "--sum-exp");
    let row_sink = arguments.iter().any(|argument| argument == "--row-sink");
    let with_sink = row_sink || arguments.iter().any(|argument| argument == "--sink");
    let explicit_layouts = arguments
        .iter()
        .any(|argument| argument == "--explicit-layouts");
    let summary_type = if arguments
        .iter()
        .any(|argument| argument == "--fp16-summaries")
    {
        DataType::F16
    } else {
        DataType::F32
    };
    let stats_base = if arguments.iter().any(|argument| argument == "--log2") {
        SoftmaxStatsBase::Log2
    } else {
        SoftmaxStatsBase::NaturalLog
    };
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let input = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([2, 2, 4, 16])?,
    ));
    let scores = if arguments.iter().any(|argument| argument == "--causal-mask") {
        let fill = graph.tensor(
            TensorSpec::constant(f16::from_f32(f32::NEG_INFINITY))
                .with_shape(Shape::contiguous([1, 1, 1, 1])?),
        );
        let masked = graph.diagonal_band_mask(
            input,
            fill,
            DiagonalBandMaskConfig::new(PointwiseMode::CmpGe).with_name("causal"),
            TensorOutputConfig::new().with_data_type(DataType::F16),
        )?;
        println!(
            "masked scores {masked}: name={:?}",
            graph.tensor_spec(masked)?.name
        );
        masked
    } else {
        input
    };
    let sink = if with_sink {
        let dimensions = if row_sink { [2, 2, 4, 1] } else { [1, 2, 1, 1] };
        Some(graph.tensor(TensorSpec::new(
            DataType::F32,
            Shape::contiguous(dimensions)?,
        )))
    } else {
        None
    };
    // Saved statistics stay FP32 even though caller-bound activations inherit FP16.
    let saved = TensorOutputConfig::new()
        .with_data_type(DataType::F32)
        .output_tensor();
    let summary = saved.with_data_type(summary_type);
    let config = SoftmaxConfig::new()
        .with_name("distribution")
        .with_stats_base(stats_base)
        .with_sink(sink);
    let outputs = if explicit_layouts {
        let output = graph.tensor_like(input, Some("probabilities"))?;
        let shape = Shape::with_strides([2, 2, 4, 1], [32, 16, 2, 1])?;
        let stats = with_stats.then(|| {
            graph.tensor(TensorSpec::new(DataType::F32, shape.clone()).with_name("stats"))
        });
        let max = with_max
            .then(|| graph.tensor(TensorSpec::new(summary_type, shape.clone()).with_name("max")));
        let sum_exp = with_sum_exp
            .then(|| graph.tensor(TensorSpec::new(summary_type, shape).with_name("sum_exp")));
        let outputs = SoftmaxOutputs {
            output,
            stats,
            max,
            sum_exp,
        };
        graph.softmax_into(scores, outputs, &config)?;
        outputs
    } else {
        graph.softmax(
            scores,
            SoftmaxOutputs {
                output: TensorOutputConfig::new().output_tensor(),
                stats: with_stats.then_some(saved),
                max: with_max.then_some(summary),
                sum_exp: with_sum_exp.then_some(summary),
            },
            &config,
        )?
    };
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let input_device = DeviceMemory::<f16>::zeroed(cuda, graph.shape(input)?.storage_span()?)?;
    let sink_device = sink
        .map(|id| -> Result<_> {
            Ok(DeviceMemory::<f32>::zeroed(
                cuda,
                graph.shape(id)?.storage_span()?,
            )?)
        })
        .transpose()?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(input, &input_device)?;
    if let (Some(id), Some(memory)) = (sink, sink_device.as_ref()) {
        bindings.set_input(id, memory)?;
    }
    let mut bound_outputs = 0;
    for id in [
        Some(outputs.output),
        outputs.stats,
        outputs.max,
        outputs.sum_exp,
    ]
    .into_iter()
    .flatten()
    {
        let tensor = graph.tensor_spec(id)?;
        println!("softmax result {id}: name={:?}", tensor.name);
        if tensor.is_virtual() {
            continue;
        }
        let memory = DeviceMemory::<u8>::zeroed(cuda, tensor.storage_byte_len(None)?)?;
        bindings.set_owned_region(
            id,
            OwnedDeviceRegion::from_memory(&memory),
            BindingAccess::ReadWrite,
        )?;
        bound_outputs += 1;
    }
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    // SAFETY: initialized inputs, retained writable outputs, plan and workspace remain
    // live until the stream completes, including when execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!(
        "softmax: {bound_outputs} caller-bound outputs, stats={}, base={stats_base:?}",
        outputs.stats.is_some(),
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_softmax", run())
}
