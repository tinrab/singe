mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f8e5m2, f8ue8m0, f16},
    error::Result,
    frontend::{
        graph::{BlockScaleQuantizeOutputs, DataTypePolicy, Graph, TensorOutputConfig},
        operation::{
            BlockScaleQuantizeConfig, LayerNormalizationAffine, LayerNormalizationConfig,
            LayerNormalizationInputs, LayerNormalizationOutputs, NormalizationSavedStatistics,
            NormalizationStatistics, ReshapeConfig, ReshapeMode,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let batch_size = 4_i64;
    let seq_length = 1024_i64;
    let hidden_size = 128_i64;
    let block_size = 32_i64;

    let mut graph = Graph::new();
    graph.set_data_type_policy(
        DataTypePolicy::new()
            .with_io(DataType::F16)
            .with_intermediate(DataType::F32)
            .with_compute(DataType::F32),
    );
    if args().any(|argument| argument == "--inherit-compute") {
        graph.set_data_type_policy(graph.data_type_policy().with_compute(None));
    }

    let x_layout = Shape::with_strides(
        [batch_size, seq_length, hidden_size, 1],
        [seq_length * hidden_size, hidden_size, 1, hidden_size],
    )?;
    let scale_layout = Shape::with_strides(
        [1, 1, hidden_size, 1],
        vec![hidden_size, hidden_size, 1, hidden_size],
    )?;
    let x = graph.tensor(TensorSpec::new(DataType::F16, x_layout.clone()));
    let scale = graph.tensor(TensorSpec::new(DataType::F32, scale_layout.clone()));
    let bias = graph.tensor(TensorSpec::new(DataType::F32, scale_layout));
    let epsilon = graph.tensor(TensorSpec::scalar_f32(1e-5));

    let y_ln = graph.tensor(TensorSpec::new(DataType::F16, x_layout).virtual_tensor());
    let mean = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides(
            [batch_size, seq_length, 1, 1],
            [seq_length, 1, seq_length, seq_length],
        )?,
    ));
    let inv_variance = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::with_strides(
            [batch_size, seq_length, 1, 1],
            [seq_length, 1, seq_length, seq_length],
        )?,
    ));
    graph.layer_normalization_into(
        LayerNormalizationInputs {
            input: x,
            affine: LayerNormalizationAffine::ScaleBias { scale, bias },
        },
        LayerNormalizationOutputs {
            output: y_ln,
            statistics: NormalizationStatistics::Training(NormalizationSavedStatistics {
                mean,
                inv_variance,
            }),
        },
        LayerNormalizationConfig::new(epsilon),
    )?;

    let y_ln_2d = graph.reshape(
        y_ln,
        [batch_size * seq_length, hidden_size],
        ReshapeConfig::new(ReshapeMode::ViewOnly).with_name("normalized_matrix"),
        TensorOutputConfig::new().with_data_type(DataType::F16),
    )?;

    let quantized_outputs = BlockScaleQuantizeOutputs::new(
        TensorOutputConfig::new()
            .with_data_type(DataType::F8E5M2)
            .output_tensor(),
        TensorOutputConfig::new()
            .with_data_type(DataType::F8UE8M0)
            .output_tensor(),
    );
    let row_config = BlockScaleQuantizeConfig::new(block_size)
        .with_name("row_quantize")
        .with_axis(1);
    let row = if args().any(|argument| argument == "--explicit-quantized-output") {
        let shapes = row_config.infer_shapes(graph.shape(y_ln_2d)?)?;
        let output = graph.tensor(
            TensorSpec::new(DataType::F8E5M2, shapes.output)
                .with_name("row_values")
                .output_tensor(),
        );
        let scale = graph.tensor(
            TensorSpec::new(DataType::F8UE8M0, shapes.scale)
                .with_name("row_scales")
                .output_tensor(),
        );
        let outputs = BlockScaleQuantizeOutputs::new(output, scale);
        graph.block_scale_quantize_into(y_ln_2d, outputs, row_config)?;
        outputs
    } else {
        graph.block_scale_quantize(y_ln_2d, quantized_outputs, row_config)?
    };
    let col = graph.block_scale_quantize(
        y_ln_2d,
        quantized_outputs,
        BlockScaleQuantizeConfig::new(block_size)
            .with_name("column_quantize")
            .with_axis(0),
    )?;

    let y_row = row.output;
    let mx_row = row.scale;
    let y_col = col.output;
    let mx_col = col.scale;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let x_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(x)?.storage_span()?,
    )?;
    let scale_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(scale)?.storage_span()?,
    )?;
    let bias_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(bias)?.storage_span()?,
    )?;
    let mut mean_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(mean)?.storage_span()?,
    )?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(inv_variance)?.storage_span()?,
    )?;
    let mut y_row_dev = DeviceMemory::<f8e5m2>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(y_row)?.storage_span()?,
    )?;
    let mut mx_row_dev = DeviceMemory::<f8ue8m0>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(mx_row)?.storage_span()?,
    )?;
    let mut y_col_dev = DeviceMemory::<f8e5m2>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(y_col)?.storage_span()?,
    )?;
    let mut mx_col_dev = DeviceMemory::<f8ue8m0>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(mx_col)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set_input(bias, &bias_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(y_row, &mut y_row_dev)?;
    bindings.set(mx_row, &mut mx_row_dev)?;
    bindings.set(y_col, &mut y_col_dev)?;
    bindings.set(mx_col, &mut mx_col_dev)?;
    // SAFETY: the plan, input buffers, outputs and workspace remain live through stream
    // completion, including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_norm_block_scale", run())
}
