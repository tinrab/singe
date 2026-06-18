mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::Graph,
        operation::{
            BatchNormalizationConfig, BatchNormalizationRunningStats, CompileConfig, HeuristicMode,
        },
    },
    scalar::ScalarValue,
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn scalar_tensor(name: &str, value: f32) -> Result<TensorSpec> {
    Ok(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    )
    .with_scalar_value(ScalarValue::F32(value))?
    .with_name(name))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if singe_cudnn::version()? < 90_300 {
        println!("frontend_sm_carveout: skipped, requires cuDNN 9.3.0+");
        return Ok(());
    }
    let properties = ctx.cudnn.cuda_context().device().properties()?;
    if properties.major < 8 {
        println!("frontend_sm_carveout: skipped, requires Ampere or newer");
        return Ok(());
    }

    let n = 8;
    let c = 32;
    let h = 16;
    let w = 16;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32)
        .with_sm_count_target(8);

    let x = graph.tensor(named_tensor(
        "X",
        DataType::F16,
        Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
    ));
    let prev_running_mean = graph.tensor(named_tensor(
        "prev_running_mean",
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let prev_running_var = graph.tensor(named_tensor(
        "prev_running_var",
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let scale = graph.tensor(named_tensor(
        "scale",
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let bias = graph.tensor(named_tensor(
        "bias",
        DataType::F32,
        Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
    ));
    let peer_stats_0 = graph.tensor(named_tensor(
        "peer_stats_0",
        DataType::F32,
        Shape::contiguous([2, 4 * c, 1, 1])?.with_strides([4 * c, 1, 4 * c, 4 * c])?,
    ));
    let peer_stats_1 = graph.tensor(named_tensor(
        "peer_stats_1",
        DataType::F32,
        Shape::contiguous([2, 4 * c, 1, 1])?.with_strides([4 * c, 1, 4 * c, 4 * c])?,
    ));
    let epsilon = graph.tensor(scalar_tensor("epsilon", 1e-5)?);
    let momentum = graph.tensor(scalar_tensor("momentum", 1e-1)?);

    let y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, w])?.with_strides([c * h * w, 1, c * w, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let inv_variance = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let next_running_mean = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );
    let next_running_var = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([1, c, 1, 1])?.with_strides([c, 1, c, c])?,
        )
        .virtual_tensor()
        .output_tensor(),
    );

    graph.batch_normalization(
        x,
        scale,
        bias,
        y,
        mean,
        inv_variance,
        BatchNormalizationConfig::new(epsilon)
            .with_peer_stats(vec![peer_stats_0, peer_stats_1])
            .with_running_stats(BatchNormalizationRunningStats::new(
                momentum,
                prev_running_mean,
                prev_running_var,
                next_running_mean,
                next_running_var,
            )),
    )?;

    let compiled = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_sm_count_target(8),
    )?;

    let mut x_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut prev_running_mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut prev_running_var_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut scale_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut bias_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut y_dev = DeviceMemory::<f16>::zeroes((n * c * h * w) as usize)?;
    let mut mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut inv_variance_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut next_running_mean_dev = DeviceMemory::<f32>::zeroes(c as usize)?;
    let mut next_running_var_dev = DeviceMemory::<f32>::zeroes(c as usize)?;

    let mut peer_stats_0_host = vec![0.0_f32; (2 * 4 * c) as usize];
    for chunk in peer_stats_0_host.chunks_exact_mut(2) {
        chunk[1] = f32::from_bits(1);
    }
    let mut peer_stats_0_dev = DeviceMemory::from_slice(&peer_stats_0_host)?;
    let mut peer_stats_1_dev = DeviceMemory::<f32>::zeroes((2 * 4 * c) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(x, &mut x_dev)?;
    bindings.set(prev_running_mean, &mut prev_running_mean_dev)?;
    bindings.set(prev_running_var, &mut prev_running_var_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(peer_stats_0, &mut peer_stats_0_dev)?;
    bindings.set(peer_stats_1, &mut peer_stats_1_dev)?;
    bindings.set(y, &mut y_dev)?;
    bindings.set(mean, &mut mean_dev)?;
    bindings.set(inv_variance, &mut inv_variance_dev)?;
    bindings.set(next_running_mean, &mut next_running_mean_dev)?;
    bindings.set(next_running_var, &mut next_running_var_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_sm_carveout", run())
}
