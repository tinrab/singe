mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{
            RmsNormalizationAffineGradients, RmsNormalizationBackwardConfig,
            RmsNormalizationBackwardInputs, RmsNormalizationBackwardOutputs,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn init_f32_image(size: usize, seed: &mut u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        *seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        values.push((*seed as f32) * 2.328_306_4e-10_f32);
    }
    values
}

fn count_nonfinite_f32(values: &[f32]) -> usize {
    values.iter().filter(|value| !value.is_finite()).count()
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let batch_size = 4;
    let seq_length = 1024;
    let hidden_size = 128;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides(
                [batch_size * seq_length, hidden_size, 1, 1],
                [hidden_size, 1, hidden_size, hidden_size],
            )?,
        )
        .with_name("X"),
    );
    let dy = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides(
                [batch_size * seq_length, hidden_size, 1, 1],
                [hidden_size, 1, hidden_size, hidden_size],
            )?,
        )
        .with_name("DY"),
    );
    let scale = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides(
                [1, hidden_size, 1, 1],
                vec![hidden_size, 1, hidden_size, hidden_size],
            )?,
        )
        .with_name("scale"),
    );
    let inv_variance = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::with_strides([batch_size * seq_length, 1, 1, 1], [1, 1, 1, 1])?,
        )
        .with_name("inv_variance"),
    );
    let outputs = graph.rms_normalization_backward(
        RmsNormalizationBackwardInputs {
            input: x,
            inv_variance,
            output_gradient: dy,
            scale: Some(scale),
        },
        RmsNormalizationBackwardOutputs {
            input_gradient: TensorOutputConfig::new().output_tensor(),
            affine: Some(RmsNormalizationAffineGradients {
                scale_gradient: TensorOutputConfig::new().output_tensor(),
                bias_gradient: None,
            }),
        },
        RmsNormalizationBackwardConfig::new().with_name("rms_norm_backward"),
    )?;
    let dx = outputs.input_gradient;
    let dscale = outputs
        .affine
        .ok_or(FrontendError::NormalizationAffineGradients {
            operation: "rms norm backward",
        })?
        .scale_gradient;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;

    let mut float_seed = 123_456_789_u32;
    let x_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(x)?.storage_span()?, &mut float_seed),
    )?;
    let dy_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(dy)?.storage_span()?, &mut float_seed),
    )?;
    let _mean_init = init_f32_image((batch_size * seq_length) as usize, &mut float_seed);
    let inv_variance_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(inv_variance)?.storage_span()?, &mut float_seed),
    )?;
    let scale_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(scale)?.storage_span()?, &mut float_seed),
    )?;
    let mut dscale_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(dscale)?.storage_span()?, &mut float_seed),
    )?;
    let _bias_gradient_init = init_f32_image(hidden_size as usize, &mut float_seed);
    let mut dx_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f32_image(graph.shape(dx)?.storage_span()?, &mut float_seed),
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(x, &x_dev)?;
    bindings.set_input(dy, &dy_dev)?;
    bindings.set_input(inv_variance, &inv_variance_dev)?;
    bindings.set_input(scale, &scale_dev)?;
    bindings.set(dscale, &mut dscale_dev)?;
    bindings.set(dx, &mut dx_dev)?;
    // SAFETY: matching bindings, buffers and workspace remain live until the helper
    // completes the cuDNN stream, including when native execution returns an error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    let dx_host = dx_dev.copy_to_host_vec()?;
    let dscale_host = dscale_dev.copy_to_host_vec()?;

    let dx_first = dx_host[0];
    let dx_last = dx_host[dx_host.len() - 1];
    let dx_checksum = dx_host.iter().copied().sum::<f32>();
    let dx_nonfinite = count_nonfinite_f32(&dx_host);
    let dscale_first = dscale_host[0];
    let dscale_last = dscale_host[dscale_host.len() - 1];
    let dscale_checksum = dscale_host.iter().copied().sum::<f32>();
    let dscale_nonfinite = count_nonfinite_f32(&dscale_host);

    println!(
        "dx: first={dx_first}, last={dx_last}, checksum={dx_checksum}, nonfinite={dx_nonfinite}, len={}",
        dx_host.len()
    );
    println!(
        "dscale: first={dscale_first}, last={dscale_last}, checksum={dscale_checksum}, nonfinite={dscale_nonfinite}, len={}",
        dscale_host.len()
    );

    assert_eq!(dx_nonfinite, 0, "dx.nonfinite mismatch");
    assert_eq!(dscale_nonfinite, 0, "dscale.nonfinite mismatch");
    assert_close!(dx_first, 0.251971, 1.0e-5, "dx.first");
    assert_close!(dx_last, 0.277723, 1.0e-5, "dx.last");
    assert_close!(dx_checksum, 55946.1, 1.0, "dx.checksum");
    assert_close!(dscale_first, 518.905, 1.0e-3, "dscale.first");
    assert_close!(dscale_last, 510.452, 1.0e-3, "dscale.last");
    assert_close!(dscale_checksum, 65435.9, 1.0, "dscale.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_rms_norm_backward", run())
}
