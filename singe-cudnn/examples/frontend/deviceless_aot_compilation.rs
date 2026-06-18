mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::device::DeviceProperties,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::{DataType, f16},
    error::{Error, Result, Status},
    frontend::{
        graph::Graph,
        operation::{ConvolutionConfig, HeuristicMode},
        plan::CompiledGraph,
    },
};

const X_ID: TensorId = TensorId::new(1000);
const W_ID: TensorId = TensorId::new(1001);
const Y_ID: TensorId = TensorId::new(1002);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn count_nonfinite_f16(values: &[f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
}

fn verify_output(y_host: &[f16]) {
    let first = y_host[0].to_f32();
    let last = y_host[y_host.len() - 1].to_f32();
    let checksum = y_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let nonfinite = count_nonfinite_f16(y_host);

    println!(
        "y: first={first}, last={last}, checksum={checksum}, nonfinite={nonfinite}, len={}",
        y_host.len()
    );

    assert_eq!(nonfinite, 0, "y.nonfinite mismatch");
    assert_close!(first, 126.0, 1.0e-5, "y.first");
    assert_close!(last, 129.0, 1.0e-5, "y.last");
    assert_close!(checksum, 4_452_151_300.0, 10.0, "y.checksum");
}

fn is_deviceless_not_supported(error: &Error) -> bool {
    matches!(
        error,
        Error::Cudnn { code, .. }
            if *code == Status::NotSupported
                || *code == Status::NotSupportedArchitectureMismatch
    )
}

fn build_graph(device_properties: DeviceProperties) -> Result<Graph> {
    let n = 16_i64;
    let c = 128_i64;
    let h = 64_i64;
    let width = 64_i64;
    let k = 256_i64;
    let r = 3_i64;
    let s = 3_i64;

    let mut graph = Graph::new()
        .with_name("deviceless-aot-demo")
        .with_io_data_type(DataType::F16)
        .with_compute_data_type(DataType::F32);
    graph.set_device_properties(device_properties)?;

    let x = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, c, h, width])?.with_strides(vec![
                c * h * width,
                1,
                c * width,
                c,
            ])?,
        )
        .with_id(X_ID)
        .with_name("image"),
    );

    let w = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([k, c, r, s])?.with_strides([c * r * s, 1, c * s, c])?,
        )
        .with_id(W_ID)
        .with_name("filter"),
    );

    let y = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([n, k, h, width])?.with_strides(vec![
                k * h * width,
                1,
                k * width,
                k,
            ])?,
        )
        .with_id(Y_ID),
    );
    graph.convolution_forward(
        x,
        w,
        y,
        ConvolutionConfig::new(DataType::F32, 2)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    Ok(graph)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let device_properties = DeviceProperties::create(&ctx.cudnn)?;
    let serialized_device = device_properties.serialize()?;
    let restored_device = DeviceProperties::deserialize(&serialized_device)?;

    let graph = build_graph(restored_device)?;
    let compiled = match graph.compile_deviceless(&[HeuristicMode::A, HeuristicMode::Fallback]) {
        Ok(compiled) => compiled,
        Err(error) if is_deviceless_not_supported(&error) => {
            println!(
                "frontend_deviceless_aot_compilation: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let serialized_compiled = compiled.serialize()?;
    let restored = CompiledGraph::deserialize(&ctx.cudnn, &serialized_compiled)?;

    assert_eq!(restored.graph().name(), Some("deviceless-aot-demo"));
    assert_eq!(restored.graph().name(), Some("deviceless-aot-demo"));
    assert_eq!(restored.names()?, compiled.names()?);

    let mut seed = 123_456_789_u32;
    let mut x_dev =
        DeviceMemory::from_slice(&init_f16_image((16 * 128 * 64 * 64) as usize, &mut seed))?;
    let mut w_dev =
        DeviceMemory::from_slice(&init_f16_image((256 * 128 * 3 * 3) as usize, &mut seed))?;
    let mut y_dev = DeviceMemory::<f16>::zeroes((16 * 256 * 64 * 64) as usize)?;
    let mut workspace = common::workspace(restored.workspace_size()?)?;

    let mut bindings = restored.bindings();
    bindings
        .set(X_ID, &mut x_dev)?
        .set(W_ID, &mut w_dev)?
        .set(Y_ID, &mut y_dev)?;
    restored.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let y_host = y_dev.copy_to_host_vec()?;
    verify_output(&y_host);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_deviceless_aot_compilation", run())
}
