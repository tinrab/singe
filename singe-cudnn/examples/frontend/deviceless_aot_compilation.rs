mod common;

use singe_core::assert_close;
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        device::{DeviceProperties, DevicePropertiesSource},
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::{Error, Result, Status},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::ConvolutionConfig,
        plan::{
            CompileConfig, CompiledGraph, ExecuteArgs, HeuristicMode, NativePlanRestoreTarget,
            NativePlanSerializationConfig, NativePlanStructure,
        },
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

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("deviceless-aot-demo")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_compute(DataType::F32),
            ),
    );
    graph.attach_device_properties(device_properties)?;

    let x = graph.insert_tensor(
        X_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([n, c, h, width], vec![c * h * width, 1, c * width, c])?,
        )
        .with_name("image"),
    )?;

    let w = graph.insert_tensor(
        W_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([k, c, r, s], [c * r * s, 1, c * s, c])?,
        )
        .with_name("filter"),
    )?;

    let y = graph.insert_tensor(
        Y_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([n, k, h, width], vec![k * h * width, 1, k * width, k])?,
        ),
    )?;
    graph.convolution_forward_into(
        x,
        w,
        y,
        ConvolutionConfig::new(2)
            .with_compute_type(DataType::F32)
            .with_padding(vec![1, 1])
            .with_strides([1, 1])
            .with_dilations(vec![1, 1]),
    )?;

    Ok(graph)
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let device_properties = DeviceProperties::create(DevicePropertiesSource::Device(
        context.cudnn.cuda_context().device(),
    ))?;
    let serialized_device = device_properties.serialize()?;
    let restored_device = DeviceProperties::deserialize(&serialized_device)?;

    let graph = build_graph(restored_device)?;
    let build = || {
        let candidates = graph.plan_candidates(
            None,
            &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
        )?;
        let mut built = candidates.into_build_candidates(&CompileConfig::new())?;
        for plan in built.plan_ids() {
            let report = built.build_plan(None, &plan)?;
            println!("deviceless candidate: {report}");
            if report.is_available() {
                return built.compile_selected(&plan);
            }
        }
        built.compile()
    };
    let compiled = match build() {
        Ok(compiled) => compiled,
        Err(error) if is_deviceless_not_supported(&error) => {
            println!(
                "frontend_deviceless_aot_compilation: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let serialized_compiled = compiled.serialize(NativePlanSerializationConfig {
        structure: NativePlanStructure::Omit,
        ..Default::default()
    })?;
    let restore_properties = DeviceProperties::deserialize(&serialized_device)?;
    // SAFETY: the artifact is unmodified, and execution below uses the same device that supplied these properties.
    let restored = unsafe {
        CompiledGraph::deserialize(
            NativePlanRestoreTarget::DeviceProperties(&restore_properties),
            &serialized_compiled,
        )
    }?;

    println!("restored graph structure: {:?}", restored.graph());
    for binding in restored.required_tensor_bindings() {
        for operand in &binding.bindings {
            let tensor = restored.tensor_info(operand.tensor_id)?;
            println!(
                "restored tensor {}: uid={:?}, name={:?}, dimensions={:?}, access={:?}",
                tensor.id,
                tensor.backend_uid,
                tensor.spec.name,
                tensor.spec.shape.dimensions(),
                tensor.access,
            );
        }
    }

    let mut seed = 123_456_789_u32;
    let mut x_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image((16 * 128 * 64 * 64) as usize, &mut seed),
    )?;
    let mut w_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &init_f16_image((256 * 128 * 3 * 3) as usize, &mut seed),
    )?;
    let mut y_dev =
        DeviceMemory::<f16>::zeroed(context.cudnn.cuda_context(), (16 * 256 * 64 * 64) as usize)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), restored.workspace_size())?;

    let mut bindings = restored.bindings()?;
    bindings
        .set(X_ID, &mut x_dev)?
        .set(W_ID, &mut w_dev)?
        .set(Y_ID, &mut y_dev)?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    unsafe {
        restored.execute(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    }?;

    let y_host = y_dev.copy_to_host_vec()?;
    verify_output(&y_host);

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_deviceless_aot_compilation", run())
}
