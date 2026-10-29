mod common;

use singe_cuda::{cuda_module, memory::DeviceMemory, module::LaunchConfig};
use singe_cudnn::{
    backend::{
        behavior::BackendBehaviorNote,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::MatmulConfig,
        plan::{CompileConfig, EngineConfigFilter, ExecuteArgs, HeuristicMode},
    },
};

cuda_module! {
    pub mod postprocess_kernel {
        source: r#"
        extern "C" __global__ void affine(
            float* values,
            int len,
            float scale,
            float bias
        ) {
            int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
            if (index < len) {
                values[index] = values[index] * scale + bias;
            }
        }
        "#,
        compile: {
            nvrtc_args: ["--std=c++20"],
        },
    }
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;

    let batch = 1_i64;
    let rows = 8_i64;
    let cols = 16_i64;
    let depth = 4_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("custom-kernel-matmul")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F32)
                    .with_intermediate(DataType::F32)
                    .with_compute(DataType::F32),
            ),
    );

    let a = graph.insert_tensor(
        TensorId::new(1000),
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, rows, depth])?),
    )?;
    let b = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, depth, cols])?),
    )?;
    let c = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, rows, cols])?),
    )?;
    graph.matmul_into(
        a,
        b,
        c,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;

    let compiled = graph
        .plan_candidates(
            Some(&context.cudnn),
            &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
        )?
        .filter(
            &EngineConfigFilter::new()
                .with_included_behavior_notes(&[BackendBehaviorNote::SupportsCudaGraphNativeApi]),
        )
        .compile(Some(&context.cudnn), &CompileConfig::new())?;

    let element_count = (batch * rows * cols) as usize;
    let mut a_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![1.0_f32; (batch * rows * depth) as usize],
    )?;
    let mut b_dev = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![1.0_f32; (batch * depth * cols) as usize],
    )?;
    let mut c_dev = DeviceMemory::<f32>::zeroed(context.cudnn.cuda_context(), element_count)?;
    let mut workspace = common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set(a, &mut a_dev)?;
    bindings.set(b, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;

    let mut main_graph = context.cudnn.cuda_context().create_graph()?;
    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    let cudnn_node = match unsafe {
        compiled.append_to_cuda_graph(
            &context.cudnn,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
            &mut main_graph,
            &[],
        )
    } {
        Ok(node) => node,
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let module = postprocess_kernel::Module::create(context.cudnn.cuda_context())?;
    let scale = 0.5_f32;
    let bias = 1.25_f32;
    let launch = LaunchConfig::for_1d_grid(element_count, 128);
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    let kernel_node = unsafe {
        module.affine_node(
            &mut main_graph,
            &[cudnn_node.clone()],
            &launch,
            &mut c_dev,
            element_count as i32,
            scale,
            bias,
        )?
    };

    let mut executable = main_graph.instantiate()?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&context.stream)?;
    }
    context.stream.synchronize()?;

    let output = c_dev.copy_to_host_vec()?;
    let expected = depth as f32 * scale + bias;
    assert!(
        output
            .iter()
            .all(|value| (*value - expected).abs() <= 1.0e-6)
    );

    let scale_new = 0.25_f32;
    let bias_new = 2.0_f32;
    let mut a_dev_new = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![2.0_f32; (batch * rows * depth) as usize],
    )?;
    let mut b_dev_new = DeviceMemory::from_slice(
        context.cudnn.cuda_context(),
        &vec![1.5_f32; (batch * depth * cols) as usize],
    )?;
    let mut c_dev_new = DeviceMemory::<f32>::zeroed(context.cudnn.cuda_context(), element_count)?;
    let mut workspace_new =
        common::workspace(context.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings_new = compiled.bindings()?;
    bindings_new.set(a, &mut a_dev_new)?;
    bindings_new.set(b, &mut b_dev_new)?;
    bindings_new.set(c, &mut c_dev_new)?;

    // SAFETY: the example retains its plan and buffers through the surrounding completion or replay sequence.
    unsafe {
        compiled.update_cuda_graph_node(
            &context.cudnn,
            ExecuteArgs::new(&bindings_new).with_workspace(&mut workspace_new),
            &cudnn_node,
            Some(&mut executable),
        )
    }?;
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        module.affine_set_node_params(
            &mut executable,
            &kernel_node,
            launch.geometry(),
            &mut c_dev_new,
            element_count as i32,
            scale_new,
            bias_new,
        )?;
    }
    // SAFETY: the example keeps these buffers alive and synchronizes before reading the results.
    unsafe {
        executable.launch(&context.stream)?;
    }
    context.stream.synchronize()?;

    let output_new = c_dev_new.copy_to_host_vec()?;
    let expected_new = (depth as f32 * 2.0_f32 * 1.5_f32) * scale_new + bias_new;
    assert!(
        output_new
            .iter()
            .all(|value| (*value - expected_new).abs() <= 1.0e-6)
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_custom_kernel_graph", run())
}
