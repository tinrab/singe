mod common;

use singe_cuda::{
    cuda_module, graph::Graph as CudaGraph, memory::DeviceMemory, module::LaunchConfig,
};
use singe_cudnn::{
    backend::behavior::BackendBehaviorNote,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{graph::Graph, operation::HeuristicMode},
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
            nvcc_args: ["--std=c++20"],
            nvrtc_args: ["--std=c++20"],
        },
    }
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let batch = 1_i64;
    let rows = 8_i64;
    let cols = 16_i64;
    let depth = 4_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::F32)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32)
        .with_name("custom-kernel-matmul");

    let a = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, rows, depth])?).with_id(1000),
    );
    let b = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, depth, cols])?).with_id(1001),
    );
    let c = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([batch, rows, cols])?).with_id(1002),
    );
    graph.matmul(a, b, c, DataType::F32)?;

    let compiled = graph
        .plan_candidates(&ctx.cudnn, &[HeuristicMode::A])?
        .require_behavior_notes(vec![BackendBehaviorNote::SupportsCudaGraphNativeApi])?
        .compile(&ctx.cudnn)?;

    let element_count = (batch * rows * cols) as usize;
    let mut a_dev = DeviceMemory::from_slice(&vec![1.0_f32; (batch * rows * depth) as usize])?;
    let mut b_dev = DeviceMemory::from_slice(&vec![1.0_f32; (batch * depth * cols) as usize])?;
    let mut c_dev = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;

    let mut main_graph = CudaGraph::create()?;
    let cudnn_node = match compiled.append_to_cuda_graph(
        &ctx.cudnn,
        &bindings,
        Some(&mut workspace),
        &mut main_graph,
        &[],
    ) {
        Ok(node) => node,
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let module = postprocess_kernel::Module::create(ctx.cudnn.cuda_context())?;
    let scale = 0.5_f32;
    let bias = 1.25_f32;
    let launch = LaunchConfig::for_1d_grid(element_count, 128);
    let kernel_node = unsafe {
        module.affine_node(
            &mut main_graph,
            &[cudnn_node],
            &launch,
            c_dev.as_mut_ptr() as _,
            element_count as i32,
            scale,
            bias,
        )?
    };

    let mut executable = main_graph.instantiate()?;
    executable.launch(&ctx.stream)?;
    ctx.stream.synchronize()?;

    let output = c_dev.copy_to_host_vec()?;
    let expected = depth as f32 * scale + bias;
    assert!(
        output
            .iter()
            .all(|value| (*value - expected).abs() <= 1.0e-6)
    );

    let scale_new = 0.25_f32;
    let bias_new = 2.0_f32;
    let mut a_dev_new = DeviceMemory::from_slice(&vec![2.0_f32; (batch * rows * depth) as usize])?;
    let mut b_dev_new = DeviceMemory::from_slice(&vec![1.5_f32; (batch * depth * cols) as usize])?;
    let mut c_dev_new = DeviceMemory::<f32>::zeroes(element_count)?;
    let mut workspace_new = common::workspace(compiled.workspace_size()?)?;

    let mut bindings_new = compiled.bindings();
    bindings_new.set(a, &mut a_dev_new)?;
    bindings_new.set(b, &mut b_dev_new)?;
    bindings_new.set(c, &mut c_dev_new)?;

    compiled.update_cuda_graph_node(
        &ctx.cudnn,
        &bindings_new,
        Some(&mut workspace_new),
        cudnn_node,
        Some(&mut executable),
    )?;
    unsafe {
        module.affine_set_node_params(
            &mut executable,
            kernel_node,
            &launch,
            c_dev_new.as_mut_ptr() as _,
            element_count as i32,
            scale_new,
            bias_new,
        )?;
    }
    executable.launch(&ctx.stream)?;
    ctx.stream.synchronize()?;

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
