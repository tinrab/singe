mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, TensorOutputConfig},
        operation::{MatmulConfig, PointwiseOperation, SliceConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

const A_ID: TensorId = TensorId::new(1001);
const B_ID: TensorId = TensorId::new(1002);
const C_ID: TensorId = TensorId::new(1003);

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn build_graph(
    save_slice: bool,
    compact_slice: bool,
    slice_strides: Vec<i64>,
) -> Result<(Graph, TensorId)> {
    let b_start = 1_i64;
    let b = 8_i64;
    let b_end = 2_i64;
    let b_actual = b_start + b + b_end;

    let m_start = 3_i64;
    let m = 16_i64;
    let m_end = 4_i64;
    let m_actual = m_start + m + m_end;

    let n = 32_i64;
    let k = 64_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("slice-gemm")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::F16)
                    .with_compute(DataType::F32),
            ),
    );

    let a = graph.insert_tensor(
        A_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([b_actual, m_actual, k], vec![m_actual * k, k, 1])?,
        ),
    )?;
    let slice_output = if save_slice {
        TensorOutputConfig::new().output_tensor()
    } else {
        TensorOutputConfig::new()
    };
    let ranges = [b_start..(b_start + b), m_start..(m_start + m), 0..k];
    let slice_config = SliceConfig::new(ranges)
        .with_strides(slice_strides)
        .with_name("matrix_window");
    let slice_dimensions = slice_config.infer_output_dimensions(graph.shape(a)?)?;
    let output_rows = slice_dimensions[1];
    let a_slice = if compact_slice {
        let output = TensorSpec::new(DataType::F16, Shape::contiguous(slice_dimensions)?);
        let output = if save_slice {
            output
        } else {
            output.virtual_tensor()
        };
        let output = graph.tensor(output);
        graph.slice_into(a, output, slice_config)?;
        output
    } else {
        graph.slice(a, slice_config, slice_output)?
    };
    let b_tensor = graph.insert_tensor(
        B_ID,
        TensorSpec::new(
            DataType::F16,
            Shape::with_strides([b, k, n], [k * n, n, 1])?,
        ),
    )?;
    let c0 = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, output_rows, n])?).virtual_tensor(),
    );
    graph.matmul_into(
        a_slice,
        b_tensor,
        c0,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;

    let c = graph.insert_tensor(
        C_ID,
        TensorSpec::new(DataType::F32, Shape::contiguous([b, output_rows, n])?),
    )?;
    graph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::ReluFwd,
        input: c0,
        output: c,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::NotPropagate,
        alpha1: 1.0,
        axis: None,
    })?;
    graph.mark_as_output(c)?;

    Ok((graph, a_slice))
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let save_slice = args().any(|argument| argument == "--save-slice");
    let compact_slice = args().any(|argument| argument == "--compact-slice");
    let strided_slice = args().any(|argument| argument == "--strided-slice");
    // Leave the last dimension's stride omitted, as allowed by the C++ slice attributes.
    let mut slice_strides = if strided_slice {
        vec![1, 2]
    } else {
        Vec::new()
    };
    if args().any(|argument| argument == "--extra-strides") {
        slice_strides.resize(3, 1);
        slice_strides.push(7);
    }
    let (graph, a_slice) = build_graph(save_slice, compact_slice, slice_strides)?;

    let serialized = serde_json::to_vec(&graph)?;
    let restored = serde_json::from_slice::<Graph>(&serialized)?;

    let compiled = restored.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(restored.shape(A_ID)?.storage_span()?, &mut seed),
    )?;
    let b_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(restored.shape(B_ID)?.storage_span()?, &mut seed),
    )?;
    let mut c_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        restored.shape(C_ID)?.storage_span()?,
    )?;
    let slice_elements = restored.shape(a_slice)?.storage_span()?;
    let mut slice_dev = save_slice
        .then(|| DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), slice_elements))
        .transpose()?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(A_ID, &a_dev)?;
    bindings.set_input(B_ID, &b_dev)?;
    bindings.set(C_ID, &mut c_dev)?;
    if let Some(memory) = &mut slice_dev {
        bindings.set(a_slice, memory)?;
    }
    // SAFETY: the plan, inputs, output and workspace remain live until stream completion,
    // including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    let c_host = c_dev.copy_to_host_vec()?;
    println!(
        "c: first={}, last={}, checksum={}, nonfinite={}, len={}",
        c_host[0],
        c_host[c_host.len() - 1],
        c_host.iter().copied().sum::<f32>(),
        common::count_nonfinite_f32(&c_host),
        c_host.len()
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_slice", run())
}
