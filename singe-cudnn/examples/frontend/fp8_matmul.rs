mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    data_type::{DataType, bf16, f8e4m3},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, MatmulFp8Inputs, TensorOutputConfig},
        operation::MatmulConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    tensor::{Shape, TensorSpec},
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

#[expect(
    clippy::many_single_char_names,
    reason = "keeps the upstream sample's single-letter dimension names"
)]
fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let b = 16_i64;
    let m = 32_i64;
    let n = 64_i64;
    let k = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::Bf16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let a = graph.tensor(named_tensor(
        "A",
        DataType::F8E4M3,
        Shape::with_strides([b, m, k], [m * k, k, 1])?,
    ));
    let b_tensor = graph.tensor(named_tensor(
        "B",
        DataType::F8E4M3,
        Shape::with_strides([b, k, n], [k * n, 1, k])?,
    ));
    let descale_a = graph.tensor(named_tensor(
        "descale_a",
        DataType::F32,
        Shape::with_strides([1, 1, 1], [1, 1, 1])?,
    ));
    let descale_b = graph.tensor(named_tensor(
        "descale_b",
        DataType::F32,
        Shape::with_strides([1, 1, 1], [1, 1, 1])?,
    ));

    let c = graph.matmul_fp8(
        MatmulFp8Inputs {
            a,
            b: b_tensor,
            descale_a,
            descale_b,
        },
        MatmulConfig::new().with_compute_type(DataType::FastFloatForFp8),
        TensorOutputConfig::new()
            .with_data_type(DataType::Bf16)
            .output_tensor(),
    )?;
    graph.mark_as_output(c)?;

    let a_input_elements = graph.shape(a)?.storage_span()?;
    let b_input_elements = graph.shape(b_tensor)?.storage_span()?;
    let output_elements = graph.shape(c)?.storage_span()?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let a_dev = DeviceMemory::<f8e4m3>::zeroed(example.cudnn.cuda_context(), a_input_elements)?;
    let b_dev = DeviceMemory::<f8e4m3>::zeroed(example.cudnn.cuda_context(), b_input_elements)?;
    let a_descale_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &[1.0_f32])?;
    let b_descale_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &[1.0_f32])?;
    let mut c_dev = DeviceMemory::<bf16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev.view())?;
    bindings.set_input(b_tensor, &b_dev.view())?;
    bindings.set_input(descale_a, &a_descale_dev.view())?;
    bindings.set_input(descale_b, &b_descale_dev.view())?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: initialized inputs, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute_async(
            context,
            ExecuteArgs::new(&bindings).with_workspace(workspace.output()),
        )
    })?;

    Ok(())
}

fn main() -> anyhow::Result<()> {
    common::finish("frontend_fp8_matmul", run())
}
