mod common;

use std::{env::args, ops::Range};

use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_vec};
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
        operation::{MatmulConfig, PointwiseOperation, SliceConfig, TransposeConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SliceMatmulSpec {
    b_start: i64,
    b: i64,
    b_end: i64,
    m_start: i64,
    m: i64,
    m_end: i64,
    n: i64,
    k: i64,
}

impl SliceMatmulSpec {
    fn create() -> Self {
        Self {
            b_start: 1,
            b: 8,
            b_end: 2,
            m_start: 3,
            m: 16,
            m_end: 4,
            n: 32,
            k: 64,
        }
    }

    fn b_actual(&self) -> i64 {
        self.b_start + self.b + self.b_end
    }

    fn m_actual(&self) -> i64 {
        self.m_start + self.m + self.m_end
    }

    fn slice_ranges(&self) -> [Range<i64>; 3] {
        [
            self.b_start..(self.b_start + self.b),
            self.m_start..(self.m_start + self.m),
            0..self.k,
        ]
    }
}

fn build_graph(
    spec: &SliceMatmulSpec,
    transpose_output: bool,
) -> Result<(Graph, TensorId, TensorId, TensorId)> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_compute(DataType::F32),
        ),
    );

    let a = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([spec.b_actual(), spec.m_actual(), spec.k])?,
        )
        .with_name("A"),
    );
    let a_slice = graph.slice(
        a,
        SliceConfig::new(spec.slice_ranges()).with_name("matrix_window"),
        TensorOutputConfig::new(),
    )?;
    let b = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::contiguous([spec.b, spec.k, spec.n])?).with_name("B"),
    );
    let c0 = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([spec.b, spec.m, spec.n])?)
            .virtual_tensor(),
    );
    graph.matmul_into(
        a_slice,
        b,
        c0,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;
    let output_spec = TensorSpec::new(DataType::F16, Shape::contiguous([spec.b, spec.m, spec.n])?);
    let c = graph.tensor(if transpose_output {
        output_spec.virtual_tensor()
    } else {
        output_spec
    });
    graph.pointwise_into(PointwiseOperation::Unary {
        name: None,
        mode: PointwiseMode::ReluFwd,
        input: c0,
        output: c,
        compute_type: Some(DataType::F32),
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    })?;

    let c = if transpose_output {
        let output = graph.transpose(
            c,
            TransposeConfig::new([0, 2, 1]).with_name("transposed_result"),
            TensorOutputConfig::new().output_tensor(),
        )?;
        // Store the transposed values contiguously instead of inheriting permuted input strides.
        let shape = Shape::contiguous([spec.b, spec.n, spec.m])?;
        graph.modify_tensor(output, |tensor| tensor.with_shape(shape))?;
        output
    } else {
        c
    };

    Ok((graph, a, b, c))
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let spec = SliceMatmulSpec::create();
    let serialized_spec = to_vec(&spec)?;
    let deserialized_spec: SliceMatmulSpec = from_slice(&serialized_spec)?;

    let transpose_output = args().any(|argument| argument == "--transpose-output");
    let (graph, a, b, c) = build_graph(&deserialized_spec, transpose_output)?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let mut half_seed = 123_456_789_u32;
    let a_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(a)?.storage_span()?, &mut half_seed),
    )?;
    let b_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_f16_image(graph.shape(b)?.storage_span()?, &mut half_seed),
    )?;
    let mut c_dev = DeviceMemory::<f16>::zeroed(
        example.cudnn.cuda_context(),
        graph.shape(c)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(a, &a_dev)?;
    bindings.set_input(b, &b_dev)?;
    bindings.set(c, &mut c_dev)?;
    // SAFETY: the plan, inputs, output and workspace remain live until stream completion,
    // including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!("output dimensions: {:?}", graph.shape(c)?.dimensions());

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_slice_matmul", run())
}
