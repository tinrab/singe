mod common;

use singe_core::assert_close;
use std::ops::Range;

use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_vec};
use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorId, TensorSpec},
    data_type::DataType,
    data_type::f16 as half_f16,
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{HeuristicMode, PointwiseOperation},
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<half_f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(half_f16::from_f32)
        .collect()
}

fn count_nonfinite_f16(values: &[half_f16]) -> usize {
    values
        .iter()
        .filter(|value| !value.to_f32().is_finite())
        .count()
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

fn build_graph(spec: &SliceMatmulSpec) -> Result<(Graph, TensorId, TensorId, TensorId)> {
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
    let [b_range, m_range, k_range] = spec.slice_ranges();
    let a_slice = graph.slice_infer(a, &[b_range, m_range, k_range])?;
    let b = graph.tensor(
        TensorSpec::new(DataType::F16, Shape::contiguous([spec.b, spec.k, spec.n])?).with_name("B"),
    );
    let c0 = graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([spec.b, spec.m, spec.n])?)
            .virtual_tensor(),
    );
    graph.matmul(a_slice, b, c0, DataType::F32)?;
    let c = graph.tensor(TensorSpec::new(
        DataType::F16,
        Shape::contiguous([spec.b, spec.m, spec.n])?,
    ));
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::ReluFwd,
        input: c0,
        output: c,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    Ok((graph, a, b, c))
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let spec = SliceMatmulSpec::create();
    let serialized_spec = to_vec(&spec).expect("slice matmul spec should serialize");
    let deserialized_spec: SliceMatmulSpec =
        from_slice(&serialized_spec).expect("slice matmul spec should deserialize");

    let (graph, a, b, c) = build_graph(&deserialized_spec)?;

    let compiled = match graph.compile(&ctx.cudnn, &[HeuristicMode::A]) {
        Ok(compiled) => compiled,
        Err(error) => return Err(error),
    };

    let mut half_seed = 123_456_789_u32;
    let mut a_dev = DeviceMemory::from_slice(&init_f16_image(
        (deserialized_spec.b_actual() * deserialized_spec.m_actual() * deserialized_spec.k)
            as usize,
        &mut half_seed,
    ))?;
    let mut b_dev = DeviceMemory::from_slice(&init_f16_image(
        (deserialized_spec.b * deserialized_spec.k * deserialized_spec.n) as usize,
        &mut half_seed,
    ))?;
    let mut c_dev = DeviceMemory::from_slice(&init_f16_image(
        (deserialized_spec.b * deserialized_spec.m * deserialized_spec.n) as usize,
        &mut half_seed,
    ))?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(a, &mut a_dev)?;
    bindings.set(b, &mut b_dev)?;
    bindings.set(c, &mut c_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let c_host = c_dev.copy_to_host_vec()?;
    let c_first = c_host[0].to_f32();
    let c_last = c_host[c_host.len() - 1].to_f32();
    let c_checksum = c_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let c_nonfinite = count_nonfinite_f16(&c_host);

    println!(
        "c: first={c_first}, last={c_last}, checksum={c_checksum}, nonfinite={c_nonfinite}, len={}",
        c_host.len()
    );

    assert_eq!(c_nonfinite, 0, "c.nonfinite mismatch");
    assert_close!(c_first, 14.8828125, 1.0e-5, "c.first");
    assert_close!(c_last, 17.109375, 1.0e-5, "c.last");
    assert_close!(c_checksum, 65834.375, 1.0e-2, "c.checksum");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_slice_matmul", run())
}
