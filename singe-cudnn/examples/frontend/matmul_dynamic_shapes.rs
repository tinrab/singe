mod common;

use std::sync::{Arc, Mutex};

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::execution::KernelCache,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::Result,
    frontend::{graph::Graph, operation::HeuristicMode},
};

#[derive(Clone, Copy)]
struct MatmulShape {
    batch: i64,
    m: i64,
    n: i64,
    k: i64,
}

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn lhs_shape(shape: MatmulShape) -> Result<Shape> {
    Shape::contiguous([shape.batch, shape.m, shape.k])?.with_strides([
        shape.m * shape.k,
        shape.k,
        1,
    ])
}

fn rhs_shape(shape: MatmulShape) -> Result<Shape> {
    Shape::contiguous([shape.batch, shape.k, shape.n])?.with_strides([
        shape.k * shape.n,
        shape.n,
        1,
    ])
}

fn output_shape(shape: MatmulShape) -> Result<Shape> {
    Shape::contiguous([shape.batch, shape.m, shape.n])?.with_strides([
        shape.m * shape.n,
        shape.n,
        1,
    ])
}

fn lhs_volume(shape: MatmulShape) -> usize {
    (shape.batch * shape.m * shape.k) as usize
}

fn rhs_volume(shape: MatmulShape) -> usize {
    (shape.batch * shape.k * shape.n) as usize
}

fn output_volume(shape: MatmulShape) -> usize {
    (shape.batch * shape.m * shape.n) as usize
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if singe_cudnn::version()? < 90400 {
        println!("skipping: dynamic shape kernel cache requires cuDNN 9.4 or newer");
        return Ok(());
    }

    let base = MatmulShape {
        batch: 2,
        m: 64,
        n: 32,
        k: 128,
    };
    let runtime = MatmulShape {
        batch: 4,
        m: 96,
        n: 48,
        k: 128,
    };
    let kernel_cache = Arc::new(Mutex::new(KernelCache::create()?));

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_compute_data_type(DataType::F32)
        .with_dynamic_shape()
        .with_override_shape()
        .with_name("matmul-dynamic-shapes");
    graph.set_shared_kernel_cache(Arc::clone(&kernel_cache))?;

    let lhs = graph.tensor(
        TensorSpec::new(DataType::BF16, lhs_shape(base)?)
            .with_id(1000)
            .with_name("lhs"),
    );
    let rhs = graph.tensor(
        TensorSpec::new(DataType::BF16, rhs_shape(base)?)
            .with_id(1001)
            .with_name("rhs"),
    );
    let output = graph.tensor(
        TensorSpec::new(DataType::F32, output_shape(base)?)
            .with_id(1002)
            .with_name("output"),
    );
    graph.matmul(lhs, rhs, output, DataType::F32)?;

    graph.set_dynamic_shape_constraints(lhs, [1, 1, base.k], [runtime.batch, runtime.m, base.k])?;
    graph.set_dynamic_shape_constraints(rhs, [1, base.k, 1], [runtime.batch, base.k, runtime.n])?;
    graph.set_dynamic_shape_constraints(
        output,
        [1, 1, 1],
        [runtime.batch, runtime.m, runtime.n],
    )?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut lhs_dev = DeviceMemory::from_slice(&init_bf16_image(lhs_volume(base), &mut seed))?;
    let mut rhs_dev = DeviceMemory::from_slice(&init_bf16_image(rhs_volume(base), &mut seed))?;
    let mut output_dev = DeviceMemory::<f32>::zeroes(output_volume(base))?;
    let mut workspace = common::workspace(compiled.workspace_size()?.max(1))?;

    let mut bindings = compiled.bindings();
    bindings.set(lhs, &mut lhs_dev)?;
    bindings.set(rhs, &mut rhs_dev)?;
    bindings.set(output, &mut output_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let mut lhs_runtime =
        DeviceMemory::from_slice(&init_bf16_image(lhs_volume(runtime), &mut seed))?;
    let mut rhs_runtime =
        DeviceMemory::from_slice(&init_bf16_image(rhs_volume(runtime), &mut seed))?;
    let mut output_runtime = DeviceMemory::<f32>::zeroes(output_volume(runtime))?;

    let mut runtime_bindings = compiled.bindings();
    runtime_bindings.set(lhs, &mut lhs_runtime)?;
    runtime_bindings.set(rhs, &mut rhs_runtime)?;
    runtime_bindings.set(output, &mut output_runtime)?;

    let mut runtime_overrides = compiled.runtime_overrides();
    runtime_overrides.set_shape(lhs, &lhs_shape(runtime)?)?;
    runtime_overrides.set_shape(rhs, &rhs_shape(runtime)?)?;
    runtime_overrides.set_shape(output, &output_shape(runtime)?)?;
    let runtime_overrides = runtime_overrides.build();

    compiled.execute_with_runtime_overrides(
        &ctx.cudnn,
        &runtime_bindings,
        Some(&mut workspace),
        &runtime_overrides,
    )?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_matmul_dynamic_shapes", run())
}
