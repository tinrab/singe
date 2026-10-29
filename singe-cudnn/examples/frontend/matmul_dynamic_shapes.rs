mod common;

use std::sync::{Arc, Mutex};

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        execution::KernelCache,
        tensor::{Shape, TensorId, TensorSpec},
    },
    data_type::{DataType, bf16},
    error::Result,
    frontend::{
        error::Error as FrontendError,
        graph::{DataTypePolicy, Graph, GraphConfig, GraphKeyShapePolicy},
        operation::MatmulConfig,
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
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
    Shape::contiguous([shape.batch, shape.m, shape.k])
}

fn rhs_shape(shape: MatmulShape) -> Result<Shape> {
    Shape::contiguous([shape.batch, shape.k, shape.n])
}

fn output_shape(shape: MatmulShape) -> Result<Shape> {
    Shape::contiguous([shape.batch, shape.m, shape.n])
}

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;

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

    let mut graph = Graph::with_config(
        GraphConfig::new()
            .with_name("matmul-dynamic-shapes")
            .with_data_type_policy(
                DataTypePolicy::new()
                    .with_io(DataType::BF16)
                    .with_compute(DataType::F32),
            ),
    );
    let lhs = graph.insert_tensor(
        TensorId::new(1000),
        TensorSpec::new(DataType::BF16, lhs_shape(base)?).with_name("lhs"),
    )?;
    let rhs = graph.insert_tensor(
        TensorId::new(1001),
        TensorSpec::new(DataType::BF16, rhs_shape(base)?).with_name("rhs"),
    )?;
    let output = graph.insert_tensor(
        TensorId::new(1002),
        TensorSpec::new(DataType::F32, output_shape(base)?).with_name("output"),
    )?;
    graph.matmul_into(
        lhs,
        rhs,
        output,
        MatmulConfig::new().with_compute_type(DataType::F32),
    )?;

    graph.record_runtime_shape_constraints(
        lhs,
        [1, 1, base.k],
        [runtime.batch, runtime.m, base.k],
    )?;
    graph.record_runtime_shape_constraints(
        rhs,
        [1, base.k, 1],
        [runtime.batch, base.k, runtime.n],
    )?;
    graph.record_runtime_shape_constraints(
        output,
        [1, 1, 1],
        [runtime.batch, runtime.m, runtime.n],
    )?;
    // Compilation options can be selected after constructing operations and runtime bounds.
    graph.set_dynamic_shape_enabled(true);
    graph.set_override_shape_enabled(true);
    graph.attach_shared_kernel_cache(Arc::clone(&kernel_cache))?;

    println!(
        "dynamic graph key: {:016x}, concrete-layout key: {:016x}",
        graph.key(GraphKeyShapePolicy::Auto)?,
        graph.key(GraphKeyShapePolicy::Include)?,
    );

    let candidates = graph.plan_candidates(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    println!(
        "native shape modes: dynamic={}, overrides={}",
        candidates.operation_graph().is_dynamic_shape_enabled(),
        candidates.operation_graph().is_override_shape_enabled(),
    );
    let built = candidates.build_candidates(Some(&example.cudnn), &CompileConfig::new())?;
    let selected = built
        .available_plan_ids()
        .into_iter()
        .next()
        .ok_or_else(|| FrontendError::NoAvailablePlans {
            reports: built.build_reports(),
        })?;
    // Built plans retain their preparation and cache independently of later recipe changes.
    graph.detach_kernel_cache();
    graph.set_dynamic_shape_enabled(false);
    graph.set_override_shape_enabled(false);
    let (cache_bytes, revision, entries) = {
        let cache = kernel_cache
            .lock()
            .map_err(|_| FrontendError::KernelCacheLockPoisoned)?;
        (cache.serialize()?, cache.revision()?, cache.entry_count()?)
    };
    let mut restored_cache = KernelCache::deserialize(&cache_bytes)?;
    restored_cache.build(None)?;
    println!(
        "kernel cache: revision={revision}, entries={entries}, native bytes={}, restored entries={}",
        cache_bytes.len(),
        restored_cache.entry_count()?,
    );

    let mut seed = 123_456_789_u32;
    let lhs_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(lhs_shape(base)?.storage_span()?, &mut seed),
    )?;
    let rhs_dev = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(rhs_shape(base)?.storage_span()?, &mut seed),
    )?;
    let mut output_dev = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        output_shape(base)?.storage_span()?,
    )?;
    let mut workspace = common::workspace(
        example.cudnn.cuda_context(),
        built.workspace_size_for(&selected)?,
    )?;

    let mut bindings = built.bindings()?;
    bindings.set_input(lhs, &lhs_dev)?;
    bindings.set_input(rhs, &rhs_dev)?;
    bindings.set(output, &mut output_dev)?;
    // SAFETY: inputs, writable output, plan and workspace remain live through stream
    // completion, including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        built.execute(
            context,
            ExecuteArgs::new(&bindings)
                .with_workspace(&mut workspace)
                .with_plan(&selected),
        )
    })?;
    drop(bindings);

    let lhs_runtime = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(lhs_shape(runtime)?.storage_span()?, &mut seed),
    )?;
    let rhs_runtime = DeviceMemory::from_slice(
        example.cudnn.cuda_context(),
        &init_bf16_image(rhs_shape(runtime)?.storage_span()?, &mut seed),
    )?;
    let mut output_runtime = DeviceMemory::<f32>::zeroed(
        example.cudnn.cuda_context(),
        output_shape(runtime)?.storage_span()?,
    )?;

    let mut runtime_bindings = built.bindings()?;
    runtime_bindings.set_input(lhs, &lhs_runtime)?;
    runtime_bindings.set_input(rhs, &rhs_runtime)?;
    runtime_bindings.set(output, &mut output_runtime)?;

    let mut runtime_overrides = built.runtime_overrides();
    runtime_overrides.set(lhs, lhs_shape(runtime)?)?;
    runtime_overrides.set(rhs, rhs_shape(runtime)?)?;
    runtime_overrides.set(output, output_shape(runtime)?)?;
    let runtime_overrides = runtime_overrides.build();
    let mut runtime_workspace = common::workspace(
        example.cudnn.cuda_context(),
        built.max_workspace_size_with_runtime_overrides(&example.cudnn, &runtime_overrides)?,
    )?;

    // SAFETY: runtime layouts match these initialized inputs and writable output;
    // all buffers, plan and workspace remain live through completion on success or error.
    example.execute_and_wait(|context| unsafe {
        built.execute(
            context,
            ExecuteArgs::new(&runtime_bindings)
                .with_workspace(&mut runtime_workspace)
                .with_plan(&selected)
                .with_overrides(&runtime_overrides),
        )
    })?;
    drop(runtime_bindings);

    let compiled = built.compile_selected(&selected)?;
    println!(
        "selected {selected}: {} runtime workspace bytes",
        compiled.workspace_size_with_runtime_overrides(&example.cudnn, &runtime_overrides)?,
    );

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_matmul_dynamic_shapes", run())
}
