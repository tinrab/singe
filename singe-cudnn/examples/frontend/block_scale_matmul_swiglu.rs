mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::{Error, Result, Status},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            BlockScaleDequantizeConfig, CompileConfig, HeuristicMode, MatmulConfig,
            PointwiseOperation,
        },
        plan::BuildPlanPolicy,
    },
    math::NanPropagation,
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn packed_len(element_count: i64, data_type: DataType) -> Result<usize> {
    let element_count = usize::try_from(element_count).map_err(|_| Error::OutOfRange {
        name: "element_count".into(),
    })?;
    let bits = match data_type {
        DataType::F4E2M1 => 4,
        DataType::F8E4M3 => 8,
        DataType::F32 => 32,
        _ => return Err(Error::UnsupportedDataType),
    };
    Ok(element_count.saturating_mul(bits).div_ceil(8))
}

fn is_not_supported(error: &Error) -> bool {
    matches!(
        error,
        Error::Cudnn { code, .. }
            if *code == Status::NotSupported
                || *code == Status::NotSupportedGraphPattern
                || *code == Status::NotSupportedShape
                || *code == Status::NotSupportedDataType
                || *code == Status::NotSupportedLayout
    )
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 1;
    let m = 256;
    let n = 256;
    let k = 256;
    let block_size: i32 = 16;
    let indestructible_128x4_block_m_n = 128;
    let indestructible_128x4_block_k = 4;
    let block_scale_dim_m = ((m + indestructible_128x4_block_m_n - 1)
        / indestructible_128x4_block_m_n)
        * indestructible_128x4_block_m_n;
    let block_scale_dim_n = ((n + indestructible_128x4_block_m_n - 1)
        / indestructible_128x4_block_m_n)
        * indestructible_128x4_block_m_n;
    let block_scale_dim_k = (((k + i64::from(block_size) - 1) / i64::from(block_size)
        + indestructible_128x4_block_k
        - 1)
        / indestructible_128x4_block_k)
        * indestructible_128x4_block_k;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let tensor_a = graph.tensor(named_tensor(
        "tensor_a",
        DataType::F4E2M1,
        Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
    ));
    let tensor_b0 = graph.tensor(named_tensor(
        "tensor_b",
        DataType::F4E2M1,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let tensor_b1 = graph.tensor(named_tensor(
        "tensor_b",
        DataType::F4E2M1,
        Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
    ));
    let block_descale_a = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_m, block_scale_dim_k])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            block_scale_dim_k,
            1,
        ])?,
    ));
    let block_descale_b0 = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_k, block_scale_dim_n])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            1,
            block_scale_dim_k,
        ])?,
    ));
    let block_descale_b1 = graph.tensor(TensorSpec::block_scale_tensor(
        DataType::F8E4M3,
        Shape::contiguous([b, block_scale_dim_k, block_scale_dim_n])?.with_strides(vec![
            block_scale_dim_m * block_scale_dim_k,
            1,
            block_scale_dim_k,
        ])?,
    ));

    let dequan_tensor_a = graph.block_scale_dequantize_infer(
        tensor_a,
        block_descale_a,
        BlockScaleDequantizeConfig::with_default_compute_type(vec![1, block_size]),
    )?;
    let dequan_tensor_b0 = graph.block_scale_dequantize_infer(
        tensor_b0,
        block_descale_b0,
        BlockScaleDequantizeConfig::with_default_compute_type(vec![block_size, 1]),
    )?;
    let dequan_tensor_b1 = graph.block_scale_dequantize_infer(
        tensor_b1,
        block_descale_b1,
        BlockScaleDequantizeConfig::with_default_compute_type(vec![block_size, 1]),
    )?;

    let tensor_c0 = graph.tensor(
        named_tensor(
            "matmul0::OUT_0",
            DataType::F32,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .virtual_tensor(),
    );
    graph.matmul_with_config(
        dequan_tensor_a,
        dequan_tensor_b0,
        tensor_c0,
        MatmulConfig::new(DataType::F32),
    )?;
    let tensor_c1 = graph.tensor(
        named_tensor(
            "matmul1::OUT_0",
            DataType::F32,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .virtual_tensor(),
    );
    graph.matmul_with_config(
        dequan_tensor_a,
        dequan_tensor_b1,
        tensor_c1,
        MatmulConfig::new(DataType::F32),
    )?;

    let tensor_after_swish = graph.tensor(
        named_tensor(
            "swish::OUT_0",
            DataType::F32,
            graph.shape(tensor_c0)?.clone(),
        )
        .virtual_tensor(),
    );
    graph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::SwishFwd,
        input: tensor_c0,
        output: tensor_after_swish,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });

    let tensor_d = graph.tensor(named_tensor(
        "mul::OUT_0",
        DataType::F4E2M1,
        Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
    ));
    graph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: tensor_after_swish,
        rhs: tensor_c1,
        output: tensor_d,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let compiled = match graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes(vec![HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupported),
    ) {
        Ok(compiled) => compiled,
        Err(error) if is_not_supported(&error) => {
            println!(
                "frontend_block_scale_matmul_swiglu: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        Err(error) => return Err(error),
    };

    let mut tensor_a_dev = DeviceMemory::<u8>::zeroes(packed_len(b * m * k, DataType::F4E2M1)?)?;
    let mut tensor_b0_dev = DeviceMemory::<u8>::zeroes(packed_len(b * k * n, DataType::F4E2M1)?)?;
    let mut tensor_b1_dev = DeviceMemory::<u8>::zeroes(packed_len(b * k * n, DataType::F4E2M1)?)?;
    let mut block_descale_a_dev = DeviceMemory::<u8>::zeroes(packed_len(
        b * block_scale_dim_m * block_scale_dim_k,
        DataType::F8E4M3,
    )?)?;
    let mut block_descale_b0_dev = DeviceMemory::<u8>::zeroes(packed_len(
        b * block_scale_dim_k * block_scale_dim_n,
        DataType::F8E4M3,
    )?)?;
    let mut block_descale_b1_dev = DeviceMemory::<u8>::zeroes(packed_len(
        b * block_scale_dim_k * block_scale_dim_n,
        DataType::F8E4M3,
    )?)?;
    let mut tensor_d_dev = DeviceMemory::<u8>::zeroes(packed_len(b * m * n, DataType::F4E2M1)?)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    bindings.set(tensor_a, &mut tensor_a_dev)?;
    bindings.set(tensor_b0, &mut tensor_b0_dev)?;
    bindings.set(tensor_b1, &mut tensor_b1_dev)?;
    bindings.set(block_descale_a, &mut block_descale_a_dev)?;
    bindings.set(block_descale_b0, &mut block_descale_b0_dev)?;
    bindings.set(block_descale_b1, &mut block_descale_b1_dev)?;
    bindings.set(tensor_d, &mut tensor_d_dev)?;
    if let Err(error) = compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace)) {
        if is_not_supported(&error) {
            println!(
                "frontend_block_scale_matmul_swiglu: skipped, not supported on current configuration"
            );
            return Ok(());
        }
        return Err(error);
    }

    let tensor_d_host = tensor_d_dev.copy_to_host_vec()?;
    let nonzero = tensor_d_host.iter().filter(|value| **value != 0).count();
    let checksum = tensor_d_host
        .iter()
        .map(|value| u64::from(*value))
        .sum::<u64>();

    println!("shape: {:?}", graph.shape(tensor_d)?.dimensions());
    println!("nonzero: {nonzero}");
    println!("checksum: {checksum}");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_block_scale_matmul_swiglu", run())
}
