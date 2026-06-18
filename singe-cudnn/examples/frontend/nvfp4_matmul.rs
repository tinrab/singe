mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f4e2m1, f8e4m3, f16},
    error::{Error, Result},
    frontend::{
        graph::Graph,
        operation::{BlockScaleDequantizeConfig, CompileConfig, HeuristicMode, MatmulConfig},
        plan::BuildPlanPolicy,
    },
    version,
};

const INDESTRUCTIBLE_128X4_BLOCK_M_N: i64 = 128;
const INDESTRUCTIBLE_128X4_BLOCK_K: i64 = 4;

struct TestParams {
    b: i64,
    m: i64,
    n: i64,
    k: i64,
    block_size: i32,
    datatype_a: DataType,
    datatype_b: DataType,
    datatype_block_scale: DataType,
    datatype_d: DataType,
}

fn packed_len(element_count: i64, data_type: DataType) -> Result<usize> {
    let element_count = usize::try_from(element_count).map_err(|_| Error::OutOfRange {
        name: "element_count".into(),
    })?;
    let bits = match data_type {
        DataType::F4E2M1 => 4,
        DataType::F8E4M3 => 8,
        DataType::F16 => 16,
        _ => return Err(Error::UnsupportedDataType),
    };
    Ok(element_count.saturating_mul(bits).div_ceil(8))
}

fn round_up(value: i64, multiple: i64) -> i64 {
    ((value + multiple - 1) / multiple) * multiple
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if !ctx.is_blackwell_device() {
        println!("frontend_nvfp4_matmul: skipped, requires Blackwell Computing GPU");
        return Ok(());
    }
    if version()? < 90_700 {
        println!("frontend_nvfp4_matmul: skipped, requires cuDNN 9.7.0+");
        return Ok(());
    }

    let test_params = TestParams {
        b: 1,
        m: 128,
        n: 128,
        k: 64,
        block_size: 16,
        datatype_a: DataType::F4E2M1,
        datatype_b: DataType::F4E2M1,
        datatype_block_scale: DataType::F8E4M3,
        datatype_d: DataType::F16,
    };

    let b = test_params.b;
    let m = test_params.m;
    let n = test_params.n;
    let k = test_params.k;
    let datatype_a = test_params.datatype_a;
    let datatype_b = test_params.datatype_b;
    let datatype_block_scale = test_params.datatype_block_scale;
    let block_size = test_params.block_size;
    let datatype_d = test_params.datatype_d;

    let block_scale_dim_m = round_up(m, INDESTRUCTIBLE_128X4_BLOCK_M_N);
    let block_scale_dim_n = round_up(n, INDESTRUCTIBLE_128X4_BLOCK_M_N);
    let block_scale_dim_k = round_up(
        (k + i64::from(block_size) - 1) / i64::from(block_size),
        INDESTRUCTIBLE_128X4_BLOCK_K,
    );

    let mut tensor_a_gpu = DeviceMemory::<f4e2m1>::zeroes(packed_len(b * m * k, datatype_a)?)?;
    let mut tensor_b_gpu = DeviceMemory::<f4e2m1>::zeroes(packed_len(b * k * n, datatype_b)?)?;

    let mut block_descale_a_gpu = DeviceMemory::<f8e4m3>::zeroes(packed_len(
        b * block_scale_dim_m * block_scale_dim_k,
        datatype_block_scale,
    )?)?;
    let mut block_descale_b_gpu = DeviceMemory::<f8e4m3>::zeroes(packed_len(
        b * block_scale_dim_k * block_scale_dim_n,
        datatype_block_scale,
    )?)?;

    let mut tensor_d_gpu = DeviceMemory::<f16>::zeroes(packed_len(b * m * n, datatype_d)?)?;

    let mut graph = Graph::new()
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let tensor_a = graph.tensor(
        TensorSpec::new(
            datatype_a,
            Shape::contiguous([b, m, k])?.with_strides([m * k, k, 1])?,
        )
        .with_name("tensor_a"),
    );
    let tensor_b = graph.tensor(
        TensorSpec::new(
            datatype_b,
            Shape::contiguous([b, k, n])?.with_strides([k * n, 1, k])?,
        )
        .with_name("tensor_b"),
    );
    let block_descale_a = graph.tensor(
        TensorSpec::block_scale_tensor(
            datatype_block_scale,
            Shape::contiguous([b, block_scale_dim_m, block_scale_dim_k])?.with_strides([
                block_scale_dim_m * block_scale_dim_k,
                block_scale_dim_k,
                1,
            ])?,
        )
        .with_name("block_descale_a"),
    );
    let block_descale_b = graph.tensor(
        TensorSpec::block_scale_tensor(
            datatype_block_scale,
            Shape::contiguous([b, block_scale_dim_k, block_scale_dim_n])?.with_strides([
                block_scale_dim_n * block_scale_dim_k,
                1,
                block_scale_dim_k,
            ])?,
        )
        .with_name("block_descale_b"),
    );

    let dequan_tensor_a = graph.block_scale_dequantize_infer(
        tensor_a,
        block_descale_a,
        BlockScaleDequantizeConfig::new(DataType::F32, [1, block_size]),
    )?;
    let dequan_tensor_b = graph.block_scale_dequantize_infer(
        tensor_b,
        block_descale_b,
        BlockScaleDequantizeConfig::new(DataType::F32, [block_size, 1]),
    )?;

    let tensor_d = graph.tensor(
        TensorSpec::new(
            datatype_d,
            Shape::contiguous([b, m, n])?.with_strides([m * n, n, 1])?,
        )
        .with_name("tensor_d"),
    );
    graph.matmul_with_config(
        dequan_tensor_a,
        dequan_tensor_b,
        tensor_d,
        MatmulConfig::new(DataType::F32).with_name("GEMM"),
    )?;

    let compiled_graph = graph.compile_with_config(
        &ctx.cudnn,
        &CompileConfig::new()
            .with_heuristic_modes([HeuristicMode::A])
            .with_build_policy(BuildPlanPolicy::FirstSupported),
    )?;

    let mut workspace = common::workspace(compiled_graph.workspace_size()?)?;

    let mut bindings = compiled_graph.bindings();
    bindings
        .set(tensor_a, &mut tensor_a_gpu)?
        .set(tensor_b, &mut tensor_b_gpu)?
        .set(block_descale_a, &mut block_descale_a_gpu)?
        .set(block_descale_b, &mut block_descale_b_gpu)?
        .set(tensor_d, &mut tensor_d_gpu)?;

    compiled_graph.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let output_host = tensor_d_gpu.copy_to_host_vec()?;
    let output_first = output_host[0];
    let output_last = output_host[output_host.len() - 1];
    let output_checksum = output_host.iter().map(|value| value.to_f32()).sum::<f32>();
    let output_nonzero = output_host
        .iter()
        .filter(|value| value.to_f32() != 0.0)
        .count();

    assert_eq!(graph.shape(tensor_d)?.dimensions(), &[b, m, n]);
    assert_eq!(output_nonzero, 0, "output.nonzero mismatch");
    assert_eq!(output_first.to_f32(), 0.0, "output.first mismatch");
    assert_eq!(output_last.to_f32(), 0.0, "output.last mismatch");
    assert_eq!(output_checksum, 0.0, "output.checksum mismatch");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_nvfp4_matmul", run())
}
