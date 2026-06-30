mod common;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{BlockScaleDequantizeConfig, HeuristicMode},
    },
};

#[derive(Clone, Copy)]
struct TestCase {
    b: i64,
    m: i64,
    n: i64,
    k: i64,
    block_size_a_m: i32,
    block_size_a_k: i32,
    block_size_b_k: i32,
    block_size_b_n: i32,
    datatype_a: DataType,
    datatype_b: DataType,
    datatype_block_scale_a: DataType,
    datatype_block_scale_b: DataType,
    after_dequant_datatype_a: DataType,
    datatype_d: DataType,
    compute_math_precision: DataType,
}

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn packed_len(element_count: i64, data_type: DataType) -> Result<usize> {
    let element_count = usize::try_from(element_count).map_err(|_| Error::OutOfRange {
        name: "element_count".into(),
    })?;
    let bits = match data_type {
        DataType::I4 | DataType::U4 | DataType::F4E2M1 => 4,
        DataType::I8 | DataType::U8 | DataType::Boolean => 8,
        DataType::F16 | DataType::BF16 => 16,
        DataType::F32 | DataType::I32 | DataType::U32 => 32,
        _ => return Err(Error::UnsupportedDataType),
    };
    Ok(element_count.saturating_mul(bits).div_ceil(8))
}

fn run_case(ctx: &common::ExampleContext, index: usize, case: TestCase) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(case.after_dequant_datatype_a)
                .with_compute(DataType::F32),
        ),
    );

    let tensor_a = graph.tensor(named_tensor(
        "tensor_a",
        case.datatype_a,
        Shape::contiguous([case.b, case.m, case.k])?.with_strides(vec![
            case.m * case.k,
            case.k,
            1,
        ])?,
    ));
    let tensor_b = graph.tensor(named_tensor(
        "tensor_b",
        case.datatype_b,
        Shape::contiguous([case.b, case.k, case.n])?.with_strides(vec![
            case.k * case.n,
            1,
            case.k,
        ])?,
    ));

    let block_scale_dim_a_m =
        (case.m + i64::from(case.block_size_a_m) - 1) / i64::from(case.block_size_a_m);
    let block_scale_dim_a_k =
        (case.k + i64::from(case.block_size_a_k) - 1) / i64::from(case.block_size_a_k);
    let block_scale_dim_b_k =
        (case.k + i64::from(case.block_size_b_k) - 1) / i64::from(case.block_size_b_k);
    let block_scale_dim_b_n =
        (case.n + i64::from(case.block_size_b_n) - 1) / i64::from(case.block_size_b_n);

    let block_descale_a = graph.tensor(named_tensor(
        "block_descale_a",
        case.datatype_block_scale_a,
        Shape::contiguous([case.b, block_scale_dim_a_m, block_scale_dim_a_k])?.with_strides(
            vec![
                block_scale_dim_a_m * block_scale_dim_a_k,
                block_scale_dim_a_k,
                1,
            ],
        )?,
    ));
    let block_descale_b = graph.tensor(named_tensor(
        "block_descale_b",
        case.datatype_block_scale_b,
        Shape::contiguous([case.b, block_scale_dim_b_k, block_scale_dim_b_n])?.with_strides(
            vec![
                block_scale_dim_b_k * block_scale_dim_b_n,
                1,
                block_scale_dim_b_k,
            ],
        )?,
    ));

    let dequant_tensor_a = graph.block_scale_dequantize_infer(
        tensor_a,
        block_descale_a,
        BlockScaleDequantizeConfig::new(
            case.compute_math_precision,
            vec![case.block_size_a_m, case.block_size_a_k],
        ),
    )?;
    let dequant_tensor_b = graph.block_scale_dequantize_infer(
        tensor_b,
        block_descale_b,
        BlockScaleDequantizeConfig::new(
            case.compute_math_precision,
            vec![case.block_size_b_k, case.block_size_b_n],
        ),
    )?;

    let tensor_d = graph.tensor(named_tensor(
        "GEMM::C",
        case.datatype_d,
        Shape::contiguous([case.b, case.m, case.n])?.with_strides(vec![
            case.m * case.n,
            case.n,
            1,
        ])?,
    ));
    graph.matmul(
        dequant_tensor_a,
        dequant_tensor_b,
        tensor_d,
        case.compute_math_precision,
    )?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let tensor_a_dev =
        DeviceMemory::<u8>::zeroes(packed_len(case.b * case.m * case.k, case.datatype_a)?)?;
    let tensor_b_dev =
        DeviceMemory::<u8>::zeroes(packed_len(case.b * case.k * case.n, case.datatype_b)?)?;
    let block_descale_a_dev = DeviceMemory::<u8>::zeroes(packed_len(
        case.b * block_scale_dim_a_m * block_scale_dim_a_k,
        case.datatype_block_scale_a,
    )?)?;
    let block_descale_b_dev = DeviceMemory::<u8>::zeroes(packed_len(
        case.b * block_scale_dim_b_k * block_scale_dim_b_n,
        case.datatype_block_scale_b,
    )?)?;
    let tensor_d_dev =
        DeviceMemory::<u8>::zeroes(packed_len(case.b * case.m * case.n, case.datatype_d)?)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;
    let mut bindings = compiled.bindings();
    // This example is data-type-parametric and includes packed 4-bit cases, so
    // the backing buffers are byte-sized and bound as raw device pointers.
    unsafe {
        bindings.set_ptr(
            tensor_a,
            DevicePtr::from_raw(tensor_a_dev.as_mut_ptr().cast()),
        )?;
        bindings.set_ptr(
            tensor_b,
            DevicePtr::from_raw(tensor_b_dev.as_mut_ptr().cast()),
        )?;
        bindings.set_ptr(
            block_descale_a,
            DevicePtr::from_raw(block_descale_a_dev.as_mut_ptr().cast()),
        )?;
        bindings.set_ptr(
            block_descale_b,
            DevicePtr::from_raw(block_descale_b_dev.as_mut_ptr().cast()),
        )?;
        bindings.set_ptr(
            tensor_d,
            DevicePtr::from_raw(tensor_d_dev.as_mut_ptr().cast()),
        )?;
    }
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let tensor_d_host = tensor_d_dev.copy_to_host_vec()?;
    let nonzero = tensor_d_host.iter().filter(|value| **value != 0).count();
    let checksum = tensor_d_host
        .iter()
        .map(|value| u64::from(*value))
        .sum::<u64>();

    println!(
        "case {index} d_bytes: first={}, last={}, checksum={checksum}, nonzero={nonzero}, len={}",
        tensor_d_host[0],
        tensor_d_host[tensor_d_host.len() - 1],
        tensor_d_host.len()
    );

    assert_eq!(
        graph.shape(tensor_d)?.dimensions(),
        &[case.b, case.m, case.n]
    );
    assert_eq!(nonzero, 0, "d_bytes.nonzero mismatch for case {index}");
    assert_eq!(checksum, 0, "d_bytes.checksum mismatch for case {index}");

    Ok(())
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;
    let cases = [
        TestCase {
            b: 2,
            m: 512,
            n: 512,
            k: 512,
            block_size_a_m: 1,
            block_size_a_k: 128,
            block_size_b_k: 128,
            block_size_b_n: 128,
            datatype_a: DataType::F16,
            datatype_b: DataType::F16,
            datatype_block_scale_a: DataType::F16,
            datatype_block_scale_b: DataType::F16,
            after_dequant_datatype_a: DataType::F16,
            datatype_d: DataType::F16,
            compute_math_precision: DataType::F16,
        },
        TestCase {
            b: 2,
            m: 512,
            n: 512,
            k: 512,
            block_size_a_m: 1,
            block_size_a_k: 128,
            block_size_b_k: 128,
            block_size_b_n: 1,
            datatype_a: DataType::I4,
            datatype_b: DataType::F16,
            datatype_block_scale_a: DataType::F32,
            datatype_block_scale_b: DataType::F32,
            after_dequant_datatype_a: DataType::F16,
            datatype_d: DataType::F16,
            compute_math_precision: DataType::F16,
        },
        TestCase {
            b: 2,
            m: 512,
            n: 512,
            k: 512,
            block_size_a_m: 47,
            block_size_a_k: 32,
            block_size_b_k: 32,
            block_size_b_n: 17,
            datatype_a: DataType::BF16,
            datatype_b: DataType::BF16,
            datatype_block_scale_a: DataType::F32,
            datatype_block_scale_b: DataType::F32,
            after_dequant_datatype_a: DataType::BF16,
            datatype_d: DataType::F32,
            compute_math_precision: DataType::F32,
        },
    ];

    for (index, case) in cases.into_iter().enumerate() {
        run_case(&ctx, index, case)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_general_block_scale_matmul", run())
}
