mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16, f16},
    error::{Error, Result},
    frontend::{
        graph::{
            BlockScaleDequantizeInputs, DataTypePolicy, Graph, GraphConfig, TensorOutputConfig,
        },
        operation::{BlockScaleDequantizeConfig, MatmulConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

#[derive(Clone, Copy)]
struct MatmulProblem {
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
    after_dequant_datatype_b: DataType,
    datatype_d: DataType,
    compute_math_precision: DataType,
}

fn encoded_ones(data_type: DataType, bytes: usize) -> Result<Vec<u8>> {
    let pattern = match data_type {
        DataType::I4 => vec![0x11],
        DataType::F16 => f16::from_f32(1.0).to_bits().to_le_bytes().to_vec(),
        DataType::BF16 => bf16::from_f32(1.0).to_bits().to_le_bytes().to_vec(),
        DataType::F32 => 1.0_f32.to_le_bytes().to_vec(),
        _ => return Err(Error::UnsupportedDataType),
    };
    Ok(pattern.into_iter().cycle().take(bytes).collect())
}

fn run_problem(example: &common::ExampleContext, index: usize, case: MatmulProblem) -> Result<()> {
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let tensor_a = graph.tensor(
        TensorSpec::new(
            case.datatype_a,
            Shape::contiguous([case.b, case.m, case.k])?,
        )
        .with_name("tensor_a"),
    );
    let tensor_b = graph.tensor(
        TensorSpec::new(
            case.datatype_b,
            common::column_major_shape([case.b, case.k, case.n])?,
        )
        .with_name("tensor_b"),
    );
    let [
        block_scale_dim_a_m,
        block_scale_dim_a_k,
        block_scale_dim_b_k,
        block_scale_dim_b_n,
    ] = [
        (case.m, case.block_size_a_m),
        (case.k, case.block_size_a_k),
        (case.k, case.block_size_b_k),
        (case.n, case.block_size_b_n),
    ]
    .map(|(dimension, block)| {
        let block = i64::from(block);
        dimension / block + i64::from(dimension % block != 0)
    });
    let block_descale_a = graph.tensor(
        TensorSpec::new(
            case.datatype_block_scale_a,
            Shape::contiguous([case.b, block_scale_dim_a_m, block_scale_dim_a_k])?,
        )
        .with_name("block_descale_a"),
    );
    let block_descale_b = graph.tensor(
        TensorSpec::new(
            case.datatype_block_scale_b,
            common::column_major_shape([case.b, block_scale_dim_b_k, block_scale_dim_b_n])?,
        )
        .with_name("block_descale_b"),
    );

    let dequant_tensor_a = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_a, block_descale_a),
        BlockScaleDequantizeConfig::new(vec![case.block_size_a_m, case.block_size_a_k])
            .with_name("dequantize_a")
            .with_compute_type(case.compute_math_precision)
            .with_output_data_type(case.after_dequant_datatype_a),
    )?;
    let dequant_tensor_b = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_b, block_descale_b),
        BlockScaleDequantizeConfig::new(vec![case.block_size_b_k, case.block_size_b_n])
            .with_name("dequantize_b")
            .with_compute_type(case.compute_math_precision)
            .with_output_data_type(case.after_dequant_datatype_b),
    )?;

    let tensor_d = graph.matmul(
        dequant_tensor_a,
        dequant_tensor_b,
        MatmulConfig::new()
            .with_compute_type(case.compute_math_precision)
            .with_name("GEMM"),
        TensorOutputConfig::new()
            .with_data_type(case.datatype_d)
            .output_tensor(),
    )?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut inputs = Vec::with_capacity(4);
    for tensor in [tensor_a, tensor_b, block_descale_a, block_descale_b] {
        let spec = graph.tensor_spec(tensor)?;
        let values = encoded_ones(spec.data_type, spec.storage_byte_len(None)?)?;
        inputs.push((tensor, DeviceMemory::from_slice(cuda, &values)?));
    }
    let mut tensor_d_dev =
        DeviceMemory::<u8>::zeroed(cuda, graph.tensor_spec(tensor_d)?.storage_byte_len(None)?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    // The selected data types determine the native encoding, including sub-byte packing.
    for (tensor, memory) in &inputs {
        bindings.set_encoded_input_bytes(*tensor, memory)?;
    }
    bindings.set_encoded_bytes(tensor_d, &mut tensor_d_dev)?;
    // SAFETY: the plan and all allocations remain live until stream completion,
    // including when native execution returns an enqueue error.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;

    println!(
        "executed block-scale matmul problem {index} with output shape: {:?}",
        graph.shape(tensor_d)?.dimensions()
    );

    Ok(())
}

fn run() -> Result<()> {
    let context = common::ExampleContext::create()?;
    let cases = [
        MatmulProblem {
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
            after_dequant_datatype_b: DataType::F16,
            datatype_d: DataType::F16,
            compute_math_precision: DataType::F16,
        },
        MatmulProblem {
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
            after_dequant_datatype_b: DataType::F16,
            datatype_d: DataType::F16,
            compute_math_precision: DataType::F16,
        },
        MatmulProblem {
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
            after_dequant_datatype_b: DataType::BF16,
            datatype_d: DataType::F32,
            compute_math_precision: DataType::F32,
        },
    ];

    for (index, case) in cases.into_iter().enumerate() {
        run_problem(&context, index, case)?;
    }

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_general_block_scale_matmul", run())
}
