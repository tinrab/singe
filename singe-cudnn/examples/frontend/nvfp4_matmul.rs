mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        error::Error as FrontendError,
        graph::{
            BlockScaleDequantizeInputs, DataTypePolicy, Graph, GraphConfig, TensorOutputConfig,
        },
        operation::{BlockScaleDequantizeConfig, MatmulConfig},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

const SCALE_TILE_ROWS: i64 = 128;
const SCALE_TILE_COLUMNS: i64 = 4;

#[derive(Clone, Copy)]
enum BlockFormat {
    Nvfp4,
    Mxfp8,
    Mxfp8E4M3E5M2,
    Mxfp8E5M2E4M3,
}

fn encoded_one(data_type: DataType) -> Result<u8> {
    Ok(match data_type {
        DataType::F4E2M1 => 0x22, // Two E2M1 ones per packed byte.
        DataType::F8E4M3 => 0x38,
        DataType::F8E5M2 => 0x3c,
        DataType::F8UE8M0 => 0x7f,
        _ => return Err(Error::UnsupportedDataType),
    })
}

fn run() -> Result<()> {
    let mut format = BlockFormat::Nvfp4;
    let mut output_type = DataType::F16;
    let mut irregular = false;
    for argument in args().skip(1) {
        match argument.as_str() {
            "--format=nvfp4" => format = BlockFormat::Nvfp4,
            "--format=mxfp8" => format = BlockFormat::Mxfp8,
            "--format=mxfp8-e4m3-e5m2" => format = BlockFormat::Mxfp8E4M3E5M2,
            "--format=mxfp8-e5m2-e4m3" => format = BlockFormat::Mxfp8E5M2E4M3,
            "--output=fp16" => output_type = DataType::F16,
            "--output=bf16" => output_type = DataType::BF16,
            "--output=fp32" => output_type = DataType::F32,
            "--irregular" => irregular = true,
            _ => {
                return Err(FrontendError::Compile(format!(
                    "unknown block-scale matmul argument: {argument}"
                ))
                .into());
            }
        }
    }
    let (datatype_a, datatype_b, datatype_block_scale, block_size) = match format {
        BlockFormat::Nvfp4 => (DataType::F4E2M1, DataType::F4E2M1, DataType::F8E4M3, 16_i32),
        BlockFormat::Mxfp8 => (DataType::F8E4M3, DataType::F8E4M3, DataType::F8UE8M0, 32),
        BlockFormat::Mxfp8E4M3E5M2 => (DataType::F8E4M3, DataType::F8E5M2, DataType::F8UE8M0, 32),
        BlockFormat::Mxfp8E5M2E4M3 => (DataType::F8E5M2, DataType::F8E4M3, DataType::F8UE8M0, 32),
    };
    let [b, m, n, k] = if irregular {
        let n = if matches!(format, BlockFormat::Nvfp4) && output_type != DataType::F32 {
            272
        } else {
            268
        };
        [3, 137, n, 160]
    } else {
        [
            1,
            128,
            128,
            if matches!(format, BlockFormat::Nvfp4) {
                64
            } else {
                128
            },
        ]
    };

    let example = common::ExampleContext::create()?;
    if example.cudnn.cuda_context().device().properties()?.major < 10 {
        println!("frontend_nvfp4_matmul: skipped, requires Blackwell or newer");
        return Ok(());
    }
    let block_scale_dim_m = common::round_up(m, SCALE_TILE_ROWS)?;
    let block_scale_dim_n = common::round_up(n, SCALE_TILE_ROWS)?;
    let block = i64::from(block_size);
    let block_scale_dim_k =
        common::round_up(k / block + i64::from(k % block != 0), SCALE_TILE_COLUMNS)?;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let tensor_a = graph
        .tensor(TensorSpec::new(datatype_a, Shape::contiguous([b, m, k])?).with_name("tensor_a"));
    let tensor_b = graph.tensor(
        TensorSpec::new(datatype_b, common::column_major_shape([b, k, n])?).with_name("tensor_b"),
    );
    let block_descale_a = graph.tensor(
        TensorSpec::new(
            datatype_block_scale,
            Shape::contiguous([b, block_scale_dim_m, block_scale_dim_k])?,
        )
        .with_f8_128x4_reordering()
        .with_name("block_descale_a"),
    );
    let block_descale_b = graph.tensor(
        TensorSpec::new(
            datatype_block_scale,
            common::column_major_shape([b, block_scale_dim_k, block_scale_dim_n])?,
        )
        .with_f8_128x4_reordering()
        .with_name("block_descale_b"),
    );
    let dequant_tensor_a = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_a, block_descale_a),
        BlockScaleDequantizeConfig::new([1, block_size]).with_name("dequantize_a"),
    )?;
    let dequant_tensor_b = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_b, block_descale_b),
        BlockScaleDequantizeConfig::new([block_size, 1]).with_name("dequantize_b"),
    )?;
    let tensor_d = graph.matmul(
        dequant_tensor_a,
        dequant_tensor_b,
        MatmulConfig::new().with_name("GEMM"),
        TensorOutputConfig::new()
            .with_data_type(output_type)
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
        // Uniform scales are invariant under the declared F8_128x4 reordering.
        let values = vec![encoded_one(spec.data_type)?; spec.storage_byte_len(None)?];
        inputs.push((tensor, DeviceMemory::<u8>::from_slice(cuda, &values)?));
    }
    let mut output =
        DeviceMemory::<u8>::zeroed(cuda, graph.tensor_spec(tensor_d)?.storage_byte_len(None)?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    for (tensor, memory) in &inputs {
        bindings.set_encoded_input_bytes(*tensor, memory)?;
    }
    bindings.set_encoded_bytes(tensor_d, &mut output)?;
    // SAFETY: initialized packed inputs/scales, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!(
        "executed {datatype_a} x {datatype_b} block-scale matmul with {output_type} output, irregular: {irregular}"
    );
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_nvfp4_matmul", run())
}
