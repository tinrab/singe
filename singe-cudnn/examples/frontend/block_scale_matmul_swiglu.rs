mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorSpec},
    },
    data_type::DataType,
    error::Result,
    frontend::{
        graph::{
            BlockScaleDequantizeInputs, DataTypePolicy, Graph, GraphConfig, TensorOutputConfig,
        },
        operation::{BlockScaleDequantizeConfig, MatmulConfig, PointwiseOperation},
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
};

fn run() -> Result<()> {
    let example = common::ExampleContext::create()?;
    if example.compute_capability()? < 100 {
        println!("frontend_block_scale_matmul_swiglu: skipped, requires Blackwell or newer");
        return Ok(());
    }
    let [b, m, n, k] = [1_i64, 256, 256, 256];
    let block_size = 16_i32;
    let block = i64::from(block_size);
    let block_scale_dim_m = common::round_up(m, 128)?;
    let block_scale_dim_n = common::round_up(n, 128)?;
    let block_scale_dim_k = common::round_up(k / block + i64::from(k % block != 0), 4)?;
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );
    let tensor_a = graph.tensor(
        TensorSpec::new(DataType::F4E2M1, Shape::contiguous([b, m, k])?).with_name("tensor_a"),
    );
    let tensor_b0 = graph.tensor(
        TensorSpec::new(DataType::F4E2M1, common::column_major_shape([b, k, n])?)
            .with_name("tensor_b0"),
    );
    let tensor_b1 = graph.tensor_like(tensor_b0, Some("tensor_b1"))?;
    let block_descale_a = graph.tensor(
        TensorSpec::new(
            DataType::F8E4M3,
            Shape::contiguous([b, block_scale_dim_m, block_scale_dim_k])?,
        )
        .with_f8_128x4_reordering()
        .with_name("block_descale_a"),
    );
    let block_descale_b0 = graph.tensor(
        TensorSpec::new(
            DataType::F8E4M3,
            common::column_major_shape([b, block_scale_dim_k, block_scale_dim_n])?,
        )
        .with_f8_128x4_reordering()
        .with_name("block_descale_b0"),
    );
    let block_descale_b1 = graph.tensor_like(block_descale_b0, Some("block_descale_b1"))?;
    let dequant_a = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_a, block_descale_a),
        BlockScaleDequantizeConfig::new([1, block_size]).with_name("dequantize_a"),
    )?;
    let dequant_b0 = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_b0, block_descale_b0),
        BlockScaleDequantizeConfig::new([block_size, 1]).with_name("dequantize_b0"),
    )?;
    let dequant_b1 = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(tensor_b1, block_descale_b1),
        BlockScaleDequantizeConfig::new([block_size, 1]).with_name("dequantize_b1"),
    )?;
    let projection0 = graph.matmul(
        dequant_a,
        dequant_b0,
        MatmulConfig::new().with_name("matmul0"),
        TensorOutputConfig::new().virtual_tensor(),
    )?;
    let projection1 = graph.matmul(
        dequant_a,
        dequant_b1,
        MatmulConfig::new().with_name("matmul1"),
        TensorOutputConfig::new().virtual_tensor(),
    )?;
    let after_swish = graph.pointwise(
        PointwiseOperation::unary(
            PointwiseMode::SwishFwd,
            projection0,
            TensorOutputConfig::new().virtual_tensor(),
        )
        .with_name("swish"),
    )?;
    let output = graph.pointwise(
        PointwiseOperation::binary(
            PointwiseMode::Mul,
            after_swish,
            projection1,
            TensorOutputConfig::new()
                .with_data_type(DataType::F4E2M1)
                .output_tensor(),
        )
        .with_name("mul"),
    )?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A, HeuristicMode::Fallback]),
    )?;
    let cuda = example.cudnn.cuda_context();
    let mut inputs = Vec::with_capacity(6);
    for (id, value) in [
        (tensor_a, 0x22_u8),
        (tensor_b0, 0x22),
        (tensor_b1, 0x22),
        (block_descale_a, 0x18),
        (block_descale_b0, 0x18),
        (block_descale_b1, 0x18),
    ] {
        // Packed data holds E2M1 ones; small uniform E4M3 scales are invariant under F8_128x4 reordering.
        let values = vec![value; graph.tensor_spec(id)?.storage_byte_len(None)?];
        inputs.push((id, DeviceMemory::from_slice(cuda, &values)?));
    }
    let mut output_device =
        DeviceMemory::<u8>::zeroed(cuda, graph.tensor_spec(output)?.storage_byte_len(None)?)?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    for (id, memory) in &inputs {
        bindings.set_encoded_input_bytes(*id, memory)?;
    }
    bindings.set_encoded_bytes(output, &mut output_device)?;
    // SAFETY: initialized packed inputs/scales, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    println!("executed block-scale matmul with swiglu and packed fp4 output");
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_block_scale_matmul_swiglu", run())
}
