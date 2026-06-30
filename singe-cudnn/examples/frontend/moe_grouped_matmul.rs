mod common;

use singe_cuda::{memory::DeviceMemory, types::DevicePtr};
use singe_cudnn::{
    backend::execution::advanced::MoeGroupedMatmulMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        graph::{DataTypePolicy, Graph, GraphConfig, MoeGroupedMatmulInputs},
        operation::{BlockScaleDequantizeConfig, HeuristicMode, MoeGroupedMatmulConfig},
    },
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let batch_size = 2_i64;
    let expert_count = 3_i64;
    let top_k = 2_i32;
    let token_num = 512_i64;
    let weight_size = 256_i64;
    let hidden_size = 512_i64;
    let block_size = 128_i64;
    let expanded_top_k = i64::from(top_k);

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_intermediate(DataType::F16)
                .with_compute(DataType::F16),
        ),
    );

    let token = graph.tensor(named_tensor(
        "token",
        DataType::F16,
        Shape::contiguous(vec![
            1,
            batch_size * token_num * expanded_top_k,
            hidden_size,
        ])?
        .with_strides(vec![
            batch_size * token_num * expanded_top_k * hidden_size,
            hidden_size,
            1,
        ])?,
    ));
    let weight = graph.tensor(named_tensor(
        "weight",
        DataType::I4,
        Shape::contiguous([expert_count, hidden_size, weight_size])?.with_strides([
            hidden_size * weight_size,
            1,
            hidden_size,
        ])?,
    ));
    let block_scale = graph.tensor(named_tensor(
        "block_scale",
        DataType::F16,
        Shape::contiguous([expert_count, hidden_size / block_size, weight_size])?.with_strides(
            vec![
                (hidden_size / block_size) * weight_size,
                1,
                hidden_size / block_size,
            ],
        )?,
    ));
    let first_token_offset = graph.tensor(named_tensor(
        "first_token_offset",
        DataType::I32,
        Shape::contiguous([batch_size * expert_count, 1, 1])?.with_strides([1, 1, 1])?,
    ));

    let dequantized_weight = graph.block_scale_dequantize_infer(
        weight,
        block_scale,
        BlockScaleDequantizeConfig::new(DataType::F16, vec![block_size as i32, 1]),
    )?;
    let output = graph.moe_grouped_matmul_infer(
        MoeGroupedMatmulInputs::new(token, dequantized_weight, first_token_offset),
        MoeGroupedMatmulConfig::new(MoeGroupedMatmulMode::None, DataType::F16).with_top_k(top_k),
    )?;
    graph.mark_as_output(output)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let token_elements = usize::try_from(batch_size * token_num * expanded_top_k * hidden_size)
        .expect("token elements");
    let weight_bytes =
        usize::try_from(expert_count * hidden_size * weight_size / 2).expect("weight bytes");
    let block_scale_elements =
        usize::try_from(expert_count * (hidden_size / block_size) * weight_size)
            .expect("block scale elements");
    let output_elements = usize::try_from(batch_size * token_num * expanded_top_k * weight_size)
        .expect("output elements");

    let mut token_dev = DeviceMemory::<f16>::zeroes(token_elements)?;
    let weight_dev = DeviceMemory::<u8>::zeroes(weight_bytes)?;
    let mut block_scale_dev = DeviceMemory::<f16>::zeroes(block_scale_elements)?;
    let mut first_token_offset_dev = DeviceMemory::from_slice(&[0_i32, 128, 512, 768, 1152, 1536])?;
    let mut output_dev = DeviceMemory::<f16>::zeroes(output_elements)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(token, &mut token_dev)?;
    // I4 weights are packed two values per byte and do not have a scalar Rust binding type, so bind the byte backing storage directly.
    unsafe {
        bindings.set_ptr(weight, DevicePtr::from_raw(weight_dev.as_mut_ptr().cast()))?;
    }
    bindings.set(block_scale, &mut block_scale_dev)?;
    bindings.set(first_token_offset, &mut first_token_offset_dev)?;
    bindings.set(output, &mut output_dev)?;

    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;
    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_moe_grouped_matmul", run())
}
