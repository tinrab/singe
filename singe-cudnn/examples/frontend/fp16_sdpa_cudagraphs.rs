mod common;

use singe_cuda::{graph::Graph as CudaGraph, memory::DeviceMemory};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16},
    error::{Error, Result, Status},
    frontend::{
        composite::sdpa::{SdpaInputs, SdpaOutputs},
        graph::Graph,
        operation::{AttentionConfig, HeuristicMode},
    },
};

fn init_bf16_image(size: usize, seed: &mut u32) -> Vec<bf16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(bf16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 3_i64;
    let h = 4_i64;
    let s_q = 1024_i64;
    let s_kv = 1024_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

    let mut graph = Graph::new()
        .with_io_data_type(DataType::BF16)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let q = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_q, d_qk])?).with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_kv, d_qk])?).with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(DataType::BF16, Shape::contiguous([b, h, s_kv, d_v])?).with_id(1003),
    );
    let scale = graph.tensor(TensorSpec::scalar_f32(0.123)?);
    let sequence_length_query = graph
        .tensor(TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?).with_id(1007));
    let sequence_length_key_value = graph
        .tensor(TensorSpec::new(DataType::I32, Shape::contiguous([b, 1, 1, 1])?).with_id(1008));

    let SdpaOutputs { output: o, stats } = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_stats()
            .with_causal_mask()
            .with_padding_mask(sequence_length_query, sequence_length_key_value),
    )?;

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([b, h, s_q, d_v])?.with_strides(vec![
                h * d_v,
                d_v,
                b * h * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(1004),
    )?;
    let stats = stats.expect("stats");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(1005);
    graph.replace_tensor(stats, stats_tensor)?;

    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h * s_kv * d_qk) as usize, &mut seed))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_bf16_image((b * h * s_kv * d_v) as usize, &mut seed))?;
    let mut seq_len_q_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut seq_len_kv_dev = DeviceMemory::from_slice(&vec![20_i32; b as usize])?;
    let mut o_dev = DeviceMemory::<bf16>::zeroes((b * h * s_q * d_v) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(sequence_length_query, &mut seq_len_q_dev)?;
    bindings.set(sequence_length_key_value, &mut seq_len_kv_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;

    let mut cuda_graph = CudaGraph::create()?;
    match compiled.populate_cuda_graph(&ctx.cudnn, &bindings, Some(&mut workspace), &mut cuda_graph)
    {
        Ok(()) => {}
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            return Ok(());
        }
        Err(error) => return Err(error),
    }

    match compiled.update_cuda_graph(&ctx.cudnn, &bindings, Some(&mut workspace), &cuda_graph) {
        Ok(()) => Ok(()),
        Err(Error::Cudnn { code, .. })
            if code == Status::NotSupported || code == Status::NotSupportedCudaGraphNativeApi =>
        {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_cudagraphs", run())
}
