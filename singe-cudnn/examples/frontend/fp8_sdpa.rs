mod common;

use singe_core::assert_close;
use singe_cuda::{
    memory::DeviceMemory,
    types::{DevicePtr, f8e4m3},
};
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    cudart_version,
    data_type::DataType,
    error::Result,
    frontend::{
        composite::sdpa::SdpaFp8Inputs,
        graph::Graph,
        operation::{HeuristicMode, SdpaConfig},
    },
    version,
};

fn init_i8_image(size: usize) -> Vec<f8e4m3> {
    let mut seed = 123_456_789_u32;
    let mut values = Vec::with_capacity(size);
    for _ in 0..size {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let sample = (seed as f32) * 2.328_306_4e-10_f32;
        let value = 2_i8 - (5.0_f32 * sample) as i8;
        values.push(f8e4m3::from_bits(value as u8));
    }
    values
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    if cudart_version()? < 12_000 {
        println!("frontend_fp8_sdpa: skipped, requires CUDA toolkit 12.0+");
        return Ok(());
    }
    if !ctx.is_hopper_device() && !ctx.is_blackwell_device() {
        println!("frontend_fp8_sdpa: skipped, requires Hopper or Blackwell Computing GPU");
        return Ok(());
    }
    if version()? < 90_100 {
        println!("frontend_fp8_sdpa: skipped, requires cuDNN 9.1.0+");
        return Ok(());
    }

    let b = 2_i64;
    let h = 2_i64;
    let s = 512_i64;
    let d = 128_i64;
    let generate_stats = false;
    let attention_scale = 0.123_f32;

    let mut mha_graph = Graph::new()
        .with_io_data_type(DataType::F8E4M3)
        .with_intermediate_data_type(DataType::F32)
        .with_compute_data_type(DataType::F32);

    let qkvo_dims = vec![b, h, s, d];
    let qkv_strides = vec![s * 3 * h * d, d, 3 * h * d, 1];
    let o_strides = vec![s * h * d, d, h * d, 1];

    let q = mha_graph.tensor(
        TensorSpec::new(
            DataType::F8E4M3,
            Shape::contiguous(&qkvo_dims)?.with_strides(&qkv_strides)?,
        )
        .with_name("Q"),
    );
    let k = mha_graph.tensor_like_named(q, "K")?;
    let v = mha_graph.tensor_like_named(q, "V")?;

    let descale_q = mha_graph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([1, 1, 1, 1])?).with_name("Descale_Q"),
    );
    let descale_k = mha_graph.tensor_like_named(descale_q, "Descale_K")?;
    let descale_v = mha_graph.tensor_like_named(descale_q, "Descale_V")?;
    let descale_s = mha_graph.tensor_like_named(descale_q, "Descale_S")?;
    let scale_s = mha_graph.tensor_like_named(descale_q, "Scale_S")?;
    let scale_o = mha_graph.tensor_like_named(descale_q, "Scale_O")?;

    let mut sdpa_fp8_options = SdpaConfig::new()
        .with_name("sdpa_fp8")
        .with_attention_scale(attention_scale)
        .with_absolute_max_s()
        .with_absolute_max_o()
        .with_causal_mask();
    if generate_stats {
        sdpa_fp8_options = sdpa_fp8_options.with_stats();
    }

    let outputs = mha_graph.sdpa_fp8_infer(
        SdpaFp8Inputs::new(
            q, k, v, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
        ),
        &sdpa_fp8_options,
    )?;
    let o = outputs.output;
    let stats = outputs.stats;
    let absolute_max_s = outputs.logit_max;
    let absolute_max_o = outputs.score_sum_exp;
    let mut o_tensor = mha_graph.tensor_config(o)?.clone();
    o_tensor = o_tensor
        .output_tensor()
        .with_shape(Shape::contiguous(&qkvo_dims)?.with_strides(&o_strides)?);
    mha_graph.replace_tensor(o, o_tensor)?;

    let absolute_max_shape = Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?;
    let absolute_max_s = absolute_max_s.expect("absolute_max_s requested");
    let mut absolute_max_s_output = mha_graph.tensor_config(absolute_max_s)?.clone();
    absolute_max_s_output = absolute_max_s_output
        .output_tensor()
        .with_shape(absolute_max_shape.clone());
    mha_graph.replace_tensor(absolute_max_s, absolute_max_s_output)?;

    let absolute_max_o = absolute_max_o.expect("absolute_max_o requested");
    let mut absolute_max_o_output = mha_graph.tensor_config(absolute_max_o)?.clone();
    absolute_max_o_output = absolute_max_o_output
        .output_tensor()
        .with_shape(absolute_max_shape);
    mha_graph.replace_tensor(absolute_max_o, absolute_max_o_output)?;

    if generate_stats {
        let stats = stats.expect("stats requested");
        mha_graph.mark_as_output(stats)?;
    } else {
        assert!(stats.is_none());
    }

    assert_eq!(mha_graph.shape(o)?.dimensions(), qkvo_dims.as_slice());
    assert_eq!(mha_graph.shape(o)?.strides(), o_strides.as_slice());

    let compiled_mha_graph = mha_graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let qkv_tensor_host = init_i8_image((b * s * 3 * h * d) as usize);
    let qkv_tensor = DeviceMemory::from_slice(&qkv_tensor_host)?;
    let q_ptr = DevicePtr::from_raw(qkv_tensor.as_mut_ptr().cast());
    let k_ptr = DevicePtr::from_raw(
        q_ptr
            .as_raw()
            .cast::<u8>()
            .wrapping_add((h * d) as usize)
            .cast(),
    );
    let v_ptr = DevicePtr::from_raw(
        q_ptr
            .as_raw()
            .cast::<u8>()
            .wrapping_add((2 * h * d) as usize)
            .cast(),
    );

    let mut o_dev = DeviceMemory::<f8e4m3>::zeroes((b * s * h * d) as usize)?;
    let mut descale_q_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_k_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_v_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut descale_s_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut scale_s_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut scale_o_dev = DeviceMemory::from_slice(&[1.0_f32])?;
    let mut absolute_max_s_tensor = DeviceMemory::<f32>::zeroes(1)?;
    let mut absolute_max_o_tensor = DeviceMemory::<f32>::zeroes(1)?;
    let mut workspace = common::workspace(compiled_mha_graph.workspace_size()?)?;
    let mut bindings = compiled_mha_graph.bindings();
    unsafe {
        bindings.set_ptr(q, q_ptr)?;
        bindings.set_ptr(k, k_ptr)?;
        bindings.set_ptr(v, v_ptr)?;
    }
    bindings
        .set(descale_q, &mut descale_q_dev)?
        .set(descale_k, &mut descale_k_dev)?
        .set(descale_v, &mut descale_v_dev)?
        .set(descale_s, &mut descale_s_dev)?
        .set(scale_s, &mut scale_s_dev)?
        .set(scale_o, &mut scale_o_dev)?
        .set(o, &mut o_dev)?
        .set(absolute_max_s, &mut absolute_max_s_tensor)?
        .set(absolute_max_o, &mut absolute_max_o_tensor)?;
    compiled_mha_graph.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    let o_host = o_dev.copy_to_host_vec()?;
    let absolute_max_s_host = absolute_max_s_tensor.copy_to_host_vec()?;
    let absolute_max_o_host = absolute_max_o_tensor.copy_to_host_vec()?;

    let o_first = o_host[0].to_bits();
    let o_last = o_host[o_host.len() - 1].to_bits();
    let o_checksum = o_host
        .iter()
        .map(|value| u64::from(value.to_bits()))
        .sum::<u64>();

    println!(
        "o_bits: first={o_first}, last={o_last}, checksum={o_checksum}, len={}",
        o_host.len()
    );
    println!(
        "absolute_max_s: first={}, last={}, checksum={}, nonfinite=0, len=1",
        absolute_max_s_host[0], absolute_max_s_host[0], absolute_max_s_host[0]
    );
    println!(
        "absolute_max_o: first={}, last={}, checksum={}, nonfinite=0, len=1",
        absolute_max_o_host[0], absolute_max_o_host[0], absolute_max_o_host[0]
    );

    assert_eq!(o_first, 127, "o_bits.first mismatch");
    assert_eq!(o_last, 127, "o_bits.last mismatch");
    assert_eq!(o_checksum, 33_292_288, "o_bits.checksum mismatch");
    assert_close!(absolute_max_s_host[0], 0.0, 1.0e-6, "absolute_max_s");
    assert_close!(absolute_max_o_host[0], 0.0, 1.0e-6, "absolute_max_o");

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp8_sdpa", run())
}
