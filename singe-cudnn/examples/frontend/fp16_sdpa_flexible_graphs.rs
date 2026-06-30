mod common;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::pointwise::PointwiseMode,
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, f16},
    error::Result,
    frontend::{
        composite::sdpa::SdpaInputs,
        graph::{DataTypePolicy, Graph, GraphConfig},
        operation::{
            AttentionConfig, AttentionScoreConfig, HeuristicMode, PointwiseOperation,
            SdpaAuxOutputRequest, SdpaScoreSubgraph,
        },
    },
    math::NanPropagation,
};

fn init_f16_image(size: usize, seed: &mut u32) -> Vec<f16> {
    common::init_f32_image(size, seed)
        .into_iter()
        .map(f16::from_f32)
        .collect()
}

fn run() -> Result<()> {
    let ctx = common::ExampleContext::create()?;

    let b = 16_i64;
    let h_q = 32_i64;
    let h_k = 32_i64;
    let h_v = 32_i64;
    let s_q = 2048_i64;
    let s_kv = 2048_i64;
    let d_qk = 128_i64;
    let d_v = 128_i64;

    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::F16)
                .with_intermediate(DataType::F32)
                .with_compute(DataType::F32),
        ),
    );

    let q = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b, h_q, s_q, d_qk])?.with_strides(vec![
                h_q * s_q * d_qk,
                s_q * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1001),
    );
    let k = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b, h_k, s_kv, d_qk])?.with_strides(vec![
                h_k * s_kv * d_qk,
                s_kv * d_qk,
                d_qk,
                1,
            ])?,
        )
        .with_id(1002),
    );
    let v = graph.tensor(
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b, h_v, s_kv, d_v])?.with_strides(vec![
                h_v * s_kv * d_v,
                s_kv * d_v,
                d_v,
                1,
            ])?,
        )
        .with_id(1003),
    );
    let scale = graph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([1, 1, 1, 1])?.with_strides([1, 1, 1, 1])?,
    ));
    let bias = graph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([b, 1, s_q, s_kv])?.with_strides(vec![
                s_q * s_kv,
                s_q * s_kv,
                s_kv,
                1,
            ])?,
        )
        .with_id(1006),
    );

    let mut score_subgraph = Graph::new();
    let score_input = score_subgraph.tensor(TensorSpec::new(
        DataType::F32,
        Shape::contiguous([b, h_q, s_q, s_kv])?,
    ));
    let captured_bias = score_subgraph.tensor(
        TensorSpec::new(
            DataType::F32,
            Shape::contiguous([b, 1, s_q, s_kv])?.with_strides(vec![
                s_q * s_kv,
                s_q * s_kv,
                s_kv,
                1,
            ])?,
        )
        .with_id(1006),
    );
    let biased = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Add,
        lhs: score_input,
        rhs: captured_bias,
        output: biased,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let soft_cap = score_subgraph.tensor(TensorSpec::scalar_f32(0.8)?);
    let divided = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Div,
        lhs: biased,
        rhs: soft_cap,
        output: divided,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });
    let activated = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise(PointwiseOperation::Unary {
        mode: PointwiseMode::TanhFwd,
        input: divided,
        output: activated,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        axis: None,
    });
    let score_output = score_subgraph.tensor(
        TensorSpec::new(DataType::F32, Shape::contiguous([b, h_q, s_q, s_kv])?).virtual_tensor(),
    );
    score_subgraph.pointwise(PointwiseOperation::Binary {
        mode: PointwiseMode::Mul,
        lhs: activated,
        rhs: soft_cap,
        output: score_output,
        compute_type: DataType::F32,
        nan_propagation: NanPropagation::Propagate,
        alpha1: 1.0,
        alpha2: 1.0,
    });

    let outputs = graph.sdpa_auto_infer(
        SdpaInputs {
            query: q,
            key: k,
            value: v,
            scale: scale,
        },
        AttentionConfig::new(DataType::F32)
            .with_aux_outputs(SdpaAuxOutputRequest::stats_only())
            .with_score_config(
                AttentionScoreConfig::new()
                    .with_bias(bias)
                    .with_score_subgraph(SdpaScoreSubgraph::new(
                        score_subgraph,
                        score_input,
                        score_output,
                    )),
            ),
    )?;
    let o = outputs.output();
    let stats = outputs.stats();

    graph.replace_tensor(
        o,
        TensorSpec::new(
            DataType::F16,
            Shape::contiguous([b, h_q, s_q, d_v])?.with_strides(vec![
                h_q * d_v,
                d_v,
                b * h_q * d_v,
                1,
            ])?,
        )
        .output_tensor()
        .with_id(1004),
    )?;
    let stats = stats.expect("stats requested");
    let stats_tensor = graph
        .tensor_config(stats)?
        .clone()
        .output_tensor()
        .with_id(1005);
    graph.replace_tensor(stats, stats_tensor)?;
    let compiled = graph.compile(&ctx.cudnn, &[HeuristicMode::A])?;

    let mut seed = 123_456_789_u32;
    let mut q_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h_q * s_q * d_qk) as usize, &mut seed))?;
    let mut k_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h_k * s_kv * d_qk) as usize, &mut seed))?;
    let mut v_dev =
        DeviceMemory::from_slice(&init_f16_image((b * h_v * s_kv * d_v) as usize, &mut seed))?;
    let mut scale_dev = DeviceMemory::from_slice(&[0.123_f32])?;
    let mut bias_dev = DeviceMemory::from_slice(&common::init_f32_image(
        (b * s_q * s_kv) as usize,
        &mut seed,
    ))?;
    let mut o_dev = DeviceMemory::<f16>::zeroes((b * h_q * s_q * d_v) as usize)?;
    let mut stats_dev = DeviceMemory::<f32>::zeroes((b * h_q * s_q) as usize)?;
    let mut workspace = common::workspace(compiled.workspace_size()?)?;

    let mut bindings = compiled.bindings();
    bindings.set(q, &mut q_dev)?;
    bindings.set(k, &mut k_dev)?;
    bindings.set(v, &mut v_dev)?;
    bindings.set(scale, &mut scale_dev)?;
    bindings.set(bias, &mut bias_dev)?;
    bindings.set(o, &mut o_dev)?;
    bindings.set(stats, &mut stats_dev)?;
    compiled.execute(&ctx.cudnn, &bindings, Some(&mut workspace))?;

    Ok(())
}

fn main() -> Result<()> {
    common::finish("frontend_fp16_sdpa_flexible_graphs", run())
}
