mod common;

use std::env::args;

use singe_cuda::memory::DeviceMemory;
use singe_cudnn::{
    backend::tensor::{Shape, TensorSpec},
    data_type::{DataType, bf16, f16},
    error::{Error, Result},
    frontend::{
        graph::{
            BlockScaleDequantizeInputs, DataTypePolicy, Graph, GraphConfig,
            MoeGroupedMatmulBackwardInputs, MoeGroupedMatmulBackwardTensors,
            MoeGroupedMatmulInputs, TensorOutputConfig,
        },
        operation::{
            BlockScaleDequantizeConfig, MoeGroupedMatmulBackwardConfig, MoeGroupedMatmulConfig,
            MoeGroupedMatmulRouting,
        },
        plan::{CompileConfig, ExecuteArgs, HeuristicMode},
    },
    version,
};

fn named_tensor(name: &str, data_type: DataType, shape: Shape) -> TensorSpec {
    TensorSpec::new(data_type, shape).with_name(name)
}

fn offsets_for_runtime(mut starts: Vec<i32>, end: i32) -> Result<Vec<i32>> {
    if version()?.raw() >= 92800 {
        starts.push(end);
    }
    Ok(starts)
}

fn offset_count(offsets: &[i32]) -> Result<i64> {
    i64::try_from(offsets.len()).map_err(|_| Error::OutOfRange {
        name: "moe offset count".into(),
    })
}

fn run_forward() -> Result<()> {
    let example = common::ExampleContext::create()?;

    let batch_size = 2_i64;
    let expert_count = 3_i64;
    let top_k = 2_i32;
    let token_num = 512_i64;
    let weight_size = 256_i64;
    let hidden_size = 512_i64;
    let block_size = 128_i64;
    let expanded_top_k = i64::from(top_k);
    let routed_capacity =
        i32::try_from(batch_size * token_num * expanded_top_k).map_err(|_| Error::OutOfRange {
            name: "moe routed capacity".into(),
        })?;
    let offsets = offsets_for_runtime(vec![0, 128, 512, 768, 1152, 1536], routed_capacity)?;

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
        Shape::with_strides(
            vec![1, batch_size * token_num * expanded_top_k, hidden_size],
            vec![
                batch_size * token_num * expanded_top_k * hidden_size,
                hidden_size,
                1,
            ],
        )?,
    ));
    let weight = graph.tensor(named_tensor(
        "weight",
        DataType::I4,
        Shape::with_strides(
            [expert_count, hidden_size, weight_size],
            [hidden_size * weight_size, 1, hidden_size],
        )?,
    ));
    let block_scale = graph.tensor(named_tensor(
        "block_scale",
        DataType::F16,
        Shape::with_strides(
            [expert_count, hidden_size / block_size, weight_size],
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
        Shape::with_strides([offset_count(&offsets)?, 1, 1], [1, 1, 1])?,
    ));

    let dequantized_weight = graph.block_scale_dequantize(
        BlockScaleDequantizeInputs::new(weight, block_scale),
        BlockScaleDequantizeConfig::new(vec![block_size as i32, 1])
            .with_compute_type(DataType::F16),
    )?;
    let output = graph.moe_grouped_matmul(
        MoeGroupedMatmulInputs::new(token, dequantized_weight, first_token_offset),
        MoeGroupedMatmulConfig::new(MoeGroupedMatmulRouting::None).with_name("experts"),
        TensorOutputConfig::new().output_tensor(),
    )?;

    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;

    let token_elements = graph.shape(token)?.storage_span()?;
    let weight_bytes = graph.tensor_spec(weight)?.storage_byte_len(None)?;
    let block_scale_elements = graph.shape(block_scale)?.storage_span()?;
    let output_elements = graph.shape(output)?.storage_span()?;

    let token_dev = DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), token_elements)?;
    let weight_dev = DeviceMemory::<u8>::zeroed(example.cudnn.cuda_context(), weight_bytes)?;
    let block_scale_dev =
        DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), block_scale_elements)?;
    let first_token_offset_dev = DeviceMemory::from_slice(example.cudnn.cuda_context(), &offsets)?;
    let mut output_dev =
        DeviceMemory::<f16>::zeroed(example.cudnn.cuda_context(), output_elements)?;
    let mut workspace = common::workspace(example.cudnn.cuda_context(), compiled.workspace_size())?;

    let mut bindings = compiled.bindings()?;
    bindings.set_input(token, &token_dev.view())?;
    bindings.set_encoded_input_bytes(weight, &weight_dev.view())?;
    bindings.set_input(block_scale, &block_scale_dev.view())?;
    bindings.set_input(first_token_offset, &first_token_offset_dev.view())?;
    bindings.set(output, &mut output_dev)?;

    // SAFETY: initialized encoded inputs, writable output and workspace remain live through stream completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn run_backward() -> Result<()> {
    let example = common::ExampleContext::create()?;
    let experts = 36_i64;
    let token_count = 2000_i64;
    let hidden_size = 520_i64;
    let output_size = 248_i64;
    let offsets = offsets_for_runtime(
        vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 127, 255, 383, 483,
            515, 643, 718, 924, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
        ],
        2000,
    )?;
    let (offset_type, offset_bytes) = if version()?.raw() >= 92800 {
        (
            DataType::I64,
            offsets
                .iter()
                .flat_map(|&offset| i64::from(offset).to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    } else {
        (
            DataType::I32,
            offsets
                .iter()
                .flat_map(|offset| offset.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
    };
    let mut graph = Graph::with_config(
        GraphConfig::new().with_data_type_policy(
            DataTypePolicy::new()
                .with_io(DataType::BF16)
                .with_compute(DataType::F32),
        ),
    );
    let token_spec = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, token_count, hidden_size])?,
    );
    let gradient_spec = TensorSpec::new(
        DataType::BF16,
        Shape::contiguous([1, token_count, output_size])?,
    );
    let token = graph.tensor(token_spec.clone().with_name("token"));
    let gradient = graph.tensor(gradient_spec.clone().with_name("doutput"));
    let offset_entries = offset_count(&offsets)?;
    let first_offset = graph.tensor(named_tensor(
        "first_token_offset",
        offset_type,
        Shape::with_strides([offset_entries, 1, 1], [1, 1, 1])?,
    ));
    let inputs = MoeGroupedMatmulBackwardInputs::new(gradient, token, first_offset);
    let config = MoeGroupedMatmulBackwardConfig::new()
        .with_name("expert_gradient")
        .with_expert_count((offset_entries != experts).then_some(experts));
    let weight_gradient = if args().any(|argument| argument == "--row-major-gradient") {
        let output = graph.tensor(TensorSpec::new(
            DataType::BF16,
            Shape::contiguous([experts, hidden_size, output_size])?,
        ));
        graph.moe_grouped_matmul_backward_into(
            MoeGroupedMatmulBackwardTensors::new(inputs, output),
            config,
        )?;
        output
    } else {
        graph.moe_grouped_matmul_backward(
            inputs,
            config,
            TensorOutputConfig::new().output_tensor(),
        )?
    };
    let weight_spec = graph.tensor_spec(weight_gradient)?;
    println!(
        "dweight: name={:?}, dimensions={:?}, strides={:?}, offset encoding={offset_type}",
        weight_spec.name,
        weight_spec.shape.dimensions(),
        weight_spec.shape.strides(),
    );
    let cuda = example.cudnn.cuda_context();
    let token_device = DeviceMemory::<bf16>::zeroed(cuda, token_spec.shape.storage_span()?)?;
    let gradient_device = DeviceMemory::<bf16>::zeroed(cuda, gradient_spec.shape.storage_span()?)?;
    let offset_device = DeviceMemory::from_slice(cuda, &offset_bytes)?;
    let mut weight_device = DeviceMemory::<bf16>::zeroed(cuda, weight_spec.shape.storage_span()?)?;
    let compiled = graph.compile(
        Some(&example.cudnn),
        &CompileConfig::from_heuristic_modes([HeuristicMode::A]),
    )?;
    let mut workspace = common::workspace(cuda, compiled.workspace_size())?;
    let mut bindings = compiled.bindings()?;
    bindings.set_input(token, &token_device.view())?;
    bindings.set_input(gradient, &gradient_device.view())?;
    bindings.set_encoded_input_bytes(first_offset, &offset_device.view())?;
    bindings.set(weight_gradient, &mut weight_device)?;
    // SAFETY: native encodings match the graph, offsets describe the routed rows, and owners remain live through completion.
    example.execute_and_wait(|context| unsafe {
        compiled.execute(
            context,
            ExecuteArgs::new(&bindings).with_workspace(&mut workspace),
        )
    })?;
    Ok(())
}

fn main() -> Result<()> {
    let result = if args().any(|argument| argument == "--backward") {
        run_backward()
    } else {
        run_forward()
    };
    common::finish("frontend_moe_grouped_matmul", result)
}
