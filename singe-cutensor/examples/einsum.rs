use singe_cuda::{context::Context as CudaContext, memory::DeviceMemory, stream::Stream};
use singe_cutensor::{
    context::Context,
    error::Result,
    operation::{ComputeDescriptor, OperationDescriptor, TensorOperand},
    plan::{Plan, PlanPreference},
    tensor::TensorDescriptor,
    types::{Operator, WorkspacePreference},
};

fn modes_from_text(text: &str) -> Vec<char> {
    text.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn parse_equation(equation: &str) -> (&str, Option<&str>, Option<&str>) {
    let (inputs, output) = match equation.split_once("->") {
        Some((inputs, output)) => (inputs, Some(output)),
        None => (equation, None),
    };
    let mut input_parts = inputs.split(',');
    let lhs = input_parts.next().unwrap_or_default().trim();
    let rhs = input_parts.next().map(str::trim);
    (lhs, rhs, output.map(str::trim))
}

fn infer_output(lhs: &[char], rhs: &[char]) -> Vec<char> {
    let mut output = lhs
        .iter()
        .copied()
        .filter(|mode| !rhs.contains(mode))
        .collect::<Vec<_>>();
    output.extend(rhs.iter().copied().filter(|mode| !lhs.contains(mode)));
    output.sort_unstable();
    output
}

fn extents_for_modes(modes: &[char], labels: &[char], extents: &[u64]) -> Vec<u64> {
    modes
        .iter()
        .map(|mode| {
            let index = labels
                .iter()
                .position(|candidate| candidate == mode)
                .unwrap();
            extents[index]
        })
        .collect()
}

fn extents_for_binary_output(
    modes: &[char],
    lhs_labels: &[char],
    lhs_extents: &[u64],
    rhs_labels: &[char],
    rhs_extents: &[u64],
) -> Vec<u64> {
    modes
        .iter()
        .map(|mode| {
            if let Some(index) = lhs_labels.iter().position(|candidate| candidate == mode) {
                lhs_extents[index]
            } else {
                let index = rhs_labels
                    .iter()
                    .position(|candidate| candidate == mode)
                    .unwrap();
                rhs_extents[index]
            }
        })
        .collect()
}

fn run_binary_einsum(
    context: &Context,
    stream: &Stream,
    equation: &str,
    lhs_shape: &[u64],
    rhs_shape: &[u64],
) -> Result<usize> {
    let (lhs_text, Some(rhs_text), output_text) = parse_equation(equation) else {
        unreachable!()
    };
    let lhs_modes = modes_from_text(lhs_text);
    let rhs_modes = modes_from_text(rhs_text);
    let output_modes = output_text
        .map(modes_from_text)
        .unwrap_or_else(|| infer_output(&lhs_modes, &rhs_modes));

    let lhs_extent = lhs_shape.to_vec();
    let rhs_extent = rhs_shape.to_vec();
    let output_extent = extents_for_binary_output(
        &output_modes,
        &lhs_modes,
        &lhs_extent,
        &rhs_modes,
        &rhs_extent,
    );

    let lhs_values = (0..lhs_extent.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let rhs_values = (0..rhs_extent.iter().product::<u64>())
        .map(|index| 0.5_f32 * (index as f32 + 1.0))
        .collect::<Vec<_>>();
    let output_values = vec![0.0_f32; output_extent.iter().product::<u64>() as usize];

    let device_lhs = DeviceMemory::from_slice(&lhs_values)?;
    let device_rhs = DeviceMemory::from_slice(&rhs_values)?;
    let device_output_input = DeviceMemory::from_slice(&output_values)?;
    let mut device_output = DeviceMemory::from_slice(&output_values)?;

    const ALIGNMENT: u32 = 128;

    let lhs_descriptor = TensorDescriptor::create_for::<f32>(context, &lhs_extent, ALIGNMENT)?;
    let rhs_descriptor = TensorDescriptor::create_for::<f32>(context, &rhs_extent, ALIGNMENT)?;
    let output_descriptor =
        TensorDescriptor::create_for::<f32>(context, &output_extent, ALIGNMENT)?;

    let lhs_modes = lhs_modes
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let rhs_modes = rhs_modes
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let output_modes = output_modes
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();

    let operation = OperationDescriptor::contraction(
        context,
        TensorOperand::identity(&lhs_descriptor, &lhs_modes),
        TensorOperand::identity(&rhs_descriptor, &rhs_modes),
        TensorOperand::identity(&output_descriptor, &output_modes),
        TensorOperand::identity(&output_descriptor, &output_modes),
        ComputeDescriptor::f32(),
    )?;
    let preference = PlanPreference::create_default(context)?;
    let workspace_size = Plan::estimate_workspace_size(
        context,
        &operation,
        &preference,
        WorkspacePreference::Default,
    )?;
    let plan = Plan::create(context, &operation, &preference, workspace_size)?;
    let mut workspace = if plan.required_workspace_size() > 0 {
        Some(DeviceMemory::<u8>::create(
            plan.required_workspace_size_bytes()?,
        )?)
    } else {
        None
    };

    let alpha = 1.0_f32;
    let beta = 0.0_f32;
    plan.contract(
        &alpha,
        &device_lhs,
        &device_rhs,
        &beta,
        &device_output_input,
        &mut device_output,
        workspace.as_mut(),
        stream,
    )?;
    stream.synchronize()?;
    Ok(device_output.len())
}

fn run_unary_einsum(
    context: &Context,
    stream: &singe_cuda::stream::Stream,
    equation: &str,
    input_shape: &[u64],
) -> Result<usize> {
    let (input_text, None, output_text) = parse_equation(equation) else {
        unreachable!()
    };
    let input_modes = modes_from_text(input_text);
    let output_modes = output_text.map(modes_from_text).unwrap_or_else(|| {
        let mut modes = input_modes.clone();
        modes.sort_unstable();
        modes
    });

    let input_extent = input_shape.to_vec();
    let output_extent = extents_for_modes(&output_modes, &input_modes, &input_extent);
    let input_values = (0..input_extent.iter().product::<u64>())
        .map(|index| index as f32 + 1.0)
        .collect::<Vec<_>>();
    let output_values = vec![0.0_f32; output_extent.iter().product::<u64>() as usize];

    let device_input = DeviceMemory::from_slice(&input_values)?;
    let device_output_input = DeviceMemory::from_slice(&output_values)?;
    let mut device_output = DeviceMemory::from_slice(&output_values)?;

    const ALIGNMENT: u32 = 128;

    let input_descriptor = TensorDescriptor::create_for::<f32>(context, &input_extent, ALIGNMENT)?;
    let output_descriptor =
        TensorDescriptor::create_for::<f32>(context, &output_extent, ALIGNMENT)?;

    let input_modes = input_modes
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();
    let output_modes = output_modes
        .iter()
        .copied()
        .map(Into::into)
        .collect::<Vec<_>>();

    let preference = PlanPreference::create_default(context)?;
    let alpha = 1.0_f32;

    if input_modes.len() == output_modes.len() {
        let operation = OperationDescriptor::permutation(
            context,
            TensorOperand::identity(&input_descriptor, &input_modes),
            TensorOperand::identity(&output_descriptor, &output_modes),
            ComputeDescriptor::f32(),
        )?;
        let workspace_size = Plan::estimate_workspace_size(
            context,
            &operation,
            &preference,
            WorkspacePreference::Min,
        )?;
        let plan = Plan::create(context, &operation, &preference, workspace_size)?;
        plan.permute(&alpha, &device_input, &mut device_output, stream)?;
    } else {
        let operation = OperationDescriptor::reduction(
            context,
            TensorOperand::identity(&input_descriptor, &input_modes),
            TensorOperand::identity(&output_descriptor, &output_modes),
            TensorOperand::identity(&output_descriptor, &output_modes),
            Operator::Add,
            ComputeDescriptor::f32(),
        )?;
        let workspace_size = Plan::estimate_workspace_size(
            context,
            &operation,
            &preference,
            WorkspacePreference::Default,
        )?;
        let plan = Plan::create(context, &operation, &preference, workspace_size)?;
        let mut workspace = if plan.required_workspace_size() > 0 {
            Some(DeviceMemory::<u8>::create(
                plan.required_workspace_size_bytes()?,
            )?)
        } else {
            None
        };
        plan.reduce(
            &alpha,
            &device_input,
            &0.0_f32,
            &device_output_input,
            &mut device_output,
            workspace.as_mut(),
            stream,
        )?;
    }

    stream.synchronize()?;
    Ok(device_output.len())
}

fn main() -> Result<()> {
    let cuda_context = CudaContext::create()?;
    let context = Context::create(&cuda_context)?;
    let stream = cuda_context.create_stream()?;

    let contraction_explicit =
        run_binary_einsum(&context, &stream, "ijn,jmk->inkm", &[2, 3, 2], &[3, 2, 2])?;
    let contraction_implicit =
        run_binary_einsum(&context, &stream, "ijn,jmk", &[2, 3, 2], &[3, 2, 2])?;
    let permutation_implicit = run_unary_einsum(&context, &stream, "nij", &[2, 3, 2])?;
    let permutation_explicit = run_unary_einsum(&context, &stream, "nij->ijn", &[2, 3, 2])?;
    let reduction = run_unary_einsum(&context, &stream, "nij->ji", &[2, 3, 2])?;

    assert_eq!(contraction_explicit, contraction_implicit);
    assert_eq!(permutation_implicit, permutation_explicit);
    assert_eq!(reduction, 6);

    println!(
        "einsum dispatcher exercised contraction, permutation, and reduction paths successfully"
    );
    Ok(())
}
