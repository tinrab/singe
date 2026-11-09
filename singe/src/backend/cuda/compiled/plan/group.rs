// The generated kernel of a fusion group, planned from the graph alone, so compiled graphs and exported programs make the same kernel.

use std::{collections::HashMap, ops::Range};

use crate::{
    backend::{
        cuda::{
            compiled::plan::fusion::{
                DequantizedWeight, FusionGroup, GroupKind, ModuleKind, RotaryAttention,
                RotaryCacheWrite, Rotation,
            },
            kernel_plan::{
                KernelInput, KernelParameters, KernelPlan, KernelPlanner, QueryRotation,
                RotaryCacheInputs, RotationNorm, parameter_key,
            },
        },
        kernel_lowering::{
            ElementGroup, Epilogue, EpilogueNode, EpilogueValue, FusedOperand, FusedStep,
            FusedStores, Lhs, RowEpilogue, Weight,
        },
        registry::{
            Operation, OperationAttributes, QuantizationAttributes,
            ScaledDotProductAttentionAttributes,
        },
        spec::{QuantizationGranularity, quantization_problem, unpacked_shape},
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    layout::Layout,
    shape::{Shape, normalize_dim_usize},
    utility::to_usize,
};

// A group's kernel: the values it reads, in the order its plan takes them, and those it stores, in the order its outputs hold them.
pub struct GroupKernel {
    pub plan: KernelPlan,
    pub inputs: Vec<ValueId>,
    pub outputs: Vec<ValueId>,
}

// The kernel input graph value `id` provides, named as reports name it when it has no name of its own.
pub fn kernel_input(ir: &Graph, id: ValueId) -> Result<KernelInput> {
    let name = ir
        .required_value_name(id)?
        .map_or_else(|| format!("%{}", id.index()), ToOwned::to_owned);
    Ok(KernelInput::new(
        ir.required_value_data_type(id)?,
        ir.required_value_layout(id)?.clone(),
    )
    .with_name(name))
}

// Plans the kernel of `group`, or returns `None` when no generated kernel runs it, so its members lower on their own.
pub fn plan_group_kernel(
    planner: &KernelPlanner,
    ir: &Graph,
    group: &FusionGroup,
) -> Result<Option<GroupKernel>> {
    match &group.kind {
        GroupKind::Contraction { matmul, weight } => {
            plan_contraction(planner, ir, group, *matmul, weight.as_ref())
        }
        GroupKind::Module(ModuleKind::RotaryCacheWrite(module)) => {
            let inputs = rotary_cache_inputs(ir, module)?;
            let mut values = module
                .query
                .iter()
                .map(|query| query.input)
                .collect::<Vec<_>>();
            values.extend([
                module.key.input,
                module.value,
                module.key.cos,
                module.key.sin,
                module.pages,
                module.slots,
            ]);
            for rotation in module.query.iter().chain([&module.key]) {
                values.extend(norm_weight(ir, rotation)?);
            }

            let mut outputs = module
                .query
                .iter()
                .map(|query| query.output)
                .collect::<Vec<_>>();
            outputs.extend([module.key_write, module.value_write]);
            Ok(Some(GroupKernel {
                plan: planner.rotary_cache_write(&inputs)?,
                inputs: values,
                outputs,
            }))
        }
        GroupKind::Module(ModuleKind::RotaryAttention(module)) => {
            let (metadata, values, attention) = rotary_attention_operands(ir, module)?;
            if !planner.takes_rotary_attention(&metadata[..7], attention)? {
                return Ok(None);
            }

            let rotation = QueryRotation {
                cos: kernel_input(ir, module.query.cos)?,
                sin: kernel_input(ir, module.query.sin)?,
                norm: rotation_norm(ir, &module.query)?,
            };
            Ok(Some(GroupKernel {
                plan: planner.rotary_paged_attention(&metadata, attention, &rotation)?,
                inputs: values,
                outputs: vec![module.attention],
            }))
        }
        _ => plan_elementwise(planner, ir, group),
    }
}

// What the fused attention kernel reads, as kernel inputs and as values, and the attention's attributes.
fn rotary_attention_operands(
    ir: &Graph,
    module: &RotaryAttention,
) -> Result<(
    Vec<KernelInput>,
    Vec<ValueId>,
    ScaledDotProductAttentionAttributes,
)> {
    let operation = ir.required_operation_for_value(module.attention)?;
    let attention = operation
        .attributes()
        .scaled_dot_product_attention()
        .copied()
        .ok_or_else(|| Error::missing_node("rotary attention"))?;
    let mut values = vec![module.query.input];
    values.extend(operation.inputs().iter().skip(1).copied());
    values.extend([module.query.cos, module.query.sin]);
    values.extend(norm_weight(ir, &module.query)?);
    let metadata = values
        .iter()
        .map(|&value| kernel_input(ir, value))
        .collect::<Result<Vec<_>>>()?;
    Ok((metadata, values, attention))
}

// The key a module's parameters are recorded under and what tuning may choose among.
pub fn module_parameters(
    planner: &KernelPlanner,
    ir: &Graph,
    kind: &ModuleKind,
) -> Result<(String, Vec<KernelParameters>)> {
    match kind {
        ModuleKind::RotaryCacheWrite(module) => {
            planner.rotary_cache_parameters(&rotary_cache_inputs(ir, module)?)
        }
        ModuleKind::RotaryAttention(module) => {
            let (metadata, _, attention) = rotary_attention_operands(ir, module)?;
            let dim = to_usize(
                ir.required_value_shape(module.query.input)?.dimensions()[2],
                "cuda rotary attention head",
            )?;
            Ok((
                parameter_key("paged_attention", &metadata, &format!("{attention:?}")),
                planner
                    .attention_candidates(dim)
                    .into_iter()
                    .map(KernelParameters::Attention)
                    .collect(),
            ))
        }
    }
}

// The weight of the norm a rotation applies, if it has one.
fn norm_weight(ir: &Graph, rotation: &Rotation) -> Result<Option<ValueId>> {
    let Some(norm) = rotation.norm else {
        return Ok(None);
    };
    Ok(ir
        .required_operation_for_value(norm)?
        .inputs()
        .get(1)
        .copied())
}

fn rotation_norm(ir: &Graph, rotation: &Rotation) -> Result<Option<RotationNorm>> {
    let Some(norm) = rotation.norm else {
        return Ok(None);
    };
    let OperationAttributes::LayerNormalization(attributes) =
        ir.required_operation_for_value(norm)?.attributes()
    else {
        return Err(Error::missing_node("rotary cache norm"));
    };
    Ok(Some(RotationNorm {
        eps: attributes.eps,
        weight: norm_weight(ir, rotation)?
            .map(|weight| kernel_input(ir, weight))
            .transpose()?,
    }))
}

fn rotary_cache_inputs(ir: &Graph, module: &RotaryCacheWrite) -> Result<RotaryCacheInputs> {
    let input = |id| kernel_input(ir, id);
    let base = |id| input(ir.required_operation_for_value(id)?.inputs()[0]);
    Ok(RotaryCacheInputs {
        query: module
            .query
            .as_ref()
            .map(|query| input(query.input))
            .transpose()?,
        key: input(module.key.input)?,
        value: input(module.value)?,
        cos: input(module.key.cos)?,
        sin: input(module.key.sin)?,
        pages: input(module.pages)?,
        slots: input(module.slots)?,
        key_cache: base(module.key_write)?,
        value_cache: base(module.value_write)?,
        query_norm: module
            .query
            .as_ref()
            .map(|query| rotation_norm(ir, query))
            .transpose()?
            .flatten(),
        key_norm: rotation_norm(ir, &module.key)?,
    })
}

// Plans a point-wise, row, or concatenation group: a row kernel computes the group's element-wise members as it loads its input, and stores that input too when something else reads it; a concatenation's kernel stores each of its inputs into its slice.
fn plan_elementwise(
    planner: &KernelPlanner,
    ir: &Graph,
    group: &FusionGroup,
) -> Result<Option<GroupKernel>> {
    let (concat, row) = match group.kind {
        GroupKind::Pointwise => (None, None),
        GroupKind::Row { row, stored } => (None, Some((row, stored))),
        GroupKind::Concat { concat, axis } => (Some((concat, axis)), None),
        GroupKind::Contraction { .. } | GroupKind::Module(_) => return Ok(None),
    };
    // The element-wise members before the row operation, in the order lowering visits them, and the epilogue after it.
    let row_position = row.and_then(|(row, _)| group.members.iter().position(|&id| id == row));
    let (before, after) = group
        .members
        .split_at(row_position.map_or(group.members.len(), |position| position + 1));
    let epilogue_members = if row.is_some() { after } else { &[][..] };
    let chain = before
        .iter()
        .copied()
        .filter(|&id| {
            Some(id) != row.map(|(row, _)| row) && Some(id) != concat.map(|(concat, _)| concat)
        })
        .collect::<Vec<_>>();
    // A row group without a prologue reads its input directly.
    let first = match (chain.first(), row) {
        (Some(&first), _) => first,
        (None, Some((row, _))) => ir.required_operation_for_value(row)?.inputs()[0],
        (None, None) => return Ok(None),
    };
    let data_type = ir.required_value_data_type(first)?;
    let shape = ir.required_value_shape(first)?.clone();
    let mut external = Vec::<ValueId>::new();
    if chain.is_empty() {
        external.push(first);
    }

    let steps = fused_steps(ir, &chain, &[], &mut external)?;
    let group_inputs = external.len();
    let (row, read_elsewhere) = row.map_or((None, false), |(id, read)| (Some(id), read));
    if let Some(id) = row {
        external.extend_from_slice(&ir.required_operation_for_value(id)?.inputs()[1..]);
    }

    // The epilogue's operands from outside the group, read after the row operation's parameters.
    let epilogue_start = external.len();
    let epilogue_plan = match row {
        Some(id) if !epilogue_members.is_empty() => Some(epilogue_values(
            ir,
            id,
            epilogue_members,
            group_inputs,
            &mut external,
        )?),
        _ => None,
    };
    // A point-wise group stores each of its outputs, which ends with its last member.
    let pointwise_outputs = (row.is_none() && group.outputs.len() > 1).then(|| {
        group
            .outputs
            .iter()
            .filter_map(|output| chain.iter().position(|member| member == output))
            .collect::<Vec<_>>()
    });
    let (stores, last) = match &concat {
        None if pointwise_outputs.is_some() => (
            FusedStores::Steps(pointwise_outputs.clone().unwrap_or_default()),
            chain[chain.len() - 1],
        ),
        None => (
            FusedStores::Last,
            epilogue_members
                .last()
                .copied()
                .or(row)
                .unwrap_or_else(|| chain[chain.len() - 1]),
        ),
        Some((id, axis)) => (
            FusedStores::Concat {
                axis: *axis,
                values: concat_values(ir, *id, &chain, &mut external)?,
            },
            *id,
        ),
    };
    let metadata = external
        .iter()
        .map(|&id| kernel_input(ir, id))
        .collect::<Result<Vec<_>>>()?;
    let plan = match row {
        None => planner.fused(data_type, &shape, &metadata, &steps, &stores),
        Some(id) => plan_row_group(
            planner,
            ir,
            id,
            &RowGroupParts {
                group: ElementGroup {
                    data_type,
                    shape,
                    inputs: Vec::new(),
                    steps: &steps,
                    stored: read_elsewhere,
                },
                group_inputs,
                epilogue_start,
                epilogue: epilogue_plan.as_deref(),
            },
            &metadata,
        ),
    };
    let Ok(plan) = plan else {
        return Ok(None);
    };
    // A row kernel that stores the group's value writes it as its second output.
    let mut outputs = vec![last];
    if read_elsewhere {
        outputs.push(chain[chain.len() - 1]);
    }

    if let Some(steps) = &pointwise_outputs {
        outputs = steps.iter().map(|&step| chain[step]).collect();
    }
    Ok(Some(GroupKernel {
        plan,
        inputs: external,
        outputs,
    }))
}

// Returns the slot of `value` among `external`, adding it if it is new.
fn external_slot(external: &mut Vec<ValueId>, value: ValueId) -> usize {
    external
        .iter()
        .position(|&input| input == value)
        .unwrap_or_else(|| {
            external.push(value);
            external.len() - 1
        })
}

// Returns what concatenation `id` stores, each input a step of `chain` or a value outside it, which joins `external`.
fn concat_values(
    ir: &Graph,
    id: ValueId,
    chain: &[ValueId],
    external: &mut Vec<ValueId>,
) -> Result<Vec<FusedOperand>> {
    Ok(ir
        .required_operation_for_value(id)?
        .inputs()
        .iter()
        .map(|&input| {
            chain
                .iter()
                .position(|&member| member == input)
                .map_or_else(
                    || FusedOperand::Input(external_slot(external, input)),
                    FusedOperand::Step,
                )
        })
        .collect())
}

// A row group's element-wise part, and where its operands fall among the kernel's inputs: the group's, the row operation's parameters, and the epilogue's.
struct RowGroupParts<'a> {
    group: ElementGroup<'a>,
    group_inputs: usize,
    epilogue_start: usize,
    epilogue: Option<&'a [(EpilogueValue<'a>, Shape)]>,
}

fn plan_row_group(
    planner: &KernelPlanner,
    ir: &Graph,
    id: ValueId,
    parts: &RowGroupParts<'_>,
    metadata: &[KernelInput],
) -> Result<KernelPlan> {
    let operation = ir.required_operation_for_value(id)?;
    let layouts = metadata
        .iter()
        .map(|input| (input.data_type(), input.layout()))
        .collect::<Vec<_>>();
    let group = ElementGroup {
        data_type: parts.group.data_type,
        shape: parts.group.shape.clone(),
        inputs: layouts[..parts.group_inputs].to_vec(),
        steps: parts.group.steps,
        stored: parts.group.stored,
    };
    let epilogue = parts.epilogue.map(|nodes| RowEpilogue {
        inputs: layouts[parts.epilogue_start..].to_vec(),
        nodes: nodes
            .iter()
            .map(|(value, shape)| EpilogueNode {
                value: value.clone(),
                shape: shape.clone(),
            })
            .collect(),
    });
    planner
        .row(
            operation
                .opcode()
                .ok_or_else(|| Error::missing_node("opcode of a row operation"))?,
            operation.attributes(),
            &group,
            &layouts[parts.group_inputs..parts.epilogue_start],
            epilogue.as_ref(),
            metadata,
        )
        .and_then(|plan| plan.ok_or_else(|| Error::missing_node("kernel of a row operation")))
}

/// Returns the epilogue of row operation `row` as the values `members` compute from its output, with their shapes, the first being the row's output, and adds the values they read from outside the group to `external`.
///
/// A reduction's epilogue reads the first `group_inputs` values of `external`, which the group loads, as the group does.
fn epilogue_values<'a>(
    ir: &'a Graph,
    row: ValueId,
    members: &[ValueId],
    group_inputs: usize,
    external: &mut Vec<ValueId>,
) -> Result<Vec<(EpilogueValue<'a>, Shape)>> {
    let first_input = external.len();
    let reduction = ir
        .required_operation_for_value(row)?
        .attributes()
        .reduction()
        .is_some();
    let mut nodes = vec![(EpilogueValue::Row, ir.required_value_shape(row)?.clone())];
    let mut positions = HashMap::from([(row, 0)]);
    for &id in members {
        let operation = ir.required_operation_for_value(id)?;
        let shape = ir.required_value_shape(id)?.clone();
        let rank = shape.rank();
        let inputs = operation.inputs();
        let position = |value: &ValueId| {
            positions
                .get(value)
                .copied()
                .ok_or_else(|| Error::missing_node("epilogue values"))
        };
        let value = match (operation.opcode(), operation.attributes()) {
            (Some(Operation::Slice), OperationAttributes::Slice(slice)) => EpilogueValue::Slice {
                value: position(&inputs[0])?,
                start: slice.start,
            },
            (Some(Operation::Transpose), OperationAttributes::Transpose(transpose)) => {
                EpilogueValue::Transpose {
                    value: position(&inputs[0])?,
                    dims: [
                        normalize_dim_usize(transpose.dim0, rank)?,
                        normalize_dim_usize(transpose.dim1, rank)?,
                    ],
                }
            }
            (Some(Operation::Concat), _) => EpilogueValue::Concat {
                values: inputs.iter().map(position).collect::<Result<_>>()?,
            },
            (Some(opcode), attributes) => {
                let mut operands = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let group_slot = external[..group_inputs]
                        .iter()
                        .position(|value| value == input)
                        .filter(|_| reduction);
                    let operand = if let Some(&known) = positions.get(input) {
                        known
                    } else if let Some(slot) = group_slot {
                        nodes.push((
                            EpilogueValue::GroupInput {
                                slot,
                                data_type: ir.required_value_data_type(*input)?,
                            },
                            ir.required_value_shape(*input)?.clone(),
                        ));
                        positions.insert(*input, nodes.len() - 1);
                        nodes.len() - 1
                    } else {
                        // Each value from outside is one input node, read in the shape of the step that reads it.
                        let slot = external[first_input..]
                            .iter()
                            .position(|value| value == input)
                            .unwrap_or_else(|| {
                                external.push(*input);
                                external.len() - first_input - 1
                            });
                        nodes.push((
                            EpilogueValue::Input(slot),
                            ir.required_value_shape(*input)?.clone(),
                        ));
                        positions.insert(*input, nodes.len() - 1);
                        nodes.len() - 1
                    };
                    operands.push(operand);
                }
                EpilogueValue::Step {
                    operation: opcode,
                    attributes,
                    operands,
                }
            }
            (None, _) => return Err(Error::missing_node("epilogue values")),
        };
        nodes.push((value, shape));
        positions.insert(id, nodes.len() - 1);
    }
    Ok(nodes)
}

// Plans matmul `id` as one generated kernel with its dequantized weight and its epilogue.
fn plan_contraction(
    planner: &KernelPlanner,
    ir: &Graph,
    group: &FusionGroup,
    id: ValueId,
    weight: Option<&DequantizedWeight>,
) -> Result<Option<GroupKernel>> {
    let operation = ir.required_operation_for_value(id)?;
    let (Some(Operation::MatMul), &[lhs, rhs]) = (operation.opcode(), operation.inputs()) else {
        return Ok(None);
    };
    // The members before the matmul that are not its weight's compute its left operand, and those after it its epilogue.
    let position = group
        .members
        .iter()
        .position(|&member| member == id)
        .ok_or_else(|| Error::missing_node("plan contraction"))?;
    let weight_members = weight.map_or(&[][..], DequantizedWeight::members);
    let prologue = group.members[..position]
        .iter()
        .copied()
        .filter(|member| !weight_members.contains(member))
        .collect::<Vec<_>>();
    let epilogue = &group.members[position + 1..];
    let data_type = ir.required_value_data_type(id)?;
    let mut external = Vec::new();
    let lhs_steps = fused_steps(ir, &prologue, &[], &mut external)?;
    if prologue.is_empty() {
        external.push(lhs);
    }

    let lhs_inputs = external.len();
    match weight {
        None => external.push(rhs),
        Some(weight) => external.extend([weight.quantized, weight.scale, weight.zero_point]),
    }

    let epilogue_start = external.len();
    let mut epilogue_external = Vec::new();
    let epilogue_steps = fused_steps(ir, epilogue, &[id], &mut epilogue_external)?;
    external.extend(epilogue_external);
    let metadata = external
        .iter()
        .map(|&node| kernel_input(ir, node))
        .collect::<Result<Vec<_>>>()?;
    let value = |node: ValueId| {
        external
            .iter()
            .position(|&input| input == node)
            .map(|index| &metadata[index])
            .ok_or_else(|| Error::missing_input("plan contraction"))
    };
    let weight_layouts;
    let plan_weight = match weight {
        None => Weight::Plain(value(rhs)?.layout()),
        Some(weight) => {
            weight_layouts = dequantized_layouts(
                value(weight.quantized)?.layout(),
                value(weight.scale)?.layout(),
                value(weight.zero_point)?.layout(),
                weight.attributes,
                weight.transposed,
            )?;
            let Some(layouts) = &weight_layouts else {
                return Ok(None);
            };
            Weight::Dequantized {
                quantized: (value(weight.quantized)?.data_type(), &layouts[0]),
                scale: &layouts[1],
                zero_point: &layouts[2],
                rounded: weight.rounded,
                packed: weight.attributes.packed,
            }
        }
    };
    let inputs = |range: Range<usize>| {
        metadata[range]
            .iter()
            .map(|value| (value.data_type(), value.layout()))
            .collect::<Vec<_>>()
    };
    let plan_lhs = Lhs {
        shape: ir.required_value_shape(lhs)?.clone(),
        inputs: inputs(0..lhs_inputs),
        steps: &lhs_steps,
    };
    let plan_epilogue = (!epilogue.is_empty()).then(|| Epilogue {
        inputs: inputs(epilogue_start..metadata.len()),
        steps: &epilogue_steps,
    });
    let Ok(plan) = planner.contraction(
        data_type,
        &plan_lhs,
        &plan_weight,
        plan_epilogue.as_ref(),
        &metadata,
    ) else {
        return Ok(None);
    };
    let produced = epilogue.last().copied().unwrap_or(id);
    Ok(Some(GroupKernel {
        plan,
        inputs: external,
        outputs: vec![produced],
    }))
}

/// Returns `members` as fused steps: an operand that is an earlier member is that step, one of `given` is the kernel input in its position, and any other is an input after those, which `external` gains.
fn fused_steps<'a>(
    ir: &'a Graph,
    members: &[ValueId],
    given: &[ValueId],
    external: &mut Vec<ValueId>,
) -> Result<Vec<FusedStep<'a>>> {
    let mut steps = Vec::with_capacity(members.len());
    for (index, &member) in members.iter().enumerate() {
        let operation = ir.required_operation_for_value(member)?;
        let operands = operation
            .inputs()
            .iter()
            .map(|input| {
                if let Some(step) = members[..index].iter().position(|earlier| earlier == input) {
                    return FusedOperand::Step(step);
                }

                if let Some(slot) = given.iter().position(|value| value == input) {
                    return FusedOperand::Input(slot);
                }

                let slot = external
                    .iter()
                    .position(|value| value == input)
                    .unwrap_or_else(|| {
                        external.push(*input);
                        external.len() - 1
                    });
                FusedOperand::Input(given.len() + slot)
            })
            .collect();
        steps.push(FusedStep {
            operation: operation
                .opcode()
                .ok_or_else(|| Error::missing_node("fused steps"))?,
            attributes: operation.attributes(),
            operands,
        });
    }
    Ok(steps)
}

/// Returns the quantized tensor's, scale's, and zero point's layouts over the matmul's right operand, `[K, N]`, or `None` for a granularity the fused kernel does not read.
fn dequantized_layouts(
    quantized: &Layout,
    scale: &Layout,
    zero_point: &Layout,
    attributes: QuantizationAttributes,
    transposed: bool,
) -> Result<Option<[Layout; 3]>> {
    // A packed weight's parameters cover its unpacked shape, and its packed axis must become the contraction.
    if attributes.packed.is_some() && !transposed {
        return Ok(None);
    }

    let unpacked = unpacked_shape(quantized.shape(), attributes.packed)?;
    let shape = &unpacked;
    let problem = quantization_problem(shape, attributes.axis, attributes.block_size)?;
    let parameter = |layout: &Layout| -> Result<Option<Layout>> {
        let mut strides = vec![0; 2];
        match problem.granularity {
            QuantizationGranularity::Tensor => {}
            QuantizationGranularity::Channel { .. } => {
                let axis = normalize_dim_usize(attributes.axis.unwrap_or(0), shape.rank())?;
                strides[axis] = layout.strides()[0];
            }
            QuantizationGranularity::Block { .. } => return Ok(None),
        }
        Ok(Some(
            Layout::with_strides(shape.dimensions(), strides)?.with_offset(layout.storage_offset()),
        ))
    };
    let (Some(scale), Some(zero_point)) = (parameter(scale)?, parameter(zero_point)?) else {
        return Ok(None);
    };
    let orient = |layout: Layout| -> Result<Layout> {
        if transposed {
            layout.transpose(0, 1)
        } else {
            Ok(layout)
        }
    };
    Ok(Some([
        orient(quantized.clone())?,
        orient(scale)?,
        orient(zero_point)?,
    ]))
}
