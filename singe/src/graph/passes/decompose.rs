use std::{collections::HashMap, sync::Arc};

use crate::{
    backend::{
        decomposition,
        registry::{CudaGraphProvider, CudaGraphProviderSet, Operation, OperationAttributes},
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeRef},
        passes::remap::remapped_node,
    },
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    utility::to_i64,
};

// The rewritten graph, its value remapping, and the operations split only because an excluded provider runs them whole.
type Decomposition = (Graph, Vec<(ValueId, ValueId)>, Vec<Operation>);

// Decompositions nest only a few levels, so a chain this deep is one that records itself.
const MAX_DECOMPOSITION_DEPTH: usize = 16;

// Returns `graph` with every operation that compiled graphs have no kernel for written as its decomposition, and the map from old values to new ones, which is empty when nothing decomposes.
// Each decomposition is traced on meta tensors laid out as the node's operands, so the operations it records see the layouts they will run on.
// A decomposition may record operations that decompose in turn, such as `conv3d`'s `pad`, so this repeats until nothing changes.
pub fn decompose_operations(
    graph: &Graph,
    cuda_providers: Option<CudaGraphProviderSet>,
) -> Result<Decomposition> {
    let (mut graph, mut remapped, operations) = decompose_once(graph, cuda_providers)?;
    let mut rounds = 1;
    while !remapped.is_empty() {
        let (next, again, _) = decompose_once(&graph, cuda_providers)?;
        if again.is_empty() {
            break;
        }

        rounds += 1;
        if rounds > MAX_DECOMPOSITION_DEPTH {
            return Err(Error::unsupported_operation_with_details(
                "graph decomposition",
                "a decomposition keeps recording operations that decompose",
            ));
        }

        let later = again.iter().copied().collect::<HashMap<_, _>>();
        for (_, current) in &mut remapped {
            if let Some(&moved) = later.get(current) {
                *current = moved;
            }
        }

        graph = next;
    }
    Ok((graph, remapped, operations))
}

// One round, also naming the operations it split only because a provider outside `cuda_providers` is the one that runs them whole.
fn decompose_once(
    graph: &Graph,
    cuda_providers: Option<CudaGraphProviderSet>,
) -> Result<Decomposition> {
    let takes_rowwise = cuda_providers.is_none_or(|providers: CudaGraphProviderSet| {
        providers.allows(CudaGraphProvider::CublasLt, Operation::ScaledMatMul)
    });
    let mut id_map = HashMap::with_capacity(graph.node_count());
    let mut nodes = Vec::with_capacity(graph.node_count());
    let mut decomposed = false;
    let mut operations = Vec::new();
    for id in graph.node_ids() {
        let node = graph.required_node(id)?;
        let new_id = if takes_rowwise && takes_rowwise_scaled_matmul(graph, &node)? {
            decomposed = true;
            let attributes = node.attributes();
            splice_traced(graph, &node, &id_map, &mut nodes, |inputs| {
                decomposition::rowwise_scaled_matmul(&attributes, inputs)
            })?
        } else if decomposes(graph, &node, cuda_providers)? {
            decomposed = true;
            if let Some(operation) = node.operation()
                && !decomposes(graph, &node, Some(CudaGraphProviderSet::all()))?
            {
                operations.push(operation);
            }
            splice(graph, &node, &id_map, &mut nodes)?
        } else {
            push(&mut nodes, remapped_node(&node, &id_map)?)
        };
        id_map.insert(id, new_id);
    }

    if !decomposed {
        return Ok((graph.clone(), Vec::new(), Vec::new()));
    }

    let remapped = graph.remapped_node_pairs(&id_map)?;
    Ok((
        Graph::from_retained_nodes(graph, &id_map, nodes)?,
        remapped,
        operations,
    ))
}

// Returns whether `node` is a `scaled_mm` with row-wise scales over operands that cuBLASLt's FP8 tensor cores take, which then run its unscaled product.
fn takes_rowwise_scaled_matmul(graph: &Graph, node: &NodeRef<'_>) -> Result<bool> {
    if node.operation() != Some(Operation::ScaledMatMul) {
        return Ok(false);
    }

    let operands = node
        .inputs()
        .iter()
        .map(|&input| graph.required_node(input))
        .collect::<Result<Vec<_>>>()?;
    let [lhs, rhs, lhs_scale, rhs_scale] = operands.as_slice() else {
        return Ok(false);
    };
    Ok(spec::scaled_matmul_problem(
        operand(lhs),
        operand(rhs),
        operand(lhs_scale),
        operand(rhs_scale),
        "graph scaled_mm",
    )
    .is_ok_and(|problem| {
        problem.scaling == spec::MatmulScaling::Rowwise
            && spec::takes_scaled_matmul(lhs.output_layout(), rhs.output_layout(), &problem)
    }))
}

fn decomposes(
    graph: &Graph,
    node: &NodeRef<'_>,
    cuda_providers: Option<CudaGraphProviderSet>,
) -> Result<bool> {
    let Some(operation) = node.operation() else {
        return Ok(false);
    };
    let allowed = |provider| {
        cuda_providers
            .is_none_or(|providers: CudaGraphProviderSet| providers.allows(provider, operation))
    };
    if !decomposition::has_decomposition(operation) {
        return Ok(false);
    }

    let Some(&first) = node.inputs().first() else {
        return Ok(false);
    };
    // Long `F32` and `F64` transforms stay whole for cuFFT; the rest decompose into a matmul with the basis.
    if matches!(operation, Operation::Rfft | Operation::Irfft) {
        let input = graph.required_node(first)?;
        let dimensions = input.output_shape().dimensions();
        let takes = match (operation, node.attributes()) {
            (Operation::Rfft, _) => {
                input.output_layout().is_contiguous()
                    && spec::transforms_with_cufft(
                        dimensions.last().copied().unwrap_or(0),
                        input.output_data_type(),
                    )
            }
            (_, OperationAttributes::Length(length)) => {
                dimensions.len() >= 2
                    && dimensions[dimensions.len() - 2] > length / 2
                    && spec::transforms_with_cufft(length, input.output_data_type())
            }
            _ => false,
        };
        return Ok(!(takes && allowed(CudaGraphProvider::Cufft)));
    }

    // Without their library, these run as their decompositions.
    if (operation == Operation::Contraction && !allowed(CudaGraphProvider::Cutensor))
        || (operation == Operation::Conv3d && !allowed(CudaGraphProvider::CudnnBackend))
    {
        return Ok(true);
    }

    // CUDA selects a few values of contiguous rows rather than sorting them.
    if matches!(operation, Operation::TopKValues | Operation::TopKIndices)
        && let Some(topk) = node.attributes().topk()
    {
        let input = graph.required_node(first)?;
        if (cuda_providers.is_some() || input.output_device().is_cuda())
            && spec::selects_topk(input.output_data_type(), input.output_layout(), topk)
        {
            return Ok(false);
        }
    }

    // `scaled_mm` stays whole where cuBLASLt's FP8 tensor cores take it, and decomposes into a dequantized `F32` product elsewhere.
    if operation == Operation::ScaledMatMul {
        let operands = node
            .inputs()
            .iter()
            .map(|&input| graph.required_node(input))
            .collect::<Result<Vec<_>>>()?;
        let [lhs, rhs, lhs_scale, rhs_scale] = operands.as_slice() else {
            return Ok(false);
        };
        let takes = spec::scaled_matmul_problem(
            operand(lhs),
            operand(rhs),
            operand(lhs_scale),
            operand(rhs_scale),
            "graph scaled_mm",
        )
        // Compiled graphs take one scale per operand, which every FP8 target scales by.
        .is_ok_and(|problem| {
            problem.scaling == spec::MatmulScaling::Tensorwise
                && spec::takes_scaled_matmul(lhs.output_layout(), rhs.output_layout(), &problem)
        });
        return Ok(!(takes && allowed(CudaGraphProvider::CublasLt)));
    }

    let data_type = graph.required_node(first)?.output_data_type();
    Ok(decomposition::decomposes_in_graphs(
        operation,
        &node.attributes(),
        data_type,
    ))
}

fn operand<'a>(node: &NodeRef<'a>) -> (&'a Shape, DataType) {
    (node.output_shape(), node.output_data_type())
}

// Pushes the traced decomposition of `node` and returns the id of its result, as the attention pass also does for attention no fused kernel runs.
pub fn splice(
    graph: &Graph,
    node: &NodeRef<'_>,
    id_map: &HashMap<ValueId, ValueId>,
    nodes: &mut Vec<Node>,
) -> Result<ValueId> {
    let operation = node
        .operation()
        .ok_or_else(|| Error::missing_node("splice"))?;
    let attributes = node.attributes();
    splice_traced(graph, node, id_map, nodes, |inputs| {
        decomposition::decompose(operation, &attributes, inputs)
    })
}

// Pushes what `trace` records for `node` over meta tensors laid out as its operands, and returns the id of its result.
fn splice_traced(
    graph: &Graph,
    node: &NodeRef<'_>,
    id_map: &HashMap<ValueId, ValueId>,
    nodes: &mut Vec<Node>,
    trace: impl FnOnce(&[&Tensor]) -> Result<Vec<Tensor>>,
) -> Result<ValueId> {
    let inputs = node
        .inputs()
        .iter()
        .map(|&input| {
            let source = graph.required_node(input)?;
            Ok((
                source.output_data_type(),
                source.output_layout().clone(),
                Graph::remapped_node(input, id_map)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    splice_traced_on(
        &inputs,
        node.output_device(),
        node.source_location(),
        node.output_layout().is_contiguous(),
        nodes,
        trace,
    )
}

// Appends what `trace` records on meta tensors laid out as `inputs`, each the type, layout, and node of an operand, and returns its one result.
// A `contiguous` result is read where later nodes' metadata places it, which may take a copy.
pub fn splice_traced_on(
    inputs: &[(DataType, Layout, ValueId)],
    device: Device,
    source_location: Option<&str>,
    contiguous: bool,
    nodes: &mut Vec<Node>,
    trace: impl FnOnce(&[&Tensor]) -> Result<Vec<Tensor>>,
) -> Result<ValueId> {
    let operands = inputs
        .iter()
        .enumerate()
        .map(|(index, (data_type, layout, _))| {
            Ok((format!("input{index}"), meta_operand(*data_type, layout)?))
        })
        .collect::<Result<Vec<_>>>()?;
    let (traced, _) = Graph::capture(operands, |inputs| {
        let mut outputs = trace(&inputs.iter().collect::<Vec<_>>())?;
        // A node has one result; the projections of a multi-result operation are nodes of their own.
        let output = outputs
            .pop()
            .ok_or_else(|| Error::missing_output("splice"))?;
        // Later nodes read the result where the node's metadata places it.
        let output = if contiguous && !output.is_contiguous() {
            output.contiguous()?
        } else {
            output
        };
        Ok(vec![output])
    })?;
    let traced = traced.placed_on(device)?;
    let mut local = HashMap::with_capacity(traced.node_count());
    for id in traced.node_ids() {
        let traced_node = traced.required_node(id)?;
        let new_id = if let Some((name, ..)) = traced_node.input_metadata() {
            name.strip_prefix("input")
                .and_then(|index| index.parse::<usize>().ok())
                .and_then(|index| inputs.get(index))
                .map(|(_, _, new_id)| *new_id)
                .ok_or_else(|| Error::missing_input("splice"))?
        } else {
            let spliced = traced_node.map_inputs(|input| Graph::remapped_node(input, &local))?;
            push(
                nodes,
                spliced.with_source_location(source_location.map(ToOwned::to_owned)),
            )
        };
        local.insert(id, new_id);
    }

    let projections = traced.output_projections();
    let [output] = projections.as_slice() else {
        return Err(Error::missing_output("splice"));
    };
    Graph::remapped_node(output.node(), &local)
}

// A meta tensor of `data_type` laid out by `layout`, over storage that spans it.
fn meta_operand(data_type: DataType, layout: &Layout) -> Result<Tensor> {
    let span = to_i64(
        layout.storage_span_element_count()?,
        "graph decomposition operand span",
    )?;
    let storage = Tensor::zeroed_on(Device::Meta, data_type, Shape::create([span])?)?;
    Tensor::create_from_storage(Arc::clone(storage.storage()), data_type, layout.clone())
}

pub fn push(nodes: &mut Vec<Node>, node: Node) -> ValueId {
    nodes.push(node);
    ValueId::new(nodes.len() - 1)
}
