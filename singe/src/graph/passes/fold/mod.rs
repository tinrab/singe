mod elementwise;
mod neural;
mod shape;

use std::collections::HashMap;

use self::{
    elementwise::{
        folded_small_dtype_transform_constant, folded_small_elementwise_constant,
        folded_small_index_reduction_constant, folded_small_matmul_constant,
        folded_small_misc_elementwise_constant, folded_small_reduction_constant,
        folded_small_softmax_constant, folded_small_topk_constant,
    },
    neural::{folded_small_backend_structural_constant, folded_small_neural_constant},
    shape::{folded_small_shape_gather_constant, folded_small_view_constant},
};
use crate::{
    backend::registry::{BackendExecution, OperationRequest},
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{Graph, ir::ValueId},
    layout::Layout,
    shape::Shape,
    tensor::{Tensor, TensorId},
};

pub fn fold_small_constants(graph: &Graph) -> Result<Graph> {
    let mut graph = graph.clone();
    let mut generated_constant_map = HashMap::new();
    for id in graph.topological_sort()? {
        let Some(tensor) = folded_small_constant(&graph, id)? else {
            continue;
        };
        let tensor = if let Some(key) = generated_constant_key(&tensor)? {
            generated_constant_map
                .entry(key)
                .or_insert_with(|| tensor.clone())
                .clone()
        } else {
            tensor
        };
        graph.update_node(id, |node| node.replace_with_constant(tensor))?;
    }
    Ok(graph)
}

fn folded_small_constant(graph: &Graph, id: ValueId) -> Result<Option<Tensor>> {
    let node = &graph.required_node(id)?;
    let operation = node.operation();
    if let Some(operation) = operation {
        let inputs = node
            .inputs()
            .iter()
            .map(|&input| Ok(graph.required_node(input)?.output_spec()))
            .collect::<Result<Vec<_>>>()?;
        let checked = OperationRequest::new(operation, node.attributes(), inputs).check()?;
        // Host folding is optional; logical analysis must also work for graphs whose operations or dtypes are supported only by another backend.
        match checked.validate_execution(BackendExecution::CpuEager) {
            Ok(()) => {}
            Err(Error::Unsupported { .. }) => return Ok(None),
            Err(error) => return Err(error),
        }
    }

    if let Some(folded) = folded_small_elementwise_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_matmul_constant(graph, node)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_softmax_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_reduction_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_index_reduction_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_topk_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_misc_elementwise_constant(graph, node)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_shape_gather_constant(graph, node)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_view_constant(graph, node)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_backend_structural_constant(graph, node)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_dtype_transform_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }

    if let Some(folded) = folded_small_neural_constant(graph, node, operation)? {
        return Ok(Some(folded));
    }
    Ok(None)
}

pub const SMALL_HOST_CONSTANT_FOLD_BYTES: usize = 4096;

pub fn constant_inputs<const N: usize>(
    graph: &Graph,
    inputs: &[ValueId; N],
) -> Result<Option<[Tensor; N]>> {
    let mut tensors = Vec::with_capacity(N);
    for input in inputs {
        let Some(tensor) = constant_input(graph, *input)? else {
            return Ok(None);
        };
        tensors.push(tensor);
    }
    tensors.try_into().map(Some).map_err(|values: Vec<Tensor>| {
        Error::length_mismatch("constant folded input count", N, values.len())
    })
}

pub enum OptionalConstantInput {
    NonConstant,
    Constant(Option<Tensor>),
}

pub fn optional_constant_input(
    graph: &Graph,
    id: Option<ValueId>,
) -> Result<OptionalConstantInput> {
    let Some(id) = id else {
        return Ok(OptionalConstantInput::Constant(None));
    };
    let Some(tensor) = constant_input(graph, id)? else {
        return Ok(OptionalConstantInput::NonConstant);
    };
    Ok(OptionalConstantInput::Constant(Some(tensor)))
}

pub fn constant_input(graph: &Graph, id: ValueId) -> Result<Option<Tensor>> {
    let Some(node) = graph.node(id) else {
        return Ok(None);
    };
    let Some(tensor) = node.constant_tensor(0) else {
        return Ok(None);
    };
    Ok(is_small_cpu_tensor(tensor)?.then_some(tensor.clone()))
}

fn is_small_cpu_tensor(tensor: &Tensor) -> Result<bool> {
    Ok(tensor.device().is_cpu() && tensor.byte_size()? <= SMALL_HOST_CONSTANT_FOLD_BYTES)
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstantKey {
    id: TensorId,
    data_type: DataType,
    device: Device,
    shape: Shape,
    layout: Layout,
    storage_identity: usize,
    data_pointer_identity: Option<usize>,
    storage_byte_size: usize,
    version: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GeneratedConstantKey {
    data_type: DataType,
    device: Device,
    shape: Shape,
    layout: Layout,
    values: GeneratedConstantValues,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum GeneratedConstantValues {
    F32Bits(Vec<u32>),
    F64Bits(Vec<u64>),
    Bool(Vec<bool>),
    I64(Vec<i64>),
}

pub fn constant_key(tensor: &Tensor) -> ConstantKey {
    ConstantKey {
        id: tensor.id(),
        data_type: tensor.data_type(),
        device: tensor.device(),
        shape: tensor.shape().clone(),
        layout: Layout::from_tensor(tensor),
        storage_identity: tensor.storage().storage_identity(),
        data_pointer_identity: tensor.storage().data_pointer_identity(),
        storage_byte_size: tensor.storage().byte_size(),
        version: tensor.version(),
    }
}

pub fn generated_constant_key(tensor: &Tensor) -> Result<Option<GeneratedConstantKey>> {
    if !tensor.device().is_cpu() || tensor.byte_size()? > SMALL_HOST_CONSTANT_FOLD_BYTES {
        return Ok(None);
    }

    let values = match tensor.data_type() {
        DataType::Bool => GeneratedConstantValues::Bool(tensor.to_vec::<bool>()?),
        DataType::F64 => GeneratedConstantValues::F64Bits(
            tensor
                .to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect(),
        ),
        DataType::F32 => GeneratedConstantValues::F32Bits(
            tensor
                .to_vec::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect(),
        ),
        DataType::I64 => GeneratedConstantValues::I64(tensor.to_vec::<i64>()?),
        _ => return Ok(None),
    };
    Ok(Some(GeneratedConstantKey {
        data_type: tensor.data_type(),
        device: tensor.device(),
        shape: tensor.shape().clone(),
        layout: Layout::from_tensor(tensor),
        values,
    }))
}
