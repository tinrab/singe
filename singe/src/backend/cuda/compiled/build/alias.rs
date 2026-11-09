use std::collections::HashMap;

use singe_cuda::graph::GraphNode;

use crate::{
    backend::cuda::{
        compiled::{
            build::{
                operation::{contiguous_input, copy_to_input},
                realized::record_realized_cuda_lowering,
            },
            executable::CudaCompiledGraphStats,
            plan::lowering::{CudaGraphProvider, LoweringPlan, RealizedLoweringPlan},
        },
        graph_binding::{validate_graph_alias_tensor_metadata, validate_graph_input_value_tensor},
        graph_value::CudaGraphValue,
    },
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
};

pub enum GraphAliasLowering {
    Added,
    Rejected,
    NotAlias,
}

pub struct GraphAliasLoweringContext<'a> {
    pub(crate) cache: &'a mut HashMap<ValueId, CudaGraphValue>,
    pub(crate) values: &'a mut HashMap<ValueId, CudaGraphValue>,
    pub(crate) native_nodes: &'a mut HashMap<ValueId, GraphNode>,
    pub(crate) stats: &'a mut CudaCompiledGraphStats,
    pub(crate) realized_lowering: &'a mut RealizedLoweringPlan,
}

pub fn try_add_cuda_graph_alias_node(
    id: ValueId,
    ir: &Graph,
    lowering_plan: &LoweringPlan,
    context: &mut GraphAliasLoweringContext<'_>,
) -> Result<GraphAliasLowering> {
    if let Some(input) = contiguous_input(ir, id)? {
        let input_tensor = context
            .cache
            .get(&input)
            .ok_or_else(|| Error::missing_input("try add cuda graph alias node"))?
            .clone();
        validate_graph_input_value_tensor(ir, input, &input_tensor)?;
        if input_tensor.is_contiguous()
            && input_tensor.storage_offset() == 0
            && lowering_plan.node_supports(id, CudaGraphProvider::Alias)
        {
            add_cuda_graph_alias_node(id, ir, input, &input_tensor, context)?;
            record_alias_lowering(id, ir, context)?;
            return Ok(GraphAliasLowering::Added);
        }

        // A strided input needs a copy, which a generated kernel or cuDNN makes.
        if lowering_plan.node_supports(id, CudaGraphProvider::Kernel)
            || lowering_plan.node_supports(id, CudaGraphProvider::CudnnBackend)
        {
            return Ok(GraphAliasLowering::NotAlias);
        }
        return Ok(GraphAliasLowering::Rejected);
    }

    if copy_to_input(ir, id)?.is_none()
        && let Some(input) = alias_input(ir, id)?
    {
        if !lowering_plan.node_supports(id, CudaGraphProvider::Alias) {
            return Ok(GraphAliasLowering::Rejected);
        }

        let tensor = context
            .cache
            .get(&input)
            .ok_or_else(|| Error::missing_input("try add cuda graph alias node"))?
            .clone();
        validate_graph_input_value_tensor(ir, input, &tensor)?;
        add_cuda_graph_alias_node(id, ir, input, &tensor, context)?;
        record_alias_lowering(id, ir, context)?;
        return Ok(GraphAliasLowering::Added);
    }

    Ok(GraphAliasLowering::NotAlias)
}

fn record_alias_lowering(
    id: ValueId,
    ir: &Graph,
    context: &mut GraphAliasLoweringContext<'_>,
) -> Result<()> {
    record_realized_cuda_lowering(context.realized_lowering, id, ir, CudaGraphProvider::Alias)
}

pub fn add_cuda_graph_alias_node(
    id: ValueId,
    ir: &Graph,
    input: ValueId,
    input_tensor: &CudaGraphValue,
    context: &mut GraphAliasLoweringContext<'_>,
) -> Result<()> {
    let output_value = id;
    ir.required_value_alias(output_value)?.ok_or_else(|| {
        Error::unsupported_operation_with_details("cuda graph alias", "missing ir alias metadata")
    })?;
    if required_cuda_graph_alias_source_node(ir, id)? != input {
        return Err(Error::unsupported_operation_with_details(
            "cuda graph alias",
            "ir alias source does not match lowering input",
        ));
    }

    let input_value = input;
    let output_device = ir.required_value(output_value)?.device();
    let input_device = ir.required_value(input_value)?.device();
    if input_device != output_device {
        return Err(Error::device_mismatch(
            "cuda graph alias input device",
            output_device,
            input_device,
        ));
    }

    let output_data_type = ir.required_value_data_type(output_value)?;
    let input_data_type = ir.required_value_data_type(input_value)?;
    if input_data_type != output_data_type {
        return Err(Error::data_type_mismatch(
            "cuda graph alias input data type",
            output_data_type,
            input_data_type,
        ));
    }

    let output_layout = ir.required_value_layout(output_value)?;
    let alias = input_tensor.alias(output_layout.clone())?;
    validate_graph_alias_tensor_metadata(output_data_type, output_layout, output_device, &alias)?;
    context.cache.insert(id, alias.clone());
    context.values.insert(id, alias);
    if let Some(native_node) = context.native_nodes.get(&input) {
        let native_node = native_node.clone();
        context.native_nodes.insert(id, native_node);
    }

    context.stats.record_alias_node();
    Ok(())
}

pub fn alias_input(ir: &Graph, id: ValueId) -> Result<Option<ValueId>> {
    let output_value = id;
    if ir.required_value_alias(output_value)?.is_none() {
        return Ok(None);
    }
    Ok(Some(required_cuda_graph_alias_source_node(ir, id)?))
}

fn required_cuda_graph_alias_source_node(ir: &Graph, id: ValueId) -> Result<ValueId> {
    let output_value = id;
    let alias = ir
        .required_value_alias(output_value)?
        .ok_or_else(|| Error::missing_node("required cuda graph alias source node"))?;
    let operation = ir.required_operation_for_value(id)?;
    operation
        .inputs()
        .iter()
        .copied()
        .find(|&value| value == alias.source())
        .ok_or_else(|| Error::missing_node("required cuda graph alias source node"))
}
