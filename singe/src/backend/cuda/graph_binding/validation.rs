use crate::{
    backend::{
        cuda::{
            graph_binding::alias::validate_graph_alias_tensor_metadata,
            tensor_region::CudaTensorSource,
        },
        spec,
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::ir::{Graph, ValueId},
    layout::Layout,
};

// Checks `tensor` against the metadata of a graph `kind`, such as an alias or a tensor.
pub fn validate_graph_tensor_metadata(
    kind: &str,
    data_type: DataType,
    layout: &Layout,
    device: Device,
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    if tensor.data_type() != data_type {
        return Err(Error::data_type_mismatch(
            format!("cuda graph {kind} data type"),
            data_type,
            tensor.data_type(),
        ));
    }

    if tensor.shape().dimensions() != layout.dimensions() {
        return Err(Error::shape_mismatch(
            format!("cuda graph {kind} dimensions"),
            layout.dimensions(),
            tensor.shape().dimensions(),
        ));
    }

    if tensor.device() != device {
        return Err(Error::device_mismatch(
            format!("cuda graph {kind} device"),
            device,
            tensor.device(),
        ));
    }

    // An axis of one element is never stepped along, so its stride addresses nothing and may differ.
    if tensor.layout().addressing_strides() != layout.addressing_strides()
        || tensor.layout().addressing_offset() != layout.addressing_offset()
    {
        return Err(Error::layout_mismatch(
            format!("cuda graph {kind} layout"),
            layout.dimensions().to_vec(),
            layout.strides().to_vec(),
            layout.storage_offset(),
            tensor.shape().dimensions().to_vec(),
            tensor.layout().strides().to_vec(),
            tensor.storage_offset(),
        ));
    }

    tensor.tensor_region()?;
    Ok(())
}

pub fn validate_graph_node_value_tensor(
    ir: &Graph,
    id: ValueId,
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    let value = id;
    validate_graph_ir_value_tensor(ir, value, tensor)
}

fn validate_graph_ir_value_tensor(
    ir: &Graph,
    value: ValueId,
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    validate_graph_tensor_metadata(
        "tensor",
        ir.required_value_data_type(value)?,
        ir.required_value_layout(value)?,
        ir.required_value(value)?.device(),
        tensor,
    )
}

pub fn validate_graph_input_value_tensor(
    ir: &Graph,
    id: ValueId,
    tensor: &(impl CudaTensorSource + ?Sized),
) -> Result<()> {
    let value = id;
    let data_type = ir.required_value_data_type(value)?;
    let layout = ir.required_value_layout(value)?;
    let device = ir.required_value(value)?.device();
    if ir.required_value_alias(value)?.is_some() {
        validate_graph_alias_tensor_metadata(data_type, layout, device, tensor)
    } else {
        validate_graph_tensor_metadata("tensor", data_type, layout, device, tensor)
    }
}

pub fn validate_graph_node_value_device(ir: &Graph, output: ValueId, input: ValueId) -> Result<()> {
    let output_value = output;
    let input_value = input;
    let output_device = ir.required_value(output_value)?.device();
    let input_device = ir.required_value(input_value)?.device();
    if input_device != output_device {
        return Err(Error::device_mismatch(
            "cuda graph operand device",
            output_device,
            input_device,
        ));
    }
    Ok(())
}

pub fn validate_graph_node_value_operand(
    ir: &Graph,
    output: ValueId,
    input: ValueId,
) -> Result<()> {
    validate_graph_node_value_device(ir, output, input)?;
    let output_value = output;
    let input_value = input;
    let input_data_type = ir.required_value_data_type(input_value)?;
    let output_data_type = ir.required_value_data_type(output_value)?;
    if input_data_type != output_data_type {
        return Err(Error::data_type_mismatch(
            format!(
                "cuda graph {} operand data type",
                graph_operation_name(ir, output)?
            ),
            output_data_type,
            input_data_type,
        ));
    }
    Ok(())
}

fn graph_operation_name(ir: &Graph, id: ValueId) -> Result<&'static str> {
    Ok(ir
        .required_operation_for_value(id)?
        .schema()
        .map_or("unsupported graph operation", |schema| schema.name()))
}

pub fn validate_graph_matmul(ir: &Graph, id: ValueId, inputs: [ValueId; 2]) -> Result<()> {
    let output_value = id;
    let output_data_type = ir.required_value_data_type(output_value)?;
    let output_device = ir.required_value(output_value)?.device();
    let a_value = inputs[0];
    let b_value = inputs[1];
    let a_data_type = ir.required_value_data_type(a_value)?;
    let b_data_type = ir.required_value_data_type(b_value)?;
    if a_data_type != output_data_type {
        return Err(Error::data_type_mismatch(
            "cuda graph matmul lhs data type",
            output_data_type,
            a_data_type,
        ));
    }

    if b_data_type != output_data_type {
        return Err(Error::data_type_mismatch(
            "cuda graph matmul rhs data type",
            output_data_type,
            b_data_type,
        ));
    }

    let a_device = ir.required_value(a_value)?.device();
    let b_device = ir.required_value(b_value)?.device();
    if a_device != output_device {
        return Err(Error::device_mismatch(
            "cuda graph matmul lhs device",
            output_device,
            a_device,
        ));
    }

    if b_device != output_device {
        return Err(Error::device_mismatch(
            "cuda graph matmul rhs device",
            output_device,
            b_device,
        ));
    }

    let a_shape = ir.required_value_shape(a_value)?;
    let b_shape = ir.required_value_shape(b_value)?;
    let output_shape = ir.required_value_shape(output_value)?;
    let problem = spec::matmul_problem(a_shape, b_shape, "cuda graph matmul")?;
    if output_shape.dimensions() != problem.output_shape.dimensions() {
        return Err(Error::shape_mismatch(
            "cuda graph matmul output dimensions",
            problem.output_shape.dimensions(),
            output_shape.dimensions(),
        ));
    }
    Ok(())
}

pub fn graph_matmul_is_zero_work(ir: &Graph, id: ValueId, inputs: [ValueId; 2]) -> Result<bool> {
    let a_value = inputs[0];
    let b_value = inputs[1];
    let output_value = id;
    let a_shape = ir.required_value_shape(a_value)?;
    let b_shape = ir.required_value_shape(b_value)?;
    let output_shape = ir.required_value_shape(output_value)?;
    let Ok(problem) = spec::matmul_problem(a_shape, b_shape, "cuda graph matmul") else {
        return Ok(false);
    };
    if output_shape.dimensions() != problem.output_shape.dimensions() {
        return Ok(false);
    }

    Ok(problem.is_zero_work())
}

pub fn validate_graph_matmul_add(
    ir: &Graph,
    matmul: ValueId,
    add: ValueId,
    bias: ValueId,
) -> Result<()> {
    validate_graph_node_value_operand(ir, add, matmul)?;
    validate_graph_node_value_operand(ir, add, bias)?;
    let matmul_value = matmul;
    let bias_value = bias;
    let add_value = add;
    let matmul_shape = ir.required_value_shape(matmul_value)?;
    let bias_shape = ir.required_value_shape(bias_value)?;
    let add_shape = ir.required_value_shape(add_value)?;
    let broadcast = spec::binary_broadcast_problem(matmul_shape, bias_shape)?;
    if broadcast.output_shape.dimensions() != add_shape.dimensions() {
        return Err(Error::shape_mismatch(
            "cuda graph matmul add output dimensions",
            broadcast.output_shape.dimensions(),
            add_shape.dimensions(),
        ));
    }
    Ok(())
}
