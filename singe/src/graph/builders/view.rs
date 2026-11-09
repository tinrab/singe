use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        Graph,
        ir::ValueId,
        node::{Node, NodeKind},
    },
    layout::Layout,
    shape::{Shape, normalize_dim_usize},
};

impl Graph {
    // Adds an operation of `kind` whose output has a contiguous layout of `shape`.
    pub(crate) fn add_named_operation(
        &mut self,
        name: Option<String>,
        kind: NodeKind,
        data_type: DataType,
        shape: Shape,
        device: Device,
    ) -> Result<ValueId> {
        self.add_named_node_with_layout(Node::create_with_shape(
            kind, name, data_type, shape, device,
        ))
    }

    pub(crate) fn add_named_slice_operation(
        &mut self,
        name: Option<String>,
        input: ValueId,
        dim: i64,
        start: i64,
        length: i64,
        data_type: DataType,
        shape: &Shape,
        device: Device,
    ) -> Result<ValueId> {
        self.ensure_node(input)?;
        let input_layout = self.required_value_layout(input)?;
        let axis = normalize_dim_usize(dim, input_layout.shape().rank())?;
        let stride = input_layout
            .strides()
            .get(axis)
            .copied()
            .ok_or(Error::DimensionOutOfBounds)?;
        let offset_delta = start
            .checked_mul(stride)
            .ok_or_else(|| Error::out_of_range("graph slice offset".to_string()))?;
        let offset_delta = usize::try_from(offset_delta).map_err(|_| Error::InvalidArgument {
            name: "graph slice start".into(),
        })?;
        let offset = input_layout
            .storage_offset()
            .checked_add(offset_delta)
            .ok_or_else(|| Error::out_of_range("graph slice storage offset".to_string()))?;
        let layout =
            Layout::with_strides(shape.dimensions(), input_layout.strides())?.with_offset(offset);
        self.add_named_node_with_layout(Node::create(
            NodeKind::slice(input, dim, start, length),
            name,
            data_type,
            layout,
            device,
        ))
    }
}
