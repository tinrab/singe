//! Epilogues of row kernels: element-wise work after a normalization, stored in place of the normalization's output.
//!
//! An epilogue may read the row's output at other points of the row through slices and concatenations along the row's axis, as the rotary embedding after a head's norm reads both halves of the head, and through transposes of the axes before it.
//! The kernel recomputes the normalized value at those points from the row's input and its reductions, so the row is never stored.
//! Every value of the epilogue is indexed by the kernel's parallel axes, in an order its transposes give, and by a point of the row's axis: the kernel's own point, or one computed from it.

use std::collections::HashMap;

use singe_codegen::ir::{AxisId, BinaryOp, ElementType, Expr, Literal, NodeId};

use crate::{
    backend::{
        kernel_lowering::{
            domain::DomainKernel,
            element_type,
            expression::{ElementValue, Expressions},
        },
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
};

/// Element-wise work after a row operation, as values in order; the last is the one the kernel stores.
#[derive(Debug, Clone)]
pub struct RowEpilogue<'a> {
    /// The operands the epilogue reads from outside the kernel.
    pub inputs: Vec<(DataType, &'a Layout)>,
    pub nodes: Vec<EpilogueNode<'a>>,
}

/// One value of an epilogue and its shape.
#[derive(Debug, Clone)]
pub struct EpilogueNode<'a> {
    pub value: EpilogueValue<'a>,
    pub shape: Shape,
}

/// What an epilogue value is, naming earlier values by position.
#[derive(Debug, Clone)]
pub enum EpilogueValue<'a> {
    /// The row operation's output.
    Row,
    /// The epilogue's input in this slot, broadcast to the shape of the step that reads it.
    Input(usize),
    /// The input of the row operation's group in this slot, of `data_type`, at the kernel's own point.
    GroupInput { slot: usize, data_type: DataType },
    /// The points of `value` along the row's axis from `start` on.
    Slice { value: usize, start: i64 },
    /// `value` with two of the axes before the row's axis swapped.
    Transpose { value: usize, dims: [usize; 2] },
    /// `values` one after another along the row's axis.
    Concat { values: Vec<usize> },
    /// An element-wise operation of earlier values.
    Step {
        operation: Operation,
        attributes: &'a OperationAttributes,
        operands: Vec<usize>,
    },
}

/// A point of the row's axis: the kernel's own, or one a node computes as an `I64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RowPoint {
    Own,
    At(NodeId),
}

impl RowEpilogue<'_> {
    /// Returns, for every value, the parallel axis of the row each of its axes before the row's axis is.
    ///
    /// # Errors
    ///
    /// Returns an error if a transpose names the row's axis, or values an operation combines are transposed differently.
    pub fn orders(&self, rank: usize) -> Result<Vec<Option<Vec<usize>>>> {
        let leading = rank
            .checked_sub(1)
            .ok_or_else(|| Error::missing_node("orders"))?;
        let mut orders: Vec<Option<Vec<usize>>> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let order = match &node.value {
                EpilogueValue::Row => Some((0..leading).collect()),
                EpilogueValue::Input(_) | EpilogueValue::GroupInput { .. } => None,
                EpilogueValue::Slice { value, .. } => orders[*value].clone(),
                EpilogueValue::Transpose { value, dims } => {
                    if dims.iter().any(|&dim| dim >= leading) {
                        return Err(Error::unsupported_operation(
                            "row epilogue transpose of the row's axis",
                        ));
                    }
                    orders[*value].clone().map(|mut order| {
                        order.swap(dims[0], dims[1]);
                        order
                    })
                }
                EpilogueValue::Concat { values }
                | EpilogueValue::Step {
                    operands: values, ..
                } => {
                    let mut known = values.iter().filter_map(|&value| orders[value].clone());
                    let first = known.next();
                    if known.any(|order| Some(&order) != first.as_ref()) {
                        return Err(Error::unsupported_operation(
                            "row epilogue values transposed differently",
                        ));
                    }
                    first
                }
            };
            orders.push(order);
        }
        Ok(orders)
    }

    /// Returns the strides over the row's domain of a layout of `shape` whose axes before the last are the row's axes `order`.
    #[must_use]
    pub fn domain_strides(layout: &Layout, order: &[usize]) -> Vec<i64> {
        let strides = layout.strides();
        let mut domain = vec![0; strides.len()];
        for (axis, &row_axis) in order.iter().enumerate() {
            domain[row_axis] = strides[axis];
        }

        if let (Some(last), Some(&stride)) = (domain.last_mut(), strides.last()) {
            *last = stride;
        }
        domain
    }
}

/// Builds an epilogue's values into a normalization kernel.
pub struct EpilogueWriter<'a, 'b> {
    pub epilogue: &'a RowEpilogue<'b>,
    /// The row operation's data type, which every value has.
    pub data_type: DataType,
    /// The kernel's axis along the row.
    pub axis: AxisId,
    /// The kernel's column of each step's input operand, by step and operand position.
    pub columns: HashMap<(usize, usize), usize>,
    /// The stride along the row's axis of each of those columns.
    pub row_strides: HashMap<usize, i64>,
    memo: HashMap<(usize, RowPoint), NodeId>,
}

impl<'a, 'b> EpilogueWriter<'a, 'b> {
    #[must_use]
    pub fn new(
        epilogue: &'a RowEpilogue<'b>,
        data_type: DataType,
        axis: AxisId,
        columns: HashMap<(usize, usize), usize>,
        row_strides: HashMap<usize, i64>,
    ) -> Self {
        Self {
            epilogue,
            data_type,
            axis,
            columns,
            row_strides,
            memo: HashMap::new(),
        }
    }

    /// Returns the value of epilogue node `node` at `point`, in the compute type, where `row` computes the row operation's output at a point.
    ///
    /// # Errors
    ///
    /// Returns an error if a step cannot be built or a value has a type kernels cannot hold.
    pub fn value(
        &mut self,
        builder: &mut DomainKernel,
        node: usize,
        point: RowPoint,
        row: &mut dyn FnMut(&mut DomainKernel, RowPoint) -> Result<NodeId>,
    ) -> Result<NodeId> {
        if let Some(&value) = self.memo.get(&(node, point)) {
            return Ok(value);
        }

        let epilogue = self.epilogue;
        let value = match &epilogue.nodes[node].value {
            EpilogueValue::Row => row(builder, point)?,
            EpilogueValue::Input(_) => {
                return Err(Error::invalid_argument("row epilogue input"));
            }
            EpilogueValue::GroupInput { slot, data_type } => {
                if point != RowPoint::Own {
                    return Err(Error::unsupported_operation(
                        "row epilogue group input at another point",
                    ));
                }

                let access = builder.shared_input(element_type(*data_type)?, *slot);
                let value = builder.kernel.load(access);
                builder.convert(value, element_type(self.data_type)?.compute())
            }
            EpilogueValue::Slice { value, start } => {
                let shifted = self.offset(builder, point, *start);
                self.value(builder, *value, RowPoint::At(shifted), row)?
            }
            EpilogueValue::Transpose { value, .. } => self.value(builder, *value, point, row)?,
            EpilogueValue::Concat { values } => {
                let index = self.index(builder, point);
                let mut start = 0;
                let mut result = None;
                for &value in values {
                    let length = epilogue.nodes[value]
                        .shape
                        .dimensions()
                        .last()
                        .copied()
                        .unwrap_or(1);
                    // The point inside this value, clamped so the values a later selection discards still read inside it.
                    let local = self.offset(builder, point, -start);
                    let zero = Self::integer(builder, 0);
                    let last = Self::integer(builder, length - 1);
                    let local = builder.kernel.binary(BinaryOp::Max, local, zero);
                    let local = builder.kernel.binary(BinaryOp::Min, local, last);
                    let part = self.value(builder, value, RowPoint::At(local), row)?;
                    result = Some(match result {
                        None => part,
                        Some(earlier) => {
                            let bound = Self::integer(builder, start);
                            let inside =
                                builder.kernel.binary(BinaryOp::GreaterEqual, index, bound);
                            builder.kernel.select(inside, part, earlier)
                        }
                    });
                    start += length;
                }
                result.ok_or_else(|| Error::missing_node("value"))?
            }
            EpilogueValue::Step {
                operation,
                attributes,
                operands,
            } => {
                let mut inputs = Vec::with_capacity(operands.len());
                for (position, &operand) in operands.iter().enumerate() {
                    let node = match epilogue.nodes[operand].value {
                        EpilogueValue::Input(slot) => {
                            self.load(builder, node, position, slot, point)?
                        }
                        _ => self.value(builder, operand, point, row)?,
                    };
                    inputs.push(ElementValue {
                        node,
                        data_type: self.data_type,
                    });
                }

                let shape = epilogue.nodes[node].shape.clone();
                let mut expressions = Expressions {
                    kernel: &mut builder.kernel,
                    bindings: &mut builder.bindings,
                    labels: &mut builder.labels,
                    shape: &shape,
                };
                let value = expressions.apply(*operation, attributes, &inputs)?;
                builder.convert(value.node, element_type(self.data_type)?.compute())
            }
        };
        self.memo.insert((node, point), value);
        Ok(value)
    }

    /// Loads the epilogue input in `slot`, read by operand `position` of step `step`, at `point`.
    fn load(
        &self,
        builder: &mut DomainKernel,
        step: usize,
        position: usize,
        slot: usize,
        point: RowPoint,
    ) -> Result<NodeId> {
        let (data_type, _) = self.epilogue.inputs[slot];
        let column = *self
            .columns
            .get(&(step, position))
            .ok_or_else(|| Error::missing_input("load"))?;
        let element = element_type(data_type)?;
        // The kernel's own point reads the input directly; any other point reads it through a computed index.
        let access = if point == RowPoint::Own {
            builder.shared_input(element, column)
        } else {
            let index = self.index(builder, point);
            let stride = self.row_strides.get(&column).copied().unwrap_or(0);
            builder.input_at(element, column, self.axis, index, stride)?
        };
        let value = builder.kernel.load(access);
        Ok(builder.convert(value, element_type(self.data_type)?.compute()))
    }

    /// Returns `point` as an `I64` node.
    pub fn index(&self, builder: &mut DomainKernel, point: RowPoint) -> NodeId {
        match point {
            RowPoint::Own => builder
                .kernel
                .add_node(ElementType::I64, Expr::AxisIndex(self.axis)),
            RowPoint::At(node) => node,
        }
    }

    /// Returns `point` moved `offset` points along the row, as an `I64` node.
    fn offset(&self, builder: &mut DomainKernel, point: RowPoint, offset: i64) -> NodeId {
        let index = self.index(builder, point);
        if offset == 0 {
            return index;
        }

        let offset = Self::integer(builder, offset);
        builder.kernel.binary(BinaryOp::Add, index, offset)
    }

    fn integer(builder: &mut DomainKernel, value: i64) -> NodeId {
        builder
            .kernel
            .literal(ElementType::I64, Literal::Int(value))
    }
}
