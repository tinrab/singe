use std::{collections::HashMap, mem};

use singe_codegen::{
    abi::Bindings,
    ir::{
        Access, AxisId, AxisKind, ElementType, Expr, Kernel, Literal, NodeId, OperandId,
        OperandRole, Size, SizeId,
    },
    schedule::Schedule,
};

use crate::{
    backend::kernel_lowering::{
        element_type,
        expression::{ElementValue, Expressions},
        group::ElementGroup,
    },
    error::{Error, Result},
    utility::to_u64,
    value::ValueSpec,
};

// What each launch value of a lowered kernel means, in the order its bindings list them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArgumentLabels {
    pub scalars: Vec<String>,
    pub sizes: Vec<String>,
}

/// The most points a reduction may have and still be part of a kernel's code.
pub const STATIC_REDUCTION: u64 = 16 * 1024;

/// A kernel lowered from one operation, with its launch values, its outputs in operand order, and the schedule it runs with.
pub struct LoweredKernel {
    pub kernel: Kernel,
    pub bindings: Bindings,
    pub labels: ArgumentLabels,
    pub outputs: Vec<ValueSpec>,
    pub schedule: Schedule,
}

/// A kernel over a domain of parallel and reduction dimensions, whose operands each address the domain through their strides.
///
/// Adjacent dimensions of one kind that are contiguous with each other in every operand share one axis.
/// The outermost parallel axis takes its extent at launch, so contiguous row kernels serve every row count, while strides and inner extents are constants of the code.
/// Reductions over at most [`STATIC_REDUCTION`] points are part of the code, so row schedules can keep a row in registers; longer ones take their outermost extent at launch too.
pub struct DomainKernel {
    pub kernel: Kernel,
    pub bindings: Bindings,
    pub labels: ArgumentLabels,
    groups: Vec<Group>,
    axes: Vec<AxisId>,
    /// Static sizes declared by value, so equal values share one.
    sizes: HashMap<u64, SizeId>,
    /// The outputs stored after the operation's own, in operand order.
    stored: Vec<ValueSpec>,
    /// The first input operand added for each column, which [`Self::shared_input`] reads again.
    column_operands: HashMap<usize, OperandId>,
}

#[derive(Debug, Clone)]
struct Group {
    extent: u64,
    kind: AxisKind,
    /// The innermost stride of the group in each operand.
    strides: Vec<u64>,
    /// The domain dimensions the group merges, innermost first.
    dimensions: Vec<usize>,
}

impl DomainKernel {
    pub fn create(
        name: &str,
        dimensions: &[i64],
        kinds: &[AxisKind],
        strides: &[Vec<i64>],
    ) -> Result<Self> {
        let mut groups: Vec<Group> = Vec::new();
        // A domain with an empty parallel dimension has no work, so it needs no strides.
        let no_work = dimensions
            .iter()
            .zip(kinds)
            .any(|(&dimension, &kind)| kind == AxisKind::Parallel && dimension == 0);
        for index in (0..dimensions.len()).rev() {
            if no_work {
                break;
            }

            let extent = to_u64(dimensions[index], "cuda generated kernel dimension")?;
            if extent == 1 {
                continue;
            }

            let operand_strides = strides
                .iter()
                .map(|strides| to_u64(strides[index], "cuda generated kernel stride"))
                .collect::<Result<Vec<_>>>()?;
            if let Some(inner) = groups.last_mut()
                && inner.kind == kinds[index]
                && operand_strides
                    .iter()
                    .zip(&inner.strides)
                    .all(|(&outer, &stride)| stride.checked_mul(inner.extent) == Some(outer))
            {
                inner.extent *= extent;
                inner.dimensions.push(index);
                continue;
            }

            groups.push(Group {
                extent,
                kind: kinds[index],
                strides: operand_strides,
                dimensions: vec![index],
            });
        }

        // Each kind keeps at least one axis, so every kernel has a parallel axis and every reduction a reduction axis.
        for kind in [AxisKind::Reduction, AxisKind::Parallel] {
            if kinds.contains(&kind) && !groups.iter().any(|group| group.kind == kind) {
                let empty = dimensions
                    .iter()
                    .zip(kinds)
                    .any(|(&dimension, &axis_kind)| axis_kind == kind && dimension == 0);
                groups.push(Group {
                    extent: u64::from(!empty),
                    kind,
                    strides: vec![0; strides.len()],
                    dimensions: Vec::new(),
                });
            }
        }

        if !groups.iter().any(|group| group.kind == AxisKind::Parallel) {
            groups.push(Group {
                extent: 1,
                kind: AxisKind::Parallel,
                strides: vec![0; strides.len()],
                dimensions: Vec::new(),
            });
        }

        groups.reverse();
        let mut builder = Self {
            kernel: Kernel::new(name),
            bindings: Bindings::default(),
            labels: ArgumentLabels::default(),
            groups,
            axes: Vec::new(),
            sizes: HashMap::new(),
            stored: Vec::new(),
            column_operands: HashMap::new(),
        };
        let reduced = builder
            .groups
            .iter()
            .filter(|group| group.kind == AxisKind::Reduction)
            .map(|group| group.extent)
            .product::<u64>();
        // A static reduction is part of the code, as if its outermost extent were already seen.
        let mut seen = if reduced <= STATIC_REDUCTION {
            vec![AxisKind::Reduction]
        } else {
            Vec::new()
        };
        let groups = mem::take(&mut builder.groups);
        for group in &groups {
            let extent = if seen.contains(&group.kind) {
                builder.size(group.extent)
            } else {
                seen.push(group.kind);
                // Strides and inner extents never share an argument, so they stay constants of the code.
                builder.bindings.sizes.push(group.extent);
                builder.labels.sizes.push(
                    match group.kind {
                        AxisKind::Parallel => "outermost parallel extent",
                        AxisKind::Reduction => "outermost reduction extent",
                    }
                    .to_owned(),
                );
                builder.kernel.add_size(Size::Argument)
            };
            let axis = builder.kernel.add_axis(extent, group.kind);
            builder.axes.push(axis);
        }

        builder.groups = groups;
        Ok(builder)
    }

    /// Adds an input operand of `element` whose strides are column `column`, and returns its access.
    pub fn input(&mut self, element: ElementType, column: usize) -> Access {
        let operand = self.kernel.add_operand(element, OperandRole::Input);
        self.column_operands.entry(column).or_insert(operand);
        self.access(operand, column)
    }

    /// Returns an access of the input operand of column `column`, adding one of `element` when the column has none, so reading a column twice reads one operand.
    pub fn shared_input(&mut self, element: ElementType, column: usize) -> Access {
        match self.column_operands.get(&column) {
            Some(&operand) => self.access(operand, column),
            None => self.input(element, column),
        }
    }

    /// Loads `group`'s inputs, whose strides are the first columns, computes its steps, and returns its value in its compute type.
    ///
    /// # Errors
    ///
    /// Returns an error if the group has no inputs, an input has a data type kernels cannot hold, or a step cannot be built.
    pub fn group_value(&mut self, group: &ElementGroup<'_>) -> Result<NodeId> {
        let mut loaded = Vec::with_capacity(group.inputs.len());
        for (column, &(data_type, _)) in group.inputs.iter().enumerate() {
            let access = self.input(element_type(data_type)?, column);
            loaded.push(ElementValue {
                node: self.kernel.load(access),
                data_type,
            });
        }

        let value = if group.steps.is_empty() {
            *loaded
                .first()
                .ok_or_else(|| Error::missing_input("group value"))?
        } else {
            let mut expressions = Expressions {
                kernel: &mut self.kernel,
                bindings: &mut self.bindings,
                labels: &mut self.labels,
                shape: &group.shape,
            };
            *expressions
                .steps(&loaded, group.steps)?
                .last()
                .ok_or_else(|| Error::missing_node("group value"))?
        };
        Ok(self.convert(value.node, element_type(group.data_type)?.compute()))
    }

    /// Adds an input operand of `element` whose strides are column `column`, read along `axis` at the point `index` computes instead of the kernel's own, `stride` elements apart.
    pub fn input_at(
        &mut self,
        element: ElementType,
        column: usize,
        axis: AxisId,
        index: NodeId,
        stride: i64,
    ) -> Result<Access> {
        let mut access = self.shared_input(element, column);
        access.strides.retain(|(walked, _)| *walked != axis);
        if stride != 0 {
            let stride = to_u64(stride, "cuda generated kernel stride")?;
            let stride = self.size(stride);
            access.indirect.push((index, stride));
        }
        Ok(access)
    }

    /// Returns `group`'s value as [`Self::group_value`] does, but at the point of `axis` that `index` computes, the group's inputs `strides` apart along it.
    ///
    /// # Errors
    ///
    /// Returns an error if the group has no inputs, an input has a data type kernels cannot hold, or a step cannot be built.
    pub fn group_value_at(
        &mut self,
        group: &ElementGroup<'_>,
        axis: AxisId,
        index: NodeId,
        strides: &[i64],
    ) -> Result<NodeId> {
        let mut loaded = Vec::with_capacity(group.inputs.len());
        for (column, &(data_type, _)) in group.inputs.iter().enumerate() {
            let access = self.input_at(
                element_type(data_type)?,
                column,
                axis,
                index,
                strides[column],
            )?;
            loaded.push(ElementValue {
                node: self.kernel.load(access),
                data_type,
            });
        }

        let value = if group.steps.is_empty() {
            *loaded
                .first()
                .ok_or_else(|| Error::missing_input("group value at"))?
        } else {
            let mut expressions = Expressions {
                kernel: &mut self.kernel,
                bindings: &mut self.bindings,
                labels: &mut self.labels,
                shape: &group.shape,
            };
            *expressions
                .steps(&loaded, group.steps)?
                .last()
                .ok_or_else(|| Error::missing_node("group value at"))?
        };
        Ok(self.convert(value.node, element_type(group.data_type)?.compute()))
    }

    /// Stores `value`, the value of `group`, into a new output whose strides are column `column`, when the group asks to be stored.
    ///
    /// # Errors
    ///
    /// Returns an error if the group's data type is one kernels cannot hold.
    pub fn store_group(
        &mut self,
        group: &ElementGroup<'_>,
        value: NodeId,
        column: usize,
    ) -> Result<()> {
        if group.stored {
            let element = element_type(group.data_type)?;
            let access = self.output(element, column);
            self.kernel.store(access, value);
            self.stored
                .push(ValueSpec::new(group.data_type, group.shape.clone()));
        }
        Ok(())
    }

    /// Adds an output operand of `element` whose strides are column `column`, and returns its access.
    pub fn output(&mut self, element: ElementType, column: usize) -> Access {
        let operand = self.kernel.add_operand(element, OperandRole::Output);
        self.access(operand, column)
    }

    fn access(&mut self, operand: OperandId, column: usize) -> Access {
        let strides = self.groups.iter().map(|group| group.strides[column]);
        strided_access(
            &mut self.kernel,
            &mut self.sizes,
            operand,
            &self.axes,
            strides,
        )
    }

    pub fn reduction_axes(&self) -> Vec<AxisId> {
        self.groups
            .iter()
            .zip(&self.axes)
            .filter(|(group, _)| group.kind == AxisKind::Reduction)
            .map(|(_, axis)| *axis)
            .collect()
    }

    /// Returns the axis of domain dimension `dimension` when it has one of its own, merged with no other; a dimension of one has none.
    pub fn dimension_axis(&self, dimension: usize) -> Option<AxisId> {
        self.groups
            .iter()
            .zip(&self.axes)
            .find(|(group, _)| group.dimensions == [dimension])
            .map(|(_, axis)| *axis)
    }

    /// Returns the innermost axis of the domain, the last dimension's unless it merged with the one before it.
    pub fn innermost_axis(&self) -> Option<AxisId> {
        self.axes.last().copied()
    }

    /// Returns the number of domain points along the axes of `kind`, as a float.
    pub fn count(&self, kind: AxisKind) -> f64 {
        self.groups
            .iter()
            .filter(|group| group.kind == kind)
            .map(|group| group.extent as f64)
            .product()
    }

    pub fn scalar(&mut self, element: ElementType, value: f64, label: &str) -> NodeId {
        let scalar = self.kernel.add_scalar(element);
        self.bindings.scalars.push(Literal::Float(value));
        self.labels.scalars.push(label.to_owned());
        self.kernel.add_node(element, Expr::Scalar(scalar))
    }

    /// Adds a floating-point constant of `element`.
    pub fn constant(&mut self, element: ElementType, value: f64) -> NodeId {
        self.kernel.literal(element, Literal::Float(value))
    }

    /// Returns `node` converted to `element`, or `node` itself when it already has that type.
    pub fn convert(&mut self, node: NodeId, element: ElementType) -> NodeId {
        converted(&mut self.kernel, node, element)
    }

    /// Returns a static size of `value`, reusing a declared one.
    pub fn size(&mut self, value: u64) -> SizeId {
        static_size(&mut self.kernel, &mut self.sizes, value)
    }

    /// Returns the finished kernel, whose own output `output` precedes the groups it stored.
    pub fn finish(self, output: ValueSpec, schedule: Schedule) -> LoweredKernel {
        let mut outputs = vec![output];
        outputs.extend(self.stored);
        LoweredKernel {
            kernel: self.kernel,
            bindings: self.bindings,
            labels: self.labels,
            outputs,
            schedule,
        }
    }
}

// A static size of `value`, reusing a declared one.
pub fn static_size(kernel: &mut Kernel, sizes: &mut HashMap<u64, SizeId>, value: u64) -> SizeId {
    *sizes
        .entry(value)
        .or_insert_with(|| kernel.add_size(Size::Static(value)))
}

// `node` converted to `element`, or `node` itself when it already has that type.
pub fn converted(kernel: &mut Kernel, node: NodeId, element: ElementType) -> NodeId {
    if kernel.nodes[node.0].element == element {
        node
    } else {
        kernel.cast(element, node)
    }
}

// An access of `operand` whose strides along `axes` are `strides`, leaving out the zero ones.
pub fn strided_access(
    kernel: &mut Kernel,
    sizes: &mut HashMap<u64, SizeId>,
    operand: OperandId,
    axes: &[AxisId],
    strides: impl Iterator<Item = u64>,
) -> Access {
    let strides = axes
        .iter()
        .zip(strides)
        .filter(|(_, stride)| *stride != 0)
        .map(|(&axis, stride)| (axis, static_size(kernel, sizes, stride)))
        .collect();
    Access {
        operand,
        base: None,
        indirect: Vec::new(),
        strides,
    }
}
