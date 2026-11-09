use std::collections::HashMap;

use singe_codegen::{
    abi::Bindings,
    ir::{
        Access, AxisId, AxisKind, ElementType, Kernel, NodeId, OperandId, OperandRole, Size, SizeId,
    },
    schedule::element_schedule,
};

use crate::{
    backend::{
        kernel_lowering::{
            ArgumentLabels, LoweredKernel,
            domain::{static_size, strided_access},
            element_type,
            expression::{ElementValue, Expressions},
            gather::bits_type,
            group::FusedStep,
        },
        registry::{Operation, OperationAttributes},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    utility::to_u64,
    value::ValueSpec,
};

/// An element-wise kernel being built over one output shape.
///
/// Dimensions that are contiguous with each other in every input and in the contiguous outputs share one axis, and the outermost axis's extent is passed at launch, so kernels over contiguous operands do not depend on their shape.
/// Values stay in their compute type between operations, as one fused kernel holds them in registers, and round only when stored.
pub struct ElementWiseKernel {
    kernel: Kernel,
    bindings: Bindings,
    labels: ArgumentLabels,
    shape: Shape,
    /// The dimensions of `shape` each axis covers, outermost first.
    groups: Vec<Group>,
    axes: Vec<AxisId>,
    /// Sizes already declared, by value, so equal strides share one size.
    sizes: HashMap<u64, SizeId>,
}

/// Dimensions of the output merged into one axis.
#[derive(Debug, Clone)]
struct Group {
    extent: u64,
    /// The stride of the group's innermost dimension in each input, then in the outputs.
    strides: Vec<u64>,
}

impl ElementWiseKernel {
    /// Starts a kernel named `name` over `shape` that reads `inputs`, each broadcast to `shape` through its layout, and returns it with the loaded input values.
    ///
    /// # Errors
    ///
    /// Returns an error if an input cannot broadcast to `shape` or has a data type kernels cannot hold.
    pub fn create(
        name: &str,
        shape: &Shape,
        inputs: &[(DataType, &Layout)],
    ) -> Result<(Self, Vec<ElementValue>)> {
        Self::create_into(name, &Layout::from_shape(shape), inputs)
    }

    /// Starts a kernel like [`Self::create`] whose outputs are stored through `output`'s strides; `output`'s storage offset is ignored.
    ///
    /// # Errors
    ///
    /// Returns an error if an input cannot broadcast to `output`'s shape, `output` writes one element at several positions, or an input has a data type kernels cannot hold.
    pub fn create_into(
        name: &str,
        output: &Layout,
        inputs: &[(DataType, &Layout)],
    ) -> Result<(Self, Vec<ElementValue>)> {
        let shape = output.shape();
        if output
            .dimensions()
            .iter()
            .zip(output.strides())
            .any(|(&dimension, &stride)| dimension > 1 && stride == 0)
        {
            return Err(Error::invalid_argument("cuda kernel output layout"));
        }

        let layouts = inputs
            .iter()
            .map(|(_, layout)| layout.expand(shape))
            .collect::<Result<Vec<_>>>()?;
        let groups = merge_dimensions(shape, layouts.iter().chain([output]))?;
        let mut builder = Self {
            kernel: Kernel::new(name),
            bindings: Bindings::default(),
            labels: ArgumentLabels::default(),
            shape: shape.clone(),
            groups,
            axes: Vec::new(),
            sizes: HashMap::new(),
        };
        let extents = builder
            .groups
            .iter()
            .map(|group| group.extent)
            .collect::<Vec<_>>();
        for (position, extent) in extents.into_iter().enumerate() {
            // The outermost extent is a launch argument; the others are part of the code.
            let extent = if position == 0 {
                builder.bindings.sizes.push(extent);
                builder.labels.sizes.push("outermost extent".to_owned());
                builder.kernel.add_size(Size::Argument)
            } else {
                builder.size(extent)
            };
            let axis = builder.kernel.add_axis(extent, AxisKind::Parallel);
            builder.axes.push(axis);
        }

        let mut values = Vec::with_capacity(inputs.len());
        for (index, (data_type, _)) in inputs.iter().enumerate() {
            let operand = builder
                .kernel
                .add_operand(element_type(*data_type)?, OperandRole::Input);
            let access = builder.access(operand, index);
            let node = builder.kernel.load(access);
            values.push(ElementValue {
                node,
                data_type: *data_type,
            });
        }
        Ok((builder, values))
    }

    /// Applies element-wise `operation` with `attributes` to `inputs` and returns its value.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Expressions::apply`] does.
    pub fn apply(
        &mut self,
        operation: Operation,
        attributes: &OperationAttributes,
        inputs: &[ElementValue],
    ) -> Result<ElementValue> {
        self.expressions().apply(operation, attributes, inputs)
    }

    /// Computes `steps` from `loaded`, the kernel's loaded inputs, and returns every step's value.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Expressions::steps`] does.
    pub fn steps(
        &mut self,
        loaded: &[ElementValue],
        steps: &[FusedStep<'_>],
    ) -> Result<Vec<ElementValue>> {
        self.expressions().steps(loaded, steps)
    }

    /// Stores `value` into a new contiguous output of `data_type` over the kernel's shape.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is one kernels cannot hold.
    pub fn store(&mut self, value: ElementValue, data_type: DataType) -> Result<()> {
        let element = element_type(data_type)?;
        let operand = self.kernel.add_operand(element, OperandRole::Output);
        // Every output shares the contiguous layout, which follows the inputs' layouts in each group.
        let column = self
            .kernel
            .operands
            .iter()
            .filter(|operand| operand.role == OperandRole::Input)
            .count();
        let access = self.access(operand, column);
        let node = self.convert(value.node, element.compute());
        self.kernel.store(access, node);
        Ok(())
    }

    /// Stores each value of `slices` into one new output of `data_type`, at its offset in elements.
    ///
    /// The offsets are passed at launch, so one compiled kernel serves every placement of the slices.
    ///
    /// # Errors
    ///
    /// Returns an error if `data_type` is one kernels cannot hold.
    pub fn store_slices(
        &mut self,
        data_type: DataType,
        slices: &[(ElementValue, u64)],
    ) -> Result<()> {
        let element = element_type(data_type)?;
        let operand = self.kernel.add_operand(element, OperandRole::Output);
        let column = self
            .kernel
            .operands
            .iter()
            .filter(|operand| operand.role == OperandRole::Input)
            .count();
        for &(value, offset) in slices {
            self.bindings.sizes.push(offset);
            self.labels.sizes.push("slice offset".to_owned());
            let base = self.kernel.add_size(Size::Argument);
            let access = Access {
                base: Some(base),
                ..self.access(operand, column)
            };
            let node = self.convert(value.node, element.compute());
            self.kernel.store(access, node);
        }
        Ok(())
    }

    /// Returns the finished kernel, the launch values of its argument sizes and scalars, and what each one means.
    pub fn finish(self) -> (Kernel, Bindings, ArgumentLabels) {
        (self.kernel, self.bindings, self.labels)
    }

    /// Returns the access of operand `operand`, whose strides are column `column` of each group.
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

    fn size(&mut self, value: u64) -> SizeId {
        static_size(&mut self.kernel, &mut self.sizes, value)
    }

    fn expressions(&mut self) -> Expressions<'_> {
        Expressions {
            kernel: &mut self.kernel,
            bindings: &mut self.bindings,
            labels: &mut self.labels,
            shape: &self.shape,
        }
    }

    /// Returns `node` converted to `element`, or `node` itself when it already has that type.
    fn convert(&mut self, node: NodeId, element: ElementType) -> NodeId {
        self.expressions().convert(node, element)
    }
}

/// Lowers copying `source`, of `data_type`, into the elements `view` addresses `offset` elements into a buffer that `buffer` describes.
///
/// The copy moves each element's bits as an unsigned integer of its size, so one kernel serves every data type of that size, and it reads and writes up to 16 bytes at once where both sides are contiguous.
///
/// # Errors
///
/// Returns an error if `source` cannot broadcast to `view`'s shape or `view` writes one element at several positions.
pub fn copy(
    buffer: ValueSpec,
    view: &Layout,
    offset: usize,
    data_type: DataType,
    source: &Layout,
) -> Result<LoweredKernel> {
    let bits = match bits_type(data_type) {
        ElementType::U8 => DataType::U8,
        ElementType::U16 => DataType::U16,
        ElementType::U32 => DataType::U32,
        _ => DataType::U64,
    };
    let offset = to_u64(offset, "cuda copy offset")?;
    let (mut kernel, values) =
        ElementWiseKernel::create_into("singe_copy", view, &[(bits, source)])?;
    kernel.store_slices(bits, &[(values[0], offset)])?;
    let (kernel, bindings, labels) = kernel.finish();
    let schedule = element_schedule(&kernel, 128);
    Ok(LoweredKernel {
        kernel,
        bindings,
        labels,
        outputs: vec![buffer],
        schedule,
    })
}

/// Merges the dimensions of `shape` that are contiguous with each other in every one of `layouts`, dropping dimensions of size one.
///
/// A shape with no elements, or one that is all size-one dimensions, has one group.
fn merge_dimensions<'a>(
    shape: &Shape,
    layouts: impl Iterator<Item = &'a Layout> + Clone,
) -> Result<Vec<Group>> {
    let dimensions = shape.dimensions();
    let mut groups: Vec<Group> = Vec::new();
    for dimension in (0..dimensions.len()).rev() {
        let extent = to_u64(dimensions[dimension], "cuda element-wise dimension")?;
        if extent == 1 {
            continue;
        }

        let strides = layouts
            .clone()
            .map(|layout| to_u64(layout.strides()[dimension], "cuda element-wise stride"))
            .collect::<Result<Vec<_>>>()?;
        if let Some(inner) = groups.last_mut()
            && strides
                .iter()
                .zip(&inner.strides)
                .all(|(&outer, &inner_stride)| {
                    inner_stride.checked_mul(inner.extent) == Some(outer)
                })
        {
            inner.extent *= extent;
            continue;
        }

        groups.push(Group { extent, strides });
    }

    if groups.is_empty() {
        let width = layouts.count();
        groups.push(Group {
            extent: u64::from(shape.element_count() != 0),
            strides: vec![0; width],
        });
    }

    // Merging walked inner dimensions first; the kernel lists axes outermost first.
    groups.reverse();
    Ok(groups)
}
