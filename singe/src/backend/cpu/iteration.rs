use crate::{
    device::Device,
    error::{Error, Result},
    layout::Layout,
    parallel,
    shape::Shape,
    storage::cpu::CpuSlice,
    tensor::{Tensor, TensorDataType},
    utility::to_usize,
};

/// Shared logical iteration over checked, allocation-relative operand layouts.
pub struct ElementwisePlan {
    shape: Shape,
    operands: Vec<Operand>,
}

struct Operand {
    layout: Layout,
    linear_start: Option<usize>,
}

impl ElementwisePlan {
    pub(crate) fn create(inputs: &[&Tensor]) -> Result<Self> {
        let mut shape = Shape::scalar();
        for input in inputs {
            if input.device() != Device::Cpu {
                return Err(Error::device_mismatch(
                    "cpu operation input",
                    Device::Cpu,
                    input.device(),
                ));
            }

            shape = shape.broadcast_with(input.shape())?;
        }

        let operands = inputs
            .iter()
            .map(|input| {
                let layout = input.layout().expand(&shape)?;
                let linear_start = layout.is_contiguous().then_some(layout.storage_offset());
                Ok(Operand {
                    layout,
                    linear_start,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { shape, operands })
    }

    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns `produce(linear)` for every logical element, in parallel when there are enough elements.
    pub(crate) fn collect<T: Send>(&self, produce: impl Fn(usize) -> T + Sync) -> Vec<T> {
        parallel::collect(self.shape.element_count(), produce)
    }

    pub(crate) fn offset(&self, operand: usize, linear: usize) -> usize {
        let operand = &self.operands[operand];
        operand.linear_start.map_or_else(
            || operand.layout.linear_offset(linear),
            |start| start + linear,
        )
    }
}

pub fn typed_values<T: TensorDataType>(input: &Tensor) -> Result<CpuSlice<'_, T>> {
    if input.data_type() != T::data_type() {
        return Err(Error::data_type_mismatch(
            "cpu storage typed slice",
            T::data_type(),
            input.data_type(),
        ));
    }
    input
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::unsupported_operation("cpu storage borrow"))?
        .as_typed_slice::<T>()
}

/// The sliding window of a two-dimensional convolution or pooling over the last two dimensions of its input.
#[derive(Debug, Clone, Copy)]
pub struct Window2d {
    input: [usize; 2],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding_start: [usize; 2],
    padding_end: [usize; 2],
    dilation: [usize; 2],
}

impl Window2d {
    /// Creates a window from heights and widths, with padding given as top, bottom, left, and right.
    ///
    /// # Errors
    ///
    /// Returns an error if a value is negative.
    pub(crate) fn create(
        input: (i64, i64),
        kernel: (i64, i64),
        stride: (i64, i64),
        padding: (i64, i64, i64, i64),
        dilation: (i64, i64),
    ) -> Result<Self> {
        let pair = |name: &'static str, (height, width): (i64, i64)| -> Result<[usize; 2]> {
            let convert = |value: i64| to_usize(value, name);
            Ok([convert(height)?, convert(width)?])
        };
        Ok(Self {
            input: pair("cpu window input size", input)?,
            kernel: pair("cpu window kernel size", kernel)?,
            stride: pair("cpu window stride", stride)?,
            padding_start: pair("cpu window padding", (padding.0, padding.2))?,
            padding_end: pair("cpu window padding", (padding.1, padding.3))?,
            dilation: pair("cpu window dilation", dilation)?,
        })
    }

    /// Calls `visit(kernel_row, kernel_column, input_row, input_column)` for each kernel tap of output position `(row, column)` that lies inside the input, in row-major kernel order.
    pub(crate) fn for_each_tap(
        &self,
        row: usize,
        column: usize,
        mut visit: impl FnMut(usize, usize, usize, usize),
    ) {
        for kernel_row in 0..self.kernel[0] {
            let Some(input_row) = self.input_position(0, row, kernel_row) else {
                continue;
            };
            for kernel_column in 0..self.kernel[1] {
                if let Some(input_column) = self.input_position(1, column, kernel_column) {
                    visit(kernel_row, kernel_column, input_row, input_column);
                }
            }
        }
    }

    /// Returns the number of kernel taps of output position `(row, column)` inside the padded input, counting padding but not positions past it.
    pub(crate) fn padded_taps(&self, row: usize, column: usize) -> usize {
        let extent = |axis: usize, position: usize| {
            let start = position * self.stride[axis];
            let end = (start + self.kernel[axis])
                .min(self.input[axis] + self.padding_start[axis] + self.padding_end[axis]);
            end.saturating_sub(start)
        };
        extent(0, row) * extent(1, column)
    }

    fn input_position(&self, axis: usize, output: usize, kernel: usize) -> Option<usize> {
        let padded = output * self.stride[axis] + kernel * self.dilation[axis];
        padded
            .checked_sub(self.padding_start[axis])
            .filter(|&position| position < self.input[axis])
    }
}
