// Window reductions: 2D pooling and depthwise convolution as one thread per output point that walks its window's taps.
//
// The domain is the output's points and the window's taps, a reduction over the taps.
// Along an axis where some window reaches past the input, a tap computes its input row or column, reads it clamped into the input, and replaces what it read with the pooling's or the convolution's padding value when it lies outside; along other axes the taps are plain strides.

use singe_codegen::{
    ir::{AxisId, AxisKind, BinaryOp, Combiner, ElementType, Expr, Literal, NodeId},
    schedule::element_schedule,
};

use crate::{
    backend::{
        kernel_lowering::{
            domain::{DomainKernel, LoweredKernel},
            element_type,
        },
        registry::{Conv2dAttributes, Operation, Pool2dAttributes},
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    utility::to_u64,
    value::ValueSpec,
};

// The domain's dimensions are the batch, the input channel, the channel multiplier, the output row and column, and the tap row and column.
const ROW: usize = 3;
const COLUMN: usize = 4;
const TAP_ROW: usize = 5;
const TAP_COLUMN: usize = 6;
const RANK: usize = 8;

// One spatial axis of a window: the input's extent and stride along it, and the window's geometry.
#[derive(Debug, Clone, Copy)]
struct Span {
    input: i64,
    stride: i64,
    output: i64,
    taps: i64,
    step: i64,
    dilation: i64,
    pad_start: i64,
    pad_end: i64,
}

impl Span {
    // Whether some tap reads outside the input, so the axis needs bounds.
    const fn reaches_out(self) -> bool {
        self.pad_start > 0
            || (self.output - 1) * self.step + (self.taps - 1) * self.dilation > self.input - 1
    }

    // Whether some tap reads past the padding too, as ceil-mode windows can.
    const fn reaches_past_padding(self) -> bool {
        (self.output - 1) * self.step + (self.taps - 1) * self.dilation
            > self.input - 1 + self.pad_start + self.pad_end
    }
}

// Where a tap lies along one axis: inside the input, inside the padded input, and its clamped index.
struct Position {
    inside: NodeId,
    inside_padding: NodeId,
}

struct Window {
    builder: DomainKernel,
    rows: Span,
    columns: Span,
    // The row and column of the tap along axes with bounds, each with its position.
    positions: Vec<Position>,
}

impl Window {
    // Builds the domain of `[batch, channels * multiplier, rows.output, columns.output]` outputs over windows of `rows` and `columns`, with `operands` holding each operand's strides over the domain, the input's first.
    fn create(
        name: &str,
        extents: [i64; 4],
        rows: Span,
        columns: Span,
        operands: &[[i64; RANK]],
    ) -> Result<Self> {
        let [batch, channels, multiplier, in_channels] = extents;
        let dimensions = [
            batch,
            channels,
            multiplier,
            rows.output,
            columns.output,
            rows.taps,
            columns.taps,
            in_channels,
        ];
        let kinds = [
            AxisKind::Parallel,
            AxisKind::Parallel,
            AxisKind::Parallel,
            AxisKind::Parallel,
            AxisKind::Parallel,
            AxisKind::Reduction,
            AxisKind::Reduction,
            AxisKind::Reduction,
        ];
        let mut strides = operands
            .iter()
            .map(|column| column.to_vec())
            .collect::<Vec<_>>();
        // Axes with bounds read through their own index, so they keep their own axes.
        let mut marker = vec![0; RANK];
        for (span, output, tap) in [(rows, ROW, TAP_ROW), (columns, COLUMN, TAP_COLUMN)] {
            if span.reaches_out() {
                marker[output] = 1;
                marker[tap] = 1;
            }
        }

        strides.push(marker);
        let builder = DomainKernel::create(name, &dimensions, &kinds, &strides)?;
        Ok(Self {
            builder,
            rows,
            columns,
            positions: Vec::new(),
        })
    }

    // The index of domain dimension `dimension` as an `I64`, zero when the dimension has one point.
    fn index(&mut self, dimension: usize) -> NodeId {
        match self.builder.dimension_axis(dimension) {
            Some(axis) => self
                .builder
                .kernel
                .add_node(ElementType::I64, Expr::AxisIndex(axis)),
            None => self.integer(0),
        }
    }

    fn integer(&mut self, value: i64) -> NodeId {
        self.builder
            .kernel
            .literal(ElementType::I64, Literal::Int(value))
    }

    // Loads the input in its compute type at the kernel's tap, clamped into the input along axes with bounds, and records where the tap lies along them.
    fn load_input(&mut self, data_type: DataType) -> Result<NodeId> {
        let element = element_type(data_type)?;
        let mut access = self.builder.input(element, 0);
        for (span, output, tap) in [
            (self.rows, ROW, TAP_ROW),
            (self.columns, COLUMN, TAP_COLUMN),
        ] {
            if !span.reaches_out() {
                continue;
            }

            let output = self.index(output);
            let tap = self.index(tap);
            let step = self.integer(span.step);
            let dilation = self.integer(span.dilation);
            let start = self.integer(span.pad_start);
            let kernel = &mut self.builder.kernel;
            let row = kernel.binary(BinaryOp::Mul, output, step);
            let offset = kernel.binary(BinaryOp::Mul, tap, dilation);
            let row = kernel.binary(BinaryOp::Add, row, offset);
            let row = kernel.binary(BinaryOp::Sub, row, start);
            let zero = self.integer(0);
            let last = self.integer(span.input - 1);
            let below = self.integer(-span.pad_start);
            let past = self.integer(span.input + span.pad_end);
            let kernel = &mut self.builder.kernel;
            let low = kernel.binary(BinaryOp::GreaterEqual, row, zero);
            let high = kernel.binary(BinaryOp::LessEqual, row, last);
            let inside = kernel.binary(BinaryOp::And, low, high);
            let low = kernel.binary(BinaryOp::GreaterEqual, row, below);
            let high = kernel.binary(BinaryOp::Less, row, past);
            let inside_padding = kernel.binary(BinaryOp::And, low, high);
            let clamped = kernel.binary(BinaryOp::Max, row, zero);
            let clamped = kernel.binary(BinaryOp::Min, clamped, last);
            let clamped = kernel.cast(ElementType::U64, clamped);
            let stride = self
                .builder
                .size(to_u64(span.stride, "cuda window input stride")?);
            access.indirect.push((clamped, stride));
            self.positions.push(Position {
                inside,
                inside_padding,
            });
        }

        let value = self.builder.kernel.load(access);
        Ok(self.builder.convert(value, element.compute()))
    }

    // Whether the tap lies inside the input along every axis with bounds, or `None` when every tap does.
    fn inside(&mut self, padding: bool) -> Option<NodeId> {
        let kernel = &mut self.builder.kernel;
        self.positions
            .iter()
            .map(|position| {
                if padding {
                    position.inside_padding
                } else {
                    position.inside
                }
            })
            .reduce(|lhs, rhs| kernel.binary(BinaryOp::And, lhs, rhs))
    }

    fn taps(&self) -> Vec<AxisId> {
        self.builder.reduction_axes()
    }
}

// The input's strides over the domain, with `channel` the stride between input channels: plain along axes without bounds, and none along axes with them, which the clamped index addresses.
fn input_strides(
    strides: [i64; 4],
    channel: i64,
    in_channel: i64,
    rows: Span,
    columns: Span,
) -> [i64; RANK] {
    let along = |span: Span, stride: i64| {
        if span.reaches_out() {
            (0, 0)
        } else {
            (span.step * stride, span.dilation * stride)
        }
    };
    let (row, tap_row) = along(rows, strides[2]);
    let (column, tap_column) = along(columns, strides[3]);
    [
        strides[0], channel, 0, row, column, tap_row, tap_column, in_channel,
    ]
}

// The contiguous output's strides over the domain.
fn output_strides(channels: i64, multiplier: i64, rows: Span, columns: Span) -> [i64; RANK] {
    let plane = rows.output * columns.output;
    [
        channels * multiplier * plane,
        multiplier * plane,
        plane,
        columns.output,
        1,
        0,
        0,
        0,
    ]
}

fn four(layout: &Layout) -> Result<[i64; 4]> {
    <[i64; 4]>::try_from(layout.strides())
        .map_err(|_| Error::invalid_argument("cuda window operand rank"))
}

// Lowers `max_pool2d` or `avg_pool2d` of a floating `[N, C, H, W]` input as a window reduction.
pub fn pool2d(
    operation: Operation,
    attributes: &Pool2dAttributes,
    data_type: DataType,
    input: &Layout,
) -> Result<LoweredKernel> {
    if !data_type.is_float() {
        return Err(Error::unsupported_data_type("cuda generated pool2d"));
    }

    let average = operation == Operation::AvgPool2d;
    let problem = spec::pool2d_problem(
        input.shape(),
        attributes.kernel_size,
        attributes.stride,
        attributes.padding,
        attributes.dilation,
        attributes.ceil_mode,
        average,
        "cuda generated pool2d",
    )?;
    // A `[C, H, W]` input is one image.
    let strides = match *input.strides() {
        [channel, row, column] => [0, channel, row, column],
        _ => four(input)?,
    };
    let rows = Span {
        input: problem.input_height,
        stride: strides[2],
        output: problem.output_height,
        taps: problem.kernel_height,
        step: problem.stride_height,
        dilation: problem.dilation_height,
        pad_start: problem.pad_height_start,
        pad_end: problem.pad_height_end,
    };
    let columns = Span {
        input: problem.input_width,
        stride: strides[3],
        output: problem.output_width,
        taps: problem.kernel_width,
        step: problem.stride_width,
        dilation: problem.dilation_width,
        pad_start: problem.pad_width_start,
        pad_end: problem.pad_width_end,
    };
    let mut window = Window::create(
        if average {
            "singe_avg_pool2d"
        } else {
            "singe_max_pool2d"
        },
        [problem.batch, problem.channels, 1, 1],
        rows,
        columns,
        &[
            input_strides(strides, strides[1], 0, rows, columns),
            output_strides(problem.channels, 1, rows, columns),
        ],
    )?;
    let element = element_type(data_type)?;
    let compute = element.compute();
    let value = window.load_input(data_type)?;
    let inside = window.inside(false);
    let taps = window.taps();
    let result = if average {
        let zero = window.builder.constant(compute, 0.0);
        let value = match inside {
            Some(inside) => window.builder.kernel.select(inside, value, zero),
            None => value,
        };
        let sum = window
            .builder
            .kernel
            .reduce(Combiner::Sum, taps.clone(), value);
        // The divisor counts the window's taps inside the input, or inside the padded input when padding counts, which ceil-mode windows can still overhang.
        let counted = if attributes.count_include_pad.unwrap_or(true) {
            if rows.reaches_past_padding() || columns.reaches_past_padding() {
                window.inside(true)
            } else {
                None
            }
        } else {
            inside
        };
        let count = match counted {
            Some(counted) => {
                let one = window.builder.constant(compute, 1.0);
                let tap = window.builder.kernel.select(counted, one, zero);
                window.builder.kernel.reduce(Combiner::Sum, taps, tap)
            }
            None => window
                .builder
                .constant(compute, (rows.taps * columns.taps) as f64),
        };
        window.builder.kernel.binary(BinaryOp::Div, sum, count)
    } else {
        let value = match inside {
            Some(inside) => {
                let lowest = window.builder.constant(compute, f64::NEG_INFINITY);
                window.builder.kernel.select(inside, value, lowest)
            }
            None => value,
        };
        window.builder.kernel.reduce(Combiner::Max, taps, value)
    };
    let output = window.builder.output(element, 1);
    window.builder.kernel.store(output, result);
    let schedule = element_schedule(&window.builder.kernel, 128);
    Ok(window
        .builder
        .finish(ValueSpec::new(data_type, problem.output_shape), schedule))
}

// Lowers a `conv2d` of a floating `[N, C, H, W]` input by `[O, C / G, KH, KW]` weights in `G` groups as a window reduction over the weight's taps and each group's input channels, when it is depthwise or `general` allows any grouping, with an optional bias of `O` values.
pub fn conv2d(
    general: bool,
    attributes: &Conv2dAttributes,
    data_type: DataType,
    input: &Layout,
    weight: (DataType, &Layout),
    bias: Option<(DataType, &Layout)>,
) -> Result<LoweredKernel> {
    let problem = spec::conv2d_problem(
        input.shape(),
        weight.1.shape(),
        attributes.stride,
        attributes.padding,
        attributes.dilation,
        attributes.groups,
        "cuda generated depthwise conv2d",
    )?;
    if !data_type.is_float()
        || weight.0 != data_type
        || bias.is_some_and(|(bias_type, _)| !bias_type.is_float())
        || (!general && problem.groups != problem.input_channels)
    {
        return Err(Error::unsupported_operation(
            "cuda generated conv2d that is not depthwise",
        ));
    }

    let groups = problem.groups;
    let multiplier = problem.output_channels / groups;
    let in_channels = problem.input_channels / groups;
    let strides = four(input)?;
    let rows = Span {
        input: problem.input_height,
        stride: strides[2],
        output: problem.output_height,
        taps: problem.kernel_height,
        step: problem.stride_height,
        dilation: problem.dilation_height,
        pad_start: problem.pad_height_start,
        pad_end: problem.pad_height_end,
    };
    let columns = Span {
        input: problem.input_width,
        stride: strides[3],
        output: problem.output_width,
        taps: problem.kernel_width,
        step: problem.stride_width,
        dilation: problem.dilation_width,
        pad_start: problem.pad_width_start,
        pad_end: problem.pad_width_end,
    };
    // Output channel `g * multiplier + m` reads input channels `g * in_channels + i` through weight `[g * multiplier + m, i]`.
    let weights = four(weight.1)?;
    let mut operands = vec![
        input_strides(strides, strides[1] * in_channels, strides[1], rows, columns),
        [
            0,
            weights[0] * multiplier,
            weights[0],
            0,
            0,
            weights[2],
            weights[3],
            weights[1],
        ],
    ];
    if let Some((_, layout)) = bias {
        let stride = layout.strides().first().copied().unwrap_or(0);
        operands.push([0, stride * multiplier, stride, 0, 0, 0, 0, 0]);
    }

    operands.push(output_strides(groups, multiplier, rows, columns));
    let mut window = Window::create(
        if in_channels == 1 {
            "singe_depthwise_conv2d"
        } else {
            "singe_conv2d"
        },
        [problem.batch_size, groups, multiplier, in_channels],
        rows,
        columns,
        &operands,
    )?;
    let element = element_type(data_type)?;
    let compute = element.compute();
    let value = window.load_input(data_type)?;
    let value = match window.inside(false) {
        Some(inside) => {
            let zero = window.builder.constant(compute, 0.0);
            window.builder.kernel.select(inside, value, zero)
        }
        None => value,
    };
    let filter = window.builder.input(element, 1);
    let filter = window.builder.kernel.load(filter);
    let filter = window.builder.convert(filter, compute);
    let product = window.builder.kernel.binary(BinaryOp::Mul, value, filter);
    let taps = window.taps();
    let mut result = window.builder.kernel.reduce(Combiner::Sum, taps, product);
    let mut column = 2;
    if let Some((bias_type, _)) = bias {
        let bias = window.builder.input(element_type(bias_type)?, column);
        let bias = window.builder.kernel.load(bias);
        let bias = window.builder.convert(bias, compute);
        result = window.builder.kernel.binary(BinaryOp::Add, result, bias);
        column += 1;
    }

    let output = window.builder.output(element, column);
    window.builder.kernel.store(output, result);
    let schedule = element_schedule(&window.builder.kernel, 128);
    Ok(window
        .builder
        .finish(ValueSpec::new(data_type, problem.output_shape), schedule))
}
