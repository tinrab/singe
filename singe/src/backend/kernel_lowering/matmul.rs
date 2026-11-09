//! A general matrix product as a reduction over the contraction dimension, for the cases no library covers.

use singe_codegen::{
    ir::{AxisId, AxisKind, BinaryOp, Combiner, ElementType, NodeId},
    schedule::{Schedule, contraction::FusedContraction, reduction_vector, size_values},
};

use crate::{
    backend::{
        kernel_lowering::{
            STATUS_INVALID_QUANTIZATION_SCALE,
            domain::{DomainKernel, LoweredKernel},
            element_type,
            expression::{ElementValue, Expressions},
            group::FusedStep,
            quantization::unpacked,
            reduction::row_schedule,
        },
        registry::PackedFormat,
        spec,
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    value::ValueSpec,
};

/// Lowers `lhs × rhs`, both of `data_type`, with batch broadcasting and vector promotion, as a row kernel per output element that splits the contraction dimension across lanes or threads.
///
/// # Errors
///
/// Returns an error if the operands are not floating-point or their shapes do not multiply.
pub fn matmul(data_type: DataType, lhs: &Layout, rhs: &Layout) -> Result<LoweredKernel> {
    contraction(
        data_type,
        &Lhs::plain(data_type, lhs),
        &Weight::Plain(rhs),
        None,
    )
}

/// The left operand of a contraction: element-wise `steps` of `inputs`, which broadcast to `shape`, computed as the kernel reads them, or the one input when there are no steps.
#[derive(Debug, Clone)]
pub struct Lhs<'a> {
    pub shape: Shape,
    pub inputs: Vec<(DataType, &'a Layout)>,
    pub steps: &'a [FusedStep<'a>],
}

impl<'a> Lhs<'a> {
    /// Returns the left operand that is the tensor of `data_type` laid out by `layout`.
    #[must_use]
    pub fn plain(data_type: DataType, layout: &'a Layout) -> Self {
        Self {
            shape: layout.shape().clone(),
            inputs: vec![(data_type, layout)],
            steps: &[],
        }
    }
}

/// The right operand of a contraction: a tensor, or one a dequantization computes as the kernel reads it.
#[derive(Debug, Clone, Copy)]
pub enum Weight<'a> {
    Plain(&'a Layout),
    /// `(quantized - zero_point) × scale`, with an `I8` or `U8` quantized tensor, an `F32` scale, and an `I64` zero point, each laid out over the operand's shape, then rounded to `rounded` when the graph converts it before the product.
    /// With `packed`, each byte of `quantized` holds two 4-bit values along the contraction axis, which it spans half of.
    Dequantized {
        quantized: (DataType, &'a Layout),
        scale: &'a Layout,
        zero_point: &'a Layout,
        rounded: Option<DataType>,
        packed: Option<PackedFormat>,
    },
}

/// Element-wise work on a contraction's product before it is stored: `steps`, whose operand `FusedOperand::Input(0)` is the product rounded to its type and `Input(k)` is `inputs[k - 1]`, which broadcast to the product's shape; the last step's value is stored.
#[derive(Debug, Clone)]
pub struct Epilogue<'a> {
    pub inputs: Vec<(DataType, &'a Layout)>,
    pub steps: &'a [FusedStep<'a>],
}

/// Lowers `lhs × weight` of `data_type`, with the left operand's steps and the weight's dequantization computed as the kernel reads them, and `epilogue` applied to the product, as [`matmul`] does.
///
/// The kernel rounds as the unfused operations would: the left operand and the product to `data_type`, and the dequantized weight to `rounded`.
/// Its inputs are the left operand's, the weight's, and the epilogue's, in that order.
///
/// # Errors
///
/// Returns an error if the operands are Boolean, a dequantized weight's product is not floating-point, their shapes do not multiply, or an epilogue operand or operation does not fit the product.
pub fn contraction(
    data_type: DataType,
    lhs: &Lhs<'_>,
    weight: &Weight<'_>,
    epilogue: Option<&Epilogue<'_>>,
) -> Result<LoweredKernel> {
    // Integer products wrap, as PyTorch's CPU computes them, and Boolean ones have no meaning.
    let dequantized = matches!(weight, Weight::Dequantized { .. });
    if matches!(data_type, DataType::Bool | DataType::F4E2M1)
        || (dequantized && !data_type.is_float())
    {
        return Err(Error::unsupported_data_type("cuda generated matmul"));
    }

    let (rhs, packed) = match *weight {
        Weight::Plain(layout) => (layout.shape().clone(), None),
        Weight::Dequantized {
            quantized, packed, ..
        } => (quantized.1.shape().clone(), packed),
    };
    // A packed weight's bytes span half its contraction axis.
    let rhs = if packed.is_some() {
        let mut dimensions = rhs.dimensions().to_vec();
        let inner = dimensions.len().checked_sub(2).ok_or_else(|| {
            Error::unsupported_operation("cuda generated matmul of a packed vector weight")
        })?;
        dimensions[inner] *= 2;
        Shape::create(dimensions)?
    } else {
        rhs
    };
    let problem = spec::matmul_problem(&lhs.shape, &rhs, "cuda matmul")?;
    if (epilogue.is_some() || !lhs.steps.is_empty()) && (lhs.shape.rank() < 2 || rhs.rank() < 2) {
        return Err(Error::unsupported_operation(
            "cuda generated matmul prologue or epilogue of a vector product",
        ));
    }

    let lhs_layouts = lhs
        .inputs
        .iter()
        .map(|(_, layout)| {
            let layout = layout.expand(&lhs.shape)?;
            if layout.rank() == 1 {
                layout.unsqueeze(0)
            } else {
                Ok(layout)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let column = |layout: &Layout| -> Result<Layout> {
        if layout.rank() == 1 {
            layout.unsqueeze(1)
        } else {
            Ok(layout.clone())
        }
    };
    let batch = &problem.batch_dimensions;
    let expanded = |layout: &Layout, rows: i64, columns: i64| -> Result<Layout> {
        let mut dimensions = batch.clone();
        dimensions.extend([rows, columns]);
        layout.expand(&Shape::create(dimensions)?)
    };
    let lhs_layouts = lhs_layouts
        .iter()
        .map(|layout| expanded(layout, problem.lhs_rows, problem.inner))
        .collect::<Result<Vec<_>>>()?;
    let weight_layouts = match *weight {
        Weight::Plain(layout) => vec![layout],
        Weight::Dequantized {
            quantized,
            scale,
            zero_point,
            ..
        } => vec![quantized.1, scale, zero_point],
    };
    let weight_layouts = weight_layouts
        .into_iter()
        .enumerate()
        .map(|(index, layout)| {
            let inner = if packed.is_some() && index == 0 {
                problem.inner / 2
            } else {
                problem.inner
            };
            expanded(&column(layout)?, inner, problem.rhs_cols)
        })
        .collect::<Result<Vec<_>>>()?;
    // The domain is the batch, the rows, the columns, and the contraction.
    let mut dimensions = batch.clone();
    dimensions.extend([problem.lhs_rows, problem.rhs_cols, problem.inner]);
    let mut kinds = vec![AxisKind::Parallel; dimensions.len() - 1];
    kinds.push(AxisKind::Reduction);
    let rank = batch.len();
    let mut columns = lhs_layouts
        .iter()
        .map(|layout| {
            let lhs_strides = layout.strides();
            let mut strides = lhs_strides[..rank].to_vec();
            strides.extend([lhs_strides[rank], 0, lhs_strides[rank + 1]]);
            strides
        })
        .collect::<Vec<_>>();
    let weight_column = columns.len();
    for layout in &weight_layouts {
        let strides = layout.strides();
        let mut column = strides[..rank].to_vec();
        column.extend([0, strides[rank + 1], strides[rank]]);
        columns.push(column);
    }

    // The epilogue operand and the output walk the product's axes and not the contraction.
    let product_columns = |layout: &Layout| -> Vec<i64> {
        let mut strides = layout.strides().to_vec();
        strides.push(0);
        strides
    };
    for (_, layout) in epilogue.iter().flat_map(|epilogue| &epilogue.inputs) {
        columns.push(product_columns(&layout.expand(&problem.output_shape)?));
    }

    columns.push(product_columns(&Layout::contiguous(
        &dimensions[..=rank + 1],
    )?));
    if packed.is_some() {
        // The contraction splits into the packed bytes and the two nibbles of each; the bytes do not move along the nibble.
        let last = dimensions.len() - 1;
        dimensions[last] /= 2;
        dimensions.push(2);
        kinds.push(AxisKind::Reduction);
        for (index, column) in columns.iter_mut().enumerate() {
            let stride = column.pop().unwrap_or(0);
            if index == weight_column {
                column.extend([stride, 0]);
            } else {
                column.extend([2 * stride, stride]);
            }
        }
    }

    let name = contraction_name(lhs, weight, epilogue);
    let mut builder = DomainKernel::create(&name, &dimensions, &kinds, &columns)?;
    let element = element_type(data_type)?;
    let left = left_value(&mut builder, lhs, element)?;
    let right = weight_value(&mut builder, weight, element, weight_column)?;
    let product = builder.kernel.binary(BinaryOp::Mul, left, right);
    let reduced = builder.reduction_axes();
    let mut value = builder.kernel.reduce(Combiner::Sum, reduced, product);
    let mut output_column = weight_column + weight_layouts.len();
    if let Some(epilogue) = epilogue {
        value = product_epilogue(
            &mut builder,
            epilogue,
            data_type,
            value,
            &problem.output_shape,
            output_column,
        )?;
        output_column += epilogue.inputs.len();
    }

    let output = builder.output(element, output_column);
    builder.kernel.store(output, value);
    let schedule = matmul_schedule(&builder);
    Ok(builder.finish(ValueSpec::new(data_type, problem.output_shape), schedule))
}

// Names a contraction by what it fuses.
fn contraction_name(lhs: &Lhs<'_>, weight: &Weight<'_>, epilogue: Option<&Epilogue<'_>>) -> String {
    let names = |steps: &[FusedStep<'_>]| {
        steps.iter().fold(String::new(), |name, step| {
            name + "_" + step.operation.name()
        })
    };
    format!(
        "singe{}{}{}{}",
        if lhs.steps.is_empty() {
            String::new()
        } else {
            format!("_fused{}", names(lhs.steps))
        },
        match weight {
            Weight::Plain(_) => "",
            Weight::Dequantized { .. } => "_dequantize",
        },
        "_matmul",
        epilogue.map_or_else(String::new, |epilogue| names(epilogue.steps)),
    )
}

// Loads the left operand, computing its steps, and returns its value in `element`'s compute type.
fn left_value(builder: &mut DomainKernel, lhs: &Lhs<'_>, element: ElementType) -> Result<NodeId> {
    if lhs.steps.is_empty() {
        let access = builder.input(element, 0);
        return Ok(builder.kernel.load(access));
    }

    // The unfused graph stores the left operand in its type before the product reads it.
    let mut loaded = Vec::with_capacity(lhs.inputs.len());
    for (column, &(input_type, _)) in lhs.inputs.iter().enumerate() {
        let access = builder.input(element_type(input_type)?, column);
        loaded.push(ElementValue {
            node: builder.kernel.load(access),
            data_type: input_type,
        });
    }

    let mut expressions = Expressions {
        kernel: &mut builder.kernel,
        bindings: &mut builder.bindings,
        labels: &mut builder.labels,
        shape: &lhs.shape,
    };
    let value = expressions.steps(&loaded, lhs.steps)?;
    let value = value
        .last()
        .ok_or_else(|| Error::missing_node("value of a fused left operand"))?
        .node;
    let rounded = builder.kernel.cast(element, value);
    Ok(builder.convert(rounded, element.compute()))
}

// Computes `epilogue` of `product`, whose operands are the columns from `first` on.
fn product_epilogue(
    builder: &mut DomainKernel,
    epilogue: &Epilogue<'_>,
    data_type: DataType,
    product: NodeId,
    shape: &Shape,
    first: usize,
) -> Result<NodeId> {
    let element = element_type(data_type)?;
    // The unfused graph stores the product in its type before the epilogue reads it.
    let rounded = builder.kernel.cast(element, product);
    let mut loaded = vec![ElementValue {
        node: builder.convert(rounded, element.compute()),
        data_type,
    }];
    for (offset, &(input_type, _)) in epilogue.inputs.iter().enumerate() {
        let access = builder.input(element_type(input_type)?, first + offset);
        loaded.push(ElementValue {
            node: builder.kernel.load(access),
            data_type: input_type,
        });
    }

    let mut expressions = Expressions {
        kernel: &mut builder.kernel,
        bindings: &mut builder.bindings,
        labels: &mut builder.labels,
        shape,
    };
    let computed = expressions.steps(&loaded, epilogue.steps)?;
    let last = computed
        .last()
        .ok_or_else(|| Error::missing_node("value of a contraction's epilogue"))?
        .node;
    Ok(builder.convert(last, element.compute()))
}

/// Loads the weight's operands, whose strides are the columns from `first` on, and returns its value in `element`'s compute type.
fn weight_value(
    builder: &mut DomainKernel,
    weight: &Weight<'_>,
    element: ElementType,
    first: usize,
) -> Result<NodeId> {
    match *weight {
        Weight::Plain(_) => {
            let access = builder.input(element, first);
            Ok(builder.kernel.load(access))
        }
        Weight::Dequantized {
            quantized,
            rounded,
            packed,
            ..
        } => {
            if !matches!(quantized.0, DataType::I8 | DataType::U8) {
                return Err(Error::unsupported_data_type("cuda dequantize input"));
            }

            let access = builder.input(element_type(quantized.0)?, first);
            let value = builder.kernel.load(access);
            // `I8` and `U8` values and their zero points subtract exactly in `F32`, which saves the 64-bit arithmetic per element.
            let value = match packed {
                Some(format) => unpacked(builder, value, quantized.0, format)?,
                None => builder.convert(value, ElementType::F32),
            };
            let access = builder.input(ElementType::F32, first + 1);
            let scale = builder.kernel.load(access);
            let access = builder.input(ElementType::I64, first + 2);
            let zero_point = builder.kernel.load(access);
            // A scale is valid when it is positive and finite, as the dequantize kernel checks.
            let zero = builder.constant(ElementType::F32, 0.0);
            let infinity = builder.constant(ElementType::F32, f64::INFINITY);
            let positive = builder.kernel.binary(BinaryOp::Greater, scale, zero);
            let finite = builder.kernel.binary(BinaryOp::Less, scale, infinity);
            let valid = builder.kernel.binary(BinaryOp::And, positive, finite);
            let scale = builder
                .kernel
                .require(scale, valid, STATUS_INVALID_QUANTIZATION_SCALE);
            let zero_point = builder.kernel.cast(ElementType::F32, zero_point);
            let centered = builder.kernel.binary(BinaryOp::Sub, value, zero_point);
            let mut weight = builder.kernel.binary(BinaryOp::Mul, centered, scale);
            if let Some(rounded) = rounded {
                let storage = element_type(rounded)?;
                let narrowed = builder.kernel.cast(storage, weight);
                weight = builder.convert(narrowed, storage.compute());
            }
            Ok(builder.convert(weight, element.compute()))
        }
    }
}

/// Returns the default schedule of a matmul: for contractions of at least 256 points, a warp per output with the widest vectors and parallel tiles that share the weight across rows and the inputs across columns, which the decoder benchmark's linears ran fastest under, and otherwise the row default.
// Smaller tiles read the weight more often, so they are worth it only below about one block per multiprocessor.
const MMA_BLOCKS: u64 = 32;

fn matmul_schedule(builder: &DomainKernel) -> Schedule {
    // Products of half-precision factors with at least 16 rows and columns run on tensor cores, in the largest tiles that still give the GPU `MMA_BLOCKS` blocks, or the smallest tiles.
    if let Some(contraction) = FusedContraction::of(&builder.kernel)
        && let Ok(sizes) = size_values(&builder.kernel, &builder.bindings.sizes)
    {
        let extent = |axis: AxisId| sizes[builder.kernel.axes[axis.0].extent.0];
        let (rows, columns) = (extent(contraction.rows), extent(contraction.columns));
        let batch = contraction
            .batch
            .iter()
            .map(|&axis| extent(axis))
            .product::<u64>();
        if rows >= 16 && columns >= 16 {
            let tiles = [
                (128, 128),
                (64, 128),
                (128, 64),
                (64, 64),
                (32, 64),
                (64, 32),
                (32, 32),
            ];
            let blocks = |(m, n): (u32, u32)| {
                batch * rows.div_ceil(u64::from(m)) * columns.div_ceil(u64::from(n))
            };
            let (m, n) = tiles
                .into_iter()
                .find(|&tile| blocks(tile) >= MMA_BLOCKS)
                .unwrap_or((32, 32));
            let schedule = Schedule::Mma { m, n, k: 64 };
            if schedule
                .launch(&builder.kernel, &builder.bindings.sizes)
                .is_ok()
            {
                return schedule;
            }
        }
    }

    if builder.count(AxisKind::Reduction) < 256.0 {
        return row_schedule(builder);
    }

    let vector = reduction_vector(&builder.kernel, 16);
    [[4, 4], [4, 2], [2, 4], [4, 1], [2, 2], [2, 1], [1, 1]]
        .into_iter()
        .map(|tiles| Schedule::Warps {
            threads: 128,
            vector,
            lanes: 32,
            tiles,
        })
        .find(|schedule| {
            schedule
                .launch(&builder.kernel, &builder.bindings.sizes)
                .is_ok()
        })
        .unwrap_or_else(|| row_schedule(builder))
}
