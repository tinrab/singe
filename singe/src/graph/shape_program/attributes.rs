use std::hash::{Hash, Hasher};

use crate::{
    backend::registry::{OperationAttributes, ResizeAttributes, SliceAttributes},
    device::Device,
    error::{Error, Result},
    graph::DimensionExpr,
    shape::{
        Shape,
        spatial::{Spatial2d, Spatial2dExpr},
    },
    tensor::{Interpolation, Tensor},
    utility::to_i64,
};

// Attributes of one node that depend on run-time axes, resolved to concrete attributes at each input shape.
#[derive(Debug, Clone)]
pub enum SymbolicAttributes {
    // A reshape's target; `allow_zero` follows ONNX's switch, under which a zero copies the input's axis unless it is set.
    Reshape {
        dimensions: Vec<DimensionExpr>,
        allow_zero: bool,
    },
    // An expansion's target shape.
    Expand(Vec<DimensionExpr>),
    // A slice of `length` elements from `start` along `dim`.
    Slice {
        dim: i64,
        start: DimensionExpr,
        length: DimensionExpr,
    },
    // A resize of the last two axes.
    Resize {
        size: Spatial2dExpr,
        interpolation: Interpolation,
    },
    // An `I64` constant counting from `start` up to, but not including, `end`.
    Range {
        start: DimensionExpr,
        end: DimensionExpr,
    },
    // An `I64` constant vector of the dimensions.
    Dimensions(Vec<DimensionExpr>),
    // A constant of the dimensions' shape, every element the one element of `value`.
    Fill {
        dimensions: Vec<DimensionExpr>,
        value: Tensor,
    },
}

// Concrete attributes at one input shape: an operation's, or the host tensor a constant holds.
pub enum ResolvedAttributes {
    Operation(OperationAttributes),
    Constant(Tensor),
}

impl SymbolicAttributes {
    pub(crate) fn expressions(&self) -> Vec<&DimensionExpr> {
        match self {
            Self::Reshape { dimensions, .. }
            | Self::Expand(dimensions)
            | Self::Dimensions(dimensions)
            | Self::Fill { dimensions, .. } => dimensions.iter().collect(),
            Self::Slice { start, length, .. } => vec![start, length],
            Self::Resize { size, .. } => vec![&size.height, &size.width],
            Self::Range { start, end } => vec![start, end],
        }
    }

    pub(crate) fn map(
        &self,
        mut map: impl FnMut(&DimensionExpr) -> Result<DimensionExpr>,
    ) -> Result<Self> {
        Ok(match self {
            Self::Reshape {
                dimensions,
                allow_zero,
            } => Self::Reshape {
                dimensions: dimensions.iter().map(map).collect::<Result<_>>()?,
                allow_zero: *allow_zero,
            },
            Self::Expand(dimensions) => {
                Self::Expand(dimensions.iter().map(map).collect::<Result<_>>()?)
            }
            Self::Slice { dim, start, length } => Self::Slice {
                dim: *dim,
                start: map(start)?,
                length: map(length)?,
            },
            Self::Resize {
                size,
                interpolation,
            } => Self::Resize {
                size: Spatial2d::new(map(&size.height)?, map(&size.width)?),
                interpolation: *interpolation,
            },
            Self::Range { start, end } => Self::Range {
                start: map(start)?,
                end: map(end)?,
            },
            Self::Dimensions(dimensions) => {
                Self::Dimensions(dimensions.iter().map(map).collect::<Result<_>>()?)
            }
            Self::Fill { dimensions, value } => Self::Fill {
                dimensions: dimensions.iter().map(map).collect::<Result<_>>()?,
                value: value.clone(),
            },
        })
    }

    // Hashes the attributes with each axis reference's value numbered by `value`.
    pub(crate) fn hash_with<H: Hasher>(
        &self,
        hasher: &mut H,
        value: &impl Fn(usize) -> Result<usize>,
    ) -> Result<()> {
        match self {
            Self::Reshape { allow_zero, .. } => (0_u8, allow_zero).hash(hasher),
            Self::Expand(_) => 1_u8.hash(hasher),
            Self::Slice { dim, .. } => (2_u8, dim).hash(hasher),
            Self::Resize { interpolation, .. } => (3_u8, interpolation).hash(hasher),
            Self::Range { .. } => 4_u8.hash(hasher),
            Self::Dimensions(_) => 5_u8.hash(hasher),
            Self::Fill { value, .. } => {
                (6_u8, value.data_type()).hash(hasher);
                value.to_bytes()?.hash(hasher);
            }
        }

        let expressions = self.expressions();
        expressions.len().hash(hasher);
        for dim in expressions {
            dim.hash_with(hasher, value)?;
        }
        Ok(())
    }

    // Evaluates the expressions through `evaluate`, against `input`, the shape of the node's first operand.
    pub(crate) fn resolve(
        &self,
        input: Option<&Shape>,
        mut evaluate: impl FnMut(&DimensionExpr) -> Result<i64>,
    ) -> Result<ResolvedAttributes> {
        let mut all = |dimensions: &[DimensionExpr]| {
            dimensions
                .iter()
                .map(&mut evaluate)
                .collect::<Result<Vec<_>>>()
        };
        Ok(match self {
            Self::Reshape {
                dimensions,
                allow_zero,
            } => {
                let input = input.ok_or_else(|| Error::missing_input("symbolic attributes"))?;
                let mut dimensions = all(dimensions)?;
                if !allow_zero {
                    for (axis, dim) in dimensions.iter_mut().enumerate() {
                        if *dim == 0 {
                            *dim = *input.dimensions().get(axis).ok_or_else(|| {
                                Error::dimension_out_of_bounds("shape program axis")
                            })?;
                        }
                    }
                } else if dimensions.contains(&0) && dimensions.contains(&-1) {
                    return Err(Error::invalid_argument(
                        "reshape zero dimensions cannot be combined with inference",
                    ));
                }
                ResolvedAttributes::Operation(OperationAttributes::Shape(
                    input.reshape(dimensions)?,
                ))
            }
            Self::Expand(dimensions) => ResolvedAttributes::Operation(OperationAttributes::Shape(
                Shape::create(all(dimensions)?)?,
            )),
            Self::Slice { dim, start, length } => {
                ResolvedAttributes::Operation(OperationAttributes::Slice(SliceAttributes {
                    dim: *dim,
                    start: evaluate(start)?,
                    length: evaluate(length)?,
                    step: 1,
                }))
            }
            Self::Resize {
                size,
                interpolation,
            } => ResolvedAttributes::Operation(OperationAttributes::Resize(ResizeAttributes {
                size: Spatial2d::new(evaluate(&size.height)?, evaluate(&size.width)?),
                interpolation: *interpolation,
            })),
            Self::Range { start, end } => {
                let (start, end) = (evaluate(start)?, evaluate(end)?);
                if end < start {
                    return Err(Error::invalid_argument(format!(
                        "range end {end} precedes its start {start}"
                    )));
                }
                ResolvedAttributes::Constant(Tensor::from_vec(
                    Device::Cpu,
                    Shape::create([end - start])?,
                    (start..end).collect(),
                )?)
            }
            Self::Dimensions(dimensions) => ResolvedAttributes::Constant(Tensor::from_vec(
                Device::Cpu,
                Shape::create([to_i64(dimensions.len(), "dimension count")?])?,
                all(dimensions)?,
            )?),
            Self::Fill { dimensions, value } => ResolvedAttributes::Constant(
                value
                    .to_device(Device::Cpu)?
                    .reshape(Vec::new())?
                    .expand(Shape::create(all(dimensions)?)?)?
                    .contiguous()?,
            ),
        })
    }
}
