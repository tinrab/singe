use std::slice;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::DimensionExpr,
    shape::Shape,
    utility::to_i64,
    value::ValueSpec,
};

/// Signed integer metadata, restricted to scalars and shape vectors.
/// This is deliberately separate from a storage-backed tensor or tensor data expression.
#[cfg_attr(
    not(feature = "onnx"),
    expect(dead_code, reason = "only onnx import and export build shape values")
)]
#[derive(Debug, Clone)]
pub enum ShapeValue {
    Scalar(DimensionExpr),
    Vector(Vec<DimensionExpr>),
}

#[cfg_attr(
    not(feature = "onnx"),
    expect(dead_code, reason = "only onnx import and export build shape values")
)]
impl ShapeValue {
    pub(crate) fn is_constant(&self) -> bool {
        self.elements().iter().all(DimensionExpr::is_constant)
    }

    pub(crate) fn elements(&self) -> &[DimensionExpr] {
        match self {
            Self::Scalar(value) => slice::from_ref(value),
            Self::Vector(values) => values,
        }
    }

    pub(crate) fn shape(&self) -> Result<Shape> {
        match self {
            Self::Scalar(_) => Ok(Shape::scalar()),
            Self::Vector(values) => Shape::create([to_i64(values.len(), "shape vector length")?]),
        }
    }

    pub(crate) fn spec(&self) -> Result<ValueSpec> {
        Ok(ValueSpec::new(DataType::I64, self.shape()?))
    }

    pub(crate) fn reshape(&self, shape: &Shape) -> Result<Self> {
        match shape.dimensions() {
            [] if self.elements().len() == 1 => Ok(Self::Scalar(self.elements()[0].clone())),
            [length]
                if usize::try_from(*length).is_ok_and(|length| length == self.elements().len()) =>
            {
                Ok(Self::Vector(self.elements().to_vec()))
            }
            _ => Err(Error::unsupported_operation(
                "shape values outside the scalar/vector domain",
            )),
        }
    }

    pub(crate) fn binary(
        &self,
        other: &Self,
        operation: impl Fn(DimensionExpr, DimensionExpr) -> DimensionExpr,
    ) -> Result<Self> {
        if let (Self::Scalar(lhs), Self::Scalar(rhs)) = (self, other) {
            return Ok(Self::Scalar(operation(lhs.clone(), rhs.clone())));
        }

        let lhs = self.elements();
        let rhs = other.elements();
        if lhs.len() != rhs.len() && lhs.len() != 1 && rhs.len() != 1 {
            return Err(Error::invalid_argument("shape arithmetic broadcast"));
        }

        let length = if lhs.len() == 1 { rhs.len() } else { lhs.len() };
        Ok(Self::Vector(
            (0..length)
                .map(|index| {
                    operation(
                        lhs[if lhs.len() == 1 { 0 } else { index }].clone(),
                        rhs[if rhs.len() == 1 { 0 } else { index }].clone(),
                    )
                })
                .collect(),
        ))
    }
}
