use std::{
    hash::{Hash, Hasher},
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::{
    error::{Error, Result},
    graph::InputDimensionGuard,
    value::ValueSpec,
};

/// A signed integer expression evaluated from logical dimensions, never tensor contents.
/// Arithmetic is checked at shape evaluation; division truncates toward zero.
/// Negative literals such as reshape's `-1` remain representable.
#[derive(Debug, Clone)]
pub struct DimensionExpr(Arc<DimensionExpression>);

#[derive(Debug)]
pub enum DimensionExpression {
    Constant(i64),
    Axis {
        owner: Option<Arc<()>>,
        value: usize,
        axis: usize,
    },
    Binary {
        operation: Arithmetic,
        lhs: DimensionExpr,
        rhs: DimensionExpr,
    },
}

#[derive(Debug, Clone, Copy, Hash)]
pub enum Arithmetic {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl DimensionExpr {
    #[cfg(feature = "onnx")]
    pub(crate) fn expression(&self) -> &DimensionExpression {
        &self.0
    }

    #[cfg_attr(
        not(feature = "onnx"),
        expect(dead_code, reason = "only onnx import and export read shape constants")
    )]
    pub(crate) fn is_constant(&self) -> bool {
        match self.0.as_ref() {
            DimensionExpression::Constant(_) => true,
            DimensionExpression::Axis { .. } => false,
            DimensionExpression::Binary { lhs, rhs, .. } => lhs.is_constant() && rhs.is_constant(),
        }
    }

    /// Creates a signed dimension literal, including zero and negative values.
    /// Expression consumers validate whether the value is permitted in their shape operation.
    #[must_use]
    pub fn constant(value: i64) -> Self {
        Self(Arc::new(DimensionExpression::Constant(value)))
    }

    pub(crate) fn axis(owner: Option<Arc<()>>, value: usize, axis: usize) -> Self {
        Self(Arc::new(DimensionExpression::Axis { owner, value, axis }))
    }

    pub(crate) fn evaluate(
        &self,
        dimension: &mut impl FnMut(usize, usize) -> Result<i64>,
    ) -> Result<i64> {
        match self.0.as_ref() {
            DimensionExpression::Constant(value) => Ok(*value),
            DimensionExpression::Axis { value, axis, .. } => dimension(*value, *axis),
            DimensionExpression::Binary {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = lhs.evaluate(dimension)?;
                let rhs = rhs.evaluate(dimension)?;
                match operation {
                    Arithmetic::Add => lhs.checked_add(rhs),
                    Arithmetic::Subtract => lhs.checked_sub(rhs),
                    Arithmetic::Multiply => lhs.checked_mul(rhs),
                    Arithmetic::Divide => lhs.checked_div(rhs),
                }
                .ok_or_else(|| Error::out_of_range("shape arithmetic"))
            }
        }
    }

    pub(crate) fn remap(
        &self,
        owner: Option<&Arc<()>>,
        values: &impl Fn(usize) -> Result<usize>,
    ) -> Result<Self> {
        match self.0.as_ref() {
            DimensionExpression::Constant(value) => Ok(Self::constant(*value)),
            DimensionExpression::Axis {
                owner: source,
                value,
                axis,
            } => {
                if source
                    .as_ref()
                    .zip(owner)
                    .is_some_and(|(source, owner)| !Arc::ptr_eq(owner, source))
                {
                    return Err(Error::invalid_argument("foreign shape expression"));
                }
                Ok(Self::axis(None, values(*value)?, *axis))
            }
            DimensionExpression::Binary {
                operation,
                lhs,
                rhs,
            } => Ok(Self::binary(
                *operation,
                lhs.remap(owner, values)?,
                rhs.remap(owner, values)?,
            )),
        }
    }

    pub(crate) fn references(&self, values: &mut Vec<usize>) {
        match self.0.as_ref() {
            DimensionExpression::Constant(_) => {}
            DimensionExpression::Axis { value, .. } => values.push(*value),
            DimensionExpression::Binary { lhs, rhs, .. } => {
                lhs.references(values);
                rhs.references(values);
            }
        }
    }

    pub(crate) fn hash_with<H: Hasher>(
        &self,
        hasher: &mut H,
        value: &impl Fn(usize) -> Result<usize>,
    ) -> Result<()> {
        match self.0.as_ref() {
            DimensionExpression::Constant(number) => {
                0_u8.hash(hasher);
                number.hash(hasher);
            }
            DimensionExpression::Axis {
                value: source,
                axis,
                ..
            } => {
                1_u8.hash(hasher);
                value(*source)?.hash(hasher);
                axis.hash(hasher);
            }
            DimensionExpression::Binary {
                operation,
                lhs,
                rhs,
            } => {
                2_u8.hash(hasher);
                operation.hash(hasher);
                lhs.hash_with(hasher, value)?;
                rhs.hash_with(hasher, value)?;
            }
        }
        Ok(())
    }

    pub(crate) fn referenced_values(&self, values: &mut Vec<usize>) {
        match self.0.as_ref() {
            DimensionExpression::Constant(_) => {}
            DimensionExpression::Axis { value, .. } => values.push(*value),
            DimensionExpression::Binary { lhs, rhs, .. } => {
                lhs.referenced_values(values);
                rhs.referenced_values(values);
            }
        }
    }

    fn binary(operation: Arithmetic, lhs: Self, rhs: Self) -> Self {
        Self(Arc::new(DimensionExpression::Binary {
            operation,
            lhs,
            rhs,
        }))
    }
}

impl From<i64> for DimensionExpr {
    fn from(value: i64) -> Self {
        Self::constant(value)
    }
}

impl Add for DimensionExpr {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::binary(Arithmetic::Add, self, rhs)
    }
}

impl Sub for DimensionExpr {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::binary(Arithmetic::Subtract, self, rhs)
    }
}

impl Mul for DimensionExpr {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::binary(Arithmetic::Multiply, self, rhs)
    }
}

impl Div for DimensionExpr {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::binary(Arithmetic::Divide, self, rhs)
    }
}

/// A representative concrete input spec and explicit runtime dimension guards.
/// The representative shape prepares the initial exact specialization.
/// Symbolic axes may vary within their inclusive bounds; other axes remain fixed.
#[derive(Debug, Clone)]
pub struct GraphInputSpec {
    spec: ValueSpec,
    dimensions: Vec<InputDimensionGuard>,
}

impl From<ValueSpec> for GraphInputSpec {
    fn from(spec: ValueSpec) -> Self {
        Self {
            spec,
            dimensions: Vec::new(),
        }
    }
}

impl GraphInputSpec {
    /// Creates an input specification with a fixed representative shape and no symbolic dimension guards.
    #[must_use]
    pub fn new(spec: ValueSpec) -> Self {
        spec.into()
    }

    /// Returns the representative logical shape and data type.
    #[must_use]
    pub fn spec(&self) -> &ValueSpec {
        &self.spec
    }

    /// Declares a named symbolic input axis with optional inclusive bounds.
    /// The representative dimension must satisfy the bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if `axis` is out of range or already declared, the symbol is empty, a bound is negative, bounds are reversed, or the representative dimension lies outside the bounds.
    pub fn with_dimension(
        mut self,
        axis: usize,
        symbol: impl Into<String>,
        min: Option<i64>,
        max: Option<i64>,
    ) -> Result<Self> {
        let symbol = symbol.into();
        let value = self
            .spec
            .shape()
            .dimensions()
            .get(axis)
            .copied()
            .ok_or(Error::DimensionOutOfBounds)?;
        if symbol.is_empty()
            || min.is_some_and(|min| min < 0 || value < min)
            || max.is_some_and(|max| max < 0 || value > max)
            || min.zip(max).is_some_and(|(min, max)| min > max)
            || self.dimensions.iter().any(|dim| dim.axis() == axis)
        {
            return Err(Error::invalid_argument("input dimension declaration"));
        }

        self.dimensions
            .push(InputDimensionGuard::new("", axis, symbol, min, max));
        Ok(self)
    }

    pub(crate) fn into_parts(self, name: &str) -> (ValueSpec, Vec<InputDimensionGuard>) {
        (
            self.spec,
            self.dimensions
                .into_iter()
                .map(|dim| {
                    InputDimensionGuard::new(name, dim.axis(), dim.name(), dim.min(), dim.max())
                })
                .collect(),
        )
    }
}
