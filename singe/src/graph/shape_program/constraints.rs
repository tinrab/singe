use std::collections::HashMap;

use crate::{
    error::{Error, Result},
    value::ValueSpec,
};

#[cfg_attr(
    not(feature = "onnx"),
    expect(dead_code, reason = "only onnx import records dimension constraints")
)]
#[derive(Debug, Clone, Hash)]
pub enum DimensionConstraint {
    Unknown,
    Fixed(i64),
    Symbol {
        name: String,
        min: Option<i64>,
        max: Option<i64>,
    },
}

#[derive(Debug, Clone)]
pub struct ValueShapeConstraint {
    pub(crate) name: String,
    pub(crate) dimensions: Vec<DimensionConstraint>,
}

impl ValueShapeConstraint {
    pub(crate) fn validate(
        &self,
        spec: &ValueSpec,
        symbols: &mut HashMap<String, i64>,
    ) -> Result<()> {
        if self.dimensions.len() != spec.shape().dimensions().len() {
            return Err(self.failure("rank does not match its annotation"));
        }

        for (declared, actual) in self.dimensions.iter().zip(spec.shape().dimensions()) {
            match declared {
                DimensionConstraint::Fixed(expected) if expected != actual => {
                    return Err(self.failure("dimension does not match its fixed annotation"));
                }
                DimensionConstraint::Symbol { name, min, max } => {
                    if min.is_some_and(|min| *actual < min)
                        || max.is_some_and(|max| *actual > max)
                        || symbols
                            .insert(name.clone(), *actual)
                            .is_some_and(|previous| previous != *actual)
                    {
                        return Err(self.failure(&format!(
                            "symbol `{name}` violates its equality or range constraint"
                        )));
                    }
                }
                DimensionConstraint::Unknown | DimensionConstraint::Fixed(_) => {}
            }
        }
        Ok(())
    }

    fn failure(&self, reason: &str) -> Error {
        Error::unsupported_operation_with_details(
            "shape annotation",
            format!("`{}`: {reason}", self.name),
        )
    }
}
