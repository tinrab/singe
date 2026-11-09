use singe_onnx::proto::onnx::tensor_shape_proto::dimension::Value;

use crate::{
    graph::{
        Arithmetic, DimensionConstraint, DimensionExpr, DimensionExpression, ShapeValue,
        ir::{OperationNode, ValueId},
    },
    onnx::{
        error::Result,
        export_contract::annotate_dimension,
        exporter::{
            Exporter,
            proto::{integer, unsupported, value_info},
        },
    },
    shape::normalize_dim_usize,
};

pub fn fixed_dimensions(dimensions: &[i64]) -> Vec<Option<Value>> {
    dimensions
        .iter()
        .map(|&dim| Some(Value::DimValue(dim)))
        .collect()
}

impl Exporter<'_> {
    pub(crate) fn squeeze(
        &mut self,
        node: &OperationNode,
        mut inputs: Vec<String>,
        output: String,
        dim: i64,
    ) -> Result<()> {
        let source = self.source;
        let input = node.inputs()[0];
        let shape = source.required_value_shape(input)?;
        let axis = normalize_dim_usize(dim, shape.rank().max(1))?;
        if shape.rank() == 0 {
            self.node("Identity", inputs, output, vec![]);
            return Ok(());
        }

        let singleton = if let Some(name) = source.required_value(input)?.binding().input_name() {
            if let Some(guard) = source.input_dimension_guard(name, axis) {
                if guard.min() == Some(1) && guard.max() == Some(1) {
                    true
                } else if guard.min().is_some_and(|min| min > 1)
                    || guard.max().is_some_and(|max| max < 1)
                {
                    false
                } else {
                    return Err(unsupported("squeeze dimension may change output rank"));
                }
            } else {
                shape.dimensions()[axis] == 1
            }
        } else if self.dynamic
            && source
                .required_operation_for_value(input)?
                .opcode()
                .is_some()
        {
            return Err(unsupported(
                "dynamic intermediate squeeze dimension is not proven invariant",
            ));
        } else {
            shape.dimensions()[axis] == 1
        };
        if singleton {
            inputs.push(self.integers(
                vec![1],
                &[i64::try_from(axis).map_err(|_| unsupported("squeeze axis"))?],
            )?);
            self.node("Squeeze", inputs, output, vec![]);
        } else {
            self.node("Identity", inputs, output, vec![]);
        }
        Ok(())
    }

    pub(crate) fn dimension(&mut self, expression: &DimensionExpr) -> Result<String> {
        Ok(match expression.expression() {
            DimensionExpression::Constant(value) => self.integers(vec![], &[*value])?,
            DimensionExpression::Axis { value, axis, .. } => {
                let source = self.emit(ValueId::new(*value))?;
                let shape = self.temporary("Shape", vec![source], vec![]);
                let axis = self.integers(
                    vec![],
                    &[i64::try_from(*axis).map_err(|_| unsupported("shape axis overflow"))?],
                )?;
                self.temporary("Gather", vec![shape, axis], vec![integer("axis", 0)])
            }
            DimensionExpression::Binary {
                operation,
                lhs,
                rhs,
            } => {
                let lhs = self.dimension(lhs)?;
                let rhs = self.dimension(rhs)?;
                let operator = match operation {
                    Arithmetic::Add => "Add",
                    Arithmetic::Subtract => "Sub",
                    Arithmetic::Multiply => "Mul",
                    Arithmetic::Divide => "Div",
                };
                self.temporary(operator, vec![lhs, rhs], vec![])
            }
        })
    }

    pub(crate) fn shape_vector(&mut self, dimensions: &[DimensionExpr]) -> Result<String> {
        if dimensions.is_empty() {
            return self.integers(vec![0], &[]);
        }

        let axes = self.integers(vec![1], &[0])?;
        let mut values = Vec::new();
        for expression in dimensions {
            let value = self.dimension(expression)?;
            values.push(self.temporary("Unsqueeze", vec![value, axes.clone()], vec![]));
        }
        Ok(self.temporary("Concat", values, vec![integer("axis", 0)]))
    }

    pub(crate) fn shape_value(&mut self, value: ShapeValue) -> Result<String> {
        match value {
            ShapeValue::Scalar(expression) => self.dimension(&expression),
            ShapeValue::Vector(expressions) => self.shape_vector(&expressions),
        }
    }

    pub(crate) fn constraints(&mut self) -> Result<()> {
        let program = self.source.shape_program();
        for (id, constraints) in &program.constraints {
            let source = self.emit(*id)?;
            let spec = self.source.required_value(*id)?;
            for constraint in constraints {
                let name = self.temporary("Identity", vec![source.clone()], vec![]);
                let dimensions = constraint
                    .dimensions
                    .iter()
                    .map(|dim| match dim {
                        DimensionConstraint::Unknown => None,
                        DimensionConstraint::Fixed(value) => Some(Value::DimValue(*value)),
                        DimensionConstraint::Symbol { name, .. } => {
                            Some(Value::DimParam(name.clone()))
                        }
                    })
                    .collect();
                let mut info = value_info(name, spec.data_type(), dimensions)?;
                info.doc_string.clone_from(&constraint.name);
                for (axis, dimension) in constraint.dimensions.iter().enumerate() {
                    if let DimensionConstraint::Symbol { min, max, .. } = dimension {
                        annotate_dimension(
                            &mut info,
                            axis,
                            spec.shape().dimensions()[axis],
                            *min,
                            *max,
                        );
                    }
                }

                self.graph.value_info.push(info);
            }
        }
        Ok(())
    }
}
