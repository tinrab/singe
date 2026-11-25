use singe_onnx::proto::onnx::tensor_shape_proto::dimension::Value;

use crate::{
    backend::registry::{Operation, OperationAttributes, OperationRequest},
    error::Error as SingeError,
    graph::{
        Arithmetic, DimensionConstraint, DimensionExpr, DimensionExpression, ResolvedAttributes,
        ShapeValue, SymbolicAttributes,
        ir::{OperationNode, ValueId},
    },
    onnx::{
        error::{Error, Result},
        export_contract::annotate_dimension,
        exporter::{
            Exporter,
            proto::{self, integer, unsupported, value_info},
        },
    },
    shape::{Shape, normalize_dim_usize},
    value::ValueSpec,
};

const GROWTH: &[i64] = &[2, 3];

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
            && let Some(producer) = source.required_operation_for_value(input)?.opcode()
        {
            let extent = shape.dimensions()[axis];
            let stays = (0..GROWTH.len()).all(|growth| {
                self.grown_shape(growth, input)
                    .is_some_and(|grown| grown.dimensions().get(axis) == Some(&extent))
            });
            if !stays {
                return Err(self.not_following(|| {
                    format!(
                        "squeeze of a dynamic {producer:?} result's axis {axis}, which does not stay {extent}"
                    )
                }));
            }
            extent == 1
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

    fn grown_shape(&mut self, growth: usize, value: ValueId) -> Option<Shape> {
        if let Some(shape) = self.grown[growth].get(&value) {
            return shape.clone();
        }

        let shape = self.grow(growth, value);
        self.grown[growth].insert(value, shape.clone());
        shape
    }

    pub(crate) fn require_growth(&mut self, value: ValueId) -> Result<()> {
        for growth in 0..GROWTH.len() {
            if self.grown_shape(growth, value).is_none() {
                return Err(self.not_following(|| {
                    "a value whose shape does not follow the run-time axes".to_owned()
                }));
            }
        }
        Ok(())
    }

    fn not_following(&self, otherwise: impl FnOnce() -> String) -> Error {
        unsupported(self.traced.as_ref().map_or_else(otherwise, |operation| {
            format!(
                "{operation} holds an extent of the traced size, so the export would not follow its run-time axes"
            )
        }))
    }

    fn grow(&mut self, growth: usize, value: ValueId) -> Option<Shape> {
        let source = self.source;
        let traced = source.required_value_shape(value).ok()?;
        if let Some(name) = source.required_value(value).ok()?.binding().input_name() {
            let dimensions = traced
                .dimensions()
                .iter()
                .enumerate()
                .map(|(axis, &extent)| {
                    // The largest multiple of the traced extent within the axis's bound, which keeps any divisibility the model needs.
                    source
                        .input_dimension_guard(name, axis)
                        .map_or(extent, |guard| {
                            (2..=GROWTH[growth])
                                .rev()
                                .filter_map(|factor| extent.checked_mul(factor))
                                .find(|&grown| guard.max().is_none_or(|max| grown <= max))
                                .unwrap_or(extent)
                        })
                })
                .collect::<Vec<_>>();
            return Shape::create(dimensions).ok();
        }

        let symbolic = source.shape_program().attributes.get(&value).cloned();
        if source.constant_tensor(value).is_some() {
            let Some(symbolic) = symbolic else {
                return Some(traced.clone());
            };
            return match self.grown_attributes(growth, &symbolic, None)? {
                ResolvedAttributes::Constant(tensor) => Some(tensor.shape().clone()),
                ResolvedAttributes::Operation(_) => None,
            };
        }

        let operation = source.required_operation_for_value(value).ok()?;
        let opcode = operation.opcode()?;
        let mut specs = Vec::new();
        for &input in operation.inputs() {
            let data_type = source.required_value_data_type(input).ok()?;
            specs.push(ValueSpec::new(data_type, self.grown_shape(growth, input)?));
        }

        let attributes = if let Some(symbolic) = &symbolic {
            let input = specs.first().map(|spec| spec.shape().clone());
            match self.grown_attributes(growth, symbolic, input.as_ref())? {
                ResolvedAttributes::Operation(attributes) => attributes,
                ResolvedAttributes::Constant(..) => return None,
            }
        } else {
            self.exported_attributes(operation, &specs)
        };
        // A slice with literal bounds that runs to the end of a growing axis would keep its traced extent.
        if symbolic.is_none()
            && let (OperationAttributes::Slice(slice), Some(input), Some(&operand)) =
                (&attributes, specs.first(), operation.inputs().first())
            && let Ok(traced) = source.required_value_shape(operand)
            && let Ok(axis) = normalize_dim_usize(slice.dim, traced.rank().max(1))
            && let (Some(&extent), Some(&grown)) = (
                traced.dimensions().get(axis),
                input.shape().dimensions().get(axis),
            )
            && grown != extent
            && slice.length > 0
            && slice
                .length
                .saturating_mul(slice.step)
                .saturating_add(slice.start)
                >= extent
        {
            self.traced.get_or_insert_with(|| {
                format!(
                    "Slice of {} elements from {} along axis {}",
                    slice.length, slice.start, slice.dim
                )
            });
            return None;
        }

        // A reshape's inference trusts its target, which must hold the grown input's elements.
        if let (Some(Operation::Reshape), OperationAttributes::Shape(target), Some(input)) =
            (operation.opcode(), &attributes, specs.first())
            && target.element_count() != input.shape().element_count()
        {
            self.traced
                .get_or_insert_with(|| format!("Reshape to {:?}", target.dimensions()));
            return None;
        }

        let Ok(inference) = OperationRequest::new(opcode, attributes, specs).infer() else {
            self.traced.get_or_insert_with(|| {
                operation.source_location().map_or_else(
                    || format!("{opcode:?} with {:?}", operation.attributes()),
                    |location| format!("{opcode:?} at {location}"),
                )
            });
            return None;
        };
        let index = operation
            .results()
            .iter()
            .position(|result| result.value() == value)?;
        Some(inference.outputs().get(index)?.shape().clone())
    }

    // Resolves `symbolic` at the grown extents of its axis references; `None` when one does not grow.
    fn grown_attributes(
        &mut self,
        growth: usize,
        symbolic: &SymbolicAttributes,
        input: Option<&Shape>,
    ) -> Option<ResolvedAttributes> {
        symbolic
            .resolve(input, |dim| {
                self.grown_extent(growth, dim)
                    .ok_or_else(|| SingeError::missing_input("grown extent"))
            })
            .ok()
    }

    fn exported_attributes(
        &self,
        operation: &OperationNode,
        specs: &[ValueSpec],
    ) -> OperationAttributes {
        let grown = specs.first().map(|spec| spec.shape().dimensions());
        match (operation.attributes().clone(), grown) {
            (OperationAttributes::LayerNormalization(mut norm), Some(grown))
                if norm.normalized_shape.len() <= grown.len() =>
            {
                norm.normalized_shape = grown[grown.len() - norm.normalized_shape.len()..].to_vec();
                OperationAttributes::LayerNormalization(norm)
            }
            (OperationAttributes::Shape(target), Some(grown))
                if operation.opcode() == Some(Operation::Expand) =>
            {
                let traced = operation
                    .inputs()
                    .first()
                    .and_then(|&input| self.source.required_value_shape(input).ok());
                let (Some(traced), Ok(offset)) = (
                    traced,
                    usize::try_from(target.rank() - i64::try_from(grown.len()).unwrap_or(0)),
                ) else {
                    return OperationAttributes::Shape(target);
                };
                let dimensions = target
                    .dimensions()
                    .iter()
                    .enumerate()
                    .map(|(axis, &extent)| {
                        axis.checked_sub(offset)
                            .filter(|&input| traced.dimensions().get(input) == Some(&extent))
                            .map_or(extent, |input| grown[input])
                    })
                    .collect::<Vec<_>>();
                Shape::create(dimensions).map_or(
                    OperationAttributes::Shape(target),
                    OperationAttributes::Shape,
                )
            }
            (attributes, _) => attributes,
        }
    }

    fn grown_extent(&mut self, growth: usize, expression: &DimensionExpr) -> Option<i64> {
        match expression.expression() {
            DimensionExpression::Constant(value) => Some(*value),
            DimensionExpression::Axis { value, axis, .. } => self
                .grown_shape(growth, ValueId::new(*value))?
                .dimensions()
                .get(*axis)
                .copied(),
            DimensionExpression::Binary {
                operation,
                lhs,
                rhs,
            } => {
                let (lhs, rhs) = (
                    self.grown_extent(growth, lhs)?,
                    self.grown_extent(growth, rhs)?,
                );
                match operation {
                    Arithmetic::Add => lhs.checked_add(rhs),
                    Arithmetic::Subtract => lhs.checked_sub(rhs),
                    Arithmetic::Multiply => lhs.checked_mul(rhs),
                    Arithmetic::Divide => lhs.checked_div(rhs),
                }
            }
        }
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

    // A constant whose values follow the run-time axes.
    pub(crate) fn symbolic_constant(&mut self, attributes: &SymbolicAttributes) -> Result<String> {
        match attributes {
            SymbolicAttributes::Range { start, end } => {
                let start = self.dimension(start)?;
                let end = self.dimension(end)?;
                let step = self.integers(vec![], &[1])?;
                Ok(self.temporary("Range", vec![start, end, step], vec![]))
            }
            SymbolicAttributes::Dimensions(dimensions) => self.shape_vector(dimensions),
            SymbolicAttributes::Fill { dimensions, value } => {
                let name = self.fresh();
                self.graph
                    .initializer
                    .push(proto::tensor(name.clone(), value)?);
                let shape = self.shape_vector(dimensions)?;
                Ok(self.temporary("Expand", vec![name, shape], vec![]))
            }
            SymbolicAttributes::Reshape { .. }
            | SymbolicAttributes::Expand(_)
            | SymbolicAttributes::Slice { .. }
            | SymbolicAttributes::Resize { .. } => Err(unsupported(
                "a constant with an operation's symbolic attributes",
            )),
        }
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
