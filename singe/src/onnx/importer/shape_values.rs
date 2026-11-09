use singe_onnx::{DimensionValue, Node, TypeValue, ValueInfo};

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error as SingeError, Result as SingeResult},
    graph::{DimensionConstraint, DimensionExpr, ShapeValue, ValueShapeConstraint, ir::ValueId},
    onnx::{
        attributes::Attributes,
        error::{Error, Result},
        importer::{ImportedValue, Importer},
        utility::node_label,
    },
    tensor::Tensor,
    utility::{to_i64, to_usize},
};

impl Importer {
    pub(crate) fn metadata_dimension(&self, source: ValueId, axis: usize) -> Result<DimensionExpr> {
        let node = &self.graph.required_node(source)?;
        let fixed = if let Some((name, ..)) = node.input_metadata() {
            self.graph
                .input_dimension_guards
                .get(&(name.to_owned(), axis))
                .is_none_or(|guard| {
                    guard
                        .min()
                        .zip(guard.max())
                        .is_some_and(|(min, max)| min == max)
                })
        } else {
            self.graph.input_dimension_guards.values().all(|guard| {
                guard
                    .min()
                    .zip(guard.max())
                    .is_some_and(|(min, max)| min == max)
            })
        };
        if fixed {
            Ok(DimensionExpr::constant(
                *node
                    .output_shape()
                    .dimensions()
                    .get(axis)
                    .ok_or(SingeError::DimensionOutOfBounds)?,
            ))
        } else {
            Ok(DimensionExpr::axis(None, source.index(), axis))
        }
    }

    pub(crate) fn record_shape_constraint(&mut self, info: &ValueInfo) -> Result<()> {
        let Some(ty) = &info.value_type else {
            return Ok(());
        };
        let TypeValue::Tensor(tensor) = &ty.value else {
            return Ok(());
        };
        let Some(shape) = &tensor.shape else {
            return Ok(());
        };
        let dimensions = shape
            .dimensions
            .iter()
            .map(|dim| match &dim.value {
                Some(DimensionValue::Value(value)) => DimensionConstraint::Fixed(*value),
                Some(DimensionValue::Parameter(name)) => {
                    let (min, max) = self.options.symbolic_bounds(name);
                    DimensionConstraint::Symbol {
                        name: name.clone(),
                        min,
                        max,
                    }
                }
                None => DimensionConstraint::Unknown,
            })
            .collect();
        let id = self.value(&info.name)?.node;
        self.graph
            .shape_program
            .constraints
            .entry(id)
            .or_default()
            .push(ValueShapeConstraint {
                name: info.name.clone(),
                dimensions,
            });
        Ok(())
    }

    pub(crate) fn lower_metadata_arithmetic(&mut self, node: &Node) -> Result<bool> {
        if !matches!(node.operator_type.as_str(), "Add" | "Sub" | "Mul" | "Div")
            || node.inputs.len() != 2
            || node.outputs.len() != 1
        {
            return Ok(false);
        }

        let tracked = node.inputs.iter().try_fold(false, |found, name| {
            Ok::<_, Error>(found || self.metadata_shape_value(name)?.is_some())
        })?;
        if !tracked {
            return Ok(false);
        }

        let Some(lhs) = self.shape_operand(&node.inputs[0])? else {
            return Ok(false);
        };
        let Some(rhs) = self.shape_operand(&node.inputs[1])? else {
            return Ok(false);
        };
        let value = lhs.binary(&rhs, |lhs, rhs| match node.operator_type.as_str() {
            "Add" => lhs + rhs,
            "Sub" => lhs - rhs,
            "Mul" => lhs * rhs,
            _ => lhs / rhs,
        })?;
        self.register_metadata_value(&node.outputs[0], value)?;
        Ok(true)
    }

    fn register_metadata_value(&mut self, name: &str, value: ShapeValue) -> Result<()> {
        let shape = value.shape()?;
        let tensor = Tensor::from_vec(
            Device::Cpu,
            shape.clone(),
            self.evaluate_shape_value(&value)?,
        )?;
        let id = self.graph.add_constant(tensor);
        self.graph.shape_program.values.insert(id, value);
        self.register_value(name, ImportedValue::new(id, DataType::I64, shape))
    }

    pub(crate) fn lower_metadata_gather(&mut self, node: &Node) -> Result<bool> {
        let Some(ShapeValue::Vector(values)) = self.metadata_shape_value(&node.inputs[0])?.cloned()
        else {
            return Ok(false);
        };
        if self.metadata_shape_value(&node.inputs[1])?.is_some() {
            return Ok(false);
        }

        let Some(indices) = self.static_input_tensor(&node.inputs[1])? else {
            return Ok(false);
        };
        if indices.data_type() != DataType::I64 || indices.shape().rank() > 1 {
            return Ok(false);
        }

        let axis = Attributes::create(node)?.int("axis", 0)?;
        if !matches!(axis, 0 | -1) {
            return Err(Error::invalid_attribute(
                node_label(node),
                "axis",
                "metadata gather requires axis 0 or -1",
            ));
        }

        let length = to_i64(values.len(), "metadata gather length")?;
        let gathered = indices
            .to_vec::<i64>()?
            .into_iter()
            .map(|index| {
                let index = if index < 0 {
                    index
                        .checked_add(length)
                        .ok_or_else(|| SingeError::out_of_range("metadata gather index"))?
                } else {
                    index
                };
                let index = to_usize(index, "metadata gather index")?;
                values
                    .get(index)
                    .cloned()
                    .ok_or_else(|| SingeError::out_of_range("metadata gather index"))
            })
            .collect::<SingeResult<Vec<_>>>()?;
        let value = if indices.shape().rank() == 0 {
            ShapeValue::Scalar(
                gathered
                    .into_iter()
                    .next()
                    .ok_or_else(|| SingeError::missing_input("lower metadata gather"))?,
            )
        } else {
            ShapeValue::Vector(gathered)
        };
        self.register_metadata_value(&node.outputs[0], value)?;
        Ok(true)
    }

    pub(crate) fn metadata_shape_value(&self, name: &str) -> Result<Option<&ShapeValue>> {
        Ok(self.graph.shape_program.values.get(&self.value(name)?.node))
    }

    /// Reads only static integer literals or previously tracked metadata.
    /// Static initializer dependencies retain the importer's explicit value guards.
    pub(crate) fn shape_operand(&self, name: &str) -> Result<Option<ShapeValue>> {
        if let Some(value) = self.metadata_shape_value(name)? {
            return Ok(Some(value.clone()));
        }

        let value = self.value(name)?;
        if value.data_type != DataType::I64 || value.shape.rank() > 1 {
            return Ok(None);
        }

        let Some(tensor) = self.static_input_tensor(name)? else {
            return Ok(None);
        };
        let elements = tensor
            .to_vec::<i64>()?
            .into_iter()
            .map(DimensionExpr::constant)
            .collect::<Vec<_>>();
        let value = if value.shape.rank() == 0 {
            ShapeValue::Scalar(
                elements
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::MissingValue(name.into()))?,
            )
        } else {
            ShapeValue::Vector(elements)
        };
        Ok(Some(value))
    }

    pub(crate) fn evaluate_shape_value(&self, value: &ShapeValue) -> Result<Vec<i64>> {
        value
            .elements()
            .iter()
            .map(|dim| {
                dim.evaluate(&mut |source, axis| {
                    self.graph
                        .required_value_shape(ValueId::new(source))?
                        .dimensions()
                        .get(axis)
                        .copied()
                        .ok_or(SingeError::DimensionOutOfBounds)
                })
                .map_err(Into::into)
            })
            .collect()
    }

    pub(crate) fn propagate_shape_value(&mut self, node: &Node) -> Result<()> {
        if node.operator_type == "Shape" {
            return Ok(());
        }

        let Some(output_name) = node.outputs.first().filter(|name| !name.is_empty()) else {
            return Ok(());
        };
        let output = self.value(output_name)?.clone();
        if self.graph.shape_program.values.contains_key(&output.node) {
            return Ok(());
        }

        if output.data_type != DataType::I64 || output.shape.rank() > 1 {
            return Ok(());
        }

        let tracked = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .try_fold(false, |found, name| {
                Ok::<_, Error>(found || self.metadata_shape_value(name)?.is_some())
            })?;
        if !tracked {
            return Ok(());
        }

        let Some(inputs) = node
            .inputs
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| self.shape_operand(name))
            .collect::<Result<Option<Vec<_>>>>()?
        else {
            return Ok(());
        };
        let Some(first) = inputs.first() else {
            return Ok(());
        };
        let value = match node.operator_type.as_str() {
            "Identity" | "Reshape" | "Squeeze" | "Unsqueeze" | "Cast" => {
                first.reshape(&output.shape)?
            }
            "Add" | "Sub" | "Mul" | "Div" if inputs.len() == 2 => {
                first.binary(&inputs[1], |lhs, rhs| match node.operator_type.as_str() {
                    "Add" => lhs + rhs,
                    "Sub" => lhs - rhs,
                    "Mul" => lhs * rhs,
                    _ => lhs / rhs,
                })?
            }
            "Neg" => ShapeValue::Scalar(0.into()).binary(first, |zero, value| zero - value)?,
            "Concat" => ShapeValue::Vector(
                inputs
                    .iter()
                    .flat_map(|value| value.elements().iter().cloned())
                    .collect(),
            ),
            _ => return Ok(()),
        };
        self.graph.shape_program.values.insert(output.node, value);
        Ok(())
    }
}
