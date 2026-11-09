mod constraints;
mod value;

use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(feature = "onnx")]
pub use self::constraints::DimensionConstraint;
pub use self::{constraints::ValueShapeConstraint, value::ShapeValue};
use crate::{
    backend::registry::{OperationAttributes, OperationRequest},
    device::Device,
    error::{Error, Result},
    graph::{DimensionExpr, Graph, ir::ValueId, outputs::GraphOutput},
    shape::Shape,
    tensor::Tensor,
    value::ValueSpec,
};

/// Reshape attributes expressed in the integer metadata domain.
/// `allow_zero` follows ONNX's explicit zero-copy switch; native module shapes use true.
#[derive(Debug, Clone)]
pub struct ReshapeExpression {
    pub(crate) dimensions: Vec<DimensionExpr>,
    pub(crate) allow_zero: bool,
}

/// Symbolic shape values, reshape expressions, and shape constraints, keyed by value.
/// Axis references inside expressions hold a value index.
#[derive(Debug, Clone, Default)]
pub struct ShapeProgram {
    pub(crate) reshapes: BTreeMap<ValueId, ReshapeExpression>,
    pub(crate) values: BTreeMap<ValueId, ShapeValue>,
    pub(crate) constraints: BTreeMap<ValueId, Vec<ValueShapeConstraint>>,
}

impl ShapeProgram {
    pub(crate) fn is_empty(&self) -> bool {
        self.reshapes.is_empty() && self.values.is_empty() && self.constraints.is_empty()
    }

    /// Rewrites every key through `key` and every axis reference through `index`.
    pub(crate) fn remap(
        &self,
        key: impl Fn(ValueId) -> Result<ValueId>,
        index: &impl Fn(usize) -> Result<usize>,
    ) -> Result<Self> {
        let dimensions = |dimensions: &[DimensionExpr]| {
            dimensions
                .iter()
                .map(|dim| dim.remap(None, index))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            reshapes: self
                .reshapes
                .iter()
                .map(|(id, expression)| {
                    Ok((
                        key(*id)?,
                        ReshapeExpression {
                            dimensions: dimensions(&expression.dimensions)?,
                            allow_zero: expression.allow_zero,
                        },
                    ))
                })
                .collect::<Result<_>>()?,
            values: self
                .values
                .iter()
                .map(|(id, value)| {
                    let value = match value {
                        ShapeValue::Scalar(dim) => ShapeValue::Scalar(dim.remap(None, index)?),
                        ShapeValue::Vector(dims) => ShapeValue::Vector(dimensions(dims)?),
                    };
                    Ok((key(*id)?, value))
                })
                .collect::<Result<_>>()?,
            constraints: self
                .constraints
                .iter()
                .map(|(id, constraints)| Ok((key(*id)?, constraints.clone())))
                .collect::<Result<_>>()?,
        })
    }

    /// Removes the entries whose key fails `keep`.
    pub(crate) fn retain(&mut self, keep: impl Fn(ValueId) -> bool) {
        self.reshapes.retain(|id, _| keep(*id));
        self.values.retain(|id, _| keep(*id));
        self.constraints.retain(|id, _| keep(*id));
    }

    /// Returns the key indices that the expressions of `id` read through axis references.
    pub(crate) fn references(&self, id: ValueId) -> Vec<usize> {
        let mut references = Vec::new();
        if let Some(expression) = self.reshapes.get(&id) {
            for dim in &expression.dimensions {
                dim.references(&mut references);
            }
        }

        if let Some(value) = self.values.get(&id) {
            for dim in value.elements() {
                dim.references(&mut references);
            }
        }
        references
    }

    fn resolve(&self, graph: &Graph) -> Result<ResolvedShapes> {
        let mut evaluator = Evaluator {
            graph,
            program: self,
            specs: vec![None; graph.node_count()],
            visiting: vec![false; graph.node_count()],
            reshapes: HashMap::new(),
            values: HashMap::new(),
        };
        for output in graph.output_projections() {
            evaluator.spec(output.node())?;
        }

        let mut symbols = HashMap::new();
        for (id, constraints) in &self.constraints {
            let spec = evaluator.spec(*id)?;
            for constraint in constraints {
                constraint.validate(&spec, &mut symbols)?;
            }
        }

        let mut live = HashSet::new();
        let mut pending = graph
            .output_projections()
            .iter()
            .map(GraphOutput::node)
            .collect::<Vec<_>>();
        while let Some(id) = pending.pop() {
            if !live.insert(id) {
                continue;
            }
            if let Some(value) = self.values.get(&id) {
                let location = graph.required_operation_for_value(id)?.source_location();
                let values = value
                    .elements()
                    .iter()
                    .map(|dim| {
                        dim.evaluate(&mut |source, axis| {
                            evaluator
                                .spec(ValueId::new(source))?
                                .shape()
                                .dimensions()
                                .get(axis)
                                .copied()
                                .ok_or(Error::DimensionOutOfBounds)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
                    .map_err(|error| error.with_graph_source(location))?;
                evaluator.values.insert(id, (value.shape()?, values));
            } else {
                pending.extend(graph.required_node(id)?.inputs());
            }
        }
        Ok(ResolvedShapes {
            reshapes: evaluator.reshapes,
            values: evaluator.values,
        })
    }
}

struct ResolvedShapes {
    reshapes: HashMap<ValueId, Shape>,
    values: HashMap<ValueId, (Shape, Vec<i64>)>,
}

struct Evaluator<'a> {
    graph: &'a Graph,
    program: &'a ShapeProgram,
    specs: Vec<Option<ValueSpec>>,
    visiting: Vec<bool>,
    reshapes: HashMap<ValueId, Shape>,
    values: HashMap<ValueId, (Shape, Vec<i64>)>,
}

impl Evaluator<'_> {
    fn spec(&mut self, id: ValueId) -> Result<ValueSpec> {
        if let Some(spec) = self.specs.get(id.index()).and_then(Option::as_ref) {
            return Ok(spec.clone());
        }

        let node = &self.graph.required_node(id)?;
        if self.visiting[id.index()] {
            return Err(Error::GraphCycleDetected);
        }

        self.visiting[id.index()] = true;
        let spec = if let Some(value) = self.program.values.get(&id).cloned() {
            value.spec()?
        } else if node.input_metadata().is_some() || node.constant_tensor(0).is_some() {
            node.output_spec()
        } else {
            let operation = node
                .operation()
                .ok_or_else(|| Error::missing_node("spec"))?;
            let inputs = node
                .inputs()
                .iter()
                .map(|&input| self.spec(input))
                .collect::<Result<Vec<_>>>()?;
            let attributes = if let Some(expression) = self.program.reshapes.get(&id).cloned() {
                let mut dimensions = expression
                    .dimensions
                    .iter()
                    .map(|dim| {
                        dim.evaluate(&mut |source, axis| {
                            let spec = self.spec(ValueId::new(source))?;
                            spec.shape()
                                .dimensions()
                                .get(axis)
                                .copied()
                                .ok_or(Error::DimensionOutOfBounds)
                        })
                    })
                    .collect::<Result<Vec<_>>>()
                    .map_err(|error| error.with_graph_source(node.source_location()))?;
                let source = inputs.first().ok_or_else(|| Error::missing_input("spec"))?;
                if !expression.allow_zero {
                    for (axis, dim) in dimensions.iter_mut().enumerate() {
                        if *dim == 0 {
                            *dim = *source
                                .shape()
                                .dimensions()
                                .get(axis)
                                .ok_or(Error::DimensionOutOfBounds)?;
                        }
                    }
                } else if dimensions.contains(&0) && dimensions.contains(&-1) {
                    return Err(Error::invalid_argument(
                        "reshape zero dimensions cannot be combined with inference",
                    ));
                }

                let shape = source
                    .shape()
                    .reshape(dimensions)
                    .map_err(|error| error.with_graph_source(node.source_location()))?;
                self.reshapes.insert(id, shape.clone());
                OperationAttributes::Shape(shape)
            } else {
                node.attributes()
            };
            OperationRequest::new(operation, attributes, inputs)
                .infer()
                .map_err(|error| error.with_graph_source(node.source_location()))?
                .outputs()
                .first()
                .cloned()
                .ok_or_else(|| Error::missing_output("spec"))?
        };
        self.visiting[id.index()] = false;
        self.specs[id.index()] = Some(spec.clone());
        Ok(spec)
    }
}

impl Graph {
    /// Resolves the shape stage entirely from logical metadata.
    /// Layout inference and materialization run afterward on the resulting concrete requests.
    pub(crate) fn resolve_shape_program(&self) -> Result<Self> {
        if self.shape_program.is_empty() {
            return Ok(self.clone());
        }

        let mut graph = self.clone();
        let resolved = self.shape_program.resolve(self)?;
        for (id, shape) in resolved.reshapes {
            let node = &graph.required_node(id)?;
            let input = *node
                .inputs()
                .first()
                .ok_or_else(|| Error::missing_input("resolve shape program"))?;
            graph.replace_node(id, node.with_reshape_operation(input, shape))?;
        }

        for (id, (shape, values)) in resolved.values {
            let tensor = Tensor::from_vec(Device::Cpu, shape, values)?
                .to_device(graph.required_value_device(id)?)?;
            graph.update_node(id, |node| node.replace_with_constant(tensor))?;
        }

        graph.shape_program = ShapeProgram::default();
        Ok(graph)
    }
}
