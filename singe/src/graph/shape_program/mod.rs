mod attributes;
mod constraints;
mod value;

use std::collections::{BTreeMap, HashMap, HashSet};

#[cfg(feature = "onnx")]
pub use self::constraints::DimensionConstraint;
pub use self::{
    attributes::{ResolvedAttributes, SymbolicAttributes},
    constraints::ValueShapeConstraint,
    value::ShapeValue,
};
use crate::{
    backend::registry::{OperationAttributes, OperationRequest},
    device::Device,
    error::{Error, Result},
    graph::{DimensionExpr, Graph, capture, ir::ValueId, outputs::GraphOutput},
    tensor::Tensor,
    value::ValueSpec,
};

/// Symbolic shape values, node attributes, and shape constraints, keyed by value.
/// Axis references inside expressions hold a value index.
#[derive(Debug, Clone, Default)]
pub struct ShapeProgram {
    pub(crate) attributes: BTreeMap<ValueId, SymbolicAttributes>,
    pub(crate) values: BTreeMap<ValueId, ShapeValue>,
    pub(crate) constraints: BTreeMap<ValueId, Vec<ValueShapeConstraint>>,
}

impl ShapeProgram {
    pub(crate) fn is_empty(&self) -> bool {
        self.attributes.is_empty() && self.values.is_empty() && self.constraints.is_empty()
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
            attributes: self
                .attributes
                .iter()
                .map(|(id, attributes)| {
                    Ok((key(*id)?, attributes.map(|dim| dim.remap(None, index))?))
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

    #[cfg(feature = "onnx")]
    pub(crate) fn extend(&mut self, other: Self) {
        self.attributes.extend(other.attributes);
        self.values.extend(other.values);
        self.constraints.extend(other.constraints);
    }

    /// Removes the entries whose key fails `keep`.
    pub(crate) fn retain(&mut self, keep: impl Fn(ValueId) -> bool) {
        self.attributes.retain(|id, _| keep(*id));
        self.values.retain(|id, _| keep(*id));
        self.constraints.retain(|id, _| keep(*id));
    }

    /// Returns the key indices that the expressions of `id` read through axis references.
    pub(crate) fn references(&self, id: ValueId) -> Vec<usize> {
        let mut references = Vec::new();
        if let Some(attributes) = self.attributes.get(&id) {
            for dim in attributes.expressions() {
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
            attributes: HashMap::new(),
            constants: HashMap::new(),
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
                    .map(|dim| dim.evaluate(&mut |source, axis| evaluator.axis(source, axis)))
                    .collect::<Result<Vec<_>>>()
                    .map_err(|error| error.with_graph_source(location))?;
                evaluator
                    .constants
                    .insert(id, Tensor::from_vec(Device::Cpu, value.shape()?, values)?);
            } else {
                pending.extend(graph.required_node(id)?.inputs());
            }
        }
        Ok(ResolvedShapes {
            attributes: evaluator.attributes,
            constants: evaluator.constants,
        })
    }
}

// Each operation's concrete attributes, and each constant's host values, at one set of input shapes.
struct ResolvedShapes {
    attributes: HashMap<ValueId, OperationAttributes>,
    constants: HashMap<ValueId, Tensor>,
}

struct Evaluator<'a> {
    graph: &'a Graph,
    program: &'a ShapeProgram,
    specs: Vec<Option<ValueSpec>>,
    visiting: Vec<bool>,
    attributes: HashMap<ValueId, OperationAttributes>,
    constants: HashMap<ValueId, Tensor>,
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
        } else if let Some(symbolic) = self.program.attributes.get(&id).cloned() {
            let inputs = node
                .inputs()
                .iter()
                .map(|&input| self.spec(input))
                .collect::<Result<Vec<_>>>()?;
            let resolved = symbolic
                .resolve(inputs.first().map(ValueSpec::shape), |dim| {
                    dim.evaluate(&mut |source, axis| self.axis(source, axis))
                })
                .map_err(|error| error.with_graph_source(node.source_location()))?;
            match resolved {
                ResolvedAttributes::Constant(tensor) => {
                    let spec = ValueSpec::new(tensor.data_type(), tensor.shape().clone());
                    self.constants.insert(id, tensor);
                    spec
                }
                ResolvedAttributes::Operation(attributes) => {
                    self.attributes.insert(id, attributes.clone());
                    self.infer(id, attributes, inputs)?
                }
            }
        } else if node.input_metadata().is_some() || node.constant_tensor(0).is_some() {
            node.output_spec()
        } else {
            let inputs = node
                .inputs()
                .iter()
                .map(|&input| self.spec(input))
                .collect::<Result<Vec<_>>>()?;
            self.infer(id, node.attributes(), inputs)?
        };
        self.visiting[id.index()] = false;
        self.specs[id.index()] = Some(spec.clone());
        Ok(spec)
    }

    fn axis(&mut self, source: usize, axis: usize) -> Result<i64> {
        self.spec(ValueId::new(source))?
            .shape()
            .dimensions()
            .get(axis)
            .copied()
            .ok_or_else(|| Error::dimension_out_of_bounds("shape program axis"))
    }

    fn infer(
        &self,
        id: ValueId,
        attributes: OperationAttributes,
        inputs: Vec<ValueSpec>,
    ) -> Result<ValueSpec> {
        let node = self.graph.required_node(id)?;
        let operation = node
            .operation()
            .ok_or_else(|| Error::missing_node("spec"))?;
        OperationRequest::new(operation, attributes, inputs)
            .infer()
            .map_err(|error| error.with_graph_source(node.source_location()))?
            .outputs()
            .first()
            .cloned()
            .ok_or_else(|| Error::missing_output("spec"))
    }
}

impl Graph {
    // Resolves the shape stage with each input named in `inputs` taking that tensor's metadata.
    pub(crate) fn resolve_shape_program_at(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let mut graph = self.clone();
        graph.rewrite_nodes(|_, mut node| {
            if let Some((name, ..)) = node.view().input_metadata()
                && let Some(tensor) = inputs.get(name)
            {
                node.update_input_metadata(tensor);
            }
            Ok(node)
        })?;
        graph.resolve_shape_program()
    }

    /// Resolves the shape stage entirely from logical metadata.
    /// Layout inference and materialization run afterward on the resulting concrete requests.
    pub(crate) fn resolve_shape_program(&self) -> Result<Self> {
        if self.shape_program.is_empty() {
            return Ok(self.clone());
        }

        let mut graph = self.clone();
        let resolved = self.shape_program.resolve(self)?;
        for (id, attributes) in resolved.attributes {
            let node = graph.required_node(id)?.with_attributes(attributes)?;
            graph.replace_node(id, node)?;
        }

        for (id, tensor) in resolved.constants {
            let device = graph.required_value_device(id)?;
            let tensor = if device.is_meta() {
                capture::meta_with_value(&tensor)?
            } else {
                tensor.to_device(device)?
            };
            graph.update_node(id, |node| node.replace_with_constant(tensor))?;
        }

        graph.shape_program = ShapeProgram::default();
        Ok(graph)
    }
}
