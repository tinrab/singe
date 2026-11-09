use std::{collections::HashSet, iter};

use crate::{
    error::{Error, Result},
    graph::{Graph, ir::ValueId},
};

/// Projection from a graph node to one graph output value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GraphOutput {
    /// Node that produces the output value.
    node: ValueId,
    /// Zero-based operation output index.
    output_index: usize,
    name: Option<String>,
}

impl GraphOutput {
    pub(crate) fn named(node: ValueId, output_index: usize, name: Option<String>) -> Self {
        Self {
            node,
            output_index,
            name,
        }
    }

    /// Creates a graph output projection from a node and output index.
    pub(crate) fn new(node: ValueId, output_index: usize) -> Self {
        Self {
            node,
            output_index,
            name: None,
        }
    }

    /// Returns the node that produces the output value.
    pub(crate) fn node(&self) -> ValueId {
        self.node
    }

    /// Returns the zero-based operation output index.
    pub(crate) fn output_index(&self) -> usize {
        self.output_index
    }

    pub(crate) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    pub(crate) fn with_node(&self, node: ValueId) -> Self {
        Self {
            node,
            ..self.clone()
        }
    }
}

impl Graph {
    /// Sets graph output node ids.
    ///
    /// # Errors
    ///
    /// Returns an error if any output id does not refer to an existing graph node.
    pub fn set_outputs(&mut self, outputs: Vec<ValueId>) -> Result<()> {
        self.set_output_values(
            outputs
                .into_iter()
                .map(|node| GraphOutput::new(node, 0))
                .collect(),
        )
    }

    /// Sets named outputs in order.
    /// Several names may project the same value.
    ///
    /// # Errors
    ///
    /// Returns an error if output names repeat, a producer node is missing, or an output index does not identify a result of its producer.
    pub fn set_named_output_projections(
        &mut self,
        outputs: Vec<(String, ValueId, usize)>,
    ) -> Result<()> {
        let mut names = HashSet::new();
        let outputs = outputs
            .into_iter()
            .map(|(name, node, output_index)| {
                if !names.insert(name.clone()) {
                    return Err(Error::invalid_argument("duplicate graph output name"));
                }
                Ok(GraphOutput {
                    node,
                    output_index,
                    name: Some(name),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.set_output_values(outputs)
    }

    // Names this graph's outputs, in order.
    pub fn with_output_names<S: Into<String>>(
        mut self,
        names: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let names = names.into_iter().map(Into::into).collect::<Vec<String>>();
        let mut unique = HashSet::new();
        let outputs = self
            .output_projections()
            .iter()
            .map(GraphOutput::node)
            .collect::<Vec<_>>();
        if names.len() > outputs.len()
            || names
                .iter()
                .any(|name| name.is_empty() || !unique.insert(name.as_str()))
        {
            return Err(Error::invalid_argument("graph output names"));
        }

        let names = names
            .into_iter()
            .map(Some)
            .chain(iter::repeat(None))
            .take(outputs.len())
            .collect();
        self.set_output_values_with_names(outputs, names);
        Ok(self)
    }

    pub(crate) fn set_output_values(&mut self, outputs: Vec<GraphOutput>) -> Result<()> {
        for output in &outputs {
            let output_count = self.node_output_count(output.node())?;
            if output.output_index() >= output_count {
                return Err(Error::missing_output("set output values"));
            }
        }

        let values = outputs
            .iter()
            .map(|output| self.node_result(output.node(), output.output_index()))
            .collect::<Result<Vec<_>>>()?;
        let names = outputs.into_iter().map(|output| output.name).collect();
        self.set_output_values_with_names(values, names);
        Ok(())
    }

    /// Replaces graph output projections without validation.
    #[cfg(test)]
    pub(crate) fn set_output_projection_unchecked_for_test(&mut self, node: ValueId) {
        self.set_output_values_with_names(vec![node], vec![None]);
    }

    /// Returns graph output value projections in declared output order.
    pub(crate) fn output_projections(&self) -> Vec<GraphOutput> {
        self.output_values()
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let (node, output_index) = self
                    .result_for_value(value)
                    .and_then(|(operation, result)| {
                        let node = self.operation(operation)?.result_values().next()?;
                        Some((node, result.index()))
                    })
                    .unwrap_or((value, 0));
                GraphOutput::named(
                    node,
                    output_index,
                    self.output_name(index).map(ToOwned::to_owned),
                )
            })
            .collect()
    }
}
