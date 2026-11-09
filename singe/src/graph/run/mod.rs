use std::collections::{HashMap, HashSet};

use crate::{
    error::{Error, Result},
    graph::{
        CompiledGraph, Graph, GraphSignature,
        generators::with_generator_states,
        interpreter::{execute_ir, value_releases},
        ir::{OperationId, ValueId},
        symbolic::SymbolicShapeEnv,
    },
    tensor::Tensor,
};

/// Work performed by one interpreted prepared run, excluding input bindings and embedded constants.
/// A multi-result invocation counts once.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphRunStats {
    pub(crate) operations_executed: usize,
    pub(crate) peak_live_values: usize,
    pub(crate) released_values: usize,
}

impl GraphRunStats {
    /// Returns the number of executed operation invocations, counting a multi-result invocation once.
    #[must_use]
    pub fn operations_executed(self) -> usize {
        self.operations_executed
    }

    /// Peak descriptors in the interpreter cache, including inputs/constants.
    /// Aliased descriptors can share storage, so this is not an allocation count.
    #[must_use]
    pub fn peak_live_values(self) -> usize {
        self.peak_live_values
    }

    /// Returns the number of value descriptors removed from the interpreter cache after their last use.
    #[must_use]
    pub fn released_values(self) -> usize {
        self.released_values
    }
}

impl CompiledGraph {
    /// Runs the prepared graph using its retained IR, guards, and symbolic analysis.
    /// Returned tensors remain readable after any later run.
    /// Compiled shape programs accept only their resolved input dimensions.
    ///
    /// # Errors
    ///
    /// Returns an error if input resolution or guards fail, required IR values or operation metadata are missing, symbolic shape evaluation fails, or an executed backend operation fails.
    pub fn run(&self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        self.run_with_stats(inputs).map(|(outputs, _)| outputs)
    }

    /// Runs the prepared IR and reports executed operation invocations.
    ///
    /// # Errors
    ///
    /// Returns an error if input resolution or guards fail, required IR values or operation metadata are missing, symbolic shape evaluation fails, or an executed backend operation fails.
    pub fn run_with_stats(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<(Vec<Tensor>, GraphRunStats)> {
        let inputs = with_generator_states(self.ir().generators(), inputs)?;
        let resolved = self.analysis.signature.resolve(&inputs)?;
        let targets = write_targets(self.ir().input_writes(), &resolved)?;
        let cache = self.ir().bind_inputs(&resolved)?;
        drop(resolved);
        let (outputs, stats) = execute_ir(
            self.ir(),
            &self.analysis.operation_order,
            &self.analysis.operation_requests,
            &self.analysis.release_after,
            &self.analysis.symbolic_shapes,
            cache,
        )?;
        let outputs = narrow_counted(self.ir().counted_outputs(), outputs)?;
        Ok((write_back(&targets, outputs)?, stats))
    }
}

impl Graph {
    /// Interprets a graph once.
    /// Prepare it with `compile` for repeated execution.
    /// Every declared input is validated, including unused inputs.
    ///
    /// # Errors
    ///
    /// Returns an error if the reachable graph cannot be topologically ordered, graph projection or symbolic analysis fails, an input violates its signature, or a backend operation fails.
    pub fn run(&self, inputs: &HashMap<String, Tensor>) -> Result<Vec<Tensor>> {
        let order = self.topological_sort_reachable()?;
        self.run_ordered(inputs, &order)
    }

    /// Interprets graph nodes in the supplied order after validating all declared inputs.
    /// The order must place each required producer before its consumers.
    /// Multiple result nodes of one operation execute that operation once.
    ///
    /// # Errors
    ///
    /// Returns an error if graph projection, symbolic shape analysis, input resolution, operation validation, value lifetime analysis, or backend execution fails.
    pub fn run_ordered(
        &self,
        inputs: &HashMap<String, Tensor>,
        order: &[ValueId],
    ) -> Result<Vec<Tensor>> {
        let ir = self.to_ir()?;
        let symbols = SymbolicShapeEnv::create_from_ir(&ir)?;
        let signature = GraphSignature::create(self)?;
        let inputs = with_generator_states(ir.generators(), inputs)?;
        let resolved = signature.resolve(&inputs)?;
        let targets = write_targets(ir.input_writes(), &resolved)?;
        let cache = ir.bind_inputs(&resolved)?;
        let operations = operation_order(&ir, order)?;
        let requests = ir.checked_requests(&operations)?;
        let releases = value_releases(&ir, &operations)?;
        let (outputs, _) = execute_ir(&ir, &operations, &requests, &releases, &symbols, cache)?;
        write_back(&targets, narrow_counted(ir.counted_outputs(), outputs)?)
    }
}

/// Returns the resolved tensors that the trailing outputs write back into, in output order.
pub fn write_targets(
    input_writes: &[String],
    resolved: &HashMap<String, Tensor>,
) -> Result<Vec<Tensor>> {
    input_writes
        .iter()
        .map(|name| {
            resolved
                .get(name)
                .cloned()
                .ok_or_else(|| Error::missing_input("write targets"))
        })
        .collect()
}

/// Writes the trailing outputs into their input tensors and returns the remaining outputs.
pub fn write_back(targets: &[Tensor], mut outputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
    let visible = outputs
        .len()
        .checked_sub(targets.len())
        .ok_or_else(|| Error::missing_output("write back"))?;
    for (target, value) in targets.iter().zip(outputs.split_off(visible)) {
        target.assign(&value)?;
    }
    Ok(outputs)
}

// Narrows each counted output to the count its trailing output holds, and returns the other outputs.
pub fn narrow_counted(counted: &[usize], mut outputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
    let visible = outputs
        .len()
        .checked_sub(counted.len())
        .ok_or_else(|| Error::missing_output("narrow counted"))?;
    let counts = outputs.split_off(visible);
    for (&position, count) in counted.iter().zip(counts) {
        let length = count
            .to_vec::<i64>()?
            .first()
            .copied()
            .ok_or_else(|| Error::missing_output("narrow counted"))?;
        let output = outputs
            .get_mut(position)
            .ok_or_else(|| Error::missing_output("narrow counted"))?;
        *output = output.narrow(0, 0, length)?;
    }
    Ok(outputs)
}

/// Returns the operations that produce `values`, in order and without repeats.
pub fn operation_order(ir: &Graph, values: &[ValueId]) -> Result<Vec<OperationId>> {
    let mut seen = HashSet::new();
    let mut order = Vec::new();
    for &value in values {
        let operation = ir.required_operation_for_value(value)?.id();
        if seen.insert(operation) {
            order.push(operation);
        }
    }
    Ok(order)
}

#[cfg(test)]
mod tests;
