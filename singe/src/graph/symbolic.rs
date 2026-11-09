use std::collections::HashMap;

use crate::{
    error::{Error, Result},
    graph::{Graph, InputDimensionGuard, ValueId, ir::Block},
    shape::Shape,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolicDimension {
    name: String,
    min: Option<i64>,
    max: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct SymbolicShapeEnv {
    values: HashMap<ValueId, Vec<Option<SymbolicDimension>>>,
}

impl SymbolicShapeEnv {
    pub fn create_from_ir(ir: &Graph) -> Result<Self> {
        let mut env = Self::default();
        let order = ir.blocks().first().map_or(&[][..], Block::operations);
        for &id in order {
            for value_id in ir.required_operation(id)?.result_values() {
                let value = ir.required_value(value_id)?;
                let binding = ir.required_value_binding_details(value_id)?;
                let symbols = if let Some(name) = binding.input_name() {
                    input_symbols(ir, name, value.shape())
                } else {
                    propagate_same_shape_symbols_from_ir(ir, value_id, &env)?
                };
                if symbols.iter().any(Option::is_some) {
                    env.values.insert(value_id, symbols);
                }
            }
        }
        Ok(env)
    }

    pub fn value_symbols(&self, id: ValueId) -> Option<&[Option<SymbolicDimension>]> {
        self.values.get(&id).map(Vec::as_slice)
    }
}

impl SymbolicDimension {
    pub fn new(name: impl Into<String>, min: Option<i64>, max: Option<i64>) -> Self {
        Self {
            name: name.into(),
            min,
            max,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn min(&self) -> Option<i64> {
        self.min
    }

    pub fn max(&self) -> Option<i64> {
        self.max
    }
}

fn input_symbols(ir: &Graph, name: &str, shape: &Shape) -> Vec<Option<SymbolicDimension>> {
    (0..shape.dimensions().len())
        .map(|axis| ir.input_dimension_guard(name, axis).map(symbol_from_guard))
        .collect()
}

fn symbol_from_guard(guard: &InputDimensionGuard) -> SymbolicDimension {
    SymbolicDimension::new(guard.name(), guard.min(), guard.max())
}

fn propagate_same_shape_symbols_from_ir(
    ir: &Graph,
    id: ValueId,
    env: &SymbolicShapeEnv,
) -> Result<Vec<Option<SymbolicDimension>>> {
    let value = ir.required_value(id)?;
    let mut output = vec![None; value.shape().dimensions().len()];
    let (operation, _) = ir
        .result_for_value(id)
        .ok_or_else(|| Error::missing_node("propagate same shape symbols from ir"))?;
    let operation = ir.required_operation(operation)?;
    for dependency in operation.inputs() {
        let dependency_value = ir.required_value(*dependency)?;
        if dependency_value.shape().dimensions() != value.shape().dimensions() {
            continue;
        }

        let Some(dependency_symbols) = env.value_symbols(*dependency) else {
            continue;
        };
        for (axis, symbol) in dependency_symbols.iter().enumerate() {
            if axis >= output.len() || symbol.is_none() {
                continue;
            }
            match (&output[axis], symbol) {
                (None, Some(symbol)) => output[axis] = Some(symbol.clone()),
                (Some(existing), Some(symbol)) if existing == symbol => {}
                _ => output[axis] = None,
            }
        }
    }
    Ok(output)
}
