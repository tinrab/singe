use std::{collections::HashMap, result::Result as StdResult};

use crate::{
    error::Result,
    graph::{
        Graph, InputDimensionGuard, ValueId,
        guards::{GraphGuardFailure, GraphGuards, validation::check_tensor_metadata},
        symbolic::SymbolicDimension,
    },
    tensor::Tensor,
};

impl GraphGuards {
    /// Validates reachable graph input initializers and passes each input name and tensor to `validate`.
    ///
    /// # Errors
    ///
    /// Returns an error if an initializer violates its input metadata guard or if `validate` rejects the guarded initializer.
    pub fn validate_initializers_with(
        &self,
        mut validate: impl FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<()> {
        for input in &self.inputs {
            if let Some(tensor) = input.initializer() {
                input.check(tensor)?;
                validate(&input.name, tensor)?;
            }
        }
        Ok(())
    }

    /// Validates runtime inputs and returns resolved input tensors by input name.
    ///
    /// Runtime inputs override graph initializers.
    /// Only reachable input names are accepted.
    ///
    /// # Errors
    ///
    /// Returns an error if an input is missing, unexpected, or fails metadata or storage-capacity guards.
    pub fn check_inputs(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        self.check_inputs_with_failure(inputs).map_err(Into::into)
    }

    /// Validates runtime inputs and returns structured guard failure details.
    ///
    /// # Errors
    ///
    /// Returns a structured failure if an input is missing, unexpected, or fails metadata or storage-capacity guards.
    pub fn check_inputs_with_failure(
        &self,
        inputs: &HashMap<String, Tensor>,
    ) -> StdResult<HashMap<String, Tensor>, GraphGuardFailure> {
        for name in inputs.keys() {
            if !self.input_names.contains(name) {
                return Err(GraphGuardFailure::UnexpectedInput(name.clone()));
            }
        }

        let mut resolved = HashMap::new();
        let mut symbol_values = HashMap::new();
        for guard in &self.inputs {
            if guard.is_static_initializer && inputs.contains_key(&guard.name) {
                return Err(GraphGuardFailure::StaticInputOverride(guard.name.clone()));
            }

            let tensor = inputs
                .get(&guard.name)
                .or(guard.initializer.as_ref())
                .ok_or(GraphGuardFailure::MissingInput)?;
            guard.check_static_storage(tensor)?;
            guard.check_with_symbol_values(tensor, &mut symbol_values)?;
            resolved.insert(guard.name.clone(), tensor.clone());
        }
        Ok(resolved)
    }
}

impl GraphGuards {
    /// Validates a tensor against IR value metadata and optional symbolic dimension guards.
    /// Symbol values are checked within this tensor, without sharing bindings with another call.
    ///
    /// # Errors
    ///
    /// Returns a guard failure if IR mappings are missing or the tensor violates data type, device, layout, storage capacity, symbolic equality, or dimension bounds.
    pub fn check_tensor_for_ir_value_with_symbols(
        ir: &Graph,
        value: ValueId,
        tensor: &Tensor,
        symbols: Option<&[Option<SymbolicDimension>]>,
    ) -> StdResult<(), GraphGuardFailure> {
        let name = ir
            .required_value_name(value)?
            .unwrap_or("input")
            .to_string();
        let dimension_guards = symbols
            .map(|symbols| {
                symbols
                    .iter()
                    .enumerate()
                    .filter_map(|(axis, symbol)| {
                        symbol.as_ref().map(|symbol| {
                            InputDimensionGuard::new(
                                name.clone(),
                                axis,
                                symbol.name(),
                                symbol.min(),
                                symbol.max(),
                            )
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let data_type = ir.required_value_data_type(value)?;
        let layout = ir.required_value_layout(value)?;
        let device = ir.required_value(value)?.device();
        check_tensor_metadata(
            &name,
            data_type,
            layout,
            device,
            &dimension_guards,
            &mut HashMap::new(),
            tensor,
        )
    }
}
