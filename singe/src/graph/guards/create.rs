use std::collections::HashSet;

#[cfg(test)]
use crate::graph::guards::ConstantGuard;
use crate::{
    error::{Error, Result},
    graph::{
        Graph,
        guards::{GraphGuards, InputGuard, storage::static_input_storage_guard},
    },
};
#[cfg(test)]
use crate::{graph::guards::storage::constant_guard, graph::guards::storage::storage_span_bytes};

impl GraphGuards {
    /// Creates guards for the inputs and constants of a prepared IR.
    ///
    /// # Errors
    ///
    /// Returns an error if reachable input names are duplicated, IR metadata is missing, or metadata requires an invalid storage span.
    pub fn create_from_ir(ir: &Graph) -> Result<Self> {
        let mut inputs = Vec::new();
        #[cfg(test)]
        let mut constants = Vec::new();
        let mut input_names = HashSet::new();

        for value in ir.values() {
            let value_id = value.id();
            let binding = ir.required_value_binding_details(value_id)?;
            if let Some(name) = binding.input_name() {
                if !input_names.insert(name.to_string()) {
                    return Err(Error::DuplicateInput);
                }

                let initializer = binding
                    .is_initializer_backed_input()
                    .then(|| ir.required_initializer_tensor(value_id).cloned())
                    .transpose()?;
                #[cfg(test)]
                if let Some(initializer) = initializer.as_ref() {
                    constants.push(constant_guard(value_id, initializer)?);
                }

                let is_static_initializer = binding.is_static_input();
                let static_storage = is_static_initializer
                    .then(|| {
                        initializer
                            .as_ref()
                            .map(static_input_storage_guard)
                            .ok_or_else(|| Error::missing_input("create from ir"))
                    })
                    .transpose()?;
                let data_type = ir.required_value_data_type(value_id)?;
                let device = ir.required_value(value_id)?.device();
                let layout = ir.required_value_layout(value_id)?.clone();
                inputs.push(InputGuard {
                    name: name.to_string(),
                    data_type,
                    device,
                    layout: layout.clone(),
                    initializer,
                    #[cfg(test)]
                    storage_span_bytes: storage_span_bytes(&layout, data_type)?,
                    is_static_initializer,
                    static_storage,
                    dimension_guards: ir.input_dimension_guards_for_input(name).cloned().collect(),
                });
            } else if binding.is_constant() {
                #[cfg(test)]
                {
                    let tensor = ir.required_constant_tensor(value_id)?;
                    constants.push(constant_guard(value_id, tensor)?);
                }
            }
        }

        Ok(Self {
            inputs,
            #[cfg(test)]
            constants,
            input_names,
        })
    }

    /// Returns reachable graph input guards.
    #[cfg(test)]
    #[must_use]
    pub fn inputs(&self) -> &[InputGuard] {
        &self.inputs
    }

    /// Returns reachable graph constant guards.
    #[cfg(test)]
    #[must_use]
    pub fn constants(&self) -> &[ConstantGuard] {
        &self.constants
    }
}
