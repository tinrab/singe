use std::{collections::HashMap, sync::Arc};

use singe_cuda::submission::AccessMode;

use crate::{
    backend::cuda::{
        compiled::executable::{CudaCompiledGraph, replay::ReplaySubmission},
        graph_binding::{copy_cuda_tensor_data_on, validate_graph_input_value_tensor},
        graph_value::CudaGraphValue,
        stream::CudaStream,
        tensor_region::CudaTensorSource,
    },
    error::{Error, Result},
    graph::ir::{ValueBindingKind, ValueId},
    tensor::Tensor,
};

impl CudaCompiledGraph {
    pub(crate) fn runtime_cache(
        &self,
        inputs: &HashMap<String, Tensor>,
        stream: &CudaStream,
        submission: &mut ReplaySubmission,
    ) -> Result<HashMap<ValueId, CudaGraphValue>> {
        let mut cache = HashMap::new();
        let mut input_copies = Vec::new();

        let resolved_inputs = self.guards.check_inputs(inputs)?;
        let resolved_inputs = self.ir.bind_inputs(&resolved_inputs)?;

        for &id in &self.execution_order {
            let value = id;
            match self.ir.required_value_binding_kind(value)? {
                ValueBindingKind::RuntimeInput | ValueBindingKind::InitializerInput => {
                    let tensor = resolved_inputs
                        .get(&id)
                        .ok_or_else(|| Error::missing_input("runtime cache"))?;
                    validate_graph_input_value_tensor(&self.ir, id, tensor)?;
                    // The graph reads an input bound in place where it is, so no copy runs; it must be the bound tensor.
                    if let Some(bound) = self.private.inputs_in_place.get(&id) {
                        if !Arc::ptr_eq(bound.storage(), tensor.storage())
                            || bound.layout() != tensor.layout()
                        {
                            return Err(Error::invalid_argument("cuda graph input bound in place"));
                        }

                        let value = self
                            .values
                            .get(&id)
                            .ok_or_else(|| Error::missing_input("runtime cache"))?;
                        cache.insert(id, value.clone());
                        continue;
                    }

                    let placeholder = self
                        .values
                        .get(&id)
                        .ok_or_else(|| Error::missing_input("runtime cache"))?;
                    input_copies.push((placeholder.clone(), tensor.clone()));
                    cache.insert(id, placeholder.clone());
                }
                ValueBindingKind::StaticInput | ValueBindingKind::Constant => {
                    let tensor = self
                        .values
                        .get(&id)
                        .ok_or_else(|| Error::missing_input("runtime cache"))?;
                    cache.insert(id, tensor.clone());
                }
                ValueBindingKind::OperationOutput => {
                    // The order names an operation by its first result, and its other results live beside it.
                    for result in self.ir.required_operation_for_value(id)?.result_values() {
                        if let Some(tensor) = self.values.get(&result) {
                            cache.insert(result, tensor.clone());
                        }
                    }
                }
            }
        }

        for (placeholder, tensor) in &input_copies {
            submission.record_input_copy(tensor.clone());
            // SAFETY: initialized runtime tensors and graph placeholders are retained through completion; replay preflight orders graph-private writers.
            let mut resources = placeholder.submission_resources(AccessMode::Write)?;
            resources.retain(Arc::new(tensor.clone()));
            resources.access(tensor.tensor_region()?.region(), AccessMode::Read);
            // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
            unsafe {
                submission.enqueue(resources, || {
                    copy_cuda_tensor_data_on(placeholder, tensor, stream)
                })?;
            }
        }

        Ok(cache)
    }
}
