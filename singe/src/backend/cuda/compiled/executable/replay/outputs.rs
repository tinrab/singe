use std::collections::HashMap;

use singe_cuda::submission::AccessMode;

use crate::{
    backend::{
        cuda::{
            compiled::executable::{CudaCompiledGraph, replay::ReplaySubmission},
            context::CudaExecutionContext,
            graph_binding::copy_cuda_tensor_data_on,
            graph_value::{CudaGraphValue, GraphOutputBuffers},
            kernel_plan::{self, KernelInput, KernelPlanner},
            stream::CudaStream,
            tensor_region::CudaTensorRegion,
        },
        registry::{Operation, OperationAttributes},
    },
    error::{Error, Result},
    graph::ir::ValueId,
    layout::Layout,
    shape::Shape,
};

impl CudaCompiledGraph {
    pub(crate) fn copy_outputs_for_run(
        &self,
        cache: &HashMap<ValueId, CudaGraphValue>,
        stream: &CudaStream,
        submission: &mut ReplaySubmission,
    ) -> Result<Vec<CudaGraphValue>> {
        let context = CudaExecutionContext::create_with_stream(self.device, Some(stream.clone()))?;
        let mut outputs = Vec::with_capacity(self.outputs.len());
        for (position, output) in self.output_tensors(cache)?.into_iter().enumerate() {
            // The input already holds this value, so a run does not return it.
            if self.private.outputs_written_in_graph.contains(&position) {
                continue;
            }

            let shape = Shape::create(output.shape().dimensions().to_vec())?;
            let storage = context
                .allocator()
                .allocate_uninit_with_shape(output.data_type(), &shape)?;
            let copied =
                CudaGraphValue::create(storage, output.data_type(), Layout::new(shape, 0))?;
            // Retain before submission, including native errors and unwinding.
            submission.record_output_copy(copied.clone());
            let mut resources = output.submission_resources(AccessMode::Read)?;
            resources.extend(copied.submission_resources(AccessMode::Write)?);
            // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
            unsafe {
                submission.enqueue(resources, || {
                    if output.is_contiguous() {
                        // SAFETY: replay orders all producers and retains both allocations;
                        // contiguous copy-out does not read uninitialized stride gaps.
                        copy_cuda_tensor_data_on(&copied, output, stream)?;
                    } else if output.element_count() != 0 {
                        copy_strided(&context, output, copied.region()?)?;
                    }
                    Ok(())
                })?;
            }

            outputs.push(copied);
        }
        Ok(outputs)
    }

    pub(crate) fn copy_outputs_into(
        &self,
        cache: &HashMap<ValueId, CudaGraphValue>,
        outputs: GraphOutputBuffers<'_>,
        stream: &CudaStream,
        submission: &mut ReplaySubmission,
    ) -> Result<Vec<CudaGraphValue>> {
        let outputs = outputs.values();
        if outputs.len() != self.outputs.len() {
            return Err(Error::length_mismatch(
                "cuda graph output buffers",
                self.outputs.len(),
                outputs.len(),
            ));
        }

        let context = CudaExecutionContext::create_with_stream(self.device, Some(stream.clone()))?;
        let mut copied = Vec::with_capacity(outputs.len());
        for (source, output) in self.output_tensors(cache)?.into_iter().zip(outputs) {
            if source.data_type() != output.data_type()
                || source.shape().dimensions() != output.shape().dimensions()
            {
                return Err(Error::invalid_argument("cuda graph output buffer metadata"));
            }

            if !output.is_contiguous() {
                return Err(Error::ContiguousRequired);
            }

            submission.record_output_copy(output.clone());
            let mut resources = source.submission_resources(AccessMode::Read)?;
            resources.extend(output.submission_resources(AccessMode::Write)?);
            // SAFETY: the declared resources retain every buffer and the stream orders all access to them.
            unsafe {
                submission.enqueue(resources, || {
                    if source.is_contiguous() {
                        // SAFETY: the destination request establishes exclusivity; replay retains allocations and orders source production before this copy.
                        copy_cuda_tensor_data_on(output, source, stream)?;
                    } else if source.element_count() != 0 {
                        copy_strided(&context, source, output.region()?)?;
                    }
                    Ok(())
                })?;
            }

            copied.push(output.clone());
        }
        Ok(copied)
    }

    pub(crate) fn graph_owned_outputs(
        &self,
        cache: &HashMap<ValueId, CudaGraphValue>,
    ) -> Result<Vec<CudaGraphValue>> {
        Ok(self.output_tensors(cache)?.into_iter().cloned().collect())
    }

    fn output_tensors<'a>(
        &self,
        cache: &'a HashMap<ValueId, CudaGraphValue>,
    ) -> Result<Vec<&'a CudaGraphValue>> {
        Ok(Self::output_value_tensors(cache, &self.outputs)?
            .into_iter()
            .map(|(_, value)| value)
            .collect())
    }
}

// Strided outputs of every type copy out through the generated contiguous kernel.
fn copy_strided(
    context: &CudaExecutionContext,
    source: &CudaGraphValue,
    destination: CudaTensorRegion<'_>,
) -> Result<()> {
    let plan = KernelPlanner::default().create(
        Operation::Contiguous,
        &OperationAttributes::None,
        &[KernelInput::from_source(source)],
    )?;
    kernel_plan::launch(&plan, &[source], context, &[destination])
}
