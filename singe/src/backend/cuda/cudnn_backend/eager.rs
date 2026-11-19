use std::time::Duration;

use singe_cuda::stream::AsStream;
use singe_cudnn::{
    backend::{
        execution::{ExecutionPlan as CudnnExecutionPlan, VariantPack},
        tensor::BackendTensorUid,
    },
    context::Context as CudnnContext,
    error::Status,
};

use crate::{
    backend::{
        cuda::{
            self, CublasLtMatmulTuningOptions, CudaEvent, CudaStream, EventTiming,
            cudnn_backend::{
                CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendGraphKey, CudnnBackendGraphKind,
                CudnnBackendGraphTemplate, CudnnBackendTensorMetadata, PlanCandidate,
                PlanSelection, cached_cudnn_backend_graph_template, cudnn_backend_tensor_binding,
                cudnn_backend_tensor_metadata_key,
            },
            diagnostics::eager_operation_error,
            eager::unsupported_eager_operation,
            stream::ensure_not_stream_capturing,
            tensor_region::CudaTensorSource,
            utility::ensure_same_device,
        },
        registry::{BackendExecution, Operation, OperationSchema, operation_name},
    },
    data_type::DataType,
    error::{Error, Result, UnsupportedTarget},
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    utility::{Locked, to_i64},
};

/// Executes an owned native template with inputs in the supplied operation-schema order.
/// Request-specific schemas describe independently optional operand roles.
/// Each input and the output retain their own storage encoding through allocation and binding.
pub fn execute_cudnn_template<'a>(
    schema: &OperationSchema,
    inputs: impl Into<Vec<(BackendTensorUid, &'a Tensor)>>,
    output_data_type: DataType,
    output_shape: &Shape,
    kind: CudnnBackendGraphKind,
    parameters: Vec<i64>,
    build: impl FnOnce(
        &CudnnContext,
        &[CudnnBackendTensorMetadata],
        &CudnnBackendTensorMetadata,
        PlanSelection,
    ) -> Result<CudnnBackendGraphTemplate>,
) -> Result<Tensor> {
    // Eager cuDNN templates are planned for contiguous operands, so strided inputs are copied first.
    let owned = inputs
        .into()
        .into_iter()
        .map(|(id, input)| Ok((id, input.contiguous()?)))
        .collect::<Result<Vec<_>>>()?;
    let inputs = owned
        .iter()
        .map(|(id, input)| (*id, input))
        .collect::<Vec<_>>();
    let input = inputs
        .first()
        .ok_or_else(|| Error::missing_input("execute cudnn template"))?
        .1;
    for &(_, operand) in &inputs {
        ensure_same_device(input, operand)?;
    }

    let data_types = inputs
        .iter()
        .map(|(_, input)| input.data_type())
        .collect::<Vec<_>>();
    let roles = schema.operands().roles_for_count(inputs.len());
    if !schema.operands().accepts_count(inputs.len()) {
        return unsupported_eager_operation(
            schema.operation(),
            output_data_type,
            output_shape,
            &data_types,
        );
    }

    if !schema.support().supports_operand_data_types(
        BackendExecution::CudaEager,
        &roles,
        &data_types,
    ) || !schema
        .support()
        .supports_data_type(BackendExecution::CudaEager, output_data_type)
    {
        return Err(eager_operation_error(
            Error::unsupported_data_type(operation_name(schema.operation())),
            schema.operation(),
            output_data_type,
            output_shape,
            &data_types,
        ));
    }

    ensure_not_stream_capturing(
        input.device(),
        "cuda eager cudnn template during stream capture",
    )?;
    let context = cuda::execution_context(input.device())?;
    context.bind()?;
    let output = context.allocate_output(output_data_type, output_shape)?;
    if output.is_empty() {
        // SAFETY: the empty output has no elements to initialize.
        return Ok(unsafe { output.assume_init() });
    }

    let output_layout = Layout::from_shape(output_shape);
    let input_metadata = inputs
        .iter()
        .map(|(_, input)| {
            CudnnBackendTensorMetadata::new(
                input.data_type(),
                input.shape(),
                &Layout::from_tensor(input),
            )
        })
        .collect::<Vec<_>>();
    let output_metadata =
        CudnnBackendTensorMetadata::new(output_data_type, output_shape, &output_layout);
    let key = CudnnBackendGraphKey::new(
        kind,
        parameters,
        inputs
            .iter()
            .map(|(_, input)| {
                cudnn_backend_tensor_metadata_key(
                    input.data_type(),
                    input.shape(),
                    &Layout::from_tensor(input),
                )
            })
            .collect(),
        cudnn_backend_tensor_metadata_key(output_data_type, output_shape, &output_layout),
    );
    let operation = schema.operation();
    let template_owner =
        cached_cudnn_backend_graph_template(context.runtime(), key.for_eager(), |cudnn| {
            build(
                cudnn,
                &input_metadata,
                &output_metadata,
                PlanSelection::EAGER,
            )
        })
        .map_err(|error| unsupported_cudnn_configuration(operation, error))?;
    let mut template = template_owner.locked("cuda runtime cache")?;
    // The first call times the plans the heuristics offered on its own operands and keeps the fastest.
    let candidates = template.take_candidates();
    let workspace_size = candidates
        .iter()
        .map(|candidate| candidate.workspace_size)
        .chain([template.workspace_size()])
        .max()
        .unwrap_or(0);
    // The workspace comes from the caching allocator, which reuses it in stream order: a workspace of its own would free with `cudaFree`, which waits for the whole device, after every call.
    let workspace_bytes = to_i64(workspace_size.max(1), "cudnn workspace byte size")?;
    let mut workspace =
        context.allocate_output(DataType::U8, &Shape::create([workspace_bytes])?)?;
    let input_regions = inputs
        .iter()
        .map(|(_, input)| Ok(input.tensor_region()?.region()))
        .collect::<Result<Vec<_>>>()?;
    let output_region = output.region().region();
    let mut bindings = inputs
        .iter()
        .zip(&input_regions)
        .map(|((id, input), region)| {
            cudnn_backend_tensor_binding(
                &template,
                context.runtime().context(),
                *id,
                input.data_type(),
                *region,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    bindings.push(cudnn_backend_tensor_binding(
        &template,
        context.runtime().context(),
        CUDNN_GRAPH_OUTPUT_TENSOR_ID,
        output_data_type,
        output_region,
    )?);
    // Graph population locks the cuDNN handle before the template.
    // Release this metadata lock before entering the handle scope to preserve that order.
    let mut plans = vec![PlanCandidate {
        plan: template.plan().clone(),
        workspace_size: template.workspace_size(),
    }];
    plans.extend(candidates);
    drop(template);
    // SAFETY: binding encodings, capacities, contexts and alignment match the template.
    // All owners remain outside the draining scope through execution, errors and unwinding.
    let pack = unsafe {
        VariantPack::create(
            context.runtime().context(),
            &bindings,
            Some(workspace.output::<u8>()?),
        )?
    };
    let mut regions = input_regions;
    regions.extend([output_region, workspace.region().region()]);
    let chosen = context.with_cudnn(|cudnn| {
        let binding = cudnn.stream_binding();
        let execute = || -> Result<usize> {
            for region in &regions {
                if let Some(producer) = region.stream()
                    && producer.as_raw() != binding.as_raw()
                {
                    producer.ensure_not_capturing()?;
                    producer.synchronize()?;
                }
            }

            let run = |plan: &CudnnExecutionPlan| -> Result<()> {
                // SAFETY: the template and initialized input remain live, the output is exclusive, and this scope completes all writes before pending storage is promoted, or on the eager stream orders every reuse of it after them.
                unsafe { plan.execute_async(cudnn, &pack) }?;
                Ok(())
            };
            if plans.len() == 1 {
                run(&plans[0].plan)?;
                return Ok(0);
            }

            // Every plan writes the same output, so the last timed run leaves it whole; with none run, the first runs now and reports why it fails.
            if let Some(index) = fastest_plan(context.stream(), &plans, run)? {
                Ok(index)
            } else {
                run(&plans[0].plan)?;
                Ok(0)
            }
        };
        if context.is_eager() {
            execute()
        } else {
            binding.sync_scope(execute)
        }
    })?;
    drop(regions);
    if chosen != 0 {
        template_owner
            .locked("cuda runtime cache")?
            .set_plan(plans.swap_remove(chosen));
    }
    // SAFETY: completed cuDNN execution initializes every contiguous output element.
    Ok(unsafe { output.assume_init() })
}

fn fastest_plan(
    stream: &CudaStream,
    plans: &[PlanCandidate],
    run: impl Fn(&CudnnExecutionPlan) -> Result<()>,
) -> Result<Option<usize>> {
    let iterations = CublasLtMatmulTuningOptions::DEFAULT;
    let start = CudaEvent::create(stream.device(), EventTiming::Recorded)?;
    let end = CudaEvent::create(stream.device(), EventTiming::Recorded)?;
    let mut best: Option<(usize, Duration)> = None;
    for (index, candidate) in plans.iter().enumerate() {
        if run(&candidate.plan).is_err() {
            continue;
        }

        for _ in 1..iterations.warmup_iterations() {
            run(&candidate.plan)?;
        }

        start.record(stream)?;
        for _ in 0..iterations.measurement_iterations() {
            run(&candidate.plan)?;
        }

        end.record(stream)?;
        end.synchronize()?;
        let elapsed = end.elapsed_since(&start)?;
        if best.is_none_or(|(_, fastest)| elapsed < fastest) {
            best = Some((index, elapsed));
        }
    }
    Ok(best.map(|(index, _)| index))
}

/// Reports cuDNN configurations without a supported engine as unsupported operations.
fn unsupported_cudnn_configuration(operation: Operation, error: Error) -> Error {
    match &error {
        Error::Cudnn(cudnn) if cudnn.status().is_some_and(Status::is_not_supported) => {
            Error::Unsupported {
                operation: format!("cuda eager cuDNN {}", operation_name(operation)).into(),
                target: UnsupportedTarget::Operation,
                details: "cudnn has no engine for this configuration".into(),
                source: Some(Box::new(error)),
            }
        }
        _ => error,
    }
}
