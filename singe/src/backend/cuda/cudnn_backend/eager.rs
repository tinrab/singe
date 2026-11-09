use singe_cuda::view::DeviceRegionProvider;
use singe_cudnn::{
    backend::{execution::VariantPack, tensor::BackendTensorUid},
    context::Context as CudnnContext,
    error::Status,
};

use crate::{
    backend::{
        cuda::{
            self,
            compiled::plan::cudnn::allocate_cudnn_workspace,
            cudnn_backend::{
                CUDNN_GRAPH_OUTPUT_TENSOR_ID, CudnnBackendGraphKey, CudnnBackendGraphKind,
                CudnnBackendGraphTemplate, CudnnBackendTensorMetadata,
                cached_cudnn_backend_graph_template, cudnn_backend_tensor_binding,
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
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    tensor::Tensor,
    utility::Locked,
};

/// Executes an owned native template with inputs in the supplied operation-schema order.
/// Request-specific schemas describe independently optional operand roles.
/// Each input and the output retain their own storage encoding through allocation and binding.
#[expect(
    clippy::too_many_lines,
    reason = "the native execution transaction keeps resource binding, submission, and access tracking together"
)]
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
    let template_owner = cached_cudnn_backend_graph_template(context.runtime(), key, |cudnn| {
        build(cudnn, &input_metadata, &output_metadata)
    })
    .map_err(|error| unsupported_cudnn_configuration(operation, error))?;
    let template = template_owner.locked("cuda runtime cache")?;
    let mut workspace =
        allocate_cudnn_workspace(context.runtime().context(), template.workspace_size())?;
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
    let plan = template.plan().clone();
    drop(template);
    // SAFETY: binding encodings, capacities, contexts and alignment match the template.
    // All owners remain outside the draining scope through execution, errors and unwinding.
    let pack = unsafe {
        VariantPack::create(context.runtime().context(), &bindings, Some(&mut workspace))?
    };
    let mut regions = input_regions;
    regions.extend([output_region, workspace.region()]);
    context.with_cudnn(|cudnn| {
        let binding = cudnn.stream()?;
        binding.sync_scope(|| {
            for region in &regions {
                if let Some(producer) = region.stream()
                    && producer.as_raw() != binding.as_raw()
                {
                    producer.ensure_not_capturing()?;
                    producer.synchronize()?;
                }
            }

            // SAFETY: the template and initialized input remain live, the output is exclusive, and this scope completes all writes before pending storage is promoted.
            unsafe { plan.execute(cudnn, &pack) }?;
            Ok(())
        })
    })?;
    drop(regions);
    // SAFETY: completed cuDNN execution initializes every contiguous output element.
    Ok(unsafe { output.assume_init() })
}

/// Reports cuDNN configurations without a supported engine as unsupported operations.
fn unsupported_cudnn_configuration(operation: Operation, error: Error) -> Error {
    match &error {
        Error::Cudnn(cudnn) if cudnn.status().is_some_and(Status::is_not_supported) => {
            Error::unsupported_operation_with_details(
                format!("cuda eager cuDNN {}", operation_name(operation)),
                error.to_string(),
            )
        }
        _ => error,
    }
}
