mod attention;
mod dot_product;
mod element_wise;
mod grouped;
mod indexing;
mod mla;
mod mma;
mod norm;
mod paged_attention;
mod paged_general;
mod parameters;
mod random;
mod reduce;
mod rotary_cache;
mod scan;
mod tcgen05;
mod tcgen05_attention;
mod tensor_core;
mod writer;

use std::{
    iter, slice,
    sync::{Arc, Mutex},
};

use singe_cuda::module::KernelFunction;

use crate::{
    backend::{
        cuda::{
            codegen::{BufferDescription, KernelArtifact, KernelDescription, PlanningContext},
            runtime::CudaRuntime,
            tensor_region::CudaTensorSource,
        },
        kernel_lowering::{self, ArgumentLabels, ElementGroup, Epilogue, Lhs, RowEpilogue, Weight},
        registry::{Operation, OperationAttributes, operation_schema_for_attributes},
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{ComputePrecision, NumericalPolicy},
    layout::Layout,
    shape::Shape,
    utility::{Locked, to_i64},
    value::ValueSpec,
};

pub use self::{
    paged_attention::{
        QueryRotation, cudnn_paging, plan_cudnn_paging_indices, plan_cudnn_paging_results,
        plan_cudnn_query_lengths,
    },
    parameters::{KernelParameters, SplitParameters, has_tcgen05, parameter_key},
    rotary_cache::{RotaryCacheInputs, RotationNorm},
    writer::value_type,
};

#[derive(Debug, Clone)]
pub struct KernelInput {
    data_type: DataType,
    layout: Layout,
    // The value's name in the graph it comes from, which only documentation reads.
    name: Option<String>,
}

impl KernelInput {
    pub(crate) fn new(data_type: DataType, layout: Layout) -> Self {
        Self {
            data_type,
            layout,
            name: None,
        }
    }

    pub(crate) fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub(crate) fn from_source(input: &(impl CudaTensorSource + ?Sized)) -> Self {
        Self::new(input.data_type(), input.layout().clone())
    }

    pub(crate) fn data_type(&self) -> DataType {
        self.data_type
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    fn value_spec(&self) -> ValueSpec {
        ValueSpec::new(self.data_type, self.layout.shape().clone())
    }
}

/// A kernel a generator wrote, and the values its output buffers hold.
///
/// Lowered kernels come from the kernel IR; hand-written ones declare their parameters through `KernelWriter`, which adds the launch argument each one takes.
#[derive(Debug)]
pub struct KernelPlan {
    /// The values the kernel's output buffers hold, in the order its ABI lists them.
    pub(crate) outputs: Vec<ValueSpec>,
    // Kernels that run before `kernel`, in order, leaving what it reads in the plan's scratch buffers.
    pub(crate) before: Vec<KernelArtifact>,
    pub(crate) kernel: KernelArtifact,
    // The plan's kernels as each device loaded them, under each precision, filled at their first launch.
    pub(crate) loaded: LoadedKernels,
}

// A plan's loaded functions by the device and precision they were loaded under, which a launch finds without rendering or hashing the code again.
#[derive(Debug, Default)]
pub struct LoadedKernels(Mutex<Vec<(LoadKey, Arc<[KernelFunction]>)>>);

// The device a plan's kernels were loaded on, and the precision they compiled under.
type LoadKey = (Device, ComputePrecision);

impl KernelPlan {
    // The plan's kernels in launch order.
    pub(crate) fn kernels(&self) -> impl Iterator<Item = &KernelArtifact> {
        self.before.iter().chain(iter::once(&self.kernel))
    }

    // The bytes of each scratch buffer the plan's kernels share, `KernelArgument::Scratch(index)` naming the same buffer in each: the most any of them declares.
    pub(crate) fn scratch(&self) -> Vec<u64> {
        let mut sizes = Vec::new();
        for kernel in self.kernels() {
            for (index, &bytes) in kernel.abi.scratch.iter().enumerate() {
                if index == sizes.len() {
                    sizes.push(bytes);
                } else {
                    sizes[index] = sizes[index].max(bytes);
                }
            }
        }
        sizes
    }
}

// Plans generated kernels under one planning context; eager calls plan under the default one.
#[derive(Debug, Clone, Default)]
pub struct KernelPlanner {
    context: PlanningContext,
}

impl KernelPlanner {
    pub(crate) const fn new(context: PlanningContext) -> Self {
        Self { context }
    }

    pub(crate) const fn context(&self) -> &PlanningContext {
        &self.context
    }
}

impl KernelPlan {
    /// Returns the functions of the plan's kernels that are not empty, in launch order, as `runtime` loaded them for `device` under the current precision.
    pub(crate) fn functions(
        &self,
        runtime: &CudaRuntime,
        device: Device,
    ) -> Result<Arc<[KernelFunction]>> {
        let key = (device, NumericalPolicy::current().precision);
        if let Some((_, functions)) = self
            .loaded
            .0
            .locked("kernel plan functions")?
            .iter()
            .find(|(loaded, _)| *loaded == key)
        {
            return Ok(Arc::clone(functions));
        }

        let functions = self
            .kernels()
            .filter(|kernel| !kernel.is_empty())
            .map(|kernel| runtime.kernel(kernel))
            .collect::<Result<Arc<[_]>>>()?;
        self.loaded
            .0
            .locked("kernel plan functions")?
            .push((key, Arc::clone(&functions)));
        Ok(functions)
    }

    /// Returns the value the kernel's first output buffer holds.
    pub(crate) fn output(&self) -> &ValueSpec {
        &self.outputs[0]
    }
}

impl KernelPlanner {
    // The parameters a record gives a kernel that splits long rows, or the default.
    fn split_parameters(
        &self,
        label: &str,
        inputs: &[KernelInput],
        threads: u32,
    ) -> Result<SplitParameters> {
        let key = parameter_key(label, inputs, "");
        match self.tuned_parameters(&key) {
            Some(KernelParameters::Split(parameters)) => Ok(parameters),
            Some(_) => Err(Self::stale_parameters(&key)),
            None => Ok(SplitParameters::with_threads(threads)),
        }
    }

    // What tuning may try for one node's kernel: the key its record goes under, and its candidates, the default first.
    // A node with no such kernel, or whose kernel cannot run these operands, has none.
    pub(crate) fn parameter_candidates(
        &self,
        operation: Operation,
        attributes: &OperationAttributes,
        inputs: &[KernelInput],
    ) -> Option<(String, Vec<KernelParameters>)> {
        let attention_key = |label: &str| {
            attributes
                .scaled_dot_product_attention()
                .map(|attention| parameter_key(label, inputs, &format!("{attention:?}")))
        };
        match (operation, inputs) {
            (Operation::MatMul, [lhs, rhs]) => {
                let mut candidates = self
                    .dot_product_candidates(lhs, rhs)
                    .ok()?
                    .into_iter()
                    .map(KernelParameters::DotProduct)
                    .collect::<Vec<_>>();
                candidates.extend(
                    self.mma_candidates(lhs, rhs)
                        .ok()?
                        .into_iter()
                        .map(KernelParameters::Mma),
                );
                candidates.push(KernelParameters::General);
                Some((parameter_key("matmul", inputs, ""), candidates))
            }
            (Operation::ScaledDotProductAttention, [query, key, value, ..]) => {
                if !self.takes_tensor_cores(&[query, key, value]) {
                    return None;
                }

                let dim = usize::try_from(*query.layout.dimensions().last()?).ok()?;
                Some((
                    attention_key("attention")?,
                    self.attention_candidates(dim)
                        .into_iter()
                        .map(KernelParameters::Attention)
                        .collect(),
                ))
            }
            (
                Operation::PagedAttention
                | Operation::PagedAttentionOutput
                | Operation::PagedAttentionLse,
                [query, key, value, ..],
            ) => {
                if !self.takes_tensor_cores(&[query, key, value]) {
                    return None;
                }

                let dim = usize::try_from(*query.layout.dimensions().last()?).ok()?;
                Some((
                    attention_key("paged_attention")?,
                    self.attention_candidates(dim)
                        .into_iter()
                        .map(KernelParameters::Attention)
                        .collect(),
                ))
            }
            (
                Operation::PagedMlaAttention
                | Operation::PagedMlaAttentionOutput
                | Operation::PagedMlaAttentionLse,
                [latent, positional, ..],
            ) => {
                let dimension =
                    |input: &KernelInput| usize::try_from(*input.layout.dimensions().last()?).ok();
                let (latent, positional) = (dimension(latent)?, dimension(positional)?);
                Some((
                    attention_key("paged_mla_attention")?,
                    self.mla_candidates(latent, latent + positional)
                        .into_iter()
                        .map(KernelParameters::Mla)
                        .collect(),
                ))
            }
            (Operation::GroupedMatMul, [input, ..])
                if parameters::half_precision(input.data_type) =>
            {
                Some((
                    parameter_key("grouped_matmul", inputs, ""),
                    self.grouped_candidates()
                        .into_iter()
                        .map(KernelParameters::Grouped)
                        .collect(),
                ))
            }
            _ => None,
        }
    }

    pub(crate) fn create(
        &self,
        operation: Operation,
        attributes: &OperationAttributes,
        inputs: &[KernelInput],
    ) -> Result<KernelPlan> {
        let schema = operation_schema_for_attributes(operation, attributes);
        if !schema.operands().accepts_count(inputs.len()) {
            return Err(Error::invalid_argument("cuda general kernel operand count"));
        }

        let first = inputs
            .first()
            .ok_or_else(|| Error::missing_input("source create"))?;
        if matches!(
            operation,
            Operation::Embedding | Operation::Take | Operation::IndexSelect
        ) {
            return indexing::plan(self, operation, attributes, inputs);
        }

        if let (Operation::GroupedMatMul, [input, weights, offsets]) = (operation, inputs) {
            return self.grouped_matmul(input, weights, offsets);
        }

        if let (Operation::MatMul, [lhs, rhs]) = (operation, inputs) {
            if lhs.data_type != rhs.data_type {
                return Err(Error::data_type_mismatch(
                    "cuda matmul rhs",
                    lhs.data_type,
                    rhs.data_type,
                ));
            }

            // A record names the kernel and its parameters, and one that no longer fits is an error.
            let key = parameter_key("matmul", inputs, "");
            match self.tuned_parameters(&key) {
                Some(KernelParameters::DotProduct(parameters)) => {
                    return self
                        .dot_product(lhs, rhs, parameters)?
                        .ok_or_else(|| Self::stale_parameters(&key));
                }
                Some(KernelParameters::Mma(parameters)) => {
                    if self.mma_candidates(lhs, rhs)?.contains(&parameters)
                        && let Some(plan) = self.mma_matmul_tuned(lhs, rhs, Some(parameters))?
                    {
                        return Ok(plan);
                    }

                    return Err(Self::stale_parameters(&key));
                }
                Some(KernelParameters::General) => {
                    return self.lowered(
                        kernel_lowering::matmul(lhs.data_type, &lhs.layout, &rhs.layout)?,
                        inputs,
                        &[operation],
                    );
                }
                Some(_) => return Err(Self::stale_parameters(&key)),
                None => {}
            }

            // Up to four rows against a weight read it once as dot products; more rows run on tensor cores, which reuse each load across more of them.
            if let Some(plan) = self.default_dot_product(lhs, rhs)? {
                return Ok(plan);
            }

            // Nothing can measure the tensor engine here, so it is the default for targets that have it.
            if has_tcgen05(self.context.target)
                && let Some(plan) = KernelPlan::tcgen05_matmul(lhs, rhs)?
            {
                return Ok(plan);
            }

            if let Some(plan) = self.mma_matmul(lhs, rhs)? {
                return Ok(plan);
            }
            return self.lowered(
                kernel_lowering::matmul(lhs.data_type, &lhs.layout, &rhs.layout)?,
                inputs,
                &[operation],
            );
        }

        match (operation, inputs) {
            (Operation::Index, [input, indices @ ..]) => {
                let dim = attributes
                    .dimension()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return self.index(input, indices, dim);
            }
            // The scatter writes into a copy of the input, which the caller makes first.
            (Operation::IndexPut, [input, values, indices @ ..]) => {
                let put = attributes
                    .index_put()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return self.index_put(input, values, indices, put.dim, put.accumulate);
            }
            // The sort reads a scratch buffer as its second operand, which the caller provides.
            (Operation::Argsort, [input]) => {
                let sort = attributes
                    .sort()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return KernelPlan::argsort(input, sort.dim, sort.descending);
            }
            // The scatter writes into a copy of the base, which the caller makes first.
            (Operation::StridedScatter, [base, source]) => {
                let strided = attributes
                    .strided()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return self.strided_scatter(base, source, &strided.layout(0)?);
            }
            (Operation::RotaryEmbedding, [input, cos, sin]) => {
                return self.lowered(
                    kernel_lowering::rotary_embedding(
                        input.data_type,
                        &input.layout,
                        (cos.data_type, &cos.layout),
                        (sin.data_type, &sin.layout),
                    )?,
                    inputs,
                    &[operation],
                );
            }
            (Operation::ScaledDotProductAttention, [query, key, value, rest @ ..]) => {
                let attention = attributes
                    .scaled_dot_product_attention()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                let mask = rest.first().filter(|_| attention.masked);
                let sinks = rest
                    .get(usize::from(attention.masked))
                    .filter(|_| attention.sinks);
                let lengths = rest
                    .get(usize::from(attention.masked) + usize::from(attention.sinks))
                    .filter(|_| attention.key_lengths);
                return self
                    .fused_attention(query, key, value, mask, sinks, lengths, *attention)?
                    .ok_or_else(|| {
                        Error::unsupported_operation("cuda fused attention of these operands")
                    });
            }
            (Operation::MaxPool2d | Operation::AvgPool2d, [input]) => {
                let pool = attributes
                    .pool2d()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return self.lowered(
                    kernel_lowering::pool2d(operation, pool, input.data_type, &input.layout)?,
                    inputs,
                    &[operation],
                );
            }
            // Depthwise convolutions are window reductions, and cuDNN runs the others, which are window reductions too when no library may.
            (Operation::Conv2d, [input, weight, bias @ ..]) => {
                let convolution = attributes
                    .conv2d()
                    .ok_or_else(|| Error::missing_node("source create"))?;
                return self.lowered(
                    kernel_lowering::conv2d(
                        self.context.generated_convolutions,
                        convolution,
                        input.data_type,
                        &input.layout,
                        (weight.data_type, &weight.layout),
                        bias.first().map(|bias| (bias.data_type, &bias.layout)),
                    )?,
                    inputs,
                    &[operation],
                );
            }
            _ => {}
        }

        if let (Some(quantization), [input, scale, zero_point]) =
            (attributes.quantization(), inputs)
        {
            return self.lowered(
                kernel_lowering::quantization(
                    operation,
                    *quantization,
                    (input.data_type, &input.layout),
                    (scale.data_type, &scale.layout),
                    (zero_point.data_type, &zero_point.layout),
                )?,
                inputs,
                &[operation],
            );
        }

        if let Some(random) = attributes.random() {
            return random::plan(operation, random, inputs);
        }

        let layouts = inputs
            .iter()
            .map(|input| (input.data_type, &input.layout))
            .collect::<Vec<_>>();
        if let Some(batch) = attributes.batch_normalization() {
            return self.lowered(
                kernel_lowering::batch_norm(batch, &layouts)?,
                inputs,
                &[operation],
            );
        }

        let group = ElementGroup::input(first.data_type, &first.layout);
        if let Some(plan) = self.row(operation, attributes, &group, &layouts[1..], None, inputs)? {
            return Ok(plan);
        }

        if matches!(operation, Operation::Contiguous | Operation::Copy) {
            let view = Layout::from_shape(first.layout.shape());
            return self.assign_into(first.value_spec(), &view, 0, first);
        }
        self.element_wise(operation, attributes, inputs)
    }

    /// Plans row operation `operation`, a normalization, softmax, or reduction, of `group`'s value, with the normalization's `parameters` after it, or returns `None` for other operations.
    pub(crate) fn row(
        &self,
        operation: Operation,
        attributes: &OperationAttributes,
        group: &ElementGroup<'_>,
        parameters: &[(DataType, &Layout)],
        epilogue: Option<&RowEpilogue<'_>>,
        inputs: &[KernelInput],
    ) -> Result<Option<KernelPlan>> {
        let lowered = if let Some(normalization) = attributes.layer_normalization() {
            // Too few long rows for one block per row split each row's reduction across blocks.
            if epilogue.is_none()
                && group.steps.is_empty()
                && !group.stored
                && let [input, parameters @ ..] = inputs
                && let Some(plan) = KernelPlan::split_normalization(
                    operation,
                    normalization,
                    input,
                    parameters,
                    self.split_parameters("split_normalization", inputs, norm::DEFAULT_THREADS)?,
                )?
            {
                return Ok(Some(plan));
            }
            match epilogue {
                Some(epilogue) => kernel_lowering::normalization_with_epilogue(
                    operation,
                    normalization,
                    group,
                    parameters,
                    epilogue,
                )?,
                None => {
                    kernel_lowering::normalization(operation, normalization, group, parameters)?
                }
            }
        } else if matches!(operation, Operation::Softmax | Operation::LogSoftmax) {
            let dim = attributes
                .dimension()
                .ok_or_else(|| Error::missing_node("row"))?;
            kernel_lowering::softmax(operation, group, dim, epilogue)?
        } else if matches!(operation, Operation::Cumsum | Operation::Cumprod) {
            if epilogue.is_some() {
                return Ok(None);
            }

            let dim = attributes
                .dimension()
                .ok_or_else(|| Error::missing_node("row"))?;
            // Few long rows of a plain input split each row across blocks.
            if group.steps.is_empty()
                && !group.stored
                && let [input] = inputs
                && let Some(plan) = KernelPlan::split_scan(
                    input,
                    dim,
                    operation,
                    self.split_parameters("split_scan", inputs, scan::DEFAULT_SPLIT_THREADS)?,
                )?
            {
                return Ok(Some(plan));
            }
            kernel_lowering::scan(operation, group, dim)?
        } else if matches!(operation, Operation::ArgMax | Operation::ArgMin) {
            let dim = attributes
                .dimension()
                .ok_or_else(|| Error::missing_node("row"))?;
            kernel_lowering::reduction(operation, group, &[dim], false, epilogue)?
        } else if let Some(reduction) = attributes.reduction() {
            let lowered = kernel_lowering::reduction(
                operation,
                group,
                &reduction.dimensions,
                reduction.keepdim,
                epilogue,
            )?;
            // A whole plain input reduced to one value splits across blocks, which one row kernel's block cannot.
            if group.steps.is_empty()
                && !group.stored
                && epilogue.is_none()
                && let [input] = inputs
                && let Some(plan) = KernelPlan::split_reduction(
                    operation,
                    input,
                    &lowered.outputs[0],
                    self.split_parameters(
                        "split_reduction",
                        inputs,
                        reduce::DEFAULT_SPLIT_THREADS,
                    )?,
                )?
            {
                return Ok(Some(plan));
            }
            lowered
        } else {
            return Ok(None);
        };
        let operations = group
            .steps
            .iter()
            .map(|step| step.operation)
            .chain([operation])
            .collect::<Vec<_>>();
        // A row kernel that stores its group's value writes it after its own output.
        let producers = iter::once(operation)
            .chain(
                group
                    .stored
                    .then(|| group.steps.last().map(|step| step.operation))
                    .flatten(),
            )
            .collect::<Vec<_>>();
        self.lowered_producing(lowered, inputs, &operations, &producers)
            .map(Some)
    }

    /// Plans `lhs × weight` of `data_type` with the left operand's steps and the weight's dequantization computed as the kernel reads them and `epilogue` applied to the product, as one generated kernel whose inputs are the left operand's, the weight's, and the epilogue's, in that order.
    pub(crate) fn contraction(
        &self,
        data_type: DataType,
        lhs: &Lhs<'_>,
        weight: &Weight<'_>,
        epilogue: Option<&Epilogue<'_>>,
        inputs: &[KernelInput],
    ) -> Result<KernelPlan> {
        let operations = lhs
            .steps
            .iter()
            .map(|step| step.operation)
            .chain([Operation::MatMul])
            .chain(
                epilogue
                    .iter()
                    .flat_map(|epilogue| epilogue.steps.iter().map(|step| step.operation)),
            )
            .collect::<Vec<_>>();
        self.lowered(
            kernel_lowering::contraction(data_type, lhs, weight, epilogue)?,
            inputs,
            &operations,
        )
    }

    /// Plans a strided copy of `source` into the elements `destination` addresses.
    ///
    /// The kernel's output pointer is the destination region, written at the destination's offsets.
    pub(crate) fn assign(
        &self,
        destination: &KernelInput,
        source: &KernelInput,
    ) -> Result<KernelPlan> {
        self.assign_into(destination.value_spec(), &destination.layout, 0, source)
    }

    /// Plans copying `source` into the elements `view` addresses, `offset` elements into the buffer `buffer` describes.
    ///
    /// Generated kernels address every operand from its first element, so `view`'s strides apply and its storage offset does not; `offset` places the view inside the buffer.
    fn assign_into(
        &self,
        buffer: ValueSpec,
        view: &Layout,
        offset: usize,
        source: &KernelInput,
    ) -> Result<KernelPlan> {
        self.lowered(
            kernel_lowering::copy(buffer, view, offset, source.data_type, &source.layout)?,
            slice::from_ref(source),
            &[Operation::Copy],
        )
    }

    /// Plans copying each nonempty input into its slice of `output` along `axis`, one plan per input.
    ///
    /// Every plan writes into the whole concatenated buffer at its slice's offset.
    pub(crate) fn concat(
        &self,
        output: &KernelInput,
        inputs: &[KernelInput],
        axis: usize,
    ) -> Result<Vec<KernelPlan>> {
        let axis_index = to_i64(axis, "cuda concat axis")?;
        let mut start = 0;
        let mut plans = Vec::with_capacity(inputs.len());
        for input in inputs {
            let extent = *input
                .layout
                .dimensions()
                .get(axis)
                .ok_or(Error::DimensionOutOfBounds)?;
            if input.layout.element_count() != 0 {
                let slice = output.layout.slice(axis_index, start, extent, 1)?;
                let offset = slice.storage_offset() - output.layout.storage_offset();
                plans.push(self.assign_into(output.value_spec(), &slice, offset, input)?);
            }

            start += extent;
        }
        Ok(plans)
    }

    /// Plans repeating each head of a `[batch, heads, sequence, head_dim]` input `repeats` times into a contiguous `output`.
    ///
    /// The source is the input with a unit axis after the heads, expanded to `repeats`, and the destination is the output viewed with the same five axes.
    pub(crate) fn repeat_kv(
        &self,
        output: &KernelInput,
        input: &KernelInput,
        repeats: i64,
    ) -> Result<KernelPlan> {
        let [batch, heads, sequence, head_dim] = input.layout.dimensions() else {
            return Err(Error::invalid_argument("cuda repeat_kv input rank"));
        };
        if !output.layout.is_contiguous() {
            return Err(Error::ContiguousRequired);
        }

        let shape = Shape::create([*batch, *heads, repeats, *sequence, *head_dim])?;
        let source = input.layout.unsqueeze(2)?.expand(&shape)?;
        self.assign_into(
            output.value_spec(),
            &Layout::contiguous(shape.dimensions())?,
            0,
            &KernelInput::new(input.data_type, source),
        )
    }

    /// Plans writing `source` at the positions `target` addresses in a buffer laid out like `base`.
    ///
    /// The plan describes the whole base buffer as its output; callers copy the base into it first.
    pub(crate) fn strided_scatter(
        &self,
        base: &KernelInput,
        source: &KernelInput,
        target: &Layout,
    ) -> Result<KernelPlan> {
        if base.data_type != source.data_type {
            return Err(Error::data_type_mismatch(
                "cuda strided scatter source",
                base.data_type,
                source.data_type,
            ));
        }

        if target.shape() != source.layout.shape() {
            return Err(Error::shape_mismatch(
                "cuda strided scatter source",
                target.dimensions().to_vec(),
                source.layout.dimensions().to_vec(),
            ));
        }

        let span = target.storage_span_byte_size(1)?;
        if span > base.layout.element_count() {
            return Err(Error::length_mismatch(
                "cuda strided scatter target span",
                base.layout.element_count(),
                span,
            ));
        }
        // Regions start at their tensor's origin, but the target addresses the base buffer, so its offset places the copy.
        self.assign_into(base.value_spec(), target, target.storage_offset(), source)
    }
}

// The description of a kernel whose input operands read `inputs` in order, and that writes `outputs` and computes `operations`.
// Output `i` holds the result of `producers[i]`; a lone output past the end of `producers` holds the last operation's.
fn describe<'a>(
    inputs: impl IntoIterator<Item = &'a KernelInput>,
    outputs: &[ValueSpec],
    operations: &[Operation],
    producers: &[Operation],
    labels: ArgumentLabels,
) -> KernelDescription {
    let producer = |index: usize| {
        producers
            .get(index)
            .or_else(|| (outputs.len() == 1).then(|| operations.last()).flatten())
            .map(|operation| operation.name().to_owned())
    };
    KernelDescription {
        operations: operations
            .iter()
            .map(|operation| operation.name().to_owned())
            .collect(),
        inputs: inputs
            .into_iter()
            .map(|input| BufferDescription {
                name: input.name.clone(),
                producer: None,
                data_type: input.data_type,
                layout: input.layout.clone(),
            })
            .collect(),
        outputs: outputs
            .iter()
            .enumerate()
            .map(|(index, output)| BufferDescription {
                name: None,
                producer: producer(index),
                data_type: output.data_type(),
                layout: Layout::from_shape(output.shape()),
            })
            .collect(),
        scalars: labels.scalars,
        sizes: labels.sizes,
    }
}

/// Returns `singe_` followed by `label` with every character that cannot appear in a C identifier replaced by `_`.
fn kernel_name(label: &str) -> String {
    let label = label
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() {
                character.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    format!("singe_{label}")
}

#[cfg(test)]
mod nvidia_reference_tests;
#[cfg(test)]
mod parameter_tests;
#[cfg(test)]
mod tests;
