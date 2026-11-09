//! Element-wise kernels, lowered to the kernel IR and emitted as CUDA C++.

use singe_codegen::schedule::{Schedule, element_schedule};

use crate::{
    backend::{
        cuda::{
            codegen::plan_kernel,
            kernel_plan::source::{KernelInput, KernelPlan, KernelPlanner, describe, kernel_name},
        },
        kernel_lowering::{ElementWiseKernel, FusedOperand, FusedStep, FusedStores, LoweredKernel},
        registry::{Operation, OperationAttributes, OperationRequest},
    },
    data_type::DataType,
    error::{Error, Result},
    layout::Layout,
    shape::Shape,
    utility::{to_i64, to_u64},
    value::ValueSpec,
};

/// The threads per block of element-wise kernels.
const THREADS: u32 = 128;

impl KernelPlanner {
    /// Plans element-wise `operation` on `inputs`, which broadcast to its output.
    pub(crate) fn element_wise(
        &self,
        operation: Operation,
        attributes: &OperationAttributes,
        inputs: &[KernelInput],
    ) -> Result<KernelPlan> {
        let output = OperationRequest::new(
            operation,
            attributes.clone(),
            inputs.iter().map(KernelInput::value_spec).collect(),
        )
        .infer()?
        .outputs()[0]
            .clone();
        let layouts = inputs
            .iter()
            .map(|input| (input.data_type, &input.layout))
            .collect::<Vec<_>>();
        let (mut kernel, values) =
            ElementWiseKernel::create(&kernel_name(operation.name()), output.shape(), &layouts)?;
        let value = kernel.apply(operation, attributes, &values)?;
        kernel.store(value, output.data_type())?;
        self.emitted(kernel, vec![output], inputs, &[operation], &[])
    }

    /// Plans one kernel that computes `steps` per element of `shape`, holding intermediate values in registers and storing what `stores` names.
    ///
    /// Inputs broadcast to `shape`, and each step keeps the data type its operation infers; a concatenation's slices all have `data_type`.
    pub(crate) fn fused(
        &self,
        data_type: DataType,
        shape: &Shape,
        inputs: &[KernelInput],
        steps: &[FusedStep<'_>],
        stores: &FusedStores,
    ) -> Result<KernelPlan> {
        if steps.is_empty() {
            return Err(Error::invalid_argument("cuda fused kernel"));
        }

        let name = steps.iter().fold(String::from("fused"), |name, step| {
            name + "_" + step.operation.name()
        });
        let layouts = inputs
            .iter()
            .map(|input| (input.data_type, &input.layout))
            .collect::<Vec<_>>();
        let (output, layout) = match stores {
            FusedStores::Last | FusedStores::Steps(_) => (shape.clone(), Layout::from_shape(shape)),
            FusedStores::Concat { axis, values } => {
                let mut dimensions = shape.dimensions().to_vec();
                let slice = dimensions
                    .get_mut(*axis)
                    .ok_or_else(|| Error::invalid_argument("cuda fused concat axis"))?;
                *slice *= to_i64(values.len(), "cuda fused concat inputs")?;
                let output = Shape::create(dimensions)?;
                let strides = Layout::from_shape(&output).strides().to_vec();
                (output, Layout::with_strides(shape.dimensions(), strides)?)
            }
        };
        let (mut kernel, loaded) =
            ElementWiseKernel::create_into(&kernel_name(&name), &layout, &layouts)?;
        let computed = kernel.steps(&loaded, steps)?;
        let mut outputs = Vec::new();
        match stores {
            FusedStores::Last => {
                let last = *computed
                    .last()
                    .ok_or_else(|| Error::missing_node("fused"))?;
                kernel.store(last, last.data_type)?;
                outputs.push(ValueSpec::new(last.data_type, output));
            }
            FusedStores::Steps(steps) => {
                for &step in steps {
                    let value = *computed
                        .get(step)
                        .ok_or_else(|| Error::missing_node("fused"))?;
                    kernel.store(value, value.data_type)?;
                    outputs.push(ValueSpec::new(value.data_type, output.clone()));
                }
            }
            FusedStores::Concat { axis, values } => {
                // Slice `k` starts `k` slice extents along the axis into the output.
                let step = to_u64(
                    shape.dimensions()[*axis] * layout.strides()[*axis],
                    "cuda fused concat offset",
                )?;
                let slices = values
                    .iter()
                    .zip(0..)
                    .map(|(operand, index)| {
                        let value = match *operand {
                            FusedOperand::Input(slot) => loaded.get(slot).copied(),
                            FusedOperand::Step(step) => computed.get(step).copied(),
                        };
                        value
                            .map(|value| (value, step * index))
                            .ok_or_else(|| Error::invalid_argument("cuda fused concat operand"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                kernel.store_slices(data_type, &slices)?;
                outputs.push(ValueSpec::new(data_type, output));
            }
        }

        let operations = steps.iter().map(|step| step.operation).collect::<Vec<_>>();
        let producers = match stores {
            FusedStores::Last => Vec::new(),
            FusedStores::Steps(indices) => indices.iter().map(|&step| operations[step]).collect(),
            FusedStores::Concat { .. } => vec![Operation::Concat],
        };
        self.emitted(kernel, outputs, inputs, &operations, &producers)
    }

    /// Plans a kernel the lowering produced, under the schedule it chose, whose input operands read `inputs` in order and which computes `operations`.
    pub(crate) fn lowered<'a>(
        &self,
        lowered: LoweredKernel,
        inputs: impl IntoIterator<Item = &'a KernelInput>,
        operations: &[Operation],
    ) -> Result<KernelPlan> {
        self.lowered_producing(lowered, inputs, operations, &[])
    }

    // Like `Self::lowered`, with output `i` holding the result of `producers[i]`.
    pub(crate) fn lowered_producing<'a>(
        &self,
        lowered: LoweredKernel,
        inputs: impl IntoIterator<Item = &'a KernelInput>,
        operations: &[Operation],
        producers: &[Operation],
    ) -> Result<KernelPlan> {
        let description = describe(
            inputs,
            &lowered.outputs,
            operations,
            producers,
            lowered.labels,
        );
        // `mma.sync` m16n8k16 needs `sm_80`; older targets run contractions on CUDA cores.
        let schedule = if matches!(lowered.schedule, Schedule::Mma { .. })
            && self
                .context
                .target
                .is_some_and(|target| target.compute_capability() < (8, 0))
        {
            Schedule::Warps {
                threads: 128,
                vector: 1,
                lanes: 32,
                tiles: [1, 1],
            }
        } else {
            lowered.schedule
        };
        Ok(KernelPlan {
            before: Vec::new(),
            kernel: plan_kernel(
                &lowered.kernel,
                schedule,
                &lowered.bindings,
                &description,
                &self.context,
            )?,
            outputs: lowered.outputs,
        })
    }

    fn emitted(
        &self,
        kernel: ElementWiseKernel,
        outputs: Vec<ValueSpec>,
        inputs: &[KernelInput],
        operations: &[Operation],
        producers: &[Operation],
    ) -> Result<KernelPlan> {
        let (kernel, bindings, labels) = kernel.finish();
        let description = describe(inputs, &outputs, operations, producers, labels);
        Ok(KernelPlan {
            before: Vec::new(),
            kernel: plan_kernel(
                &kernel,
                element_schedule(&kernel, THREADS),
                &bindings,
                &description,
                &self.context,
            )?,
            outputs,
        })
    }
}
