use std::{collections::HashMap, mem, sync::Mutex};

use singe_cuda::graph::Graph as NativeCudaGraph;

use crate::{
    backend::{
        cuda::{
            codegen::{KernelArtifact, PlanningContext},
            compiled::{
                executable::GraphPrivateBuffers,
                plan::lowering::{LoweringCandidate, LoweringProvider},
            },
            graph_binding::validate_graph_node_value_tensor,
            graph_value::CudaGraphValue,
            kernel_plan::{self, GraphStep, KernelInput, KernelPlan, KernelPlanner},
            provider_selection::is_elementwise,
            tensor_region::CudaTensorSource,
        },
        registry::{
            CudaGraphProvider, LayoutSupport, Operation, OperationAttributes, operation_support,
        },
    },
    error::{Error, Result},
    graph::ir::{Graph, OperationNode, ValueId},
    shape::normalize_dim_usize,
    utility::Locked,
};

// Above every library's cost.
const GENERAL_CONVOLUTION_COST: u32 = 1000;

// Plans each node's generated kernels to learn whether it can have any, and keeps the plans, so that when planning has nothing to repeat a node is planned once.
//
// Tuning and collecting kernels wait for the nodes that keep their kernels, since fusion and libraries take many.
#[derive(Debug)]
pub struct KernelFeasibility {
    planner: KernelPlanner,
    reusable: bool,
    plans: Mutex<HashMap<ValueId, NodeKernelPlans>>,
}

impl KernelFeasibility {
    #[must_use]
    pub fn new(context: &PlanningContext) -> Self {
        Self {
            planner: KernelPlanner::new(PlanningContext {
                recorder: None,
                generated_convolutions: false,
                ..context.clone()
            }),
            reusable: context.recorder.is_none(),
            plans: Mutex::new(HashMap::new()),
        }
    }

    // Plans the node producing `id`, and says whether it can have generated kernels.
    fn plan(&self, ir: &Graph, id: ValueId) -> Result<bool> {
        let Some(plans) = node_kernel_plans(&self.planner, ir, id) else {
            return Ok(false);
        };
        if self.reusable {
            self.plans.locked("kernel feasibility")?.insert(id, plans);
        }
        Ok(true)
    }

    pub(crate) const fn planner(&self) -> &KernelPlanner {
        &self.planner
    }

    // Takes the plans a compile can launch as they are, leaving none behind.
    pub fn take_plans(&self) -> Result<HashMap<ValueId, NodeKernelPlans>> {
        Ok(mem::take(&mut *self.plans.locked("kernel feasibility")?))
    }
}

/// CUDA graph lowering provider for generated kernels.
#[derive(Debug, Clone, Copy)]
pub struct KernelProvider<'a>(pub &'a KernelFeasibility);

impl LoweringProvider for KernelProvider<'_> {
    fn kind(&self) -> CudaGraphProvider {
        CudaGraphProvider::Kernel
    }

    fn cost(&self) -> u32 {
        20
    }

    fn supports_operation(&self, operation: Operation) -> bool {
        operation_support(operation).supports_cuda_graph_provider(CudaGraphProvider::Kernel)
    }

    fn candidate_for_operation(
        &self,
        ir: &Graph,
        id: ValueId,
        operation: &OperationNode,
    ) -> Option<LoweringCandidate> {
        let opcode = operation.opcode()?;
        let schema = operation.schema()?;
        if !schema
            .support()
            .supports_cuda_graph_provider(CudaGraphProvider::Kernel)
        {
            return None;
        }

        let planned = self.0.plan(ir, id).unwrap_or(false);
        // A convolution that is not depthwise runs as a generated kernel only when no library may, so it costs more than any library.
        if !planned && opcode == Operation::Conv2d {
            let general = KernelPlanner::new(PlanningContext {
                generated_convolutions: true,
                ..self.0.planner.context().clone()
            });
            return node_kernel_plan(&general, ir, id, operation)
                .is_ok()
                .then_some(LoweringCandidate::new(
                    self.kind(),
                    LayoutSupport::AnyStrided,
                    GENERAL_CONVOLUTION_COST,
                ));
        }

        let layout = if is_generated(opcode, operation.attributes()) {
            LayoutSupport::AnyStrided
        } else {
            schema.support().layout()
        };
        planned.then_some(LoweringCandidate::new(self.kind(), layout, self.cost()))
    }
}

/// The generated kernels one node runs: copies into slices of its output, each with the input it reads, or one kernel over the node's inputs.
#[derive(Debug)]
pub enum NodeKernelPlans {
    Copies(Vec<(KernelPlan, ValueId)>),
    Kernel(Box<KernelPlan>),
}

impl NodeKernelPlans {
    pub fn artifacts(&self) -> impl Iterator<Item = &KernelArtifact> {
        let (copies, kernel) = match self {
            Self::Copies(copies) => (copies.as_slice(), None),
            Self::Kernel(plan) => (&[][..], Some(plan)),
        };
        copies
            .iter()
            .flat_map(|(plan, _)| plan.kernels())
            .chain(kernel.into_iter().flat_map(|plan| plan.kernels()))
    }
}

/// Returns the generated kernels [`build_node_kernel_graph`] launches for the node producing `id`, planned once so they can compile before lowering reaches it and lowering can launch them.
///
/// Nodes whose plan cannot be made have none.
pub fn node_kernel_plans(
    planner: &KernelPlanner,
    ir: &Graph,
    id: ValueId,
) -> Option<NodeKernelPlans> {
    let operation = ir.required_operation_for_value(id).ok()?;
    let opcode = operation.opcode()?;
    if matches!(opcode, Operation::Concat | Operation::RepeatKv) {
        return copy_plans(planner, ir, id, operation)
            .ok()
            .map(NodeKernelPlans::Copies);
    }
    node_kernel_plan(planner, ir, id, operation)
        .ok()
        .map(|plan| NodeKernelPlans::Kernel(Box::new(plan)))
}

/// Builds the native graph for one node from its `prepared` kernels, or plans them with `planner` when they were not planned ahead, allocating the compiled graph's status word and scratch buffers in `private` as kernels need them.
pub fn build_node_kernel_graph(
    prepared: Option<NodeKernelPlans>,
    planner: &KernelPlanner,
    ir: &Graph,
    id: ValueId,
    output: &CudaGraphValue,
    cache: &HashMap<ValueId, CudaGraphValue>,
    private: &mut GraphPrivateBuffers,
) -> Result<NativeCudaGraph> {
    let operation = ir.required_operation_for_value(id)?;
    let opcode = operation
        .opcode()
        .ok_or_else(|| Error::missing_node("build node kernel graph"))?;
    validate_graph_node_value_tensor(ir, id, output)?;

    // Concatenation and repeated heads are generated copies into slices of the output.
    if matches!(opcode, Operation::Concat | Operation::RepeatKv) {
        let plans = match prepared {
            Some(NodeKernelPlans::Copies(plans)) => plans,
            _ => copy_plans(planner, ir, id, operation)?,
        };
        let steps = plans
            .iter()
            .map(|(plan, input)| {
                let input: &dyn CudaTensorSource = cache
                    .get(input)
                    .ok_or_else(|| Error::missing_input("build node kernel graph"))?;
                Ok(GraphStep {
                    plan,
                    inputs: vec![input],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        return kernel_plan::build_graph_steps(&steps, &[output], private);
    }

    let plan = match prepared {
        Some(NodeKernelPlans::Kernel(plan)) => *plan,
        _ => node_kernel_plan(planner, ir, id, operation)?,
    };
    let values = operation
        .inputs()
        .iter()
        .map(|input| {
            cache
                .get(input)
                .ok_or_else(|| Error::missing_input("build node kernel graph"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut inputs = values
        .iter()
        .map(|value| *value as &dyn CudaTensorSource)
        .collect::<Vec<_>>();
    let mut steps = Vec::new();
    let copy;
    // `index_put` and `strided_scatter` write into a copy of their first input, unless nothing else reads it and they write it in place.
    if matches!(opcode, Operation::IndexPut | Operation::StridedScatter) {
        if !values[0].shares_storage(output) {
            copy = planner.assign(
                &KernelInput::from_source(output),
                &KernelInput::from_source(values[0]),
            )?;
            steps.push(GraphStep {
                plan: &copy,
                inputs: vec![values[0]],
            });
        }

        inputs.remove(0);
    }

    steps.push(GraphStep {
        plan: &plan,
        inputs,
    });
    kernel_plan::build_graph_steps(&steps, &[output], private)
}

/// Returns the copy plans of a concatenation or a head repeat, each with the input it reads.
fn copy_plans(
    planner: &KernelPlanner,
    ir: &Graph,
    id: ValueId,
    operation: &OperationNode,
) -> Result<Vec<(KernelPlan, ValueId)>> {
    let input = |value: ValueId| -> Result<KernelInput> {
        Ok(KernelInput::new(
            ir.required_value_data_type(value)?,
            ir.required_value_layout(value)?.clone(),
        ))
    };
    let output = input(id)?;
    if !output.layout().is_contiguous() {
        return Err(Error::ContiguousRequired);
    }
    match (operation.opcode(), operation.attributes()) {
        (Some(Operation::Concat), &OperationAttributes::Concat(dim)) => {
            let inputs = operation
                .inputs()
                .iter()
                .map(|&value| input(value))
                .collect::<Result<Vec<_>>>()?;
            let axis = normalize_dim_usize(dim, output.layout().shape().rank())?;
            let nonempty = operation
                .inputs()
                .iter()
                .zip(&inputs)
                .filter(|(_, input)| input.layout().element_count() != 0)
                .map(|(&value, _)| value);
            Ok(planner
                .concat(&output, &inputs, axis)?
                .into_iter()
                .zip(nonempty)
                .collect())
        }
        (Some(Operation::RepeatKv), &OperationAttributes::RepeatKv(repeats)) => {
            let [source] = operation.inputs() else {
                return Err(Error::missing_input("copy plans"));
            };
            Ok(vec![(
                planner.repeat_kv(&output, &input(*source)?, repeats)?,
                *source,
            )])
        }
        _ => Err(Error::unsupported_operation("cuda graph copy plan")),
    }
}

fn node_kernel_plan(
    planner: &KernelPlanner,
    ir: &Graph,
    id: ValueId,
    operation: &OperationNode,
) -> Result<KernelPlan> {
    let inputs = operation
        .inputs()
        .iter()
        .map(|value| {
            Ok(KernelInput::new(
                ir.required_value_data_type(*value)?,
                ir.required_value_layout(*value)?.clone(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let plan = planner.create(
        operation
            .opcode()
            .ok_or_else(|| Error::missing_node("node kernel plan"))?,
        operation.attributes(),
        &inputs,
    )?;
    let output = id;
    if ir.required_value_data_type(output)? != plan.output().data_type()
        || ir.required_value_shape(output)? != plan.output().shape()
        || !ir.required_value_layout(output)?.is_contiguous()
    {
        return Err(Error::invalid_argument(
            "cuda general kernel result metadata",
        ));
    }
    Ok(plan)
}

// Whether the operation always runs its generated kernel, which reads operands of any strides: element-wise operations, the one definition of their arithmetic, copies, and the reductions, softmax, and normalizations measured at least as fast as the built-ins they replaced.
fn is_generated(operation: Operation, attributes: &OperationAttributes) -> bool {
    is_elementwise(operation)
        || attributes.reduction().is_some()
        || matches!(
            operation,
            Operation::Concat
                | Operation::RepeatKv
                | Operation::Softmax
                | Operation::LogSoftmax
                | Operation::LayerNorm
                | Operation::RmsNorm
                | Operation::BatchNorm
        )
}
