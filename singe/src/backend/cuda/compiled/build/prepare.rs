use std::{collections::HashMap, sync::Arc};

use crate::{
    backend::{
        cuda::{
            codegen::PlanningContext,
            compiled::{
                executable::CudaCompiledGraph,
                plan::{
                    cache_key::{DynamicShapeMode, DynamicShapePolicy, GraphCacheKey},
                    kernel::KernelFeasibility,
                    lowering::{
                        LoweringCandidate, LoweringDecision, LoweringDecisionRecord, LoweringPlan,
                        LoweringPolicy,
                    },
                    pipeline::CompilePipeline,
                },
            },
            diagnostics::describe_cuda_graph_lowering_failure,
            graph_binding::{infer_cuda_graph_device, validate_graph_node_value_tensor},
            graph_value::CudaGraphValue,
        },
        registry::{
            BackendExecution, Operation, OperationAttributes, OperationSchema, OperationSupport,
        },
    },
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{
        CompiledGraph, GraphFingerprintOptions,
        guards::GraphGuards,
        ir::{Graph, OperationNode, ValueId},
    },
    layout::Layout,
    shape::Shape,
};

pub struct PreparedCudaGraphCompile {
    pub(crate) execution_order: Vec<ValueId>,
    pub(crate) lowering_plan: LoweringPlan,
    pub(crate) ir: Arc<Graph>,
    pub(crate) device: Device,
    pub(crate) guards: GraphGuards,
    pub(crate) cache_key: GraphCacheKey,
    pub(crate) planning: PlanningContext,
    pub(crate) kernels: KernelFeasibility,
}

pub fn prepare_cuda_graph_compile(
    compiled: &CompiledGraph,
    dynamic_shape_policy: &DynamicShapePolicy,
    lowering_policy: LoweringPolicy,
    planning: impl FnOnce(Device) -> Result<PlanningContext>,
) -> Result<PreparedCudaGraphCompile> {
    let pipeline = CompilePipeline::default();
    let ir = channels_last_convolution_chains(compiled.shared_ir())?;
    // Lowering visits each node's primary result in the compiled execution order.
    let execution_order = compiled.value_order().to_vec();
    let device = infer_cuda_graph_device(&ir, &execution_order)?;
    let planning = planning(device)?;
    let kernels = KernelFeasibility::new(&planning);
    let lowering_plan = LoweringPlan::create(&ir, &execution_order, &lowering_policy, &kernels)?;
    if !ir.has_outputs() {
        return Err(Error::missing_output("prepare cuda graph compile"));
    }

    if ir.has_input_dimension_guards() && dynamic_shape_policy.mode() == DynamicShapeMode::Reject {
        return Err(Error::unsupported_operation_with_details(
            "cuda graph dynamic input dimensions",
            "cuda graph dynamic-shape policy reject",
        ));
    }

    let guards = GraphGuards::create_from_ir(&ir)?;
    let cache_key = GraphCacheKey::create_from_compiled_graph(
        compiled,
        &ir,
        &execution_order,
        &lowering_plan,
        &GraphFingerprintOptions::with_compile_options(compiled.compile_options()),
        pipeline,
        lowering_policy,
        dynamic_shape_policy.clone(),
    )?;
    guards.validate_initializers_with(|name, tensor| {
        let value = ir
            .input_values()
            .find_map(|(value, input)| (input == name).then_some(value))
            .ok_or_else(|| Error::missing_input("prepare cuda graph compile"))?;
        if ir.required_value_binding_details(value)?.is_static_input() {
            validate_graph_node_value_tensor(&ir, value, tensor)?;
        }
        // Overridable defaults are copied only when selected at replay.
        // Their logical guards permit other concrete shapes; runtime binding checks validate the selected tensor against this plan's physical placeholder.
        Ok(())
    })?;

    Ok(PreparedCudaGraphCompile {
        execution_order,
        lowering_plan,
        ir,
        device,
        guards,
        cache_key,
        planning,
        kernels,
    })
}

pub struct GraphNodeSupport<'a> {
    pub(crate) operation_support: Option<OperationSupport>,
    pub(crate) operation_name: &'static str,
    pub(crate) lowering_decision: Option<&'a LoweringDecisionRecord>,
    pub(crate) lowering_candidates: &'a [LoweringCandidate],
    pub(crate) input_data_types: Vec<DataType>,
}

pub fn validate_graph_node_support<'a>(
    ir: &Graph,
    id: ValueId,
    lowering_plan: &'a LoweringPlan,
    cache: &HashMap<ValueId, CudaGraphValue>,
) -> Result<GraphNodeSupport<'a>> {
    let ir_operation = ir.required_operation_for_value(id)?;
    let output_value = id;
    let output_data_type = ir.required_value_data_type(output_value)?;
    let output_shape = ir.required_value_shape(output_value)?;
    let operation_schema = ir_operation.schema();
    let operation_support = operation_schema.map(OperationSchema::support);
    let operation_name =
        operation_schema.map_or("unsupported graph operation", |schema| schema.name());
    let lowering_decision = lowering_plan.decision(id);
    let lowering_candidates = lowering_plan.candidates(id);
    let input_data_types = graph_node_ir_input_data_types(ir, Some(ir_operation))?;
    let support_check = GraphSupportCheck {
        operation: Some(ir_operation),
        operation_name,
        output_data_type,
        output_shape,
        lowering_decision,
        lowering_candidates,
        input_data_types: &input_data_types,
    };

    support_check.validate_lowering_decision()?;

    if let Some(support) = operation_support {
        support_check.validate_operation_support(ir, id, &support, cache)?;
    }

    Ok(GraphNodeSupport {
        operation_support,
        operation_name,
        lowering_decision,
        lowering_candidates,
        input_data_types,
    })
}

pub struct GraphSupportCheck<'a> {
    pub(crate) operation: Option<&'a OperationNode>,
    pub(crate) operation_name: &'static str,
    pub(crate) output_data_type: DataType,
    pub(crate) output_shape: &'a Shape,
    pub(crate) lowering_decision: Option<&'a LoweringDecisionRecord>,
    pub(crate) lowering_candidates: &'a [LoweringCandidate],
    pub(crate) input_data_types: &'a [DataType],
}

impl GraphSupportCheck<'_> {
    pub(crate) fn validate_lowering_decision(&self) -> Result<()> {
        if let Some(LoweringDecision::Unsupported { .. }) =
            self.lowering_decision.map(LoweringDecisionRecord::decision)
        {
            return Err(self.failure(self.lowering_candidates, None));
        }
        Ok(())
    }

    pub(crate) fn validate_operation_support(
        &self,
        ir: &Graph,
        id: ValueId,
        support: &OperationSupport,
        cache: &HashMap<ValueId, CudaGraphValue>,
    ) -> Result<()> {
        if !support.supports(BackendExecution::CudaGraph) {
            return Err(self.failure(&[], Some(support)));
        }

        if !support.supports_data_type(BackendExecution::CudaGraph, self.output_data_type) {
            return Err(self.failure(self.lowering_candidates, Some(support)));
        }

        if !graph_node_input_data_types_supported(self.operation, self.input_data_types, support) {
            return Err(self.failure(self.lowering_candidates, Some(support)));
        }
        CudaCompiledGraph::validate_graph_layout_requirements(
            ir,
            id,
            support,
            self.lowering_decision
                .and_then(|record| record.decision().provider()),
            cache,
        )
    }

    fn failure(
        &self,
        candidates: &[LoweringCandidate],
        operation_support: Option<&OperationSupport>,
    ) -> Error {
        describe_cuda_graph_lowering_failure(
            self.operation_name,
            self.output_data_type,
            self.output_shape,
            candidates,
            self.lowering_decision,
            operation_support,
            self.input_data_types,
            &[],
        )
    }
}

pub fn graph_node_ir_input_data_types(
    ir: &Graph,
    operation: Option<&OperationNode>,
) -> Result<Vec<DataType>> {
    operation.map_or_else(
        || Ok(Vec::new()),
        |operation| {
            operation
                .inputs()
                .iter()
                .map(|value| ir.required_value_data_type(*value))
                .collect()
        },
    )
}

pub fn graph_node_input_data_types_supported(
    operation: Option<&OperationNode>,
    input_data_types: &[DataType],
    support: &OperationSupport,
) -> bool {
    let Some(operation) = operation else {
        return input_data_types.iter().all(|data_type| {
            support.supports_input_data_type(BackendExecution::CudaGraph, *data_type)
        });
    };
    support.supports_operand_data_types(
        BackendExecution::CudaGraph,
        operation.input_roles(),
        input_data_types,
    )
}

// Lays out each cuDNN convolution's output channels-last when only other convolutions read it, as their input, since cuDNN's convolutions read and write those strides directly.
// Nothing else sees these values, so no copy is needed anywhere in a chain.
pub fn channels_last_convolution_chains(ir: Arc<Graph>) -> Result<Arc<Graph>> {
    let outputs = ir.output_value_set();
    let mut readers = HashMap::<ValueId, Vec<&OperationNode>>::new();
    for operation in ir.operations() {
        for &input in operation.inputs() {
            readers.entry(input).or_default().push(operation);
        }
    }

    let mut chained = Vec::new();
    for operation in ir.operations() {
        if operation.opcode() != Some(Operation::Conv2d) {
            continue;
        }

        let OperationAttributes::Conv2d(conv) = operation.attributes() else {
            continue;
        };
        let (Some(value), Some(&input)) =
            (operation.result_values().next(), operation.inputs().first())
        else {
            continue;
        };
        let layout = ir.required_value_layout(value)?;
        let &[batch, channels, height, width] = layout.shape().dimensions() else {
            continue;
        };
        // A depthwise convolution is a generated kernel, which writes contiguous results.
        let source_channels = ir.required_value_shape(input)?.dimensions().get(1).copied();
        let read_as_input = readers.get(&value).is_some_and(|readers| {
            readers.iter().all(|reader| {
                reader.opcode() == Some(Operation::Conv2d)
                    && reader.inputs().first() == Some(&value)
                    && !reader.inputs()[1..].contains(&value)
            })
        });
        if conv.groups == 1
            && channels > 1
            && source_channels.is_some_and(|channels| channels > 1)
            && layout.is_contiguous()
            && !outputs.contains(&value)
            && read_as_input
        {
            chained.push((
                value,
                Layout::with_strides(
                    [batch, channels, height, width],
                    [channels * height * width, 1, channels * width, channels],
                )?,
            ));
        }
    }

    if chained.is_empty() {
        return Ok(ir);
    }

    let mut changed = Graph::clone(&ir);
    for (value, layout) in chained {
        changed.set_value_layout(value, layout)?;
    }
    Ok(Arc::new(changed))
}
