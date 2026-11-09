use crate::{
    backend::{
        cuda::compiled::{
            executable::report::{
                GraphCompileProviderCandidate, GraphCompileProviderDecision,
                GraphCompileProviderRejection, GraphCompileValueMaterialization,
            },
            plan::lowering::{
                CudaGraphProvider, LoweringCandidate, LoweringDecision, LoweringDecisionRecord,
                LoweringRejection,
            },
        },
        registry::operation_name,
    },
    error::Result,
    graph::ir::{Graph, ValueBindingKind, ValueId},
};

pub fn compile_report_selected_algorithm(
    realized_decision: Option<&LoweringDecision>,
) -> Option<String> {
    match realized_decision? {
        LoweringDecision::Kernel { operation } => {
            Some(format!("kernel {}", operation_name(*operation)))
        }
        LoweringDecision::CudnnBackend { operation } => Some(format!(
            "cudnn backend {} execution plan",
            operation_name(*operation)
        )),
        LoweringDecision::CublasLt { .. } => Some("cublaslt matmul".to_string()),
        LoweringDecision::Cublas { .. } => Some("cublas sgemm".to_string()),
        LoweringDecision::Cutensor { operation } => {
            Some(format!("cutensor {} plan", operation_name(*operation)))
        }
        LoweringDecision::Cufft { operation } => {
            Some(format!("cufft {} plan", operation_name(*operation)))
        }
        LoweringDecision::Fusion { operation } => {
            Some(format!("fused kernel with {}", operation_name(*operation)))
        }
        LoweringDecision::Alias { .. } | LoweringDecision::Unsupported { .. } => None,
    }
}

pub fn compile_report_provider_candidate(
    candidate: LoweringCandidate,
) -> GraphCompileProviderCandidate {
    GraphCompileProviderCandidate::new(
        candidate.provider(),
        candidate.layout_support(),
        candidate.cost(),
    )
}

pub fn compile_report_planned_provider_decision(
    decision: &LoweringDecisionRecord,
) -> GraphCompileProviderDecision {
    compile_report_provider_decision(
        decision.decision(),
        decision.rejected_candidates(),
        decision.rejections(),
    )
}

pub fn compile_report_realized_provider_decision(
    decision: &LoweringDecision,
) -> GraphCompileProviderDecision {
    compile_report_provider_decision(decision, &[], &[])
}

fn compile_report_provider_rejection(
    rejection: &LoweringRejection,
) -> GraphCompileProviderRejection {
    GraphCompileProviderRejection::new(
        compile_report_provider_candidate(rejection.candidate()),
        rejection.reason(),
    )
}

fn compile_report_provider_decision(
    decision: &LoweringDecision,
    rejected_candidates: &[LoweringCandidate],
    rejections: &[LoweringRejection],
) -> GraphCompileProviderDecision {
    let (provider, operation, unsupported_reason) = match decision {
        LoweringDecision::Alias { operation } => (
            Some(CudaGraphProvider::Alias),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Kernel { operation } => (
            Some(CudaGraphProvider::Kernel),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::CudnnBackend { operation } => (
            Some(CudaGraphProvider::CudnnBackend),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Fusion { operation } => (
            Some(CudaGraphProvider::Fusion),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::CublasLt { operation } => (
            Some(CudaGraphProvider::CublasLt),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Cublas { operation } => (
            Some(CudaGraphProvider::Cublas),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Cutensor { operation } => (
            Some(CudaGraphProvider::Cutensor),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Cufft { operation } => (
            Some(CudaGraphProvider::Cufft),
            Some(operation_name(*operation)),
            None,
        ),
        LoweringDecision::Unsupported { reason } => (None, None, Some(reason.clone())),
    };
    GraphCompileProviderDecision::new(
        provider,
        operation,
        unsupported_reason,
        rejected_candidates
            .iter()
            .copied()
            .map(compile_report_provider_candidate)
            .collect(),
        rejections
            .iter()
            .map(compile_report_provider_rejection)
            .collect(),
    )
}

pub fn compile_report_node_materialization(
    ir: &Graph,
    output_values: &[ValueId],
    realized_decision: Option<&LoweringDecision>,
) -> Result<GraphCompileValueMaterialization> {
    let Some(output) = output_values.first() else {
        return Ok(compile_report_provider_materialization(realized_decision));
    };
    match ir.required_value_binding_kind(*output)? {
        ValueBindingKind::RuntimeInput | ValueBindingKind::InitializerInput => {
            Ok(GraphCompileValueMaterialization::RuntimeInput)
        }
        ValueBindingKind::StaticInput => Ok(GraphCompileValueMaterialization::StaticInput),
        ValueBindingKind::Constant => Ok(GraphCompileValueMaterialization::Constant),
        ValueBindingKind::OperationOutput => {
            Ok(compile_report_provider_materialization(realized_decision))
        }
    }
}

fn compile_report_provider_materialization(
    realized_decision: Option<&LoweringDecision>,
) -> GraphCompileValueMaterialization {
    match realized_decision.and_then(LoweringDecision::provider) {
        Some(CudaGraphProvider::Alias) => GraphCompileValueMaterialization::Alias,
        Some(_) => GraphCompileValueMaterialization::NativeGraphOutput,
        None => GraphCompileValueMaterialization::None,
    }
}
