use std::fmt::Write as _;

use crate::{
    backend::{
        cuda::{
            compiled::plan::lowering::{
                CudaGraphProvider, LoweringCandidate, LoweringDecision, LoweringDecisionRecord,
            },
            diagnostics::{graph_layout_name, graph_provider_name},
        },
        registry::{BackendExecution, OperationSupport, operation_name},
    },
    data_type::DataType,
    error::Error,
    shape::Shape,
};

/// Creates an unsupported-operation error with candidate providers, planned decisions, type and layout requirements, shape, and retained provider rejection messages.
pub fn describe_cuda_graph_lowering_failure(
    operation: &'static str,
    output_data_type: DataType,
    output_shape: &Shape,
    candidates: &[LoweringCandidate],
    decision: Option<&LoweringDecisionRecord>,
    operation_support: Option<&OperationSupport>,
    input_data_types: &[DataType],
    provider_failures: &[(CudaGraphProvider, String)],
) -> Error {
    let providers = candidate_provider_list(candidates);
    let planned_provider = planned_provider_name(candidates);
    let rejected_candidate_providers = rejected_candidate_provider_list(candidates);
    let lowering_decision = decision.map_or_else(|| "none".to_string(), lowering_decision_name);

    let layout = operation_support.map_or("unknown", |support| graph_layout_name(support.layout()));
    let execution_family = operation_support.map_or("unknown", |support| {
        support.execution_family_name(BackendExecution::CudaGraph)
    });
    let supported_data_types = operation_support.map_or_else(
        || "unknown".into(),
        |support| support.data_type_list(BackendExecution::CudaGraph),
    );
    let provider_failure_details = provider_failure_details(provider_failures);
    let mut common_details = format!(
        "providers [{providers}], planned provider {planned_provider}, rejected candidate \
         providers [{rejected_candidate_providers}], lowering decision {lowering_decision}, \
         execution family {execution_family}, dtype {:?}, supported dtypes \
         [{supported_data_types}], layout {layout}, shape {:?}",
        output_data_type,
        output_shape.dimensions()
    );
    if let Some(support) = operation_support
        && support.has_distinct_input_data_types(BackendExecution::CudaGraph)
    {
        write!(
            common_details,
            ", supported input dtypes [{}]",
            support.input_data_type_list(BackendExecution::CudaGraph)
        )
        .expect("formatting into a string cannot fail");
    }

    if !input_data_types.is_empty() {
        write!(common_details, ", input dtypes {input_data_types:?}")
            .expect("formatting into a string cannot fail");
    }

    let details = if provider_failure_details.is_empty() {
        common_details
    } else {
        format!("{common_details}, provider failures [{provider_failure_details}]")
    };

    Error::unsupported_operation_with_details(operation, details)
}

pub fn lowering_decision_name(record: &LoweringDecisionRecord) -> String {
    match record.decision() {
        LoweringDecision::Alias { operation } => {
            format!("alias({})", operation_name(*operation))
        }
        LoweringDecision::Kernel { operation } => {
            format!("kernel({})", operation_name(*operation))
        }
        LoweringDecision::CudnnBackend { operation } => {
            format!("cudnn({})", operation_name(*operation))
        }
        LoweringDecision::CublasLt { operation } => {
            format!("cublaslt({})", operation_name(*operation))
        }
        LoweringDecision::Cublas { operation } => {
            format!("cublas({})", operation_name(*operation))
        }
        LoweringDecision::Cutensor { operation } => {
            format!("cutensor({})", operation_name(*operation))
        }
        LoweringDecision::Cufft { operation } => {
            format!("cufft({})", operation_name(*operation))
        }
        LoweringDecision::Fusion { operation } => {
            format!("fusion({})", operation_name(*operation))
        }
        LoweringDecision::Unsupported { reason } => {
            format!("unsupported({reason})")
        }
    }
}

pub fn candidate_provider_list(candidates: &[LoweringCandidate]) -> String {
    let mut providers = Vec::new();
    for candidate in candidates {
        let name = graph_provider_name(candidate.provider());
        if !providers.contains(&name) {
            providers.push(name);
        }
    }

    if providers.is_empty() {
        providers.push("none");
    }
    providers.join(", ")
}

pub fn planned_provider_name(candidates: &[LoweringCandidate]) -> &'static str {
    candidates.first().map_or("none", |candidate| {
        graph_provider_name(candidate.provider())
    })
}

pub fn rejected_candidate_provider_list(candidates: &[LoweringCandidate]) -> String {
    let rejected = candidates
        .iter()
        .skip(1)
        .map(|candidate| graph_provider_name(candidate.provider()))
        .collect::<Vec<_>>();
    if rejected.is_empty() {
        "none".to_string()
    } else {
        rejected.join(", ")
    }
}

pub fn provider_failure_details(provider_failures: &[(CudaGraphProvider, String)]) -> String {
    provider_failures
        .iter()
        .map(|(provider, message)| format!("{}: {}", graph_provider_name(*provider), message))
        .collect::<Vec<_>>()
        .join("; ")
}
