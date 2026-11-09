use super::*;
use crate::backend::cuda::CudaCompiledGraph;

pub fn assert_cutensor_contraction_report(compiled: &CudaCompiledGraph) {
    let stats = compiled.stats();
    assert_eq!(stats.input_nodes(), 3);
    assert_eq!(stats.cutensor_child_graph_nodes(), 1);
    let report = compiled.report();
    let contraction_report = report
        .operations()
        .iter()
        .find(|node| node.operation() == operation_name(Operation::Contraction))
        .expect("missing contraction report");
    assert_eq!(
        contraction_report.operation(),
        operation_name(Operation::Contraction)
    );
    assert_eq!(
        contraction_report.input_data_types(),
        vec![DataType::F32, DataType::F32, DataType::F32],
    );
    assert_eq!(contraction_report.output_data_types(), vec![DataType::F32]);
    assert_eq!(
        contraction_report.layout_support(),
        LayoutSupport::AnyStrided
    );
    assert_eq!(
        contraction_report.input_layout_requirement(),
        GraphCompileInputLayoutRequirement::AnyStrided
    );
    assert!(
        contraction_report
            .candidates()
            .iter()
            .any(
                |candidate| candidate.provider() == CudaGraphProvider::Cutensor
                    && candidate.layout_support() == LayoutSupport::AnyStrided
                    && candidate.input_layout_requirement()
                        == GraphCompileInputLayoutRequirement::AnyStrided
            ),
        "{:?}",
        contraction_report.candidates()
    );
    assert_eq!(
        contraction_report
            .realized_decision()
            .as_ref()
            .and_then(|decision| decision.provider()),
        Some(CudaGraphProvider::Cutensor),
    );
    assert_eq!(
        contraction_report.selected_algorithm(),
        Some("cutensor contraction plan")
    );
}
