use crate::{
    backend::{
        cuda::compiled::plan::{
            cache_key::{DynamicShapePolicy, GraphCacheKeySummary},
            lowering::{CudaGraphProvider, LoweringCandidate, LoweringPolicy},
            pipeline::{CompilePipeline, CompileStage},
        },
        registry::LayoutSupport,
    },
    graph::{Graph, NumericalPolicy},
};

#[test]
fn test_summary_exposes_stage_and_provider_key_material() {
    let pipeline = CompilePipeline::default().with_stage_version(CompileStage::Fusion, 7);
    let policy = LoweringPolicy::new(vec![
        CudaGraphProvider::CudnnBackend,
        CudaGraphProvider::CublasLt,
    ]);
    let candidates = vec![(
        3,
        vec![
            LoweringCandidate::new(
                CudaGraphProvider::CudnnBackend,
                LayoutSupport::RequiresContiguous,
                10,
            ),
            LoweringCandidate::new(
                CudaGraphProvider::CublasLt,
                LayoutSupport::RequiresContiguous,
                20,
            ),
        ],
    )];
    let graph_fingerprint = Graph::new().fingerprint().unwrap();

    let summary = GraphCacheKeySummary::new(
        graph_fingerprint,
        pipeline.stage_versions().to_vec(),
        policy.clone(),
        candidates.clone(),
        true,
        DynamicShapePolicy::reject(),
        Vec::new(),
        None,
        NumericalPolicy::default(),
    );

    assert!(
        summary
            .compile_stage_versions()
            .iter()
            .any(|stage| stage.stage() == CompileStage::Fusion && stage.version() == 7)
    );
    assert_eq!(summary.lowering_policy(), &policy);
    assert_eq!(summary.lowering_candidates(), candidates.as_slice());
}
