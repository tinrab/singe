use crate::{
    backend::cuda::compiled::plan::{
        cache_key::{
            CUDA_BACKEND_FEATURE_VERSION, CUDA_GRAPH_CACHE_KEY_VERSION, DynamicShapePolicy,
            GraphCacheKey,
            dynamic::{concrete_shape_bucket_from_ir, symbolic_dimension_key_material_from_ir},
            static_input::static_input_identity_key_material_from_ir,
        },
        lowering::{LoweringPlan, LoweringPolicy},
        pipeline::CompilePipeline,
    },
    error::Result,
    graph::{
        CompiledGraph, GraphFingerprintOptions, NumericalPolicy,
        ir::{Graph, ValueId},
    },
};

impl GraphCacheKey {
    /// Builds a CUDA cache identity from the compiled graph, provider candidates, static input identities, numerical policy, and shape bucket.
    ///
    /// # Errors
    ///
    /// Returns an error if execution nodes cannot be resolved, static input identities are invalid, dynamic shape bucket construction fails, or graph fingerprinting fails.
    pub fn create_from_compiled_graph(
        compiled: &CompiledGraph,
        ir: &Graph,
        execution_order: &[ValueId],
        lowering_plan: &LoweringPlan,
        fingerprint_config: &GraphFingerprintOptions,
        pipeline: CompilePipeline,
        lowering_policy: LoweringPolicy,
        dynamic_shape_policy: DynamicShapePolicy,
    ) -> Result<Self> {
        for &id in execution_order {
            ir.required_operation_for_value(id)?;
        }

        let lowering_candidates = lowering_plan
            .candidate_key_material(execution_order)
            .into_iter()
            .map(|(id, candidates)| (id.index(), candidates))
            .collect();
        let static_input_identities = static_input_identity_key_material_from_ir(ir)?;
        let symbolic_dimensions = symbolic_dimension_key_material_from_ir(ir);
        let shape_bucket = concrete_shape_bucket_from_ir(ir, &dynamic_shape_policy)?;
        Ok(Self {
            numerics: NumericalPolicy::current(),
            graph: compiled.fingerprint_with_options(fingerprint_config)?,
            compile_stage_versions: pipeline.stage_versions().to_vec(),
            pipeline,
            lowering_policy,
            lowering_candidates,
            dynamic_shape_policy,
            static_input_identities,
            symbolic_dimensions,
            shape_bucket,
            backend_feature_version: CUDA_BACKEND_FEATURE_VERSION,
            compiler_version_salt: CUDA_GRAPH_CACHE_KEY_VERSION,
        })
    }
}
