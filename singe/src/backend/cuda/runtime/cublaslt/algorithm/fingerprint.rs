use serde::{Deserialize, Serialize};
use singe_cublas::lt::{matmul::MatmulAlgorithm, types::MatmulAlgorithmConfigAttribute};

use crate::error::Result;

/// Comparable cuBLASLt algorithm configuration attributes used to identify a cached tuning selection.
/// Includes algorithm, tile, reduction, split-K, swizzle, custom option, stage, inner shape, and cluster shape identifiers.
#[allow(
    clippy::unsafe_derive_deserialize,
    reason = "cuBLASLt checks attribute values, and `unsafe` here only states their native types"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CublasLtMatmulAlgorithmFingerprint {
    pub(crate) id: i32,
    tile_id: u32,
    split_k_count: i32,
    reduction_scheme: u32,
    cta_swizzling: u32,
    custom_option: u32,
    stages_id: u32,
    inner_shape_id: u16,
    cluster_shape_id: u16,
}

impl CublasLtMatmulAlgorithmFingerprint {
    /// Queries the algorithm configuration attributes used for tuning matching.
    ///
    /// # Errors
    ///
    /// Returns an error if cuBLASLt cannot read a required configuration attribute or its native value has an unexpected byte size.
    pub fn from_algorithm(algorithm: &MatmulAlgorithm) -> Result<Self> {
        Ok(Self {
            id: algorithm.config_attribute(MatmulAlgorithmConfigAttribute::Id)?,
            tile_id: algorithm.config_attribute(MatmulAlgorithmConfigAttribute::TileId)?,
            split_k_count: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::SplitKCount)?,
            reduction_scheme: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::ReductionScheme)?,
            cta_swizzling: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::CtaSwizzling)?,
            custom_option: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::CustomOption)?,
            stages_id: algorithm.config_attribute(MatmulAlgorithmConfigAttribute::StagesId)?,
            inner_shape_id: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::InnerShapeId)?,
            cluster_shape_id: algorithm
                .config_attribute(MatmulAlgorithmConfigAttribute::ClusterShapeId)?,
        })
    }

    // Writes this configuration onto `algorithm`, one initialized for this configuration's id.
    //
    // # Errors
    //
    // Returns an error if cuBLASLt rejects an attribute.
    pub fn apply_to(&self, algorithm: &mut MatmulAlgorithm) -> Result<()> {
        // SAFETY: each value has the native type of its attribute, as `from_algorithm` reads it.
        unsafe {
            algorithm
                .set_config_attribute(MatmulAlgorithmConfigAttribute::TileId, &self.tile_id)?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::SplitKCount,
                &self.split_k_count,
            )?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::ReductionScheme,
                &self.reduction_scheme,
            )?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::CtaSwizzling,
                &self.cta_swizzling,
            )?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::CustomOption,
                &self.custom_option,
            )?;
            algorithm
                .set_config_attribute(MatmulAlgorithmConfigAttribute::StagesId, &self.stages_id)?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::InnerShapeId,
                &self.inner_shape_id,
            )?;
            algorithm.set_config_attribute(
                MatmulAlgorithmConfigAttribute::ClusterShapeId,
                &self.cluster_shape_id,
            )?;
        }
        Ok(())
    }

    // The C++ that writes this configuration onto the algorithm `name`, leaving `ok` false if cuBLASLt rejects any of it, with `value_of` writing each value.
    #[must_use]
    pub fn cpp_settings(&self, name: &str, ok: &str, value_of: &dyn Fn(i64) -> String) -> String {
        let set = |attribute: &str, ty: &str, value: String| {
            format!(
                "            {{ const {ty} value = {value}; {ok} = {ok} && cublasLtMatmulAlgoConfigSetAttribute(&{name}, {attribute}, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS; }}\n"
            )
        };
        [
            set(
                "CUBLASLT_ALGO_CONFIG_TILE_ID",
                "uint32_t",
                value_of(i64::from(self.tile_id)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
                "int32_t",
                value_of(i64::from(self.split_k_count)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",
                "uint32_t",
                value_of(i64::from(self.reduction_scheme)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",
                "uint32_t",
                value_of(i64::from(self.cta_swizzling)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",
                "uint32_t",
                value_of(i64::from(self.custom_option)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_STAGES_ID",
                "uint32_t",
                value_of(i64::from(self.stages_id)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",
                "uint16_t",
                value_of(i64::from(self.inner_shape_id)),
            ),
            set(
                "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
                "uint16_t",
                value_of(i64::from(self.cluster_shape_id)),
            ),
        ]
        .concat()
    }

    // This configuration with its depth split into `split` parts.
    #[must_use]
    pub const fn with_split(mut self, split: i32) -> Self {
        self.split_k_count = split;
        self
    }

    #[must_use]
    pub const fn split(&self) -> i32 {
        self.split_k_count
    }

    #[cfg(test)]
    #[must_use]
    pub fn new(seed: i32) -> Self {
        Self {
            id: seed,
            tile_id: (seed + 1).try_into().unwrap(),
            split_k_count: seed + 2,
            reduction_scheme: (seed + 3).try_into().unwrap(),
            cta_swizzling: (seed + 4).try_into().unwrap(),
            custom_option: (seed + 5).try_into().unwrap(),
            stages_id: (seed + 6).try_into().unwrap(),
            inner_shape_id: (seed + 7).try_into().unwrap(),
            cluster_shape_id: (seed + 8).try_into().unwrap(),
        }
    }
}
