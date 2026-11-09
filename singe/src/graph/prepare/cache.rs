use std::collections::HashMap;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::{GraphSignature, prepare::PreparedPlan},
    layout::Layout,
    tensor::Tensor,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeMetadataKey(Vec<(String, DataType, Layout, Device)>);

impl RuntimeMetadataKey {
    pub(crate) fn from_signature(signature: &GraphSignature) -> Self {
        Self(
            signature
                .inputs()
                .iter()
                .filter(|input| !input.is_static_initializer())
                .map(|input| {
                    (
                        input.name().to_owned(),
                        input.data_type(),
                        input.layout().clone(),
                        input.device(),
                    )
                })
                .collect(),
        )
    }

    pub(crate) fn from_inputs(
        signature: &GraphSignature,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<Self> {
        let resolved = signature.resolve(inputs)?;
        let mut values = Vec::new();
        for input in signature
            .inputs()
            .iter()
            .filter(|input| !input.is_static_initializer())
        {
            let tensor = resolved
                .get(&input.name)
                .ok_or_else(|| Error::missing_input("graph input of the signature"))?;
            values.push((
                input.name().to_owned(),
                tensor.data_type(),
                tensor.layout().clone(),
                tensor.device(),
            ));
        }
        Ok(Self(values))
    }
}

#[derive(Debug)]
pub struct CachedPlan {
    pub(crate) key: RuntimeMetadataKey,
    pub(crate) plan: PreparedPlan,
    pub(crate) last_used: u64,
}

/// Specialization cache activity.
/// Initial preparation counts as one miss and one analysis build.
/// Instance creation shares analysis rather than rebuilding it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphExecutionCacheStats {
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) evictions: u64,
    pub(crate) analysis_builds: u64,
    pub(crate) entries: usize,
}

impl GraphExecutionCacheStats {
    /// Returns the saturated count of active or inactive specialization cache hits.
    #[must_use]
    pub fn hits(self) -> u64 {
        self.hits
    }

    /// Returns the saturated count of successful new specialization preparations, including initial preparation.
    #[must_use]
    pub fn misses(self) -> u64 {
        self.misses
    }

    /// Returns the saturated number of inactive plans evicted to enforce cache capacity.
    #[must_use]
    pub fn evictions(self) -> u64 {
        self.evictions
    }

    /// Returns the saturated number of compiled analyses built by this instance.
    ///
    /// Instances created from shared preparation start at zero.
    #[must_use]
    pub fn analysis_builds(self) -> u64 {
        self.analysis_builds
    }

    /// Returns the number of retained active and inactive specializations.
    #[must_use]
    pub fn entries(self) -> usize {
        self.entries
    }
}
