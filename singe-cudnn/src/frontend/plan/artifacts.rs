use std::collections::BTreeMap;

use crate::{
    backend::tensor::TensorId,
    frontend::{
        graph::TensorRecord,
        plan::{BindingReplacement, RequiredTensor},
    },
    scalar::ScalarValue,
};

/// Graph metadata carried unchanged from plan candidates through built and compiled plans.
#[derive(Debug, Clone)]
pub struct PlanArtifacts {
    pub graph_name: Option<String>,
    pub tensors: BTreeMap<TensorId, TensorRecord>,
    pub scalar_bindings: Vec<(TensorId, ScalarValue)>,
    pub binding_replacements: Vec<BindingReplacement>,
    pub required_tensors: Vec<RequiredTensor>,
}
