use crate::{
    data_type::DataType,
    frontend::{
        graph::DataTypePolicy,
        operation::{
            AttentionBackwardConfig, AttentionConfig, DirectSdpaBackwardConfig,
            DirectSdpaForwardConfig, Operation, SdpaOperation, SdpaScoreSubgraph,
        },
    },
};

macro_rules! impl_sdpa_scope_policy {
    ($($config:ty),+ $(,)?) => {$(
        impl $config {
            pub(crate) fn data_type_policy(&self) -> DataTypePolicy {
                self.data_type_policy
            }

            pub(crate) fn inherit_data_type_policy(&mut self, inherited: DataTypePolicy) {
                self.data_type_policy = self.data_type_policy.with_defaults(inherited);
            }
        }
    )+};
}

impl_sdpa_scope_policy!(DirectSdpaForwardConfig, DirectSdpaBackwardConfig);

impl AttentionConfig {
    pub(crate) fn inherit_compute_type(&mut self, inherited: Option<DataType>) {
        self.softmax.inherit_compute_type(inherited);
    }
}

impl AttentionBackwardConfig {
    pub(crate) fn inherit_compute_type(&mut self, inherited: Option<DataType>) {
        self.softmax.inherit_compute_type(inherited);
    }
}

impl SdpaScoreSubgraph {
    pub(crate) fn inherit_data_type_policy(&mut self, inherited: DataTypePolicy) {
        self.graph.config.data_type_policy =
            self.graph.config.data_type_policy.with_defaults(inherited);
    }
}

impl SdpaOperation {
    pub(crate) fn inherit_data_type_policy(&mut self, inherited: DataTypePolicy) {
        match self {
            Self::Forward { config, .. } => config.inherit_data_type_policy(inherited),
            Self::Backward { config, .. } => config.inherit_data_type_policy(inherited),
        }
    }
}

impl Operation {
    /// Owned child graphs whose missing defaults inherit this operation's graph context.
    pub(crate) fn score_subgraphs_mut(&mut self) -> impl Iterator<Item = &mut SdpaScoreSubgraph> {
        let graphs = match self {
            Self::Sdpa(SdpaOperation::Forward { score_subgraph, .. }) => {
                [score_subgraph.as_deref_mut(), None]
            }
            Self::Sdpa(SdpaOperation::Backward { config, .. }) => {
                [config.score_subgraph.as_mut(), None]
            }
            Self::AttentionForward(request) => [request.config.score.score_subgraph.as_mut(), None],
            Self::AttentionBackward(request) => {
                let config = request.config_mut();
                [
                    config.score.score_subgraph.as_mut(),
                    config.score_subgraph_bprop.as_mut(),
                ]
            }
            _ => [None, None],
        };
        graphs.into_iter().flatten()
    }
}
