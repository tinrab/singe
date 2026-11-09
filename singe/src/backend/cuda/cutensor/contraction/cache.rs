use std::sync::{Arc, Mutex};

use singe_cutensor::plan::Plan as CutensorPlan;

#[derive(Debug, Clone)]
pub struct CachedCutensorContractionPlan {
    plan: Arc<Mutex<CutensorPlan>>,
    workspace_size: usize,
}

impl CachedCutensorContractionPlan {
    pub(crate) fn new(plan: CutensorPlan, workspace_size: usize) -> Self {
        Self {
            plan: Arc::new(Mutex::new(plan)),
            workspace_size,
        }
    }

    pub fn plan(&self) -> &Arc<Mutex<CutensorPlan>> {
        &self.plan
    }

    pub fn workspace_size(&self) -> usize {
        self.workspace_size
    }
}
