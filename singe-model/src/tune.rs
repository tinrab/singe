//! Tuning presets and overrides, mapped onto `GraphTuneOptions` the same way for `singe export` and `singe serve`.

#[cfg(feature = "cuda_13_4")]
use std::path::Path;
use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda_13_4")]
use singe::backend::cuda::{
    LoweringPolicy,
    tuning::{GraphTuneOptions, Measurement, TuningRecords},
};

use crate::error::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TuneLevel {
    Quick,
    #[default]
    Default,
    Thorough,
}

#[derive(Debug, Clone, Default)]
pub struct TuneSettings {
    pub level: TuneLevel,
    pub budget: Option<Duration>,
    pub runs: Option<usize>,
    pub schedules: Option<bool>,
    pub plan_budget: Option<Duration>,
    pub seed: Option<u64>,
    pub memory_limit: Option<u64>,
    pub jobs: Option<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct Tuning {
    pub settings: TuneSettings,
    pub records: Option<PathBuf>,
    pub save: Option<PathBuf>,
    pub inputs: Option<PathBuf>,
}

#[cfg(feature = "cuda_13_4")]
impl TuneSettings {
    #[must_use]
    pub fn options(&self, policy: LoweringPolicy, measurement: Measurement) -> GraphTuneOptions {
        let defaults = GraphTuneOptions::default();
        let (budget, runs, schedules, plan_budget) = match self.level {
            TuneLevel::Quick => (Duration::from_secs(1), 5, false, Duration::ZERO),
            TuneLevel::Default => (
                defaults.choice_budget,
                defaults.runs,
                defaults.schedules,
                defaults.plan_budget,
            ),
            TuneLevel::Thorough => (Duration::from_secs(30), 50, true, Duration::from_secs(60)),
        };
        GraphTuneOptions {
            policy,
            measurement,
            choice_budget: self.budget.unwrap_or(budget),
            runs: self.runs.unwrap_or(runs),
            schedules: self.schedules.unwrap_or(schedules),
            plan_budget: self.plan_budget.unwrap_or(plan_budget),
            plan_seed: self.seed.unwrap_or(defaults.plan_seed),
            memory_limit: self.memory_limit.or(defaults.memory_limit),
            build_jobs: self.jobs.unwrap_or(defaults.build_jobs),
            ..defaults
        }
    }
}

#[cfg(feature = "cuda_13_4")]
pub fn read_records(
    path: &Path,
    architecture: Option<&str>,
    measurement: Measurement,
) -> Result<TuningRecords> {
    let records = TuningRecords::read(path)?;
    if let Some(mismatch) = records.mismatch(architecture, measurement) {
        return Err(Error::invalid(format!(
            "tuning records {} made {mismatch}",
            path.display()
        )));
    }
    Ok(records)
}
