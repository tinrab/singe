use std::num::NonZeroUsize;

use crate::{
    error::{Error, Result},
    frontend::plan::{PlanFailure, PlanId, TunedPlan},
};

/// Chooses how measured execution times become a candidate's reported latency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutotuneSampling {
    /// Measures one execution.
    Once,
    /// Reports the median when all three attempts execute successfully.
    MedianOfThree,
    /// Reports the fastest successful execution within `samples` attempts.
    Minimum {
        /// Maximum number of timed execution attempts; must be nonzero.
        samples: usize,
    },
    /// Stops when the next execution no longer improves the previous minimum by the threshold.
    /// A threshold of `0.95` continues only when the next time is less than 95 percent of the previous minimum.
    /// Failed timed executions consume attempts but do not discard earlier successful measurements.
    UntilStable {
        /// Maximum number of timed execution attempts; must be nonzero.
        max_samples: usize,
        /// Finite ratio strictly between zero and one required to continue after an improved minimum.
        improvement_threshold: f32,
    },
}

impl Default for AutotuneSampling {
    fn default() -> Self {
        Self::UntilStable {
            max_samples: 100,
            improvement_threshold: 0.95,
        }
    }
}

impl AutotuneSampling {
    pub(crate) fn sample_limit(self) -> Result<usize> {
        let count = match self {
            Self::Once => 1,
            Self::MedianOfThree => 3,
            Self::Minimum { samples } => samples,
            Self::UntilStable {
                max_samples,
                improvement_threshold,
            } => {
                if !improvement_threshold.is_finite()
                    || improvement_threshold <= 0.0
                    || improvement_threshold >= 1.0
                {
                    return Err(Error::out_of_range("autotune improvement threshold"));
                }
                max_samples
            }
        };
        if count == 0 {
            return Err(Error::out_of_range("autotune samples"));
        }
        Ok(count)
    }
}

/// Controls candidate measurements and the number of successfully timed plans.
#[derive(Debug, Clone, Copy)]
pub struct AutotuneConfig {
    /// Latency estimator and stopping rule applied to timed execution attempts.
    pub sampling: AutotuneSampling,
    /// Number of unmeasured executions before timing a candidate plan.
    pub warmup: usize,
    /// Stops after this many candidates have valid timings; `None` measures every plan.
    pub max_plans: Option<NonZeroUsize>,
}

impl AutotuneConfig {
    /// Uses up to 100 timing attempts with a 0.95 improvement threshold, one warmup, and no candidate-count limit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            sampling: AutotuneSampling::default(),
            warmup: 1,
            max_plans: None,
        }
    }

    /// Replaces the timing estimator and stopping rule without validating its parameters.
    #[must_use]
    pub fn with_sampling(mut self, sampling: AutotuneSampling) -> Self {
        self.sampling = sampling;
        self
    }

    /// Sets the number of unmeasured executions before timing each candidate; zero skips warmup.
    #[must_use]
    pub fn with_warmup(mut self, warmup: usize) -> Self {
        self.warmup = warmup;
        self
    }

    /// Caps successfully timed candidates, or restores measurement of every plan with `None`.
    /// Sampling and warmup settings remain unchanged.
    #[must_use]
    pub fn with_max_plans(mut self, max_plans: impl Into<Option<NonZeroUsize>>) -> Self {
        self.max_plans = max_plans.into();
        self
    }
}

impl Default for AutotuneConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing result keyed by stable plan identity.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanTiming {
    /// Stable identity of the measured candidate in its retained collection.
    pub plan: PlanId,
    /// Reported execution latency in milliseconds according to the selected sampling policy.
    pub elapsed_ms: f32,
    /// Number of successful timed executions used to compute the reported latency.
    pub samples: usize,
}

/// Timings are sorted without changing the stored plans or their identities.
#[derive(Debug, Clone)]
pub struct AutotuneResult {
    /// Retains the measured target, compiled collection and runtime layouts independently of the input graph handle.
    pub winner: TunedPlan,
    /// Successful candidate timings sorted from lowest to highest reported latency.
    pub timings: Vec<PlanTiming>,
    /// Failed trials, including recoverable measurements from plans that were successfully timed.
    pub failures: Vec<PlanFailure>,
}
