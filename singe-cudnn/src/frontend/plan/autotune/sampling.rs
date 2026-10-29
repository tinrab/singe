use crate::{
    error::{Error, Result},
    frontend::plan::{AutotuneSampling, PlanId, PlanTiming},
};

/// Validates measured latency and gives numerically equal zero timings the same sort key.
pub fn normalize_elapsed(elapsed_ms: f32) -> Result<f32> {
    if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
        return Err(Error::out_of_range("autotune elapsed time"));
    }
    Ok(if elapsed_ms == 0.0 { 0.0 } else { elapsed_ms })
}

pub struct TimingSamples {
    sampling: AutotuneSampling,
    count: usize,
    minimum: f32,
    median: [f32; 3],
}

impl TimingSamples {
    pub fn new(sampling: AutotuneSampling) -> Self {
        Self {
            sampling,
            count: 0,
            minimum: f32::INFINITY,
            median: [0.0; 3],
        }
    }

    /// Records a completed measurement and reports whether sampling should continue.
    pub fn record(&mut self, elapsed_ms: f32) -> bool {
        let keep_sampling = match self.sampling {
            AutotuneSampling::UntilStable {
                improvement_threshold,
                ..
            } => elapsed_ms / self.minimum < improvement_threshold,
            AutotuneSampling::MedianOfThree => {
                self.median[self.count] = elapsed_ms;
                true
            }
            _ => true,
        };
        self.minimum = self.minimum.min(elapsed_ms);
        self.count += 1;
        keep_sampling
    }

    pub fn finish(mut self, plan: PlanId) -> Option<PlanTiming> {
        if self.count == 0 || (self.sampling == AutotuneSampling::MedianOfThree && self.count != 3)
        {
            return None;
        }
        let elapsed_ms = if self.sampling == AutotuneSampling::MedianOfThree {
            self.median.sort_unstable_by(f32::total_cmp);
            self.median[1]
        } else {
            self.minimum
        };
        Some(PlanTiming {
            plan,
            elapsed_ms,
            samples: self.count,
        })
    }
}
