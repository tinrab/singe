// The one rule for whether a candidate beats what it would replace.
// Both are timed in the same rounds, so what drifts over a tuning run moves them together, and each round gives one ratio.

use std::time::Duration;

use serde::{Deserialize, Serialize};

const ONE_PPM: u32 = 1_000_000;

// A candidate replaces another only when it is at least this much faster.
const FASTER_PPM: u32 = 970_000;

// What timing a candidate against a baseline in rounds showed, as the candidate's time over the baseline's, in millionths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comparison {
    pub median_ppm: u32,
    pub lower_quartile_ppm: u32,
    pub upper_quartile_ppm: u32,
    // Whether both were alive and took turns, rather than alternating in blocks because they did not fit together.
    pub together: bool,
}

impl Comparison {
    // What the rounds of `baseline` and `candidate` show, pairing them in order, or `None` without a round to read.
    #[must_use]
    pub fn of(baseline: &[Duration], candidate: &[Duration]) -> Option<Self> {
        let mut ratios = baseline
            .iter()
            .zip(candidate)
            .filter(|(baseline, _)| !baseline.is_zero())
            .map(|(baseline, candidate)| {
                let ratio = candidate.as_secs_f64() / baseline.as_secs_f64();
                ppm(ratio)
            })
            .collect::<Vec<_>>();
        ratios.sort_unstable();
        let at = |quarters: usize| ratios.get((ratios.len().checked_sub(1)?) * quarters / 4);
        Some(Self {
            median_ppm: *at(2)?,
            lower_quartile_ppm: *at(1)?,
            upper_quartile_ppm: *at(3)?,
            together: true,
        })
    }

    // What rounds in alternating order show, reading two at a time, since going first has a cost of its own that a round alone would count against one of them.
    #[must_use]
    pub fn of_alternating(baseline: &[Duration], candidate: &[Duration]) -> Option<Self> {
        let pairs = |times: &[Duration]| {
            times
                .as_chunks::<2>()
                .0
                .iter()
                .map(|pair| pair.iter().sum())
                .collect::<Vec<Duration>>()
        };
        Self::of(&pairs(baseline), &pairs(candidate))
    }

    // What fresh pairs of instances show together: the median of their medians, and the widest either way as the quartiles, so `wins` then asks that every pair agrees.
    // One instance of a plan can stay a few percent from another of the same plan for as long as it lives, which no number of rounds averages out.
    #[must_use]
    pub fn across(pairs: &[Self]) -> Option<Self> {
        let mut medians = pairs.iter().map(|pair| pair.median_ppm).collect::<Vec<_>>();
        medians.sort_unstable();
        Some(Self {
            median_ppm: *medians.get(medians.len().checked_sub(1)? / 2)?,
            lower_quartile_ppm: *medians.first()?,
            upper_quartile_ppm: *medians.last()?,
            together: pairs.iter().all(|pair| pair.together),
        })
    }

    // Whether the candidate is measurably faster: across pairs of instances, every pair over the margin, since noise that widens the pairs can carry the median of a smaller win past it.
    #[must_use]
    pub const fn wins(&self) -> bool {
        self.upper_quartile_ppm < FASTER_PPM
    }

    // This comparison with its two sides swapped, whose order pairs of instances alternate.
    #[must_use]
    pub fn inverted(&self) -> Self {
        let inverse = |ppm: u32| {
            let ratio = f64::from(ONE_PPM) / f64::from(ppm.max(1));
            self::ppm(ratio)
        };
        Self {
            median_ppm: inverse(self.median_ppm),
            lower_quartile_ppm: inverse(self.upper_quartile_ppm),
            upper_quartile_ppm: inverse(self.lower_quartile_ppm),
            together: self.together,
        }
    }

    // Whether the candidate is measurably slower, by the same rule from the other side.
    #[must_use]
    pub fn loses(&self) -> bool {
        let needed = f64::from(ONE_PPM) / f64::from(FASTER_PPM);
        f64::from(self.median_ppm) > f64::from(ONE_PPM) * needed
            && self.lower_quartile_ppm > ONE_PPM
    }

    // How far the median round sits from no difference, as a fraction.
    #[must_use]
    pub fn difference(&self) -> f64 {
        f64::from(self.median_ppm) / f64::from(ONE_PPM) - 1.0
    }
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a ratio of times is not negative, and a cast takes what is too large to the largest value"
)]
fn ppm(ratio: f64) -> u32 {
    (ratio * f64::from(ONE_PPM)).round() as u32
}
