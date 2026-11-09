use std::{collections::BTreeMap, error::Error as StdError, fmt, time::Duration};

use serde::{Deserialize, Serialize};
use singe_core::ErrorChain;

use crate::backend::cuda::tuning::comparison::Comparison;

// Where a candidate stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FailureStage {
    // Making the small graph of a choice point.
    Graph,
    // Writing the candidate's kernel or program.
    Export,
    // Compiling and loading a program.
    Build,
    // Compiling the graph.
    Prepare,
    // Running it.
    Run,
    // Reading back what it computed.
    ReadBack,
}

// Why a candidate has no time.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateFailure {
    pub stage: FailureStage,
    pub message: String,
}

impl CandidateFailure {
    pub(crate) fn at(stage: FailureStage, message: impl Into<String>) -> Self {
        Self {
            stage,
            message: message.into(),
        }
    }

    // A failure at `stage`, whose message keeps everything `error` was caused by.
    pub(crate) fn from_error(stage: FailureStage, error: &(dyn StdError + 'static)) -> Self {
        Self {
            stage,
            message: ErrorChain(error).to_string(),
        }
    }
}

// What became of one candidate of a choice point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CandidateOutcome {
    Timed { median_ns: u64, bytes: u64 },
    Failed(CandidateFailure),
    // Its values differ from the evaluated graph's beyond what rounding allows.
    Disagreed,
    // It allocates more than the whole plan may.
    OverMemory { bytes: u64 },
    // The budget ran out before it was timed.
    Skipped,
    // A library refused the configuration it states, so nothing could time that configuration.
    Refused,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CandidateReport {
    // What the candidate chose, as a reader of the file sees it.
    pub choice: String,
    pub outcome: CandidateOutcome,
}

// Where the values a choice point's second inputs should give came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondReference {
    // Eager evaluation on the host, which shares no library with a candidate.
    Host,
    // Eager evaluation on the device, as the first inputs' values are, since the inputs did not fit on the host or the host could not evaluate them.
    Device,
}

// What a choice point's candidates came to, or why it has none.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChoiceReport {
    pub key: String,
    pub candidates: Vec<CandidateReport>,
    pub failure: Option<CandidateFailure>,
    // Where the values of the second inputs every candidate is also judged on came from, or `None` when it had no floating input to change.
    pub second_reference: Option<SecondReference>,
    // What timing the fastest candidates against the first in rounds showed.
    pub comparisons: Vec<ComparisonReport>,
    // What timing a later winner against the winner it would replace showed.
    pub between_winners: Vec<ComparisonReport>,
}

// What timing one candidate against the first in rounds showed, or why that could not finish.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub choice: String,
    pub comparison: Option<Comparison>,
    pub failure: Option<CandidateFailure>,
}

// What timing a plan against itself showed, which is how far timing alone moves a comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelfComparison {
    pub key: String,
    pub in_rounds: Comparison,
    // The second of two timings in turn over the first, in millionths, as candidates were once judged.
    pub in_turn_ppm: u32,
}

// Where a tuning step's time went, in nanoseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSpent {
    pub total_ns: u64,
    pub export_ns: u64,
    pub build_ns: u64,
    pub create_ns: u64,
    pub run_ns: u64,
    pub check_ns: u64,
}

impl TimeSpent {
    pub(crate) const fn add(&mut self, other: &Self) {
        self.export_ns += other.export_ns;
        self.build_ns += other.build_ns;
        self.create_ns += other.create_ns;
        self.run_ns += other.run_ns;
        self.check_ns += other.check_ns;
    }
}

// Why a plan lost a choice point's records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DropReason {
    Failed(CandidateFailure),
    Disagreed,
    OverMemory,
    Slower,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DroppedChoice {
    pub key: String,
    pub reason: DropReason,
}

// How one check of a whole plan went.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanCheckReport {
    pub default_ns: Option<u64>,
    pub default_bytes: Option<u64>,
    pub plan_ns: Option<u64>,
    pub plan_bytes: Option<u64>,
    // Why the path with no records could not be timed or does not agree, which a reader should look into.
    pub default_failure: Option<CandidateFailure>,
    pub default_disagreed: bool,
    pub default_over_limit: bool,
    pub dropped: Vec<DroppedChoice>,
}

// How the search of whole plans went.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSearchReport {
    pub plans_run: usize,
    pub duplicates: usize,
    pub failed: usize,
    pub disagreed: usize,
    pub over_memory: usize,
    pub abandoned: usize,
    pub accepted: usize,
    pub start_ns: Option<u64>,
    pub end_ns: Option<u64>,
    pub first_failure: Option<CandidateFailure>,
}

// What a tuning step tried, and why what lost lost.
// Compiling and exporting never read this, so it does not count towards records being equal.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningReport {
    pub choices: Vec<ChoiceReport>,
    pub checks: Vec<PlanCheckReport>,
    pub searches: Vec<PlanSearchReport>,
    // Only when `GraphTuneOptions::self_comparisons` asked.
    pub self_comparisons: Vec<SelfComparison>,
    pub spent: TimeSpent,
}

// How many candidates ended each way, with the first failure of each stage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OutcomeCounts {
    pub timed: usize,
    pub failed: usize,
    pub disagreed: usize,
    pub over_memory: usize,
    pub skipped: usize,
    pub refused: usize,
    pub compared: usize,
    pub won: usize,
    pub between_winners: usize,
    pub replaced: usize,
    pub by_stage: BTreeMap<FailureStage, (usize, String)>,
}

impl OutcomeCounts {
    fn note_failure(&mut self, failure: &CandidateFailure) {
        self.by_stage
            .entry(failure.stage)
            .or_insert_with(|| (0, failure.message.clone()))
            .0 += 1;
    }
}

impl TuningReport {
    #[must_use]
    pub fn counts(&self) -> OutcomeCounts {
        let mut counts = OutcomeCounts::default();
        for choice in &self.choices {
            if let Some(failure) = &choice.failure {
                counts.failed += 1;
                counts.note_failure(failure);
            }

            for candidate in &choice.candidates {
                match &candidate.outcome {
                    CandidateOutcome::Timed { .. } => counts.timed += 1,
                    CandidateOutcome::Failed(failure) => {
                        counts.failed += 1;
                        counts.note_failure(failure);
                    }
                    CandidateOutcome::Disagreed => counts.disagreed += 1,
                    CandidateOutcome::OverMemory { .. } => counts.over_memory += 1,
                    CandidateOutcome::Skipped => counts.skipped += 1,
                    CandidateOutcome::Refused => counts.refused += 1,
                }
            }

            for compared in &choice.comparisons {
                counts.compared += 1;
                if compared
                    .comparison
                    .is_some_and(|comparison| comparison.wins())
                {
                    counts.won += 1;
                }

                if let Some(failure) = &compared.failure {
                    counts.failed += 1;
                    counts.note_failure(failure);
                }
            }

            for compared in &choice.between_winners {
                counts.between_winners += 1;
                if compared
                    .comparison
                    .is_some_and(|comparison| comparison.wins())
                {
                    counts.replaced += 1;
                }

                if let Some(failure) = &compared.failure {
                    counts.failed += 1;
                    counts.note_failure(failure);
                }
            }
        }
        counts
    }

    // The candidates that failed or disagreed, each with its choice point's key, which are what a reader should look into.
    pub fn problems(&self) -> impl Iterator<Item = (&str, &str, &CandidateOutcome)> {
        self.choices.iter().flat_map(|choice| {
            choice
                .candidates
                .iter()
                .filter(|candidate| {
                    matches!(
                        candidate.outcome,
                        CandidateOutcome::Failed(_) | CandidateOutcome::Disagreed
                    )
                })
                .map(|candidate| {
                    (
                        choice.key.as_str(),
                        candidate.choice.as_str(),
                        &candidate.outcome,
                    )
                })
        })
    }
}

impl fmt::Display for TuningReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let counts = self.counts();
        writeln!(
            f,
            "{} choice points: {} timed, {} failed, {} disagreed, {} over memory, {} skipped, {} refused; {} compared in rounds, {} won; {} between winners, {} replaced",
            self.choices.len(),
            counts.timed,
            counts.failed,
            counts.disagreed,
            counts.over_memory,
            counts.skipped,
            counts.refused,
            counts.compared,
            counts.won,
            counts.between_winners,
            counts.replaced,
        )?;
        for (stage, (count, first)) in &counts.by_stage {
            writeln!(f, "  {stage:?}: {count}, first: {first}")?;
        }

        let referenced = |reference| {
            self.choices
                .iter()
                .filter(|choice| choice.second_reference == reference)
                .count()
        };
        writeln!(
            f,
            "second inputs: {} judged on the host, {} on the device, {} without",
            referenced(Some(SecondReference::Host)),
            referenced(Some(SecondReference::Device)),
            referenced(None),
        )?;

        for check in &self.checks {
            writeln!(
                f,
                "plan check: default {:?} ns, {:?} bytes; plan {:?} ns, {:?} bytes; {} dropped",
                check.default_ns,
                check.default_bytes,
                check.plan_ns,
                check.plan_bytes,
                check.dropped.len(),
            )?;
            if let Some(failure) = &check.default_failure {
                writeln!(f, "  default failed: {failure:?}")?;
            }

            if check.default_disagreed {
                writeln!(f, "  default disagrees with the evaluated graph")?;
            }

            if check.default_over_limit {
                writeln!(f, "  default exceeds the memory limit")?;
            }
            for dropped in &check.dropped {
                writeln!(f, "  dropped {}: {:?}", dropped.key, dropped.reason)?;
            }
        }

        for search in &self.searches {
            writeln!(
                f,
                "plan search: {} run, {} duplicates, {} failed, {} disagreed, {} over memory, {} abandoned, {} accepted, {:?} ns to {:?} ns",
                search.plans_run,
                search.duplicates,
                search.failed,
                search.disagreed,
                search.over_memory,
                search.abandoned,
                search.accepted,
                search.start_ns,
                search.end_ns,
            )?;
            if let Some(failure) = &search.first_failure {
                writeln!(f, "  first failure: {failure:?}")?;
            }
        }

        if !self.self_comparisons.is_empty() {
            let widest = |ratio: fn(&SelfComparison) -> f64| {
                self.self_comparisons
                    .iter()
                    .map(|compared| ratio(compared).abs())
                    .fold(0.0, f64::max)
                    * 100.0
            };
            writeln!(
                f,
                "self comparisons: {}, {} won, {} with sides swapped; widest median round {:.2}%, widest in turn {:.2}%",
                self.self_comparisons.len(),
                self.self_comparisons
                    .iter()
                    .filter(|compared| compared.in_rounds.wins())
                    .count(),
                self.self_comparisons
                    .iter()
                    .filter(|compared| compared.in_rounds.inverted().wins())
                    .count(),
                widest(|compared| compared.in_rounds.difference()),
                widest(|compared| f64::from(compared.in_turn_ppm) / 1e6 - 1.0),
            )?;
        }

        let seconds = |nanoseconds: u64| Duration::from_nanos(nanoseconds).as_secs_f64();
        let spent = &self.spent;
        writeln!(
            f,
            "spent {:.1}s: export {:.1}s, build {:.1}s, create {:.1}s, run {:.1}s, check {:.1}s",
            seconds(spent.total_ns),
            seconds(spent.export_ns),
            seconds(spent.build_ns),
            seconds(spent.create_ns),
            seconds(spent.run_ns),
            seconds(spent.check_ns),
        )?;
        Ok(())
    }
}
