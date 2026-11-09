// Tuning is one explicit step: `tune_graph` measures what a graph can choose, and compiling and exporting read the records it returns and never measure.
//
// A choice point is a set of nodes and the mutually exclusive ways to run them, and a record states its winner outright, keyed by everything the winner depends on.
// The tuner forces one candidate at a time through the ordinary compile path, so what it times is what a graph runs.
// Every candidate is compared against what the graph evaluates to, and the whole plan is checked and searched before records are returned.

mod comparison;
mod keys;
mod library;
mod measure;
mod plan;
mod records;
mod report;
mod tune;

pub use self::{
    comparison::Comparison,
    keys::{
        dependent_launch_key, fused_attention_key, group_key, library_key, merged_projections_key,
        schedule_key, way_key,
    },
    library::{ForcedLibrary, LibraryRecorder, build_forced},
    records::{
        Measurement, RecordCoverage, TunedChoice, TunedRecord, TunedWay, TuningEnvironment,
        TuningRecords,
    },
    report::{
        CandidateFailure, CandidateOutcome, CandidateReport, ChoiceReport, ComparisonReport,
        DropReason, DroppedChoice, FailureStage, OutcomeCounts, PlanCheckReport, PlanSearchReport,
        SecondReference, SelfComparison, TimeSpent, TuningReport,
    },
    tune::{GraphTuneOptions, tune_graph},
};

#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod tests;
