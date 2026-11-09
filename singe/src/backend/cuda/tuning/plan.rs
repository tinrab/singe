use std::{
    cmp::Reverse,
    collections::HashSet,
    time::{Duration, Instant},
};

use crate::backend::cuda::tuning::{
    comparison::Comparison,
    records::{TunedChoice, TunedRecord, TunedWay, TuningEnvironment, TuningRecords},
    report::{
        CandidateFailure, CandidateOutcome, CandidateReport, ChoiceReport, ComparisonReport,
        DropReason, DroppedChoice, FailureStage, PlanCheckReport, PlanSearchReport,
    },
};

// What must be this much faster in one timing before two plans are worth timing in rounds, which `Comparison` then decides.
const WORTH_COMPARING: f64 = 0.97;

// How many of a choice point's fastest candidates are timed in rounds, since the fastest of many alike is only the luckiest.
const FINALISTS: usize = 8;

// How many mutations in a row may find nothing new before the search stops, having seen what it can reach.
const FRUITLESS_MUTATIONS: usize = 64;

// One candidate of a choice point, as the records that force it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub choices: Vec<(String, TunedChoice)>,
}

impl Candidate {
    // The way this candidate runs its nodes, if it says.
    fn way(&self) -> Option<TunedWay> {
        self.choices.iter().find_map(|(_, choice)| match choice {
            TunedChoice::Way(way) => Some(*way),
            _ => None,
        })
    }

    fn describe(&self) -> String {
        if self.choices.is_empty() {
            return "default".to_owned();
        }
        self.choices
            .iter()
            .map(|(_, choice)| format!("{choice:?}"))
            .collect::<Vec<_>>()
            .join(" + ")
    }
}

// A candidate's values judged, its time, and what it allocates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measured {
    pub agrees: bool,
    pub median: Duration,
    pub bytes: u64,
    // Its timing stopped early, once it had already lost.
    pub abandoned: bool,
}

// What running one candidate can come to, short of being judged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasureError {
    Failed(CandidateFailure),
    OverMemory(u64),
    // A library refused the configuration this candidate states.
    Refused,
}

impl From<CandidateFailure> for MeasureError {
    fn from(failure: CandidateFailure) -> Self {
        Self::Failed(failure)
    }
}

// A candidate that ran and agreed, which search may move to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alternative {
    pub candidate: Candidate,
    pub median_ns: u64,
    pub bytes: u64,
}

// One choice point, with what its candidates came to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChoicePoint {
    pub key: String,
    // The first candidate's time, which nothing has when it failed or disagreed.
    pub default_ns: Option<u64>,
    pub default_bytes: Option<u64>,
    pub alternatives: Vec<Alternative>,
    // The alternative this choice point keeps, if any.
    pub chosen: Option<usize>,
}

// Which alternative each choice point runs, or none for the path with no record.
pub type Plan = Vec<Option<usize>>;

fn nanoseconds(time: Duration) -> u64 {
    u64::try_from(time.as_nanos()).unwrap_or(u64::MAX)
}

// Whether `time` is measurably faster than `other`.
fn faster(time: Duration, other: Duration) -> bool {
    time.as_secs_f64() < other.as_secs_f64() * WORTH_COMPARING
}

// Whether `time` is measurably slower than `other`.
fn slower(time: Duration, other: Duration) -> bool {
    faster(other, time)
}

// Times `candidates` of one choice point, the first being what runs with no record, and returns what they came to.
// The fastest is only provisional until `confirm` times it against the first, and every candidate is free to win when the first failed or disagreed, since no record would then run something broken.
pub fn decide(
    key: String,
    candidates: &[Candidate],
    budget: Duration,
    measure: impl FnMut(&Candidate) -> Result<Measured, MeasureError>,
) -> (ChoicePoint, ChoiceReport) {
    decide_in_two(key, candidates, candidates.len(), budget, measure)
}

// As `decide`, with the candidates from `second` on taking a budget of their own, so what is costly to time and comes last still gets its turn.
pub fn decide_in_two(
    key: String,
    candidates: &[Candidate],
    second: usize,
    budget: Duration,
    mut measure: impl FnMut(&Candidate) -> Result<Measured, MeasureError>,
) -> (ChoicePoint, ChoiceReport) {
    let mut started = Instant::now();
    let mut point = ChoicePoint {
        key: key.clone(),
        default_ns: None,
        default_bytes: None,
        alternatives: Vec::new(),
        chosen: None,
    };
    let mut reports = Vec::with_capacity(candidates.len());
    let mut timed_before_second = 0;
    for (index, candidate) in candidates.iter().enumerate() {
        if index == second {
            started = Instant::now();
            timed_before_second = point.alternatives.len();
        }

        // Each part times at least one of its candidates.
        let timed_in_part = point.alternatives.len() > timed_before_second;
        let outcome = if started.elapsed() > budget && timed_in_part {
            CandidateOutcome::Skipped
        } else {
            let measured = measure(candidate);
            if let Ok(measured) = &measured
                && measured.agrees
            {
                let median_ns = nanoseconds(measured.median);
                if index == 0 {
                    point.default_ns = Some(median_ns);
                    point.default_bytes = Some(measured.bytes);
                }

                point.alternatives.push(Alternative {
                    candidate: candidate.clone(),
                    median_ns,
                    bytes: measured.bytes,
                });
            }

            outcome_of(&measured)
        };
        reports.push(CandidateReport {
            choice: candidate.describe(),
            outcome,
        });
    }

    let fastest = point
        .alternatives
        .iter()
        .enumerate()
        .min_by_key(|(_, alternative)| alternative.median_ns)
        .map(|(index, _)| index);
    point.chosen = fastest;
    let report = ChoiceReport {
        key,
        candidates: reports,
        failure: None,
        comparisons: Vec::new(),
        between_winners: Vec::new(),
        second_reference: None,
    };
    (point, report)
}

// What came of a comparison, for a report to say.
fn compared_or_failed(
    compared: Result<Comparison, MeasureError>,
) -> (Option<Comparison>, Option<CandidateFailure>) {
    match compared {
        Ok(comparison) => (Some(comparison), None),
        Err(MeasureError::Failed(failure)) => (None, Some(failure)),
        Err(MeasureError::OverMemory(bytes)) => (
            None,
            Some(CandidateFailure::at(
                FailureStage::Run,
                format!("refused {bytes} bytes"),
            )),
        ),
        Err(MeasureError::Refused) => (
            None,
            Some(CandidateFailure::at(
                FailureStage::Run,
                "a library refused its configuration",
            )),
        ),
    }
}

// Settles what `decide` found: the first stands unless one of the fastest wins when both are timed in rounds, which `compare` does, given what stands and then the candidate.
// A later winner must also win against the winner it would replace, since a cost both sides share moves a ratio between one comparison and the next, so two ratios against the first cannot say which of two winners is faster.
pub fn confirm(
    point: &mut ChoicePoint,
    report: &mut ChoiceReport,
    mut compare: impl FnMut(&Candidate, &Candidate) -> Result<Comparison, MeasureError>,
) {
    let Some(default_ns) = point.default_ns else {
        return;
    };

    let mut finalists = (1..point.alternatives.len())
        .filter(|&index| point.alternatives[index].median_ns < default_ns)
        .collect::<Vec<_>>();
    finalists.sort_by_key(|&index| point.alternatives[index].median_ns);
    finalists.truncate(FINALISTS);
    // In the order they were listed, which does not depend on how fast one timing made them look.
    finalists.sort_unstable();
    let first = point.alternatives[0].candidate.clone();
    let mut best = None;
    for index in finalists {
        let candidate = &point.alternatives[index].candidate;
        let (comparison, failure) = compared_or_failed(compare(&first, candidate));
        report.comparisons.push(ComparisonReport {
            choice: candidate.describe(),
            comparison,
            failure,
        });
        if !comparison.is_some_and(|comparison| comparison.wins()) {
            continue;
        }

        let Some(standing) = best else {
            best = Some(index);
            continue;
        };
        let standing = &point.alternatives[standing].candidate;
        let (comparison, failure) = compared_or_failed(compare(standing, candidate));
        report.between_winners.push(ComparisonReport {
            choice: format!("{} over {}", candidate.describe(), standing.describe()),
            comparison,
            failure,
        });
        if comparison.is_some_and(|comparison| comparison.wins()) {
            best = Some(index);
        }
    }

    point.chosen = Some(best.unwrap_or(0));
}

fn outcome_of(measured: &Result<Measured, MeasureError>) -> CandidateOutcome {
    match measured {
        Err(MeasureError::Failed(failure)) => CandidateOutcome::Failed(failure.clone()),
        Err(MeasureError::OverMemory(bytes)) => CandidateOutcome::OverMemory { bytes: *bytes },
        Err(MeasureError::Refused) => CandidateOutcome::Refused,
        Ok(measured) if !measured.agrees => CandidateOutcome::Disagreed,
        Ok(measured) => CandidateOutcome::Timed {
            median_ns: nanoseconds(measured.median),
            bytes: measured.bytes,
        },
    }
}

// What runs whole plans and says what they did, which is all that the logic here needs of a device.
pub trait PlanRunner {
    // Identifies what these records make the graph run, so a plan is timed once.
    //
    // # Errors
    //
    // Returns why the plan could not be written or compiled.
    fn fingerprint(&mut self, records: &TuningRecords) -> Result<String, CandidateFailure>;

    // Runs the graph under `records`, stopping its timing early if its first time passes `abandon_above`.
    //
    // # Errors
    //
    // Returns why the plan could not be built or run.
    fn run(
        &mut self,
        records: &TuningRecords,
        abandon_above: Option<Duration>,
    ) -> Result<Measured, CandidateFailure>;

    // Times `candidate` against `baseline` in rounds.
    //
    // # Errors
    //
    // Returns why either could not be built or run.
    fn compare(
        &mut self,
        baseline: &TuningRecords,
        candidate: &TuningRecords,
    ) -> Result<Comparison, CandidateFailure>;
}

// The records a plan makes.
#[must_use]
pub fn plan_records(
    points: &[ChoicePoint],
    plan: &[Option<usize>],
    environment: Option<&TuningEnvironment>,
) -> TuningRecords {
    let mut records = TuningRecords::new(environment.cloned());
    for (point, chosen) in points.iter().zip(plan) {
        let Some(alternative) = chosen.and_then(|index| point.alternatives.get(index)) else {
            continue;
        };
        for (key, choice) in &alternative.candidate.choices {
            records.insert(
                key.clone(),
                TunedRecord {
                    choice: choice.clone(),
                    default_ns: point.default_ns,
                    best_ns: alternative.median_ns,
                    bytes: alternative.bytes,
                },
            );
        }
    }
    records
}

// What a check of a whole plan may do.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlanLimits {
    pub memory: Option<u64>,
    // Whether a plan slower than the path with no records loses its records.
    pub speed: bool,
}

struct Checker<'a, R: PlanRunner> {
    points: &'a [ChoicePoint],
    plan: Plan,
    environment: Option<&'a TuningEnvironment>,
    runner: &'a mut R,
    dropped: Vec<DroppedChoice>,
}

impl<R: PlanRunner> Checker<'_, R> {
    fn run(&mut self, plan: &[Option<usize>]) -> Result<Measured, CandidateFailure> {
        let records = plan_records(self.points, plan, self.environment);
        self.runner.run(&records, None)
    }

    fn drop_choice(&mut self, index: usize, reason: DropReason) {
        self.plan[index] = None;
        self.dropped.push(DroppedChoice {
            key: self.points[index].key.clone(),
            reason,
        });
    }

    // Keeps those of `testing` that hold together with `kept`, halving to find those at fault, which it drops.
    fn keep_what_holds(&mut self, kept: &mut Vec<usize>, testing: &[usize]) {
        if testing.is_empty() {
            return;
        }

        let mut plan = vec![None; self.plan.len()];
        for &index in kept.iter().chain(testing) {
            plan[index] = self.plan[index];
        }

        let reason = match self.run(&plan) {
            Err(failure) => DropReason::Failed(failure),
            Ok(measured) if !measured.agrees => DropReason::Disagreed,
            Ok(_) => {
                kept.extend_from_slice(testing);
                return;
            }
        };
        if let [only] = testing {
            self.drop_choice(*only, reason);
            return;
        }

        let (left, right) = testing.split_at(testing.len() / 2);
        self.keep_what_holds(kept, left);
        self.keep_what_holds(kept, right);
    }

    // Drops what adds the most bytes over the path with no records, until what is left fits.
    fn fit(&mut self, kept: &[usize], limit: u64) -> Result<Measured, CandidateFailure> {
        let mut adding = kept.to_vec();
        adding.sort_by_key(|&index| {
            let point = &self.points[index];
            let bytes = self.plan[index]
                .and_then(|chosen| point.alternatives.get(chosen))
                .map_or(0, |alternative| alternative.bytes);
            Reverse(bytes.saturating_sub(point.default_bytes.unwrap_or(0)))
        });
        let mut measured = self.run(&self.plan.clone())?;
        for index in adding {
            if measured.bytes <= limit {
                break;
            }

            self.drop_choice(index, DropReason::OverMemory);
            measured = self.run(&self.plan.clone())?;
        }
        Ok(measured)
    }

    // Whether the plan is slower than the path with no records when both are timed in rounds, since one timing can be noise.
    fn is_slower(&mut self, plan_time: Duration, default_time: Duration) -> bool {
        if !slower(plan_time, default_time) {
            return false;
        }

        let empty = plan_records(self.points, &vec![None; self.plan.len()], self.environment);
        let plan = plan_records(self.points, &self.plan, self.environment);
        self.runner
            .compare(&empty, &plan)
            .map_or(true, |comparison| comparison.loses())
    }
}

// Runs the plan as a whole, against the path with no records, and returns the plan that holds, which loses what is at fault.
// What this returns does not fail, disagree, or exceed the limit, unless the path with no records does.
pub fn check_plan<R: PlanRunner>(
    points: &[ChoicePoint],
    plan: Plan,
    environment: Option<&TuningEnvironment>,
    limits: PlanLimits,
    runner: &mut R,
) -> (Plan, PlanCheckReport) {
    let mut report = PlanCheckReport::default();
    let chosen = (0..plan.len())
        .filter(|&index| plan[index].is_some())
        .collect::<Vec<_>>();
    if chosen.is_empty() {
        return (plan, report);
    }

    let empty = vec![None; plan.len()];
    let mut checker = Checker {
        points,
        plan,
        environment,
        runner,
        dropped: Vec::new(),
    };
    let default = match checker.run(&empty) {
        Ok(default) => {
            report.default_ns = Some(nanoseconds(default.median));
            report.default_bytes = Some(default.bytes);
            report.default_disagreed = !default.agrees;
            Some(default)
        }
        Err(failure) => {
            report.default_failure = Some(failure);
            None
        }
    };
    if let (Some(limit), Some(default)) = (limits.memory, &default)
        && default.bytes > limit
    {
        report.default_over_limit = true;
        for index in chosen {
            checker.drop_choice(index, DropReason::OverMemory);
        }

        report.dropped = checker.dropped;
        return (empty, report);
    }

    let mut kept = Vec::new();
    checker.keep_what_holds(&mut kept, &chosen);
    let mut measured = match limits.memory {
        Some(limit) => checker.fit(&kept, limit),
        None => checker.run(&checker.plan.clone()),
    };
    // What held in pieces should hold whole, and when it does not, nothing here can say which is at fault.
    let broken = match &measured {
        Ok(run) if run.agrees => None,
        Ok(_) => Some(DropReason::Disagreed),
        Err(failure) => Some(DropReason::Failed(failure.clone())),
    };
    let lose_everything = broken.or_else(|| {
        let plan_time = measured.as_ref().ok()?.median;
        let default = default.as_ref().filter(|default| default.agrees)?;
        (limits.speed
            && checker.plan.iter().any(Option::is_some)
            && checker.is_slower(plan_time, default.median))
        .then_some(DropReason::Slower)
    });
    if let Some(reason) = lose_everything {
        for index in 0..checker.plan.len() {
            if checker.plan[index].is_some() {
                checker.drop_choice(index, reason.clone());
            }
        }

        measured = default.ok_or_else(|| CandidateFailure::at(FailureStage::Run, "no default"));
    }

    if let Ok(run) = &measured {
        report.plan_ns = Some(nanoseconds(run.median));
        report.plan_bytes = Some(run.bytes);
    }

    report.dropped = checker.dropped;
    (checker.plan, report)
}

// A small generator, so one seed gives one search everywhere.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).unwrap_or(u64::MAX).max(1);
        usize::try_from(self.next() % bound).unwrap_or(0)
    }
}

// How a search of whole plans runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanSearchOptions {
    pub budget: Duration,
    pub seed: u64,
    // How many choice points a mutation moves.
    pub mutations: usize,
    pub memory_limit: Option<u64>,
}

// Plans moving every choice point that can to one way of running, tried first because independent moves rarely find a coherent plan on a repeated graph.
fn correlated_plans(points: &[ChoicePoint], start: &[Option<usize>]) -> Vec<Plan> {
    let mut ways = Vec::new();
    for point in points {
        for alternative in &point.alternatives {
            if let Some(way) = alternative.candidate.way()
                && !ways.contains(&way)
            {
                ways.push(way);
            }
        }
    }
    ways.iter()
        .map(|&way| {
            points
                .iter()
                .zip(start)
                .map(|(point, &chosen)| {
                    point
                        .alternatives
                        .iter()
                        .position(|alternative| alternative.candidate.way() == Some(way))
                        .or(chosen)
                })
                .collect()
        })
        .collect()
}

fn mutated(
    points: &[ChoicePoint],
    plan: &[Option<usize>],
    random: &mut SplitMix,
    mutations: usize,
) -> Plan {
    let movable = points
        .iter()
        .enumerate()
        .filter(|(_, point)| point.alternatives.len() > 1)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut next = plan.to_vec();
    if movable.is_empty() {
        return next;
    }

    for _ in 0..mutations.max(1) {
        let index = movable[random.below(movable.len())];
        next[index] = Some(random.below(points[index].alternatives.len()));
    }
    next
}

// Searches whole plans from `start`, moving only among alternatives that agreed alone, and returns the best that agreed as a whole.
pub fn search_plans<R: PlanRunner>(
    points: &[ChoicePoint],
    start: Plan,
    environment: Option<&TuningEnvironment>,
    options: PlanSearchOptions,
    runner: &mut R,
) -> (Plan, PlanSearchReport) {
    let mut report = PlanSearchReport::default();
    let started = Instant::now();
    let mut best = start;
    let mut best_records = plan_records(points, &best, environment);
    if options.budget.is_zero() {
        return (best, report);
    }

    let mut seen = HashSet::new();
    let mut best_time = match runner.fingerprint(&best_records) {
        Ok(fingerprint) => {
            seen.insert(fingerprint);
            match runner.run(&best_records, None) {
                Ok(measured) if measured.agrees => measured.median,
                _ => return (best, report),
            }
        }
        Err(failure) => {
            report.first_failure = Some(failure);
            return (best, report);
        }
    };
    report.plans_run = 1;
    report.start_ns = Some(nanoseconds(best_time));
    let mut random = SplitMix(options.seed);
    let mut queued = correlated_plans(points, &best);
    queued.reverse();
    let mut fruitless = 0;
    while started.elapsed() < options.budget && fruitless < FRUITLESS_MUTATIONS {
        let mutation = queued.is_empty();
        let next = queued
            .pop()
            .unwrap_or_else(|| mutated(points, &best, &mut random, options.mutations));
        let records = plan_records(points, &next, environment);
        let fingerprint = match runner.fingerprint(&records) {
            Ok(fingerprint) => fingerprint,
            Err(failure) => {
                report.failed += 1;
                report.first_failure.get_or_insert(failure);
                fruitless += usize::from(mutation);
                continue;
            }
        };
        if !seen.insert(fingerprint) {
            report.duplicates += 1;
            fruitless += usize::from(mutation);
            continue;
        }

        fruitless = 0;
        report.plans_run += 1;
        let abandon_above = Some(best_time.mul_f64(2.0));
        match runner.run(&records, abandon_above) {
            Err(failure) => {
                report.failed += 1;
                report.first_failure.get_or_insert(failure);
            }
            Ok(measured) if !measured.agrees => report.disagreed += 1,
            Ok(measured)
                if options
                    .memory_limit
                    .is_some_and(|limit| measured.bytes > limit) =>
            {
                report.over_memory += 1;
            }
            Ok(measured) if measured.abandoned => report.abandoned += 1,
            Ok(measured) => {
                if measured.median < best_time
                    && runner
                        .compare(&best_records, &records)
                        .is_ok_and(|comparison| comparison.wins())
                {
                    best = next;
                    best_records = records;
                    best_time = measured.median;
                    report.accepted += 1;
                }
            }
        }
    }

    report.end_ns = Some(nanoseconds(best_time));
    (best, report)
}
