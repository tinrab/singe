use std::{collections::HashMap, thread, time::Duration};

use super::{
    comparison::Comparison,
    plan::{
        Alternative, Candidate, ChoicePoint, MeasureError, Measured, Plan, PlanLimits, PlanRunner,
        PlanSearchOptions, check_plan, confirm, decide, decide_in_two, plan_records, search_plans,
    },
    records::{TunedChoice, TunedWay, TuningRecords},
    report::{CandidateFailure, CandidateOutcome, DropReason, FailureStage, TuningReport},
};
use crate::backend::registry::CudaGraphProvider;

fn way(key: &str, way: TunedWay) -> Candidate {
    Candidate {
        choices: vec![(key.to_owned(), TunedChoice::Way(way))],
    }
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "this answers for a measurement, which can fail"
)]
fn timed(median_ms: u64) -> Result<Measured, MeasureError> {
    Ok(Measured {
        agrees: true,
        median: Duration::from_millis(median_ms),
        bytes: 0,
        abandoned: false,
    })
}

fn failed() -> CandidateFailure {
    CandidateFailure::at(FailureStage::Run, "broke")
}

fn candidates() -> Vec<Candidate> {
    vec![
        way("a", TunedWay::Library(CudaGraphProvider::Cublas)),
        way("a", TunedWay::Library(CudaGraphProvider::CublasLt)),
        way("a", TunedWay::Generated),
    ]
}

fn showing(median_ppm: u32) -> Comparison {
    Comparison {
        median_ppm,
        lower_quartile_ppm: median_ppm - 1_000,
        upper_quartile_ppm: median_ppm + 1_000,
        together: true,
    }
}

#[test]
fn test_a_win_takes_both_size_and_consistency() {
    let ms = |times: &[u64]| {
        times
            .iter()
            .map(|&time| Duration::from_millis(time))
            .collect::<Vec<_>>()
    };
    let compared = |baseline: &[u64], candidate: &[u64]| {
        Comparison::of(&ms(baseline), &ms(candidate)).unwrap()
    };
    assert!(compared(&[100, 100, 100, 100, 100], &[90, 90, 90, 90, 90]).wins());
    // Too small to count, however consistent.
    assert!(!compared(&[100, 100, 100, 100, 100], &[98, 98, 98, 98, 98]).wins());
    // Large in the median round, and slower in most of the others' spread.
    let scattered = compared(&[100, 100, 100, 100, 100], &[60, 80, 90, 120, 130]);
    assert_eq!(scattered.median_ppm, 900_000);
    assert!(!scattered.wins());
    // Rounds pair in order, so drift that moves both cancels.
    let drifting = compared(&[100, 200, 300, 400, 500], &[90, 180, 270, 360, 450]);
    assert_eq!(drifting.median_ppm, 900_000);
    assert!(drifting.wins());

    // Going first costs the same whichever goes first, which reading rounds two at a time cancels, as one at a time does not.
    let first = ms(&[120, 100, 120, 100, 120, 100, 120, 100]);
    let second = ms(&[100, 120, 100, 120, 100, 120, 100, 120]);
    assert_eq!(
        Comparison::of_alternating(&first, &second)
            .unwrap()
            .median_ppm,
        1_000_000
    );
    assert_ne!(
        Comparison::of(&first, &second).unwrap().lower_quartile_ppm,
        1_000_000
    );

    // A pair of instances that stays apart counts for one, and a win must hold in every pair.
    let pair = |median_ppm| showing(median_ppm);
    assert!(
        Comparison::across(&[pair(900_000), pair(950_000), pair(920_000)])
            .unwrap()
            .wins()
    );
    let one_lucky = Comparison::across(&[pair(900_000), pair(950_000), pair(1_010_000)]).unwrap();
    assert_eq!(one_lucky.median_ppm, 950_000);
    assert!(!one_lucky.wins());
    // A small win that noise carries past the margin in the median, which real Qwen3 once kept at the cost of 587 MB.
    let inflated = Comparison::across(&[
        pair(937_502),
        pair(945_000),
        pair(953_588),
        pair(960_000),
        pair(988_279),
    ])
    .unwrap();
    assert!(inflated.median_ppm < 970_000);
    assert!(!inflated.wins());
    assert_eq!(Comparison::across(&[]), None);
    let inverted = pair(800_000).inverted();
    assert_eq!(inverted.median_ppm, 1_250_000);
    assert!(inverted.lower_quartile_ppm < inverted.upper_quartile_ppm);

    assert!(compared(&[90, 90, 90, 90, 90], &[100, 100, 100, 100, 100]).loses());
    assert!(!compared(&[100, 100, 100, 100, 100], &[102, 102, 102, 102, 102]).loses());
    assert_eq!(Comparison::of(&[], &[]), None);
}

#[test]
fn test_what_decides_is_timing_in_rounds_against_what_stands() {
    let decided = |showing_of: fn(&Candidate, &Candidate) -> Comparison| {
        let mut times = [100, 99, 98].into_iter();
        let (mut point, mut report) = decide("a".to_owned(), &candidates(), Duration::MAX, |_| {
            timed(times.next().unwrap())
        });
        confirm(&mut point, &mut report, |standing, candidate| {
            Ok(showing_of(standing, candidate))
        });
        (point, report)
    };
    // What one timing showed faster stays with the first when rounds do not show it.
    let (point, report) = decided(|_, _| showing(990_000));
    assert_eq!(point.chosen, Some(0));
    assert_eq!(report.comparisons.len(), 2);
    assert!(report.between_winners.is_empty());
    // Winners tied against each other stay with the one listed first, even when their ratios against the first differ by more than the margin, as a cost they share can make them.
    let (point, report) = decided(|standing, candidate| {
        if standing != &candidates()[0] {
            showing(990_000)
        } else if candidate == &candidates()[1] {
            showing(900_000)
        } else {
            showing(800_000)
        }
    });
    assert_eq!(point.chosen, Some(1));
    assert_eq!(report.between_winners.len(), 1);
    // A later winner takes over by winning against the one it would replace, even if its ratio against the first looked worse.
    let (point, report) = decided(|standing, candidate| {
        if standing != &candidates()[0] {
            showing(900_000)
        } else if candidate == &candidates()[1] {
            showing(800_000)
        } else {
            showing(900_000)
        }
    });
    assert_eq!(point.chosen, Some(2));
    let whole = TuningReport {
        choices: vec![report],
        ..TuningReport::default()
    };
    assert_eq!(whole.counts().between_winners, 1);
    assert_eq!(whole.counts().replaced, 1);
}

#[test]
fn test_a_candidate_slower_in_one_timing_costs_no_comparison() {
    let mut times = [100, 110, 120].into_iter();
    let (mut point, mut report) = decide("a".to_owned(), &candidates(), Duration::MAX, |_| {
        timed(times.next().unwrap())
    });
    confirm(&mut point, &mut report, |_, _| unreachable!());
    assert_eq!(point.chosen, Some(0));
    assert!(report.comparisons.is_empty());
}

#[test]
fn test_a_comparison_that_fails_loses_and_says_so() {
    let two = &candidates()[..2];
    let mut times = [100, 50].into_iter();
    let (mut point, mut report) = decide("a".to_owned(), two, Duration::MAX, |_| {
        timed(times.next().unwrap())
    });
    confirm(&mut point, &mut report, |_, _| {
        Err(MeasureError::Failed(failed()))
    });
    assert_eq!(point.chosen, Some(0));
    assert!(report.comparisons[0].failure.is_some());
    let whole = TuningReport {
        choices: vec![report],
        ..TuningReport::default()
    };
    assert_eq!(whole.counts().failed, 1);

    // One that fails against the winner it would replace leaves that winner standing.
    let mut times = [100, 99, 98].into_iter();
    let (mut point, mut report) = decide("a".to_owned(), &candidates(), Duration::MAX, |_| {
        timed(times.next().unwrap())
    });
    confirm(&mut point, &mut report, |standing, _| {
        if standing == &candidates()[0] {
            Ok(showing(900_000))
        } else {
            Err(MeasureError::Failed(failed()))
        }
    });
    assert_eq!(point.chosen, Some(1));
    assert!(report.between_winners[0].failure.is_some());
    let whole = TuningReport {
        choices: vec![report],
        ..TuningReport::default()
    };
    assert_eq!(whole.counts().failed, 1);
}

// The second candidate is no reference: it is judged like the rest, and with the first gone the fastest that agrees wins.
#[test]
fn test_a_first_candidate_that_fails_makes_no_other_the_reference() {
    let mut results = vec![
        Err(MeasureError::Failed(failed())),
        Ok(Measured {
            agrees: false,
            median: Duration::from_millis(1),
            bytes: 0,
            abandoned: false,
        }),
        timed(99),
    ]
    .into_iter();
    let (point, report) = decide("a".to_owned(), &candidates(), Duration::MAX, |_| {
        results.next().unwrap()
    });
    assert_eq!(point.default_ns, None);
    assert_eq!(point.alternatives.len(), 1);
    assert_eq!(point.chosen, Some(0));
    assert_eq!(
        point.alternatives[0].candidate,
        way("a", TunedWay::Generated)
    );
    assert_eq!(
        report.candidates[0].outcome,
        CandidateOutcome::Failed(failed())
    );
    assert_eq!(report.candidates[1].outcome, CandidateOutcome::Disagreed);
    assert!(matches!(
        report.candidates[2].outcome,
        CandidateOutcome::Timed { .. }
    ));
}

#[test]
fn test_every_candidate_leaves_its_outcome_in_the_report() {
    let mut results = vec![
        timed(5),
        Err(MeasureError::OverMemory(9)),
        Err(MeasureError::Failed(failed())),
    ]
    .into_iter();
    let (point, report) = decide("a".to_owned(), &candidates(), Duration::MAX, |_| {
        results.next().unwrap()
    });
    assert_eq!(point.chosen, Some(0));
    assert_eq!(
        report.candidates[1].outcome,
        CandidateOutcome::OverMemory { bytes: 9 }
    );
    assert_eq!(
        report.candidates[2].outcome,
        CandidateOutcome::Failed(failed())
    );
    // A candidate past the budget is skipped, once something has been timed.
    let (_, report) = decide("a".to_owned(), &candidates(), Duration::ZERO, |_| {
        thread::sleep(Duration::from_millis(1));
        timed(5)
    });
    assert!(matches!(
        report.candidates[0].outcome,
        CandidateOutcome::Timed { .. }
    ));
    assert_eq!(report.candidates[1].outcome, CandidateOutcome::Skipped);
    assert_eq!(report.candidates[2].outcome, CandidateOutcome::Skipped);
}

// Choice points named `p0`, `p1`, and so on, whose alternatives are `Generated` then `Separate`.
fn points(count: usize, bytes: &[u64]) -> Vec<ChoicePoint> {
    (0..count)
        .map(|index| {
            let key = format!("p{index}");
            ChoicePoint {
                key: key.clone(),
                default_ns: Some(100),
                default_bytes: Some(0),
                alternatives: vec![
                    Alternative {
                        candidate: way(&key, TunedWay::Generated),
                        median_ns: 100,
                        bytes: bytes.get(index).copied().unwrap_or(0),
                    },
                    Alternative {
                        candidate: way(&key, TunedWay::Separate),
                        median_ns: 100,
                        bytes: 0,
                    },
                ],
                chosen: Some(0),
            }
        })
        .collect()
}

fn chosen(points: &[ChoicePoint]) -> Plan {
    points.iter().map(|point| point.chosen).collect()
}

// A fake device: what a plan does is what `behave` says of its records.
struct Fake<F> {
    behave: F,
    runs: usize,
    fingerprints: HashMap<String, usize>,
    delay: Duration,
}

impl<F: FnMut(&TuningRecords) -> Result<Measured, CandidateFailure>> Fake<F> {
    fn new(behave: F) -> Self {
        Self {
            behave,
            runs: 0,
            fingerprints: HashMap::new(),
            delay: Duration::ZERO,
        }
    }
}

impl<F: FnMut(&TuningRecords) -> Result<Measured, CandidateFailure>> PlanRunner for Fake<F> {
    fn fingerprint(&mut self, records: &TuningRecords) -> Result<String, CandidateFailure> {
        let print = format!("{:?}", records.entries().collect::<Vec<_>>());
        *self.fingerprints.entry(print.clone()).or_default() += 1;
        Ok(print)
    }

    fn run(
        &mut self,
        records: &TuningRecords,
        _abandon_above: Option<Duration>,
    ) -> Result<Measured, CandidateFailure> {
        self.runs += 1;
        thread::sleep(self.delay);
        (self.behave)(records)
    }

    fn compare(
        &mut self,
        baseline: &TuningRecords,
        candidate: &TuningRecords,
    ) -> Result<Comparison, CandidateFailure> {
        let baseline = self.run(baseline, None)?.median;
        let candidate = self.run(candidate, None)?.median;
        Ok(Comparison::of(&[baseline], &[candidate]).unwrap())
    }
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "this answers for a plan's run, which can fail"
)]
fn plain(median_ms: u64, bytes: u64) -> Result<Measured, CandidateFailure> {
    Ok(Measured {
        agrees: true,
        median: Duration::from_millis(median_ms),
        bytes,
        abandoned: false,
    })
}

fn wrong(mut measured: Result<Measured, CandidateFailure>) -> Result<Measured, CandidateFailure> {
    if let Ok(measured) = &mut measured {
        measured.agrees = false;
    }
    measured
}

fn has(records: &TuningRecords, key: &str, choice: TunedWay) -> bool {
    records.get(key) == Some(TunedChoice::Way(choice))
}

#[test]
fn test_one_bad_choice_among_many_is_all_a_plan_loses() {
    let points = points(8, &[]);
    let mut fake = Fake::new(|records: &TuningRecords| {
        if has(records, "p5", TunedWay::Generated) {
            wrong(plain(10, 0))
        } else if has(records, "p2", TunedWay::Generated) {
            Err(failed())
        } else {
            plain(10, 0)
        }
    });
    let (plan, report) = check_plan(
        &points,
        chosen(&points),
        None,
        PlanLimits::default(),
        &mut fake,
    );
    let expected = (0..8)
        .map(|index| (index != 2 && index != 5).then_some(0))
        .collect::<Plan>();
    assert_eq!(plan, expected);
    let reason = |key: &str| {
        report
            .dropped
            .iter()
            .find(|dropped| dropped.key == key)
            .map(|dropped| dropped.reason.clone())
    };
    assert_eq!(reason("p5"), Some(DropReason::Disagreed));
    assert_eq!(reason("p2"), Some(DropReason::Failed(failed())));
    assert_eq!(report.dropped.len(), 2);
}

#[test]
fn test_a_plan_slower_as_a_whole_loses_its_records() {
    let points = points(3, &[]);
    let mut fake = Fake::new(|records: &TuningRecords| plain(10 + records.len() as u64, 0));
    let (plan, report) = check_plan(
        &points,
        chosen(&points),
        None,
        PlanLimits {
            memory: None,
            speed: true,
        },
        &mut fake,
    );
    assert_eq!(plan, vec![None, None, None]);
    assert_eq!(report.dropped.len(), 3);
    assert!(
        report
            .dropped
            .iter()
            .all(|dropped| dropped.reason == DropReason::Slower)
    );
    // With speed left out of it, the same plan keeps its records.
    let mut fake = Fake::new(|records: &TuningRecords| plain(10 + records.len() as u64, 0));
    let (plan, _) = check_plan(
        &points,
        chosen(&points),
        None,
        PlanLimits::default(),
        &mut fake,
    );
    assert_eq!(plan, chosen(&points));
}

#[test]
fn test_a_plan_keeps_its_records_when_only_the_default_fails() {
    let points = points(2, &[]);
    let mut fake = Fake::new(|records: &TuningRecords| {
        if records.is_empty() {
            Err(failed())
        } else {
            plain(10, 0)
        }
    });
    let (plan, report) = check_plan(
        &points,
        chosen(&points),
        None,
        PlanLimits {
            memory: None,
            speed: true,
        },
        &mut fake,
    );
    assert_eq!(plan, chosen(&points));
    assert_eq!(report.default_failure, Some(failed()));
}

#[test]
fn test_what_adds_the_most_memory_goes_first_until_a_plan_fits() {
    let points = points(3, &[100, 400, 200]);
    let bytes = |records: &TuningRecords| {
        [("p0", 100), ("p1", 400), ("p2", 200)]
            .into_iter()
            .filter(|(key, _)| has(records, key, TunedWay::Generated))
            .map(|(_, bytes)| bytes)
            .sum::<u64>()
    };
    let limits = |memory| PlanLimits {
        memory: Some(memory),
        speed: false,
    };
    let mut fake = Fake::new(|records: &TuningRecords| plain(10, 50 + bytes(records)));
    let (plan, report) = check_plan(&points, chosen(&points), None, limits(400), &mut fake);
    assert_eq!(plan, vec![Some(0), None, Some(0)]);
    assert_eq!(report.plan_bytes, Some(350));
    assert_eq!(report.dropped.len(), 1);
    assert_eq!(report.dropped[0].reason, DropReason::OverMemory);
    assert!(!report.default_over_limit);
    // A limit under the default leaves nothing to choose from, and says so.
    let mut fake = Fake::new(|records: &TuningRecords| plain(10, 50 + bytes(records)));
    let (plan, report) = check_plan(&points, chosen(&points), None, limits(40), &mut fake);
    assert_eq!(plan, vec![None, None, None]);
    assert!(report.default_over_limit);
}

// Each choice point's second alternative is slower alone, yet the pair is fastest, which only a whole plan shows.
fn interacting(records: &TuningRecords) -> Result<Measured, CandidateFailure> {
    let first = has(records, "p0", TunedWay::Separate);
    let second = has(records, "p1", TunedWay::Separate);
    plain(
        match (first, second) {
            (false, false) => 100,
            (true, true) => 40,
            _ => 120,
        },
        0,
    )
}

fn search_options(budget: Duration, seed: u64) -> PlanSearchOptions {
    PlanSearchOptions {
        budget,
        seed,
        mutations: 2,
        memory_limit: None,
    }
}

#[test]
fn test_search_finds_a_pair_that_only_wins_together() {
    let points = points(2, &[]);
    let mut fake = Fake::new(interacting);
    let (plan, report) = search_plans(
        &points,
        chosen(&points),
        None,
        search_options(Duration::from_secs(5), 7),
        &mut fake,
    );
    assert_eq!(plan, vec![Some(1), Some(1)]);
    assert!(report.accepted >= 1);
    assert_eq!(report.start_ns, Some(100_000_000));
    assert_eq!(report.end_ns, Some(40_000_000));
    // It ran out of plans long before it ran out of time.
    assert!(report.duplicates > 0);
}

#[test]
fn test_search_times_a_fingerprint_once_and_follows_its_seed() {
    let run = |seed| {
        let points = points(4, &[]);
        let mut fake = Fake::new(|records: &TuningRecords| {
            // Every plan is new and none is faster, so nothing changes which moves come next.
            plain(100 + records.len() as u64, 0)
        });
        let (plan, report) = search_plans(
            &points,
            chosen(&points),
            None,
            search_options(Duration::from_secs(5), seed),
            &mut fake,
        );
        (plan, report.plans_run, fake.runs, fake.fingerprints)
    };
    let (plan, plans_run, runs, fingerprints) = run(3);
    assert_eq!(runs, plans_run);
    // The start, one or two moves from it, and the plan with every choice point moved.
    assert_eq!(plans_run, 1 + 4 + 6 + 1);
    assert_eq!(fingerprints.len(), plans_run);
    let (again, _, _, again_fingerprints) = run(3);
    assert_eq!(plan, again);
    assert_eq!(fingerprints, again_fingerprints);
}

#[test]
fn test_search_respects_its_budget() {
    let points = points(16, &[]);
    let mut fake = Fake::new(interacting);
    let (plan, report) = search_plans(
        &points,
        chosen(&points),
        None,
        search_options(Duration::ZERO, 1),
        &mut fake,
    );
    assert_eq!(plan, chosen(&points));
    assert_eq!(fake.runs, 0);
    assert_eq!(report.plans_run, 0);

    let mut fake = Fake::new(|_: &TuningRecords| plain(100, 0));
    fake.delay = Duration::from_millis(20);
    let (_, report) = search_plans(
        &points,
        chosen(&points),
        None,
        search_options(Duration::from_millis(100), 1),
        &mut fake,
    );
    assert!(report.plans_run <= 7, "{report:?}");
}

#[test]
fn test_search_never_accepts_a_plan_that_disagrees_or_does_not_fit() {
    let points = points(2, &[]);
    let mut fake = Fake::new(|records: &TuningRecords| wrong_when_paired(records, false));
    let (plan, report) = search_plans(
        &points,
        chosen(&points),
        None,
        search_options(Duration::from_secs(5), 1),
        &mut fake,
    );
    assert_eq!(plan, chosen(&points));
    assert!(report.disagreed >= 1);
    assert_eq!(report.accepted, 0);

    let mut fake = Fake::new(|records: &TuningRecords| wrong_when_paired(records, true));
    let mut options = search_options(Duration::from_secs(5), 1);
    options.memory_limit = Some(10);
    let (plan, report) = search_plans(&points, chosen(&points), None, options, &mut fake);
    assert_eq!(plan, chosen(&points));
    assert!(report.over_memory >= 1);
    assert_eq!(report.accepted, 0);
}

// The pair of second alternatives is fastest, but disagrees, or allocates too much.
fn wrong_when_paired(
    records: &TuningRecords,
    too_large: bool,
) -> Result<Measured, CandidateFailure> {
    let paired = has(records, "p0", TunedWay::Separate) && has(records, "p1", TunedWay::Separate);
    if !paired {
        return plain(100, 0);
    }
    if too_large {
        plain(1, 11)
    } else {
        wrong(plain(1, 0))
    }
}

#[test]
fn test_a_plan_makes_only_the_records_it_chose() {
    let points = points(2, &[]);
    let records = plan_records(&points, &[None, Some(1)], None);
    assert!(records.get("p0").is_none());
    assert!(has(&records, "p1", TunedWay::Separate));
}

// What comes last still gets its turn when the first candidates spend the budget, and each part times one at least.
#[test]
fn test_a_second_part_of_candidates_takes_a_budget_of_its_own() {
    let (_, report) = decide_in_two("a".to_owned(), &candidates(), 2, Duration::ZERO, |_| {
        thread::sleep(Duration::from_millis(1));
        Ok(Measured {
            median: Duration::from_millis(1),
            bytes: 0,
            agrees: true,
            abandoned: false,
        })
    });
    let timed = report
        .candidates
        .iter()
        .map(|candidate| matches!(candidate.outcome, CandidateOutcome::Timed { .. }))
        .collect::<Vec<_>>();
    assert_eq!(timed[..3], [true, false, true]);
}
