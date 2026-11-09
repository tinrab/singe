//! Searches a kernel's schedules for the fastest one whose outputs match the default schedule's.

use std::time::{Duration, Instant};

use half::{bf16, f16};
use singe_codegen::{
    abi::Bindings,
    ir::{ElementType, Kernel, OperandRole},
    schedule::{Schedule, space::candidates},
};

use crate::{
    backend::cuda::{
        codegen::{KernelArtifact, KernelDescription, emit_kernel},
        kernel_tuning::measure::{KernelBench, Timing},
        runtime::runtime_for_device,
        tuning::{CandidateFailure, Comparison, FailureStage},
    },
    device::Device,
    error::Result,
};

/// How a search runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TuneOptions {
    /// Whether Tile IR schedules are candidates; `tileiras` takes about 0.2 s per kernel, so searches at graph compile time leave them out.
    pub tile: bool,
    /// How many interleaved rounds each candidate is timed over.
    pub rounds: usize,
    /// The time after which no further candidates are compiled or timed.
    pub budget: Duration,
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self {
            tile: false,
            rounds: 25,
            budget: Duration::from_secs(10),
        }
    }
}

/// What a search found for one candidate schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Timed(Timing),
    /// The emitter rejected the schedule, or its kernel failed to compile or launch.
    Failed(CandidateFailure),
    /// Its outputs differ from the default schedule's beyond the outputs' tolerance.
    Mismatched,
    /// The budget ran out before it was timed.
    Skipped,
}

/// The candidates of one search and the schedule it chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningResult {
    pub default: Schedule,
    pub best: Schedule,
    /// Every candidate, the default first, with what the search found for it.
    pub candidates: Vec<(Schedule, Outcome)>,
}

impl TuningResult {
    /// Returns the timing of `schedule`, when the search timed it.
    #[must_use]
    pub fn timing(&self, schedule: Schedule) -> Option<Timing> {
        self.candidates
            .iter()
            .find_map(|(candidate, outcome)| match outcome {
                Outcome::Timed(timing) if *candidate == schedule => Some(*timing),
                _ => None,
            })
    }
}

/// Returns the fastest schedule for `kernel` with `bindings` on `device`, among `default`, which the result never does worse than, and the candidates its space lists.
///
/// Every candidate's outputs are compared with the default's before it is timed, and a candidate replaces the default only when `Comparison` says it wins.
///
/// # Errors
///
/// Returns an error if the default schedule fails or its buffers cannot be made.
pub fn tune(
    device: Device,
    kernel: &Kernel,
    default: Schedule,
    bindings: &Bindings,
    options: TuneOptions,
) -> Result<TuningResult> {
    let started = Instant::now();
    let mut bench = KernelBench::create(device, kernel, bindings)?;
    let reference_kernel = emit_kernel(kernel, default, bindings, &KernelDescription::default())?;
    let reference = bench.outputs(&reference_kernel)?;
    let mut schedules = vec![default];
    for schedule in candidates(kernel, &bindings.sizes, options.tile) {
        if !schedules.contains(&schedule) {
            schedules.push(schedule);
        }
    }

    let runtime = runtime_for_device(device)?;
    let mut outcomes = vec![Outcome::Skipped; schedules.len()];
    let mut timed = vec![(0, reference_kernel)];
    let emitted = schedules
        .iter()
        .map(|&schedule| emit_kernel(kernel, schedule, bindings, &KernelDescription::default()))
        .collect::<Vec<_>>();
    // CUDA C++ candidates compile together; a candidate that fails says why below, when it compiles alone.
    let _ = runtime.prepare_kernels(emitted.iter().flatten());
    let outputs = kernel
        .operands
        .iter()
        .filter(|operand| operand.role == OperandRole::Output)
        .map(|operand| operand.element)
        .collect::<Vec<_>>();
    for (index, generated) in emitted.into_iter().enumerate().skip(1) {
        if started.elapsed() > options.budget {
            break;
        }

        let generated = match generated {
            Ok(generated) => generated,
            Err(error) => {
                outcomes[index] =
                    Outcome::Failed(CandidateFailure::from_error(FailureStage::Export, &error));
                continue;
            }
        };
        match bench.outputs(&generated) {
            Err(error) => {
                outcomes[index] =
                    Outcome::Failed(CandidateFailure::from_error(FailureStage::Run, &error));
            }
            Ok(values) if !matches(&outputs, &values, &reference) => {
                outcomes[index] = Outcome::Mismatched;
            }
            Ok(_) => timed.push((index, generated)),
        }
    }

    let kernels = timed
        .iter()
        .map(|(_, generated)| generated.clone())
        .collect::<Vec<KernelArtifact>>();
    let rounds = bench.time_rounds(&kernels, options.rounds, false)?;
    let timings = rounds.iter().cloned().map(Timing::of).collect::<Vec<_>>();
    for ((index, _), timing) in timed.iter().zip(&timings) {
        outcomes[*index] = Outcome::Timed(*timing);
    }

    let best = timed
        .iter()
        .zip(&timings)
        .zip(&rounds)
        .min_by_key(|((_, timing), _)| timing.median)
        .filter(|(_, candidate)| {
            Comparison::of(&rounds[0], candidate).is_some_and(|comparison| comparison.wins())
        })
        .map_or(default, |(((index, _), _), _)| schedules[*index]);
    Ok(TuningResult {
        default,
        best,
        candidates: schedules.into_iter().zip(outcomes).collect(),
    })
}

/// Returns whether every output in `values` matches `reference` within its element type's tolerance: exact for integers and `bool`, and relative to the larger of one and the reference's magnitude for floats.
fn matches(elements: &[ElementType], values: &[Vec<u8>], reference: &[Vec<u8>]) -> bool {
    elements
        .iter()
        .zip(values.iter().zip(reference))
        .all(|(&element, (value, reference))| {
            let tolerance = match element {
                ElementType::F16 => 2e-3,
                ElementType::Bf16 => 1.6e-2,
                ElementType::F32 => 1e-4,
                ElementType::F64 => 1e-10,
                _ => return value == reference,
            };
            let (value, reference) = (decode(element, value), decode(element, reference));
            value.len() == reference.len()
                && value.iter().zip(&reference).all(|(value, reference)| {
                    value == reference
                        || (value.is_nan() && reference.is_nan())
                        || (value - reference).abs() <= tolerance * reference.abs().max(1.0)
                })
        })
}

fn decode(element: ElementType, bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(element.size_bytes())
        .map(|chunk| match element {
            ElementType::F16 => f16::from_le_bytes([chunk[0], chunk[1]]).to_f64(),
            ElementType::Bf16 => bf16::from_le_bytes([chunk[0], chunk[1]]).to_f64(),
            ElementType::F32 => {
                f64::from(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            }
            _ => f64::from_le_bytes(chunk.try_into().unwrap_or([0; 8])),
        })
        .collect()
}
