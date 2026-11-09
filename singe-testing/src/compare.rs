use std::fmt::{self, Display, Formatter};

use singe::{DataType, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    kind: Kind,
    reason: &'static str,
}

#[derive(Debug, Clone, Copy)]
enum Kind {
    Exact,
    Absolute(f32),
    Relative { absolute: f32, relative: f32 },
}

impl Tolerance {
    #[must_use]
    pub const fn exact(reason: &'static str) -> Self {
        Self {
            kind: Kind::Exact,
            reason,
        }
    }

    #[must_use]
    pub const fn absolute(bound: f32, reason: &'static str) -> Self {
        Self {
            kind: Kind::Absolute(bound),
            reason,
        }
    }

    #[must_use]
    pub const fn relative(absolute: f32, relative: f32, reason: &'static str) -> Self {
        Self {
            kind: Kind::Relative { absolute, relative },
            reason,
        }
    }

    fn allows(self, actual: f32, expected: f32) -> bool {
        if actual == expected {
            return true;
        }

        let difference = (actual - expected).abs();
        match self.kind {
            Kind::Exact => actual.to_bits() == expected.to_bits() || actual == expected,
            Kind::Absolute(bound) => difference <= bound,
            Kind::Relative { absolute, relative } => {
                difference <= relative.mul_add(expected.abs(), absolute)
            }
        }
    }
}

impl Display for Tolerance {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self.kind {
            Kind::Exact => write!(formatter, "exact")?,
            Kind::Absolute(bound) => write!(formatter, "within {bound}")?,
            Kind::Relative { absolute, relative } => {
                write!(formatter, "within {absolute} + {relative} of the reference")?;
            }
        }
        write!(formatter, ", since {}", self.reason)
    }
}

#[must_use]
pub fn host(tensor: &Tensor) -> Vec<f32> {
    tensor.cast(DataType::F32).unwrap().to_vec::<f32>().unwrap()
}

pub fn assert_matches(actual: &[f32], expected: &[f32], tolerance: Tolerance, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: element counts");
    let worst = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (actual, expected))| !tolerance.allows(**actual, **expected))
        .max_by(|(_, left), (_, right)| {
            (left.0 - left.1)
                .abs()
                .total_cmp(&(right.0 - right.1).abs())
        });
    if let Some((index, (actual, expected))) = worst {
        panic!(
            "{what}: element {index} is {actual} where the reference is {expected}, a difference of {}; the tolerance is {tolerance}",
            (actual - expected).abs()
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfGap {
    pub worst: f32,
    pub mean: f32,
}

fn distance(actual: &[f32], expected: &[f32]) -> HalfGap {
    assert_eq!(actual.len(), expected.len(), "element counts");
    let differences = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs());
    #[expect(
        clippy::cast_precision_loss,
        reason = "references hold few enough values"
    )]
    let count = actual.len().max(1) as f32;
    HalfGap {
        worst: differences.clone().fold(0.0, f32::max),
        mean: differences.sum::<f32>() / count,
    }
}

pub fn assert_within_half_gap(actual: &[f32], wide: &[f32], gap: HalfGap, factor: f32, what: &str) {
    let singe = distance(actual, wide);
    assert!(
        singe.worst <= factor * gap.worst && singe.mean <= factor * gap.mean,
        "{what}: {singe:?} from F32, where PyTorch's half precision is {gap:?}, and the bound is {factor} times it"
    );
}
