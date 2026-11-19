// The inverse of the piecewise rational quadratic spline VITS's duration flows apply, as `_unconstrained_rational_quadratic_spline` computes it in `F32` with `reverse=True`: identity outside `[-bound, bound]`, and inside, the root of each bin's quadratic.
use std::iter;

const MIN_BIN_WIDTH: f32 = 1e-3;
const MIN_BIN_HEIGHT: f32 = 1e-3;
const MIN_DERIVATIVE: f32 = 1e-3;

fn softmax(values: &[f32]) -> Vec<f32> {
    let most = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponents = values
        .iter()
        .map(|value| (value - most).exp())
        .collect::<Vec<_>>();
    let total = exponents.iter().sum::<f32>();
    exponents.iter().map(|value| value / total).collect()
}

// `log(1 + exp(x))`, linear past 20 as `softplus` is.
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        value.exp().ln_1p()
    }
}

// A bin edge's positions in `[-bound, bound]`, from unnormalized sizes, each at least `least`.
fn edges(unnormalized: &[f32], least: f32, bound: f32) -> Vec<f32> {
    let count = unnormalized.len() as f32;
    let mut cumulative = vec![0.0_f32];
    let mut total = 0.0_f32;
    for size in softmax(unnormalized) {
        total += least + (1.0 - least * count) * size;
        cumulative.push(total);
    }

    let mut edges = cumulative
        .into_iter()
        .map(|value| 2.0 * bound * value - bound)
        .collect::<Vec<_>>();
    edges[0] = -bound;
    let end = edges.len() - 1;
    edges[end] = bound;
    edges
}

// Inverts the spline at `input`, given one point's unnormalized widths and heights, `bins` each, and its unnormalized interior derivatives, `bins - 1`.
#[must_use]
pub fn inverse(
    input: f32,
    widths: &[f32],
    heights: &[f32],
    derivatives: &[f32],
    bound: f32,
) -> f32 {
    if input < -bound || input > bound {
        return input;
    }

    let width_edges = edges(widths, MIN_BIN_WIDTH, bound);
    let height_edges = edges(heights, MIN_BIN_HEIGHT, bound);
    // The outer derivatives are those that make the spline meet the identity's slope.
    #[expect(
        clippy::imprecise_flops,
        reason = "`transformers` computes `log(exp(1 - min) - 1)`"
    )]
    let edge = ((1.0 - f64::from(MIN_DERIVATIVE)).exp() - 1.0).ln() as f32;
    let slopes = iter::once(edge)
        .chain(derivatives.iter().copied())
        .chain(iter::once(edge))
        .map(|value| MIN_DERIVATIVE + softplus(value))
        .collect::<Vec<_>>();
    let mut locations = height_edges.clone();
    let last = locations.len() - 1;
    locations[last] += 1e-6;
    let bin = locations
        .iter()
        .filter(|&&location| input >= location)
        .count()
        .saturating_sub(1);
    let bin_width = width_edges[bin + 1] - width_edges[bin];
    let bin_height = height_edges[bin + 1] - height_edges[bin];
    let delta = bin_height / bin_width;
    let (slope, next_slope) = (slopes[bin], slopes[bin + 1]);
    let first = slope + next_slope - 2.0 * delta;
    let shifted = input - height_edges[bin];
    let scaled = shifted * first;
    let a = bin_height * (delta - slope) + scaled;
    let b = bin_height * slope - scaled;
    let c = -delta * shifted;
    let discriminant = b * b - 4.0 * a * c;
    let root = (2.0 * c) / (-b - discriminant.sqrt());
    root * bin_width + width_edges[bin]
}
