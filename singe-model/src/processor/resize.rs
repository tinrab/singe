//! Resizing of RGB bytes as `transformers`' processors resize them through torchvision on the CPU: `PyTorch`'s antialiased separable filter with weights quantized to `i16`, a horizontal pass into bytes and then a vertical one, ported from `aten/src/ATen/native/cpu/UpSampleKernel.cpp` and `UpSampleKernelAVXAntialias.h`.

use std::f64::consts::PI;

use rayon::prelude::*;

use crate::processor::image::{Image, Resample};

/// Each output position's first input, its weights, and the bits they are scaled by.
struct Weights {
    starts: Vec<usize>,
    rows: Vec<Vec<i32>>,
    precision: u32,
}

fn triangle(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 { 1.0 - x } else { 0.0 }
}

/// Evaluates Keys' cubic with `a = -0.5`, as PIL and antialiased `PyTorch` use it.
fn keys_cubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
    } else {
        0.0
    }
}

/// Evaluates Lanczos with three lobes, `sinc(x) · sinc(x / 3)`, as PIL and `PyTorch` use it.
fn lanczos(x: f64) -> f64 {
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            let x = x * PI;
            x.sin() / x
        }
    };
    let x = x.abs();
    if x < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

/// Returns the weights `_compute_index_ranges_int16_weights` gives a resize of `input` positions to `output`.
#[expect(
    clippy::cast_possible_wrap,
    reason = "image sides are far below 2^53, and the truncations are PyTorch's"
)]
fn weights(input: usize, output: usize, filter: fn(f64) -> f64, span: f64) -> Weights {
    let scale = input as f64 / output as f64;
    let support = if scale >= 1.0 {
        span * 0.5 * scale
    } else {
        span * 0.5
    };
    let most = support.ceil() as usize * 2 + 1;
    let inverse = if scale >= 1.0 { 1.0 / scale } else { 1.0 };
    let mut starts = Vec::with_capacity(output);
    let mut real = Vec::with_capacity(output);
    let mut largest = 0.0_f64;
    for index in 0..output {
        let center = scale * (index as f64 + 0.5);
        let start = ((center - support + 0.5) as i64).max(0) as usize;
        let end = ((center + support + 0.5) as i64).min(input as i64) as usize;
        let count = end.saturating_sub(start).min(most);
        let mut row = (0..count)
            .map(|offset| filter(((start + offset) as f64 - center + 0.5) * inverse))
            .collect::<Vec<f64>>();
        let total = row.iter().sum::<f64>();
        if total != 0.0 {
            for weight in &mut row {
                *weight /= total;
                largest = largest.max(*weight);
            }
        }

        starts.push(start);
        real.push(row);
    }

    // The most bits, below 22, that keep the largest weight within `i16`.
    let mut precision = 0;
    while precision < 22 && ((0.5 + largest * f64::from(1_u32 << (precision + 1))) as i32) < 1 << 15
    {
        precision += 1;
    }

    let weights = real
        .into_iter()
        .map(|row| {
            row.into_iter()
                .map(|weight| {
                    let value = weight * f64::from(1_u32 << precision);
                    // Rounded half away from zero, then held as `i16`, as PyTorch stores it.
                    i32::from(if value < 0.0 {
                        (value - 0.5) as i16
                    } else {
                        (value + 0.5) as i16
                    })
                })
                .collect()
        })
        .collect();
    Weights {
        starts,
        rows: weights,
        precision,
    }
}

fn byte(sum: i32, precision: u32) -> u8 {
    (sum >> precision).clamp(0, 255) as u8
}

/// A filter's kernel and the width in input positions it spans at scale 1.
type Filter = (fn(f64) -> f64, f64);

/// Returns the filter `resample` names, or `None` for nearest.
fn filter(resample: Resample) -> Option<Filter> {
    match resample {
        Resample::Nearest => None,
        Resample::Bilinear => Some((triangle, 2.0)),
        Resample::Bicubic => Some((keys_cubic, 4.0)),
        Resample::Lanczos => Some((lanczos, 6.0)),
    }
}

/// Returns the input position `PyTorch`'s `nearest-exact` mode reads for each of `output` positions, the one under each output position's center, from an `F32` scale.
/// Torchvision resizes so for PIL's `NEAREST`, which `transformers` maps to `NEAREST_EXACT`.
fn nearest(input: usize, output: usize) -> Vec<usize> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|index| {
            // `(index + 0.5) * scale` in `F64`, floored in `F32`, as `nearest_neighbor_exact_compute_source_index` computes it.
            let center = (index as f64 + 0.5) * f64::from(scale);
            ((center as f32).floor() as usize).min(input - 1)
        })
        .collect()
}

pub(crate) fn pixels(image: &Image, height: u32, width: u32, resample: Resample) -> Vec<u8> {
    let (in_height, in_width) = (image.height as usize, image.width as usize);
    let (out_height, out_width) = (height as usize, width as usize);
    let Some((filter, span)) = filter(resample) else {
        let rows = nearest(in_height, out_height);
        let columns = nearest(in_width, out_width);
        return rows
            .iter()
            .flat_map(|&row| {
                columns.iter().flat_map(move |&column| {
                    let at = (row * in_width + column) * 3;
                    image.pixels[at..at + 3].iter().copied()
                })
            })
            .collect();
    };

    // Each pass runs only when its side changes, as PyTorch's does.
    let mut pixels = image.pixels.clone();
    if out_width != in_width {
        let columns = weights(in_width, out_width, filter, span);
        let rounding = 1 << (columns.precision - 1);
        let mut resized = vec![0; in_height * out_width * 3];
        // Rows resize apart, so they run on all cores.
        resized
            .par_chunks_mut(out_width * 3)
            .zip(pixels.par_chunks_exact(in_width * 3))
            .for_each(|(out, row)| {
                let mut out = out.iter_mut();
                for (start, weights) in columns.starts.iter().zip(&columns.rows) {
                    for channel in 0..3 {
                        let sum =
                            weights
                                .iter()
                                .enumerate()
                                .fold(rounding, |sum, (offset, weight)| {
                                    sum + i32::from(row[(start + offset) * 3 + channel]) * weight
                                });
                        if let Some(slot) = out.next() {
                            *slot = byte(sum, columns.precision);
                        }
                    }
                }
            });

        pixels = resized;
    }

    if out_height != in_height {
        let rows = weights(in_height, out_height, filter, span);
        let rounding = 1 << (rows.precision - 1);
        let stride = out_width * 3;
        let mut resized = vec![0; out_height * stride];
        // Each output row sums whole input rows, which reads memory in order and vectorizes, and rows run on all cores.
        resized
            .par_chunks_mut(stride)
            .zip(rows.starts.par_iter().zip(rows.rows.par_iter()))
            .for_each(|(out, (start, weights))| {
                let mut sums = vec![rounding; stride];
                for (offset, &weight) in weights.iter().enumerate() {
                    let row = &pixels[(start + offset) * stride..(start + offset + 1) * stride];
                    for (sum, &pixel) in sums.iter_mut().zip(row) {
                        *sum += i32::from(pixel) * weight;
                    }
                }
                for (slot, sum) in out.iter_mut().zip(sums) {
                    *slot = byte(sum, rows.precision);
                }
            });

        pixels = resized;
    }
    pixels
}
