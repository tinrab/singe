//! Small FFT helpers, interleaved complex transforms, and normalization.

use num_complex::Complex32;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

pub fn c2c(input: &[Complex32], batch: usize, n: usize, direction: FftDirection) -> Vec<Complex32> {
    let sign = match direction {
        FftDirection::Forward => -1.0,
        FftDirection::Inverse => 1.0,
    };
    let mut output = vec![Complex32::new(0.0, 0.0); batch * n];
    for batch_index in 0..batch {
        let base = batch_index * n;
        for frequency in 0..n {
            let mut real = 0.0;
            let mut imag = 0.0;
            for sample in 0..n {
                let value = input[base + sample];
                let angle =
                    sign * 2.0 * std::f32::consts::PI * frequency as f32 * sample as f32 / n as f32;
                real += value.re * angle.cos() - value.im * angle.sin();
                imag += value.re * angle.sin() + value.im * angle.cos();
            }
            output[base + frequency] = Complex32::new(real, imag);
        }
    }
    output
}

pub fn r2c(input: &[f32], batch: usize, n: usize) -> Vec<Complex32> {
    let frequency_bins = n / 2 + 1;
    let mut output = vec![Complex32::new(0.0, 0.0); batch * frequency_bins];
    for batch_index in 0..batch {
        let input_base = batch_index * n;
        let output_base = batch_index * frequency_bins;
        for frequency in 0..frequency_bins {
            let mut real = 0.0;
            let mut imag = 0.0;
            for sample in 0..n {
                let value = input[input_base + sample];
                let angle =
                    -2.0 * std::f32::consts::PI * frequency as f32 * sample as f32 / n as f32;
                real += value * angle.cos();
                imag += value * angle.sin();
            }
            output[output_base + frequency] = Complex32::new(real, imag);
        }
    }
    output
}

pub fn c2r_unscaled(input: &[Complex32], batch: usize, n: usize) -> Vec<f32> {
    let frequency_bins = n / 2 + 1;
    let mut output = vec![0.0; batch * n];
    for batch_index in 0..batch {
        let input_base = batch_index * frequency_bins;
        let output_base = batch_index * n;
        for sample in 0..n {
            let mut value = input[input_base].re;
            let nyquist = input[input_base + n / 2].re;
            value += nyquist * if sample % 2 == 0 { 1.0 } else { -1.0 };
            for frequency in 1..n / 2 {
                let bin = input[input_base + frequency];
                let angle =
                    2.0 * std::f32::consts::PI * frequency as f32 * sample as f32 / n as f32;
                value += 2.0 * (bin.re * angle.cos() - bin.im * angle.sin());
            }
            output[output_base + sample] = value;
        }
    }
    output
}

pub fn dft_c2c_f32_interleaved(input: &[f32], batch: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for k in 0..n {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;
            for j in 0..n {
                let angle = -2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                let twiddle_real = angle.cos();
                let twiddle_imag = angle.sin();
                let input_offset = (batch_index * n + j) * 2;
                let input_real = input[input_offset];
                let input_imag = input[input_offset + 1];
                real += input_real * twiddle_real - input_imag * twiddle_imag;
                imag += input_real * twiddle_imag + input_imag * twiddle_real;
            }
            let output_offset = (batch_index * n + k) * 2;
            output[output_offset] = real;
            output[output_offset + 1] = imag;
        }
    }
    output
}

pub fn twiddle_tables(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut real = Vec::with_capacity(n);
    let mut imag = Vec::with_capacity(n);
    for phase in 0..n {
        let angle = -2.0 * std::f32::consts::PI * phase as f32 / n as f32;
        real.push(angle.cos());
        imag.push(angle.sin());
    }
    (real, imag)
}
