use std::f64::consts::TAU;

use crate::{
    data_type::DataType,
    error::{Error, Result},
    graph::capture,
    shape::Shape,
    tensor::Tensor,
};

// Discrete Fourier transforms of real signals along the last axis, as matmuls with the transform's basis, which tensor cores run quickly at the lengths sequence models use.
// Complex values are pairs along a trailing axis of two, `[real, imaginary]`, as PyTorch's `view_as_real` lays them out.
impl Tensor {
    // The `n / 2 + 1` frequencies of the last axis's length-`n` real signal, unnormalized, as `torch.fft.rfft` computes them.
    pub fn rfft_by_basis(&self) -> Result<Self> {
        let length = self.last_extent()?;
        let frequencies = length / 2 + 1;
        let basis = self.basis(length, frequencies * 2, |time, column| {
            let angle = angle((column / 2) * time, length);
            if column % 2 == 0 {
                angle.cos()
            } else {
                -angle.sin()
            }
        })?;
        let mut shape = self.shape().dimensions().to_vec();
        shape.pop();
        shape.extend([frequencies, 2]);
        self.matmul(&basis)?.reshape(shape)
    }

    // The length-`length` real signal of `[..., k, 2]` frequencies, as `torch.fft.irfft` computes it: the imaginary parts of the constant and, for even lengths, the highest frequency are ignored, and the result is divided by `length`.
    pub fn irfft_by_basis(&self, length: i64) -> Result<Self> {
        let dimensions = self.shape().dimensions();
        let &[.., frequencies, pair] = dimensions else {
            return Err(Error::invalid_argument(
                "irfft needs [..., k, 2] frequencies",
            ));
        };
        if pair != 2 || length <= 0 {
            return Err(Error::invalid_argument(
                "irfft needs [..., k, 2] frequencies and a positive length",
            ));
        }

        // Frequencies past those a length-`length` signal has are dropped, and missing ones are zero.
        let used = frequencies.min(length / 2 + 1);
        let basis = self.basis(used * 2, length, |row, time| {
            let frequency = row / 2;
            let weight = if frequency == 0 || (length % 2 == 0 && frequency == length / 2) {
                1.0
            } else {
                2.0
            };
            let angle = angle(frequency * time, length);
            let unscaled = if row % 2 == 0 {
                angle.cos()
            } else if weight == 1.0 {
                0.0
            } else {
                -angle.sin()
            };
            weight * unscaled / count(length)
        })?;
        let mut flat = dimensions[..dimensions.len() - 2].to_vec();
        flat.push(used * 2);
        self.narrow(-2, 0, used)?.reshape(flat)?.matmul(&basis)
    }

    fn last_extent(&self) -> Result<i64> {
        self.shape()
            .dimensions()
            .last()
            .copied()
            .filter(|&extent| extent > 0)
            .ok_or_else(|| Error::invalid_argument("fourier transforms need a nonempty last axis"))
    }

    // A `[rows, columns]` basis of this tensor's floating type, computed in `F64`, whose entries `entry` gives.
    fn basis(&self, rows: i64, columns: i64, entry: impl Fn(i64, i64) -> f64) -> Result<Self> {
        let data_type = self.data_type();
        if !data_type.is_float() {
            return Err(Error::unsupported_data_type(
                "fourier transform of integers",
            ));
        }

        let values = (0..rows)
            .flat_map(|row| (0..columns).map(move |column| (row, column)))
            .map(|(row, column)| entry(row, column))
            .collect::<Vec<_>>();
        let shape = Shape::create([rows, columns])?;
        if data_type == DataType::F64 {
            return capture::literal(self.device(), |device| {
                Self::from_vec(device, shape, values)
            });
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "basis entries lie in [-2, 2]"
        )]
        let values = values.iter().map(|&value| value as f32).collect::<Vec<_>>();
        let basis = capture::literal(self.device(), |device| {
            Self::from_vec(device, shape, values)
        })?;
        if data_type == DataType::F32 {
            Ok(basis)
        } else {
            basis.cast(data_type)
        }
    }
}

// The phase of `product` of a length-`length` transform's turns, reduced first so it stays exact.
fn angle(product: i64, length: i64) -> f64 {
    TAU * count(product % length) / count(length)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "signal lengths are far below 2^53"
)]
const fn count(value: i64) -> f64 {
    value as f64
}
