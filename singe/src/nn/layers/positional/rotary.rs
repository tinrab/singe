use std::f64::consts::PI;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    nn::TensorGroup,
    shape::Shape,
    tensor::Tensor,
    utility::to_usize,
};

// How a head's features pair up for rotation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotaryLayout {
    // Each feature pairs with the one half a head away, as `rotate_half` writes it.
    HalfSplit,
    // Adjacent features pair, as DeepSeek-V3's `rope_interleave` writes it.
    Interleaved,
}

// YaRN's scaling of rotary frequencies for positions beyond what a model was trained on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YarnFrequencies {
    pub factor: f64,
    pub original_max_positions: i64,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub attention_factor: Option<f64>,
    pub mscale: Option<f64>,
    pub mscale_all_dim: Option<f64>,
    pub truncate: bool,
}

impl YarnFrequencies {
    fn correction_dim(&self, rotations: f64, head_dim: f64, theta: f64) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "position counts round to the frequency's precision"
        )]
        let positions = self.original_max_positions as f64;
        head_dim * (positions / (rotations * 2.0 * PI)).ln() / (2.0 * theta.ln())
    }

    // The scale cosines and sines take.
    fn table_scale(&self) -> f64 {
        self.attention_factor
            .unwrap_or_else(|| match (self.mscale, self.mscale_all_dim) {
                (Some(mscale), Some(all_dim)) if mscale != 0.0 && all_dim != 0.0 => {
                    mscale_of(self.factor, mscale) / mscale_of(self.factor, all_dim)
                }
                _ => mscale_of(self.factor, 1.0),
            })
    }
}

fn mscale_of(factor: f64, mscale: f64) -> f64 {
    if factor <= 1.0 {
        1.0
    } else {
        (0.1 * mscale).mul_add(factor.ln(), 1.0)
    }
}

// The frequencies a rotary table is computed from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotaryFrequencies {
    pub theta: f64,
    pub yarn: Option<YarnFrequencies>,
}

impl RotaryFrequencies {
    #[must_use]
    pub const fn new(theta: f64) -> Self {
        Self { theta, yarn: None }
    }

    #[must_use]
    pub const fn with_yarn(mut self, yarn: YarnFrequencies) -> Self {
        self.yarn = Some(yarn);
        self
    }

    // The correction latent attention multiplies its scale by, since YaRN's positions change the scores' variance.
    #[must_use]
    pub fn score_correction(&self) -> f64 {
        self.yarn.map_or(1.0, |yarn| match yarn.mscale_all_dim {
            Some(all_dim) if all_dim != 0.0 => {
                let mscale = mscale_of(yarn.factor, all_dim);
                mscale * mscale
            }
            _ => 1.0,
        })
    }

    // Computes `1 / theta^(2i / D)` in F32, as the references do, with YaRN's interpolation when enabled.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "rotary phases are defined in F32, as the references compute them"
    )]
    fn inverse(&self, head_dim: i64, pairs: usize) -> Vec<f32> {
        let extrapolation = (0..pairs)
            .map(|pair| 1.0 / (self.theta as f32).powf((2 * pair) as f32 / head_dim as f32))
            .collect::<Vec<_>>();
        let Some(yarn) = self.yarn else {
            return extrapolation;
        };
        let dim = head_dim as f64;
        let low = yarn.correction_dim(yarn.beta_fast, dim, self.theta);
        let high = yarn.correction_dim(yarn.beta_slow, dim, self.theta);
        let (low, high) = if yarn.truncate {
            (low.floor(), high.ceil())
        } else {
            (low, high)
        };
        let (low, high) = (low.max(0.0), high.min(dim - 1.0));
        let high = if low == high { high + 0.001 } else { high };
        extrapolation
            .iter()
            .enumerate()
            .map(|(pair, &frequency)| {
                let ramp = ((pair as f32 - low as f32) / (high - low) as f32).clamp(0.0, 1.0);
                let extrapolating = 1.0 - ramp;
                (frequency / yarn.factor as f32)
                    .mul_add(1.0 - extrapolating, frequency * extrapolating)
            })
            .collect()
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "the table holds the scale at its own precision"
    )]
    fn table_scale(&self) -> f32 {
        self.yarn.map_or(1.0, |yarn| yarn.table_scale() as f32)
    }
}

// Rotary cosines and sines for a range of positions and a head of `head_dim` features.
#[derive(Debug, Clone)]
pub struct RotaryTable {
    cos: Tensor,
    sin: Tensor,
    layout: RotaryLayout,
}

// The cosines and sines of one step's positions, `[T, 1, D / 2]`.
#[derive(Debug, Clone)]
pub struct RotaryAngles {
    cos: Tensor,
    sin: Tensor,
    layout: RotaryLayout,
}

// What a rotary table needs besides its capacity and placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RotarySpec {
    pub head_dim: i64,
    pub frequencies: RotaryFrequencies,
    pub layout: RotaryLayout,
}

impl RotaryTable {
    // Computes the table for `capacity` positions, in `data_type` on `device`.
    pub fn create(
        device: Device,
        data_type: DataType,
        spec: RotarySpec,
        capacity: i64,
    ) -> Result<Self> {
        let RotarySpec {
            head_dim,
            frequencies,
            layout,
        } = spec;
        if head_dim <= 0 || head_dim % 2 != 0 || capacity <= 0 {
            return Err(Error::invalid_argument("rotary table size"));
        }

        let shape = Shape::create([capacity, head_dim / 2])?;
        if device.is_meta() {
            return Ok(Self {
                cos: Tensor::zeroed_on(device, data_type, shape.clone())?,
                sin: Tensor::zeroed_on(device, data_type, shape)?,
                layout,
            });
        }

        let pairs = to_usize(head_dim / 2, "rotary pairs")?;
        let positions = to_usize(capacity, "rotary positions")?;
        let inverse = frequencies.inverse(head_dim, pairs);
        let scale = frequencies.table_scale();
        let mut cos = Vec::with_capacity(positions * pairs);
        let mut sin = Vec::with_capacity(positions * pairs);
        for position in 0..positions {
            #[expect(
                clippy::cast_precision_loss,
                reason = "rotary phases are defined in F32, as the references compute them"
            )]
            let position = position as f32;
            for frequency in &inverse {
                let (sine, cosine) = (frequency * position).sin_cos();
                cos.push(cosine * scale);
                sin.push(sine * scale);
            }
        }

        let table = |values: Vec<f32>| {
            Tensor::from_vec(Device::Cpu, shape.clone(), values)?
                .cast(data_type)?
                .to_device(device)
        };
        Ok(Self {
            cos: table(cos)?,
            sin: table(sin)?,
            layout,
        })
    }

    // Looks up the angles of `positions`, `[T]`.
    pub fn angles(&self, positions: &Tensor) -> Result<RotaryAngles> {
        Ok(RotaryAngles {
            cos: self.cos.index_select(0, positions)?.unsqueeze(1)?,
            sin: self.sin.index_select(0, positions)?.unsqueeze(1)?,
            layout: self.layout,
        })
    }
}

impl TensorGroup for RotaryTable {
    fn named_tensors(&self) -> Vec<(String, Tensor)> {
        vec![
            ("rotary.cos".to_owned(), self.cos.clone()),
            ("rotary.sin".to_owned(), self.sin.clone()),
        ]
    }
}

impl RotaryAngles {
    // Rotates `input`, `[..., T, H, D]`: each feature of a pair turns against the other.
    pub fn rotate(&self, input: &Tensor) -> Result<Tensor> {
        let half = *self
            .cos
            .shape()
            .dimensions()
            .last()
            .ok_or(Error::DimensionOutOfBounds)?;
        let ordered = match self.layout {
            RotaryLayout::HalfSplit => input.clone(),
            // Bringing each pair's features half a head apart reuses the same rotation, and only scores read the result, whose order they do not depend on.
            RotaryLayout::Interleaved => {
                let rank = input.shape().dimensions().len();
                let leading = (0..rank.saturating_sub(1))
                    .map(|axis| input.dimension(axis))
                    .collect::<Result<Vec<_>>>()?;
                let pairs = leading
                    .iter()
                    .cloned()
                    .chain([half.into(), 2.into()])
                    .collect::<Vec<_>>();
                let whole = leading
                    .into_iter()
                    .chain([(half * 2).into()])
                    .collect::<Vec<_>>();
                input
                    .reshape_with(&pairs)?
                    .transpose(-1, -2)?
                    .reshape_with(&whole)?
            }
        };
        let first = ordered.narrow(-1, 0, half)?;
        let second = ordered.narrow(-1, half, half)?;
        let rotated_first = first.mul(&self.cos)?.sub(&second.mul(&self.sin)?)?;
        let rotated_second = second.mul(&self.cos)?.add(&first.mul(&self.sin)?)?;
        Tensor::concat(&[rotated_first, rotated_second], -1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(shape: &[i64], values: Vec<f32>) -> Tensor {
        Tensor::from_vec(Device::Cpu, Shape::create(shape.to_vec()).unwrap(), values).unwrap()
    }

    fn spec(layout: RotaryLayout) -> RotarySpec {
        RotarySpec {
            head_dim: 4,
            frequencies: RotaryFrequencies::new(100.0),
            layout,
        }
    }

    #[test]
    fn test_rotary_tables_rotate_half_split_pairs_by_position() {
        let table =
            RotaryTable::create(Device::Cpu, DataType::F32, spec(RotaryLayout::HalfSplit), 4)
                .unwrap();
        let positions =
            Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![3_i64]).unwrap();
        let angles = table.angles(&positions).unwrap();
        let rotated = angles
            .rotate(&tensor(&[1, 1, 4], vec![1., 2., 3., 4.]))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        let phases = [3.0_f32, 3.0 / 10.0];
        let [(sin_low, cos_low), (sin_high, cos_high)] = phases.map(f32::sin_cos);
        let expected = [
            (-3.0_f32).mul_add(sin_low, cos_low),
            2.0_f32.mul_add(cos_high, -4.0 * sin_high),
            3.0_f32.mul_add(cos_low, sin_low),
            4.0_f32.mul_add(cos_high, 2.0 * sin_high),
        ];
        for (actual, expected) in rotated.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-5, "{actual} and {expected}");
        }
    }

    // Interleaving changes which features pair, and the dot product of two rotated rows is what attention reads.
    #[test]
    fn test_interleaved_rotation_pairs_adjacent_features() {
        let table = RotaryTable::create(
            Device::Cpu,
            DataType::F32,
            spec(RotaryLayout::Interleaved),
            4,
        )
        .unwrap();
        let positions =
            Tensor::from_vec(Device::Cpu, Shape::create([1]).unwrap(), vec![0_i64]).unwrap();
        let rotated = table
            .angles(&positions)
            .unwrap()
            .rotate(&tensor(&[1, 1, 4], vec![1., 2., 3., 4.]))
            .unwrap()
            .to_vec::<f32>()
            .unwrap();
        // Position zero turns nothing, so this shows the order alone: evens, then odds.
        assert_eq!(rotated, [1., 3., 2., 4.]);
    }

    #[test]
    fn test_yarn_leaves_a_factor_of_one_unchanged() {
        let yarn = YarnFrequencies {
            factor: 1.0,
            original_max_positions: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
            mscale: None,
            mscale_all_dim: None,
            truncate: true,
        };
        let plain = RotaryFrequencies::new(100.0);
        let scaled = plain.with_yarn(yarn);
        assert_eq!(scaled.table_scale(), 1.0);
        assert_eq!(scaled.score_correction(), 1.0);
        for (left, right) in plain.inverse(8, 4).iter().zip(scaled.inverse(8, 4)) {
            assert!((left - right).abs() < 1e-6, "{left} and {right}");
        }
    }

    #[test]
    fn test_rotary_tables_refuse_odd_heads() {
        let odd = RotarySpec {
            head_dim: 3,
            ..spec(RotaryLayout::HalfSplit)
        };
        assert!(RotaryTable::create(Device::Cpu, DataType::F32, odd, 4).is_err());
    }
}
