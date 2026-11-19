// PyTorch's CPU generator, so a pipeline seeded as `torch.Generator("cpu").manual_seed(seed)` draws the noise `diffusers` draws: MT19937 and the normal fill of `aten/src/ATen/native/cpu/DistributionTemplates.h`, whose AVX2 build transforms by `avx_mathfun.h`'s `log256_ps` and `sincos256_ps`, ported here lane by lane, and, for fewer than 16 values, the serial `normal_distribution<double>` of `aten/src/ATen/core/DistributionsHelper.h`.

use std::{
    f32::consts::{FRAC_1_SQRT_2, PI},
    f64::consts::TAU,
};

use singe::{Device, Shape, Tensor};

use crate::error::Result;

const STATE: usize = 624;

#[derive(Debug, Clone)]
pub struct TorchGenerator {
    state: [u32; STATE],
    index: usize,
    // The second value of the serial normal draw's last Box–Muller pair, which the generator caches for the next serial draw.
    next_normal: Option<f64>,
}

impl TorchGenerator {
    // A generator as `manual_seed(seed)` leaves one, whose MT19937 reads the seed's low 32 bits.
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        let mut state = [0_u32; STATE];
        let first = seed as u32;
        state[0] = first;
        for index in 1..STATE {
            let previous = state[index - 1];
            let step = index as u32;
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(step);
        }
        Self {
            state,
            index: STATE,
            next_normal: None,
        }
    }

    fn twist(&mut self) {
        for index in 0..STATE {
            let mixed =
                (self.state[index] & 0x8000_0000) | (self.state[(index + 1) % STATE] & 0x7fff_ffff);
            let mut next = self.state[(index + 397) % STATE] ^ (mixed >> 1);
            if mixed & 1 != 0 {
                next ^= 0x9908_b0df;
            }

            self.state[index] = next;
        }

        self.index = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= STATE {
            self.twist();
        }

        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    // `uniform_real_distribution<float>(0, 1)`: the draw's low 24 bits over `2^24`.
    fn uniform(&mut self) -> f32 {
        (self.next_u32() & 0x00ff_ffff) as f32 / 16_777_216.0
    }

    // `uniform_real_distribution<double>(0, 1)`: the low 53 bits of two draws, the first high, over `2^53`.
    fn uniform_f64(&mut self) -> f64 {
        let high = u64::from(self.next_u32());
        let low = u64::from(self.next_u32());
        (((high << 32) | low) & ((1_u64 << 53) - 1)) as f64 / 9_007_199_254_740_992.0
    }

    // `normal_distribution<double>` with mean 0 and deviation 1, rounded to `F32`: the cached value where there is one, or else Box–Muller over two uniforms, caching the sine's value.
    fn serial_normal(&mut self) -> f32 {
        if let Some(value) = self.next_normal.take() {
            return value as f32;
        }

        let first = self.uniform_f64();
        let second = self.uniform_f64();
        let radius = (-2.0 * (-second).ln_1p()).sqrt();
        let angle = TAU * first;
        self.next_normal = Some(radius * angle.sin());
        (radius * angle.cos()) as f32
    }

    // `torch.randn(count)` in `F32`: for 16 or more values, uniforms for every value, Box–Muller over each run of 16, and the last 16 drawn again when the count is not a multiple of 16; for fewer, each value drawn serially.
    pub fn normal(&mut self, count: usize) -> Vec<f32> {
        if count < 16 {
            return (0..count).map(|_| self.serial_normal()).collect();
        }

        let mut values = (0..count).map(|_| self.uniform()).collect::<Vec<f32>>();
        for start in (0..=count - 16).step_by(16) {
            box_muller(&mut values[start..start + 16]);
        }

        if !count.is_multiple_of(16) {
            let tail = &mut values[count - 16..];
            for value in tail.iter_mut() {
                *value = self.uniform();
            }

            box_muller(tail);
        }
        values
    }

    // `torch.randn(shape)` in `F32`, as a tensor on `device`.
    pub fn normal_tensor(&mut self, device: Device, shape: &[i64]) -> Result<Tensor> {
        let shape = Shape::create(shape)?;
        let values = self.normal(shape.element_count());
        Ok(Tensor::from_vec(device, shape, values)?)
    }
}

// `NormalFill16` with mean 0 and deviation 1, as its AVX2 build computes it.
fn box_muller(values: &mut [f32]) {
    const TWO_PI: f32 = 2.0 * PI;
    for lane in 0..8 {
        let radius = (-2.0 * log(1.0 - values[lane])).sqrt();
        let (sin, cos) = sincos(TWO_PI * values[lane + 8]);
        values[lane] = radius * cos;
        values[lane + 8] = radius * sin;
    }
}

// `log256_ps`: Cephes' polynomial on the mantissa.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the bit manipulations are `avx_mathfun.h`'s"
)]
fn log(value: f32) -> f32 {
    let invalid = value <= 0.0;
    let value = value.max(f32::from_bits(0x0080_0000));
    let exponent = ((value.to_bits() >> 23) as i32) - 0x7f;
    let mut x = f32::from_bits((value.to_bits() & !0x7f80_0000) | 0.5_f32.to_bits());
    let mut e = exponent as f32 + 1.0;
    let below = x < FRAC_1_SQRT_2;
    let tmp = if below { x } else { 0.0 };
    x -= 1.0;
    if below {
        e -= 1.0;
    }

    x += tmp;
    let z = x * x;
    let mut y = 7.037_683_629_2e-2_f32;
    for coefficient in [
        -1.151_461_031_0e-1_f32,
        1.167_699_874_0e-1,
        -1.242_014_084_6e-1,
        1.424_932_278_7e-1,
        -1.666_805_766_5e-1,
        2.000_071_476_5e-1,
        -2.499_999_399_3e-1,
        3.333_333_117_4e-1,
    ] {
        y = y * x + coefficient;
    }

    y *= x;
    y *= z;
    y += e * -2.121_944_40e-4;
    y -= z * 0.5;
    x += y;
    x += e * 0.693_359_375;
    if invalid { f32::NAN } else { x }
}

// `sincos256_ps`: Cephes' sine and cosine after reducing by multiples of `π / 4`.
fn sincos(value: f32) -> (f32, f32) {
    let sign_sin = value.to_bits() & 0x8000_0000;
    let mut x = value.abs();
    let mut y = x * 1.273_239_544_735_16;
    let mut octant = y as i32;
    octant = (octant + 1) & !1;
    y = octant as f32;
    let swap_sin = ((octant & 4) as u32) << 29;
    let polynomial = octant & 2 == 0;
    x += y * -0.785_156_25;
    x += y * -2.418_756_484_985_351_562_5e-4;
    x += y * -3.774_894_977_445_941_08e-8;
    let sign_cos = ((!(octant - 2) & 4) as u32) << 29;
    let sign_sin = sign_sin ^ swap_sin;
    let z = x * x;
    let mut cosine = 2.443_315_711_809_948e-5_f32;
    cosine = cosine * z + -1.388_731_625_493_765e-3;
    cosine = cosine * z + 4.166_664_568_298_827e-2;
    cosine *= z;
    cosine *= z;
    cosine -= z * 0.5;
    cosine += 1.0;
    let mut sine = -1.951_529_589_1e-4_f32;
    sine = sine * z + 8.332_160_873_6e-3;
    sine = sine * z + -1.666_665_461_1e-1;
    sine *= z;
    sine *= x;
    sine += x;
    let (sin, cos) = if polynomial {
        (sine, cosine)
    } else {
        (cosine, sine)
    };
    (
        f32::from_bits(sin.to_bits() ^ sign_sin),
        f32::from_bits(cos.to_bits() ^ sign_cos),
    )
}
