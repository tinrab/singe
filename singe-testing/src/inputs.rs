use std::f64::consts::PI;

use singe::{Device, Shape, Tensor};

#[must_use]
pub fn pattern_image(size: i64, device: Device) -> Tensor {
    let mut values = Vec::new();
    for channel in 0..3 {
        for y in 0..size {
            for x in 0..size {
                let value = match channel {
                    0 => (x * 3 + y) % 256,
                    1 => (x * y / 7 + 2 * y) % 256,
                    _ => ((x / 16 + y / 16) % 2) * 160 + (x + y) % 96,
                };
                #[expect(clippy::cast_precision_loss, reason = "pixel values are below 256")]
                values.push(value as f32 / 127.5 - 1.0);
            }
        }
    }
    Tensor::from_vec(device, Shape::create([1, 3, size, size]).unwrap(), values).unwrap()
}

#[must_use]
pub fn tone_and_chirp(samples: usize, device: Device) -> Tensor {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the signal stays within one, so single precision keeps all but its last digits"
    )]
    let values = (0..samples)
        .map(|index| {
            let time = index as f64 / 16_000.0;
            let chirp = 300.0 + 1_700.0 * index as f64 / samples as f64;
            let tau = 2.0 * PI;
            0.25_f64.mul_add((tau * chirp * time).sin(), 0.5 * (tau * 440.0 * time).sin()) as f32
        })
        .collect::<Vec<_>>();
    let count = i64::try_from(samples).unwrap();
    Tensor::from_vec(device, Shape::create([count]).unwrap(), values).unwrap()
}
