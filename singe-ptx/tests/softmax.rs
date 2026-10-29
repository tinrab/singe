//! Stable row softmax compared with an independent binary64 host reference.

use anyhow::Result;
use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

mod support;

use support::Observe;

#[test]
fn test_stable_softmax_handles_odd_rows_and_partial_warps_through_both_waiting_styles() -> Result<()>
{
    let parsed = ParsedModule::parse(include_str!("fixtures/execution/softmax.ptx"))?;
    for asynchronous in [false, true] {
        let device = Emulator::create(EmulatorOptions::default())?;
        let module = device.load(parsed.clone())?.observe(asynchronous)?;
        let kernel = module.kernel("softmax")?;
        for block in [1u32, 17, 32, 33, 64, 127, 128] {
            for width in [1usize, 17, 31, 32, 33, 127, 128, 129, 257] {
                let rows = 5;
                let input: Vec<f32> = (0..rows * width)
                    .map(|index| {
                        let column = index % width;
                        match index / width {
                            0 => 0.0,
                            1 => ((column % 19) as f32).mul_add(0.25, 1000.0),
                            2 => ((column % 17) as f32).mul_add(-0.5, -1000.0),
                            3 => ((column * 37) % 241) as f32 - 120.0,
                            _ => (column as f32 - width as f32 / 2.0) / 16.0,
                        }
                    })
                    .collect();
                let source = device.upload(&input)?.observe(asynchronous)?;
                let output = device.zeroed::<f32>(input.len())?.observe(asynchronous)?;
                kernel
                    .launch(
                        LaunchConfig::create(
                            [u32::try_from(rows).unwrap(), 1, 1],
                            [block, 1, 1],
                            0,
                        )?,
                        [
                            (&source).into(),
                            (&output).into(),
                            u32::try_from(width).unwrap().into(),
                        ],
                    )?
                    .observe(asynchronous)?;
                let actual = device.download(&output)?.observe(asynchronous)?;
                for (row, (input, actual)) in
                    input.chunks(width).zip(actual.chunks(width)).enumerate()
                {
                    let maximum = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let reference: Vec<f64> = input
                        .iter()
                        .map(|&value| (f64::from(value) - f64::from(maximum)).exp())
                        .collect();
                    let denominator: f64 = reference.iter().sum();
                    for (column, (&actual, reference)) in actual.iter().zip(reference).enumerate() {
                        let expected = reference / denominator;
                        assert!(
                            actual.is_finite() && actual >= 0.0,
                            "block={block}, width={width}, row={row}, column={column}, actual={actual}"
                        );
                        assert!(
                            (f64::from(actual) - expected).abs() <= expected.mul_add(1e-5, 1e-7),
                            "block={block}, width={width}, row={row}, column={column}: {actual} != {expected}"
                        );
                    }
                    let sum: f64 = actual.iter().map(|&value| f64::from(value)).sum();
                    assert!(
                        (sum - 1.0).abs() <= 2e-6,
                        "block={block}, width={width}, row={row}, sum={sum}"
                    );
                }
            }
        }
        device.shutdown(ShutdownMode::Drain).observe(asynchronous)?;
    }
    Ok(())
}
