use std::{error::Error, thread, time::Duration};

use singe_nvml::{library::Library, types::TemperatureSensor};

const GPU_INDEX: u32 = 0;
const POLL_INTERVAL: Duration = Duration::from_secs(5);
const POLLS: u32 = 1;
const APPLY_CHANGES: bool = false;

const CURVE: &[(u32, u32)] = &[(35, 25), (50, 40), (65, 60), (75, 80), (85, 100)];

fn main() -> Result<(), Box<dyn Error>> {
    let nvml = Library::create()?;
    let device = nvml.device(GPU_INDEX)?;
    let name = device.name()?;
    let fans = device.num_fans()?;
    let fan_limits = device.min_max_fan_speed()?;

    println!("controlling GPU {GPU_INDEX}: {name}");
    println!("fan speed range: {}%-{}%", fan_limits.min, fan_limits.max);
    println!("curve: {CURVE:?}");
    if !APPLY_CHANGES {
        println!("dry run: set APPLY_CHANGES to true to write fan speeds");
    }

    for _ in 0..POLLS {
        // Read the GPU temperature and choose a speed from the curve.
        let temperature = device.temperature_reading(TemperatureSensor::Gpu)?;
        let target_speed =
            fan_speed_for_temperature(temperature).clamp(fan_limits.min, fan_limits.max);

        // Apply the same target to every fan controller on the selected GPU.
        for fan in 0..fans {
            let current_speed = device.fan_speed(fan)?;
            println!("fan {fan}: {temperature} C -> {target_speed}% (currently {current_speed}%)");

            if APPLY_CHANGES && current_speed != target_speed {
                device.set_fan_speed(fan, target_speed)?;
            }
        }

        thread::sleep(POLL_INTERVAL);
    }

    Ok(())
}

fn fan_speed_for_temperature(temperature: u32) -> u32 {
    let first = CURVE[0];
    if temperature <= first.0 {
        return first.1;
    }

    for window in CURVE.windows(2) {
        let (low_temperature, low_speed) = window[0];
        let (high_temperature, high_speed) = window[1];
        if temperature <= high_temperature {
            let span = high_temperature - low_temperature;
            let offset = temperature - low_temperature;
            let speed_delta = high_speed - low_speed;
            return low_speed + speed_delta * offset / span;
        }
    }

    CURVE[CURVE.len() - 1].1
}
