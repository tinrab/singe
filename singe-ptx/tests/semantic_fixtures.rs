//! Initialized instruction cases with independently computed results.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};

use serde::Deserialize;

use singe_ptx::emulator::{Emulator, EmulatorOptions, LaunchConfig, ParsedModule, ShutdownMode};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    source: String,
    expected: Vec<u8>,
    #[serde(default = "one_lane")]
    block: u32,
    #[serde(default)]
    error: Option<String>,
}

fn one_lane() -> u32 {
    1
}

#[test]
fn test_initialized_instruction_fixtures() -> Result<()> {
    let cases: Vec<Case> = serde_json::from_str(include_str!("fixtures/execution/cases.json"))?;
    let mut names = BTreeSet::new();
    for case in &cases {
        ensure!(names.insert(&case.name), "duplicate case {}", case.name);
        let run = || -> Result<()> {
            let device = Emulator::create(EmulatorOptions::default())?;
            let module = device.load(ParsedModule::parse(&case.source)?)?.wait()?;
            let result = module
                .kernel("kernel")?
                .launch(LaunchConfig::create([1; 3], [case.block, 1, 1], 0)?, [])?
                .wait();
            match (&case.error, result) {
                (None, Ok(report)) => ensure!(
                    report.completed_lanes == u64::from(case.block),
                    "incomplete execution"
                ),
                (Some(expected), Err(error)) => ensure!(
                    error.to_string().contains(expected),
                    "expected {expected}, got {error}"
                ),
                (None, Err(error)) => return Err(error.into()),
                (Some(expected), Ok(_)) => anyhow::bail!("expected {expected}"),
            }
            let output = device.download(&module.global("output")?)?.wait()?;
            ensure!(
                output.starts_with(&case.expected)
                    && output[case.expected.len()..].iter().all(|&byte| byte == 0),
                "output {output:?}, expected {:?} followed by zeros",
                case.expected
            );
            let shutdown = device.shutdown(ShutdownMode::Drain).wait();
            if case.error.is_none() {
                shutdown?;
            } else {
                ensure!(shutdown.is_err(), "kernel fault did not reach shutdown");
            }
            Ok(())
        };
        run().with_context(|| format!("instruction case {}", case.name))?;
    }
    Ok(())
}
