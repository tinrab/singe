//! What the examples share: reading their arguments, and starting the runtime.

#![allow(dead_code, reason = "each example uses its own part of this module")]

use std::{
    env,
    error::Error as StdError,
    path::{Path, PathBuf},
    str::FromStr,
    time::{Duration, Instant},
};

use anyhow::{Context as _, Result, anyhow, bail};
use singe::{
    DataType, Device,
    backend::cuda::{export::Library, synchronize},
};
use singe_testing::snapshot_if_set;

/// The arguments every example takes: `--snapshot DIR`, or `--model NAME` with `SINGE_SNAPSHOT_<NAME>` set, and `--cpu`, and the rest by name, each with a value but the flags an example names.
pub struct Arguments {
    snapshot: PathBuf,
    device: Device,
    named: Vec<(String, String)>,
    flags: Vec<String>,
    positional: Vec<String>,
}

impl Arguments {
    /// Reads the process's arguments, where each of `flags` takes no value.
    pub fn parse(flags: &[&str]) -> Result<Self> {
        let mut snapshot = None;
        let mut device = Device::Cuda(0);
        let (mut named, mut set, mut positional) = (Vec::new(), Vec::new(), Vec::new());
        let mut arguments = env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--cpu" => device = Device::Cpu,
                "--snapshot" => snapshot = arguments.next().map(PathBuf::from),
                name if flags.contains(&name.trim_start_matches("--")) => {
                    set.push(name.trim_start_matches("--").to_owned());
                }
                name if name.starts_with("--") => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| anyhow!("{name} needs a value"))?;
                    named.push((name.trim_start_matches("--").to_owned(), value));
                }
                _ => positional.push(argument),
            }
        }

        let snapshot = if let Some(snapshot) = snapshot {
            snapshot
        } else {
            let (_, model) = named
                .iter()
                .find(|(key, _)| key == "model")
                .context("pass --snapshot DIR, or --model NAME with SINGE_SNAPSHOT_<NAME> set")?;
            let name = model.to_ascii_uppercase();
            snapshot_if_set(&name).with_context(|| format!("set SINGE_SNAPSHOT_{name}"))?
        };
        Ok(Self {
            snapshot,
            device,
            named,
            flags: set,
            positional,
        })
    }

    pub fn snapshot(&self) -> &Path {
        &self.snapshot
    }

    pub const fn device(&self) -> Device {
        self.device
    }

    pub fn positional(&self) -> &[String] {
        &self.positional
    }

    /// Returns whether the flag `name` was passed.
    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }

    /// Returns every value given for `name`, in order, for an argument that may repeat.
    pub fn values(&self, name: &str) -> Vec<&str> {
        self.named
            .iter()
            .filter(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
            .collect()
    }

    pub fn value(&self, name: &str) -> Option<&str> {
        self.named
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }

    pub fn required(&self, name: &str) -> Result<&str> {
        self.value(name).with_context(|| format!("pass --{name}"))
    }

    pub fn parsed<T>(&self, name: &str) -> Result<Option<T>>
    where
        T: FromStr,
        T::Err: StdError + Send + Sync + 'static,
    {
        self.value(name)
            .map(|value| {
                value
                    .parse()
                    .with_context(|| format!("parse --{name} {value}"))
            })
            .transpose()
    }

    /// Returns the libraries each `--library NAME` names, or every one when none is named.
    pub fn libraries(&self) -> Result<Vec<Library>> {
        let libraries = self.values("library");
        if libraries.is_empty() {
            return Ok(Library::ALL.to_vec());
        }

        libraries
            .into_iter()
            .map(|library| {
                library
                    .parse()
                    .with_context(|| format!("parse --library {library}"))
            })
            .collect()
    }

    /// Returns the weights' type `--type f32|f16|bf16` names, or `default` without one.
    pub fn data_type(&self, default: DataType) -> Result<DataType> {
        Ok(match self.value("type") {
            None => default,
            Some("f32") => DataType::F32,
            Some("f16") => DataType::F16,
            Some("bf16") => DataType::Bf16,
            Some(other) => bail!("unknown --type {other}"),
        })
    }
}

/// Starts `device`'s runtime apart from a load, which would otherwise start it, so each phase shows on its own, and returns how long it took.
pub fn start_runtime(device: Device) -> Result<Duration> {
    let started = Instant::now();
    if device.is_cuda() {
        synchronize(device)?;
    }
    Ok(started.elapsed())
}
