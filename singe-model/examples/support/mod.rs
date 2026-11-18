// What the examples share: reading their arguments.

#![allow(dead_code, reason = "each example uses its own part of this module")]

use std::{env, path::PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use singe::{DataType, Device};

// The arguments every example takes: `--snapshot DIR`, or `SNAPSHOT_PATH`, and `--cpu`, and the rest by name, each with a value but the flags an example names.
pub struct Arguments {
    pub snapshot: PathBuf,
    pub device: Device,
    pub named: Vec<(String, String)>,
    pub flags: Vec<String>,
    pub positional: Vec<String>,
}

impl Arguments {
    // Reads the process's arguments, where each of `flags` takes no value.
    pub fn parse(flags: &[&str]) -> Result<Self> {
        let mut snapshot = env::var_os("SNAPSHOT_PATH").map(PathBuf::from);
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
        Ok(Self {
            snapshot: snapshot.context("set SNAPSHOT_PATH or pass --snapshot")?,
            device,
            named,
            flags: set,
            positional,
        })
    }

    // Returns whether the flag `name` was passed.
    pub fn flag(&self, name: &str) -> bool {
        self.flags.iter().any(|flag| flag == name)
    }

    // Returns every value given for `name`, in order, for an argument that may repeat.
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

    // Returns the weights' type `--type f32|f16|bf16` names, or `default` without one.
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
