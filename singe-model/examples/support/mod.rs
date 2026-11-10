// What the examples share: reading their arguments and files, which Singe leaves to its callers.

#![allow(dead_code, reason = "each example uses its own part of this module")]

use std::{env, fs, path::PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use image::{ImageBuffer, Rgb, imageops::FilterType};
use singe::Device;

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
}

// Reads a mono 16-bit PCM WAV file at 16 kHz, as Whisper hears, to samples in [-1, 1).
pub fn read_wav(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {path}"))?;
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        bail!("{path} is not a WAV file");
    }

    let (mut format, mut data) = (None, None);
    let mut offset = 12;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = usize::try_from(u32::from_le_bytes(
            bytes[offset + 4..offset + 8].try_into()?,
        ))?;
        let body = &bytes[offset + 8..(offset + 8 + size).min(bytes.len())];
        match id {
            b"fmt " => format = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }

        offset += 8 + size + size % 2;
    }

    let format = format.context("a WAV file without a format chunk")?;
    let field = |at: usize| u16::from_le_bytes([format[at], format[at + 1]]);
    let rate = u32::from_le_bytes(format[4..8].try_into()?);
    if field(0) != 1 || field(2) != 1 || rate != 16_000 || field(14) != 16 {
        bail!("{path} is not mono 16-bit PCM at 16 kHz");
    }
    Ok(data
        .context("a WAV file without data")?
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from(i16::from_le_bytes(*pair)) / 32_768.0)
        .collect())
}

// Reads an image file resized to `width` by `height`, as channel-major RGB in [0, 1].
pub fn read_image(path: &str, width: u32, height: u32) -> Result<Vec<f32>> {
    let image = image::open(path)
        .with_context(|| format!("reading {path}"))?
        .resize_exact(width, height, FilterType::Triangle)
        .to_rgb8();
    let plane = usize::try_from(width * height)?;
    let mut values = vec![0.0; 3 * plane];
    for (index, pixel) in image.pixels().enumerate() {
        for channel in 0..3 {
            values[channel * plane + index] = f32::from(pixel[channel]) / 255.0;
        }
    }
    Ok(values)
}

// Writes channel-major RGB in [0, 1] as an image file, whose extension picks the format.
pub fn write_image(path: &str, values: &[f32], width: u32, height: u32) -> Result<()> {
    let plane = usize::try_from(width * height)?;
    let image = ImageBuffer::from_fn(width, height, |x, y| {
        let index = usize::try_from(y * width + x).unwrap_or(0);
        Rgb([0, 1, 2].map(|channel| {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "the value is clamped to a byte's range"
            )]
            let byte = (values[channel * plane + index].clamp(0.0, 1.0) * 255.0).round() as u8;
            byte
        }))
    });
    image.save(path).with_context(|| format!("writing {path}"))
}
