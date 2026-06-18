#![allow(dead_code)]

use std::{
    error::Error,
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
};

use image::{GrayImage, ImageBuffer};
use singe_cuda::{context::Context, memory::DeviceMemory, stream::Stream};
use singe_npp::{context::StreamContext, types::Size};

pub type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub fn create_stream() -> Result<(Stream, StreamContext)> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let stream_context = StreamContext::create(&stream)?;
    Ok((stream, stream_context))
}

pub fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples")
}

pub fn image_path(group: &str, name: &str) -> PathBuf {
    examples_dir().join("images").join(group).join(name)
}

pub fn output_dir(group: &str) -> Result<PathBuf> {
    let path = examples_dir().join("output").join(group);
    fs::create_dir_all(&path)?;
    Ok(path)
}

pub fn read_raw_u8(path: impl AsRef<Path>, size: Size) -> Result<Vec<u8>> {
    let expected = (size.width as usize)
        .checked_mul(size.height as usize)
        .ok_or("raw image dimensions overflow")?;
    let mut data = Vec::new();
    File::open(path.as_ref())?.read_to_end(&mut data)?;
    if data.len() != expected {
        return Err(format!(
            "{} has {} bytes, expected {expected}",
            path.as_ref().display(),
            data.len()
        )
        .into());
    }
    Ok(data)
}

pub fn write_png(path: impl AsRef<Path>, data: &[u8], size: Size) -> Result<()> {
    let image: GrayImage =
        ImageBuffer::from_raw(size.width as u32, size.height as u32, data.to_vec())
            .ok_or("failed to build output image")?;
    image.save(path)?;
    Ok(())
}

pub fn upload<T: Copy>(host: &[T]) -> Result<DeviceMemory<T>> {
    Ok(DeviceMemory::from_slice(host)?)
}

pub fn download<T: Copy>(device: &DeviceMemory<T>) -> Result<Vec<T>> {
    Ok(device.copy_to_host_vec()?)
}

pub fn count_nonzero<T>(data: &[T]) -> usize
where
    T: Copy + Default + PartialEq,
{
    data.iter().filter(|&&value| value != T::default()).count()
}

pub fn normalize_u16_to_u8(data: &[u16]) -> Vec<u8> {
    normalize_f64(data.iter().map(|&value| value as f64))
}

pub fn normalize_u32_to_u8(data: &[u32]) -> Vec<u8> {
    normalize_f64(data.iter().map(|&value| value as f64))
}

pub fn normalize_f32_to_u8(data: &[f32]) -> Vec<u8> {
    normalize_f64(data.iter().map(|&value| value as f64))
}

pub fn normalize_i16_c2_magnitude_to_u8(data: &[i16]) -> Vec<u8> {
    normalize_f64(data.chunks_exact(2).map(|xy| {
        let x = xy[0] as f64;
        let y = xy[1] as f64;
        x.hypot(y)
    }))
}

fn normalize_f64(values: impl Iterator<Item = f64>) -> Vec<u8> {
    let values = values.collect::<Vec<_>>();
    let max = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(0.0_f64, f64::max);
    if max <= 0.0 {
        return vec![0; values.len()];
    }

    values
        .iter()
        .map(|&value| {
            if value.is_finite() {
                ((value.max(0.0) / max) * 255.0).round() as u8
            } else {
                0
            }
        })
        .collect()
}
