#![allow(dead_code)]

use cutile::tile_kernel::DevicePointer;

use crate::{
    error::{Error, Result},
    utility::checked_i32_value,
};

const RAW_VECTOR_TILE_SIZE: usize = 128;

pub fn checked_device_pointer<T>(pointer: DevicePointer<T>) -> Result<()> {
    if pointer.cu_deviceptr() == 0 {
        Err(Error::NullPointer)
    } else {
        Ok(())
    }
}

pub fn vector_tile_size(len: usize) -> usize {
    match len {
        0..=32 => 32,
        33..=64 => 64,
        65..=128 => 128,
        _ => 256,
    }
}

pub fn raw_vector_grid(len: usize) -> Result<(u32, u32, u32)> {
    let blocks = len.div_ceil(RAW_VECTOR_TILE_SIZE);
    let blocks = u32::try_from(blocks).map_err(|_| Error::SizeOverflow)?;
    Ok((blocks, 1, 1))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VectorLaunch {
    pub len: usize,
    pub len_i32: i32,
    pub grid: (u32, u32, u32),
}

impl VectorLaunch {
    pub fn create(len: usize) -> Result<Self> {
        Ok(Self {
            len,
            len_i32: checked_i32_value(len)?,
            grid: raw_vector_grid(len)?,
        })
    }
}
