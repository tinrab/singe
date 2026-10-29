//! Native image operations on an explicitly selected stream.
//!
//! # Safety
//! Retain each allocation until completion or its last captured replay and establish access ordering.
//! Input and read-write storage must contain valid initialized values for the complete native footprint.
//! Non-border spatial operations require initialized source halo beyond the output ROI.
//! Outputs promise writable capacity only; promotion requires all logical samples to be initialized.
//! Helpers owning temporary scratch or device descriptors complete their native use before releasing it.
//! Such helpers reject capture before allocating those temporary resources.

use crate::{
    error::{Error, Result},
    types::{Point, Size},
};

mod composite;

pub use composite::*;

mod mask;

pub use mask::*;

mod border;

pub use border::*;

mod three_by_three;

pub use three_by_three::*;

mod three_by_three_border;

pub use three_by_three_border::*;

fn validate_positive_size(size: Size, name: &str) -> Result<()> {
    if size.width > 0 && size.height > 0 {
        return Ok(());
    }
    Err(Error::out_of_range(name.to_owned()))
}

fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }
    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

fn validate_mask(mask: &[u8], mask_size: Size, anchor: Point) -> Result<()> {
    validate_mask_shape(mask, mask_size, anchor)
}

fn validate_mask_shape<T>(mask: &[T], mask_size: Size, anchor: Point) -> Result<()> {
    if mask_size.width <= 0 || mask_size.height <= 0 {
        return Err(Error::out_of_range("mask size"));
    }

    if anchor.x < 0 || anchor.y < 0 || anchor.x >= mask_size.width || anchor.y >= mask_size.height {
        return Err(Error::out_of_range("anchor"));
    }

    let expected = (mask_size.width as usize)
        .checked_mul(mask_size.height as usize)
        .ok_or_else(|| Error::out_of_range("mask size"))?;

    if mask.len() == expected {
        return Ok(());
    }

    Err(Error::LengthMismatch {
        name: "mask".into(),
        expected,
        actual: mask.len(),
    })
}

fn validate_border_roi(source: Size, source_offset: Point, roi: Size) -> Result<()> {
    if source_offset.x < 0 || source_offset.y < 0 {
        return Err(Error::out_of_range("source offset"));
    }

    let right = source_offset
        .x
        .checked_add(roi.width)
        .ok_or_else(|| Error::out_of_range("source offset"))?;
    let bottom = source_offset
        .y
        .checked_add(roi.height)
        .ok_or_else(|| Error::out_of_range("source offset"))?;

    if right <= source.width && bottom <= source.height {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "border roi".into(),
        expected: source,
        actual: roi,
    })
}
