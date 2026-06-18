use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::{Error, Result},
    image::view::{AC4, C1, C3, C4, ImageView, ImageViewMut},
    try_ffi,
    types::{BorderType, DataTypeLike, Point, Size},
    utility::to_usize,
    workspace::ScratchBuffer,
};

#[path = "morphology_composite.rs"]
mod composite;
pub use composite::*;

#[path = "morphology_mask.rs"]
mod mask;
pub use mask::*;

#[path = "morphology_border.rs"]
mod border;
pub use border::*;

#[path = "morphology_3x3.rs"]
mod three_by_three;
pub use three_by_three::*;

#[path = "morphology_3x3_border.rs"]
mod three_by_three_border;
pub use three_by_three_border::*;

fn validate_positive_size(size: Size, name: &str) -> Result<()> {
    if size.width > 0 && size.height > 0 {
        return Ok(());
    }
    Err(Error::OutOfRange { name: name.into() })
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
        return Err(Error::OutOfRange {
            name: "mask size".into(),
        });
    }

    if anchor.x < 0 || anchor.y < 0 || anchor.x >= mask_size.width || anchor.y >= mask_size.height {
        return Err(Error::OutOfRange {
            name: "anchor".into(),
        });
    }

    let expected = (mask_size.width as usize)
        .checked_mul(mask_size.height as usize)
        .ok_or_else(|| Error::OutOfRange {
            name: "mask size".into(),
        })?;

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
        return Err(Error::OutOfRange {
            name: "source offset".into(),
        });
    }

    let right = source_offset
        .x
        .checked_add(roi.width)
        .ok_or_else(|| Error::OutOfRange {
            name: "source offset".into(),
        })?;
    let bottom = source_offset
        .y
        .checked_add(roi.height)
        .ok_or_else(|| Error::OutOfRange {
            name: "source offset".into(),
        })?;

    if right <= source.width && bottom <= source.height {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "border roi".into(),
        expected: source,
        actual: roi,
    })
}
