use crate::{
    error::{Error, Result},
    types::Size,
    utility::to_i32,
};

pub fn validate_same_size(source: Size, destination: Size) -> Result<()> {
    if source == destination {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "image size".into(),
        expected: source,
        actual: destination,
    })
}

pub fn validate_mask_size(roi: Size, mask: Size) -> Result<()> {
    if roi == mask {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "mask size".into(),
        expected: roi,
        actual: mask,
    })
}

pub fn validate_border_placement(
    source: Size,
    destination: Size,
    top: usize,
    left: usize,
) -> Result<(i32, i32)> {
    let top = to_i32(top, "top")?;
    let left = to_i32(left, "left")?;

    let bottom = top
        .checked_add(source.height)
        .ok_or_else(|| Error::out_of_range("border offset"))?;
    let right = left
        .checked_add(source.width)
        .ok_or_else(|| Error::out_of_range("border offset"))?;

    if bottom > destination.height || right > destination.width {
        return Err(Error::SizeMismatch {
            name: "border placement".into(),
            expected: destination,
            actual: source,
        });
    }

    Ok((top, left))
}

pub fn validate_subpixel_copy(
    source: Size,
    destination: Size,
    x_shift: f32,
    y_shift: f32,
) -> Result<()> {
    if !x_shift.is_finite()
        || !y_shift.is_finite()
        || !(0.0..1.0).contains(&x_shift)
        || !(0.0..1.0).contains(&y_shift)
    {
        return Err(Error::out_of_range("subpixel shift"));
    }

    if destination.width > source.width || destination.height > source.height {
        return Err(Error::SizeMismatch {
            name: "subpixel source extent".into(),
            expected: source,
            actual: destination,
        });
    }

    Ok(())
}

pub fn validate_transpose_size(source: Size, destination: Size) -> Result<()> {
    if source.width == destination.height && source.height == destination.width {
        return Ok(());
    }

    Err(Error::SizeMismatch {
        name: "transpose image size".into(),
        expected: source,
        actual: destination,
    })
}

pub fn validate_scale_range(min: f32, max: f32) -> Result<()> {
    if !min.is_finite() || !max.is_finite() || max <= min {
        return Err(Error::out_of_range("scale range"));
    }

    Ok(())
}

pub fn validate_channel_order(order: &[i32], channels: i32) -> Result<()> {
    let mut seen = vec![false; channels as usize];
    for &channel in order {
        if !(0..channels).contains(&channel) {
            return Err(Error::out_of_range("channel order"));
        }

        let index = channel as usize;
        if seen[index] {
            return Err(Error::out_of_range("channel order"));
        }
        seen[index] = true;
    }

    Ok(())
}

pub fn validate_channel_order_with_fill(
    order: &[i32],
    source_channels: i32,
    destination_channels: i32,
) -> Result<()> {
    let mut seen = vec![false; source_channels as usize];
    for &channel in order {
        if channel == source_channels {
            continue;
        }
        if !(0..source_channels).contains(&channel) {
            return Err(Error::out_of_range("channel order"));
        }

        let index = channel as usize;
        if seen[index] {
            return Err(Error::out_of_range("channel order"));
        }
        seen[index] = true;
    }

    if order.len() != destination_channels as usize {
        return Err(Error::out_of_range("channel order"));
    }

    Ok(())
}

pub fn validate_channel_index(channel: usize, channels: usize) -> Result<()> {
    if channel < channels {
        return Ok(());
    }

    Err(Error::out_of_range("channel"))
}
