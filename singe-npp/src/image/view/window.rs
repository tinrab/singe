use std::{marker::PhantomData, sync::Arc};

use singe_cuda::{
    context::Context,
    view::{DeviceRegion, DeviceRegionProvider},
};

use crate::{
    error::{Error, Result},
    image::view::{ChannelLayout, ImageView, image_byte_len},
    types::{Point, Rectangle, Size},
};

/// An image ROI retaining the initialized parent pixels needed by spatial kernels.
/// Its allocation region covers the parent image, including every possible checked halo.
#[derive(Debug)]
pub struct ImageWindow<'a, T, L> {
    image: ImageView<'a, T, L>,
    roi: Rectangle,
    pixels: ImageView<'a, T, L>,
}

impl<'a, T, L: ChannelLayout> ImageView<'a, T, L> {
    /// Selects the remainder of the initialized image starting at the supplied pixel offset.
    ///
    /// # Errors
    ///
    /// Returns an error if the offset is negative or outside the image, the remaining ROI is empty, or ROI-coordinate or byte-range calculations fail.
    pub fn window_from(self, offset: Point) -> Result<ImageWindow<'a, T, L>> {
        let width = self
            .size
            .width
            .checked_sub(offset.x)
            .ok_or_else(|| invalid("image window width"))?;
        let height = self
            .size
            .height
            .checked_sub(offset.y)
            .ok_or_else(|| invalid("image window height"))?;
        self.window(Rectangle {
            x: offset.x,
            y: offset.y,
            width,
            height,
        })
    }

    /// Selects an initialized ROI while retaining the parent extent for halo checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the ROI has a negative origin, nonpositive dimensions, or extends outside the parent image.
    /// Coordinate overflow, byte-offset or span overflow, and backing-region range errors are also propagated.
    pub fn window(self, roi: Rectangle) -> Result<ImageWindow<'a, T, L>> {
        roi.validate_within(self.size)?;
        let offset = (roi.y as usize)
            .checked_mul(self.step as usize)
            .and_then(|rows| {
                (roi.x as usize)
                    .checked_mul(L::CHANNELS)
                    .and_then(|samples| samples.checked_mul(size_of::<T>()))
                    .and_then(|columns| rows.checked_add(columns))
            })
            .ok_or_else(|| invalid("image window offset"))?;
        let bytes = image_byte_len::<T, L>(roi.size(), self.step)?;
        let end = offset
            .checked_add(bytes)
            .ok_or_else(|| invalid("image window end"))?;
        let pixels = ImageView {
            region: self.region.slice(offset..end)?,
            size: roi.size(),
            step: self.step,
            _t: PhantomData,
        };
        Ok(ImageWindow {
            image: self,
            roi,
            pixels,
        })
    }

    /// Selects every output position whose entire mask fits within initialized source pixels.
    /// The resulting dimensions are the source dimensions minus the mask dimensions plus one.
    ///
    /// # Errors
    ///
    /// Returns an error if the mask is nonpositive or larger than the image, the anchor lies outside the mask, or ROI-coordinate or byte-range calculations fail.
    pub fn valid_window(self, mask: Size, anchor: Point) -> Result<ImageWindow<'a, T, L>> {
        validate_anchor(mask, anchor)?;
        let width = self
            .size
            .width
            .checked_sub(mask.width)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("image window width"))?;
        let height = self
            .size
            .height
            .checked_sub(mask.height)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| invalid("image window height"))?;
        self.window(Rectangle {
            x: anchor.x,
            y: anchor.y,
            width,
            height,
        })
    }
}

impl<'a, T, L: ChannelLayout> ImageWindow<'a, T, L> {
    /// Returns the ROI width and height.
    #[must_use]
    pub fn size(&self) -> Size {
        self.roi.size()
    }

    /// Returns the ROI rectangle in parent-image coordinates.
    #[must_use]
    pub fn roi(&self) -> Rectangle {
        self.roi
    }

    /// Returns the full initialized parent-image dimensions.
    #[must_use]
    pub fn source_size(&self) -> Size {
        self.image.size
    }

    /// Returns the CUDA context retained by the parent image.
    #[must_use]
    pub fn context(&self) -> &Arc<Context> {
        self.image.context()
    }

    /// Borrows the initialized center pixels; raw spatial calls still require the checked halo.
    #[must_use]
    pub fn view(&self) -> ImageView<'a, T, L> {
        ImageView {
            region: self.pixels.region,
            size: self.pixels.size,
            step: self.pixels.step,
            _t: PhantomData,
        }
    }

    /// Checks that the mask's entire footprint stays within the initialized parent image.
    ///
    /// # Errors
    ///
    /// Returns an error if the mask is nonpositive, the anchor lies outside the mask, footprint arithmetic overflows, or the required footprint extends outside the initialized parent image.
    pub fn require_halo(&self, mask: Size, anchor: Point) -> Result<()> {
        validate_anchor(mask, anchor)?;
        let x = self
            .roi
            .x
            .checked_sub(anchor.x)
            .ok_or_else(|| invalid("image halo origin"))?;
        let y = self
            .roi
            .y
            .checked_sub(anchor.y)
            .ok_or_else(|| invalid("image halo origin"))?;
        let width = self
            .roi
            .width
            .checked_add(mask.width - 1)
            .ok_or_else(|| invalid("image halo width"))?;
        let height = self
            .roi
            .height
            .checked_add(mask.height - 1)
            .ok_or_else(|| invalid("image halo height"))?;
        Rectangle {
            x,
            y,
            width,
            height,
        }
        .validate_within(self.image.size)
    }
}

// SAFETY: the region retains the original initialized source borrow and its actual allocation metadata.
// Spatial adapters check each kernel footprint against that source before reading its halo.
unsafe impl<T, L> DeviceRegionProvider for ImageWindow<'_, T, L> {
    fn region(&self) -> DeviceRegion<'_> {
        self.image.region
    }
}

fn validate_anchor(mask: Size, anchor: Point) -> Result<()> {
    mask.validate()?;
    if anchor.x < 0 || anchor.y < 0 || anchor.x >= mask.width || anchor.y >= mask.height {
        return Err(invalid("image mask anchor"));
    }
    Ok(())
}

fn invalid(name: &str) -> Error {
    Error::out_of_range(name.to_owned())
}
