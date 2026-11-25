use singe::shape::spatial::Spatial2dI64;

use crate::processor::image::Image;

/// Returns a mask of `width × height` pixels, white where `inside` holds.
#[must_use]
pub fn mask(width: u32, height: u32, inside: impl IntoIterator<Item = bool>) -> Image {
    Image::grayscale(
        width,
        height,
        inside
            .into_iter()
            .map(|inside| if inside { 255 } else { 0 }),
    )
}

/// Returns a keypoint `[x, y]` at the pixels of an image of `from` pixels, moved to `image`'s, through coordinates relative to each side in `F32`, truncated to whole pixels, as `post_process_keypoint_detection` and `post_process_keypoint_matching` compute them.
#[must_use]
pub(crate) fn rescaled_point([x, y]: [f32; 2], from: Spatial2dI64, image: &Image) -> [i32; 2] {
    [
        (x / from.width as f32 * image.width as f32) as i32,
        (y / from.height as f32 * image.height as f32) as i32,
    ]
}
