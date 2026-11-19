use crate::processor::image::Image;

// A mask of `width × height` pixels, white where `inside` holds.
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

// Moves a keypoint at the pixels of an image of `from` pixels, `(height, width)`, to `image`'s, `[x, y]`, through coordinates relative to each side in `F32`, truncated to whole pixels, as `post_process_keypoint_detection` and `post_process_keypoint_matching` compute them.
#[must_use]
pub(crate) fn rescaled_point([x, y]: [f32; 2], from: (i64, i64), image: &Image) -> [i32; 2] {
    let (height, width) = (from.0 as f32, from.1 as f32);
    [
        (x / width * image.width as f32) as i32,
        (y / height * image.height as f32) as i32,
    ]
}
