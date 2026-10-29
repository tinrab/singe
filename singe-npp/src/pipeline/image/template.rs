use crate::{
    error::{Error, Result},
    types::Size,
};

mod traits;

pub use traits::*;

#[macro_use]
mod macros;

mod dispatch;
mod square_distance_dispatch;

pub fn template_full_size(source: Size, template: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_add(template.width)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| Error::out_of_range("template full destination width"))?,
        height: source
            .height
            .checked_add(template.height)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| Error::out_of_range("template full destination height"))?,
    })
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "all template size calculators share one fallible callback signature"
)]
pub fn template_same_size(source: Size, _template: Size) -> Result<Size> {
    Ok(source)
}

pub fn template_valid_size(source: Size, template: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_sub(template.width)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::out_of_range("template valid destination width"))?,
        height: source
            .height
            .checked_sub(template.height)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::out_of_range("template valid destination height"))?,
    })
}
