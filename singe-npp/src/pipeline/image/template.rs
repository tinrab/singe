use crate::{
    error::{Error, Result},
    types::Size,
};

#[path = "template_traits.rs"]
mod traits;
pub use traits::*;

#[macro_use]
#[path = "template_macros.rs"]
mod macros;

#[path = "template_dispatch.rs"]
mod dispatch;
#[path = "template_square_distance_dispatch.rs"]
mod square_distance_dispatch;

pub(super) fn template_full_size(source: Size, template: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_add(template.width)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| Error::OutOfRange {
                name: "template full destination width".into(),
            })?,
        height: source
            .height
            .checked_add(template.height)
            .and_then(|value| value.checked_sub(1))
            .ok_or_else(|| Error::OutOfRange {
                name: "template full destination height".into(),
            })?,
    })
}

pub(super) fn template_same_size(source: Size, _template: Size) -> Result<Size> {
    Ok(source)
}

pub(super) fn template_valid_size(source: Size, template: Size) -> Result<Size> {
    Ok(Size {
        width: source
            .width
            .checked_sub(template.width)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::OutOfRange {
                name: "template valid destination width".into(),
            })?,
        height: source
            .height
            .checked_sub(template.height)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| Error::OutOfRange {
                name: "template valid destination height".into(),
            })?,
    })
}
