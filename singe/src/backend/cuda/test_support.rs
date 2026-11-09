use crate::error::{Error, Result};

pub fn is_shape_mismatch(error: &Error) -> bool {
    matches!(
        error,
        Error::ShapeMismatch { .. } | Error::LengthMismatch { .. }
    )
}

pub fn is_device_mismatch(error: &Error) -> bool {
    matches!(error, Error::DeviceMismatch { .. })
}

pub fn assert_shape_mismatch<T>(result: Result<T>) {
    assert!(result.is_err(), "expected shape mismatch, got ok");
    let Err(error) = result else {
        return;
    };
    assert!(
        is_shape_mismatch(&error),
        "expected shape mismatch, got {error:?}"
    );
}

pub fn assert_device_mismatch<T>(result: Result<T>) {
    assert!(result.is_err(), "expected device mismatch, got ok");
    let Err(error) = result else {
        return;
    };
    assert!(
        is_device_mismatch(&error),
        "expected device mismatch, got {error:?}"
    );
}
