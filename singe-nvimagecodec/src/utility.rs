use std::ffi::CString;

use singe_core::{checked_arithmetic, checked_conversions};

use crate::{
    error::{Error, Result},
    types::ProcessingStatus,
};

checked_conversions!(Error {
    to_u32 -> u32,
    to_i32 -> i32,
    to_u64 -> u64,
    to_usize -> usize,
});

checked_arithmetic!(Error { validate_same_len });

pub(crate) fn optional_cstring(value: Option<&str>) -> Result<Option<CString>> {
    value.map(CString::new).transpose().map_err(Into::into)
}

pub(crate) fn options_to_string<T>(
    entries: impl Iterator<Item = T>,
    mut write_entry: impl FnMut(T, &mut String),
) -> Option<String> {
    let mut options = String::new();
    for entry in entries {
        if !options.is_empty() {
            options.push(' ');
        }

        write_entry(entry, &mut options);
    }

    if options.is_empty() {
        None
    } else {
        Some(options)
    }
}

pub(crate) fn bool_option_value(value: bool) -> &'static str {
    if value { "1" } else { "0" }
}

pub(crate) fn validate_option_key(value: &str, name: &str) -> Result<()> {
    validate_option_field(value, name, |byte| byte == b':' || byte == b'=')
}

pub(crate) fn validate_option_value(value: &str, name: &str) -> Result<()> {
    validate_option_field(value, name, |byte| byte == b':' || byte == b'=')
}

fn validate_option_field(value: &str, name: &str, is_delimiter: impl Fn(u8) -> bool) -> Result<()> {
    if value.is_empty()
        || value
            .bytes()
            .any(|byte| byte.is_ascii_whitespace() || byte == 0 || is_delimiter(byte))
    {
        return Err(Error::InvalidOption {
            name: name.to_string().into(),
        });
    }

    Ok(())
}

pub(crate) fn collect_raw<T, R>(items: &[T], mut raw: impl FnMut(&T) -> R) -> Vec<R> {
    items.iter().map(&mut raw).collect()
}

pub(crate) fn validate_batch(left: usize, right: usize) -> Result<()> {
    if left == 0 {
        return Err(Error::Empty {
            name: "batch".into(),
        });
    }

    validate_same_len(left, right, "batch")
}

pub(crate) fn validate_processing_statuses(statuses: &[ProcessingStatus]) -> Result<()> {
    for (index, status) in statuses.iter().copied().enumerate() {
        if status != ProcessingStatus::SUCCESS {
            return Err(Error::OperationFailed {
                operation: format!("image {index}").into(),
                status,
            });
        }
    }

    Ok(())
}
