use std::{
    ffi::{CString, c_void},
    mem::size_of,
    panic::{self, AssertUnwindSafe},
    ptr,
};

use singe_core::checked_conversions;
use singe_nvimagecodec_sys as sys;

use crate::{
    debug::{DebugCallback, DebugMessage},
    error::{Error, Result},
    image::ImagePlaneInfo,
    types::{DebugMessageCategory, DebugMessageSeverity, ProcessingStatus},
};

checked_conversions!(Error {
    to_u32 -> u32,
    to_i32 -> i32,
    to_u64 -> u64,
    to_usize -> usize,
});

pub fn optional_cstring(value: Option<&str>) -> Result<Option<CString>> {
    value.map(CString::new).transpose().map_err(Into::into)
}

pub fn options_to_string<T>(
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

pub fn bool_option_value(value: bool) -> &'static str {
    if value { "1" } else { "0" }
}

pub fn validate_option_key(value: &str, name: &str) -> Result<()> {
    validate_option_field(value, name, |byte| byte == b':' || byte == b'=')
}

pub fn validate_option_value(value: &str, name: &str) -> Result<()> {
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

pub fn collect_raw<T, R>(items: &[T], mut raw: impl FnMut(&T) -> R) -> Vec<R> {
    items.iter().map(&mut raw).collect()
}

pub fn validate_batch(left: usize, right: usize) -> Result<()> {
    if left == 0 {
        return Err(Error::Empty {
            name: "batch".into(),
        });
    }

    if left != right {
        return Err(Error::LengthMismatch {
            name: "batch".into(),
            expected: left,
            actual: right,
        });
    }

    Ok(())
}

pub fn validate_processing_statuses(statuses: &[ProcessingStatus]) -> Result<()> {
    for (index, status) in statuses.iter().copied().enumerate() {
        if status != ProcessingStatus::SUCCESS {
            return Err(Error::ImageStatus { index, status });
        }
    }

    Ok(())
}

pub fn validate_jpeg2k_code_block_size(value: u32, name: &str) -> Result<()> {
    match value {
        32 | 64 => Ok(()),
        _ => Err(Error::Unsupported {
            name: name.to_string().into(),
        }),
    }
}

pub fn validate_host_buffer_len(buffer_len: usize, planes: &[ImagePlaneInfo]) -> Result<()> {
    validate_image_planes(planes)?;
    let required = host_buffer_len(planes)?;
    if buffer_len < required {
        return Err(Error::LengthMismatch {
            name: "buffer".into(),
            expected: required,
            actual: buffer_len,
        });
    }

    Ok(())
}

pub fn validate_image_planes(planes: &[ImagePlaneInfo]) -> Result<()> {
    if planes.is_empty() {
        return Err(Error::Empty {
            name: "planes".into(),
        });
    }

    if planes.len() > sys::NVIMGCODEC_MAX_NUM_PLANES as usize {
        return Err(Error::LengthMismatch {
            name: "planes".into(),
            expected: sys::NVIMGCODEC_MAX_NUM_PLANES as usize,
            actual: planes.len(),
        });
    }

    Ok(())
}

fn host_buffer_len(planes: &[ImagePlaneInfo]) -> Result<usize> {
    let mut len = 0_usize;
    for (index, plane) in planes.iter().enumerate() {
        len = len
            .checked_add(host_plane_len(index, *plane)?)
            .ok_or_else(|| Error::out_of_range("buffer"))?;
    }

    Ok(len)
}

fn host_plane_len(index: usize, plane: ImagePlaneInfo) -> Result<usize> {
    let width = usize::try_from(plane.width())
        .map_err(|_| Error::out_of_range(format!("planes[{index}].width")))?;
    let height = usize::try_from(plane.height())
        .map_err(|_| Error::out_of_range(format!("planes[{index}].height")))?;
    let channels = usize::try_from(plane.num_channels())
        .map_err(|_| Error::out_of_range(format!("planes[{index}].num_channels")))?;
    if width == 0 || height == 0 || channels == 0 {
        return Err(Error::Empty {
            name: format!("planes[{index}]").into(),
        });
    }

    let sample_bytes = plane
        .sample_type()
        .byte_size()
        .ok_or_else(|| Error::Unsupported {
            name: format!("planes[{index}].sample_type").into(),
        })?;
    let row_bytes = width
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(sample_bytes))
        .ok_or_else(|| Error::out_of_range(format!("planes[{index}].row_bytes")))?;
    let row_stride = usize::try_from(plane.row_stride())
        .map_err(|_| Error::out_of_range(format!("planes[{index}].row_stride")))?;
    if row_stride < row_bytes {
        return Err(Error::LengthMismatch {
            name: format!("planes[{index}].row_stride").into(),
            expected: row_bytes,
            actual: row_stride,
        });
    }

    row_stride
        .checked_mul(height)
        .ok_or_else(|| Error::out_of_range(format!("planes[{index}]")))
}

#[derive(Debug, Default)]
pub struct HostOutputBuffer {
    pub(crate) data: Vec<u8>,
}

pub unsafe extern "C" fn resize_host_output_buffer(
    context: *mut c_void,
    req_size: sys::size_t,
) -> *mut u8 {
    if context.is_null() {
        return ptr::null_mut();
    }

    // SAFETY: nvImageCodec passes back the `HostOutputBuffer` registered for this call, which outlives it.
    let output = unsafe { &mut *context.cast::<HostOutputBuffer>() };
    let Ok(req_size) = usize::try_from(req_size) else {
        return ptr::null_mut();
    };
    if req_size > output.data.capacity()
        && output
            .data
            .try_reserve_exact(req_size - output.data.len())
            .is_err()
    {
        return ptr::null_mut();
    }

    output.data.resize(req_size, 0);
    if output.data.is_empty() {
        ptr::NonNull::<u8>::dangling().as_ptr()
    } else {
        output.data.as_mut_ptr()
    }
}

pub unsafe extern "C" fn debug_callback(
    message_severity: sys::nvimgcodecDebugMessageSeverity_t,
    message_category: sys::nvimgcodecDebugMessageCategory_t,
    callback_data: *const sys::nvimgcodecDebugMessageData_t,
    user_data: *mut c_void,
) -> i32 {
    if user_data.is_null() || callback_data.is_null() {
        return 1;
    }

    // SAFETY: nvImageCodec passes back the `DebugCallback` registered with this messenger, which outlives it.
    let callback = unsafe { &*(user_data as *const DebugCallback) };
    let message =
        // SAFETY: nvImageCodec passes back the `DebugCallback` registered with this messenger, which outlives it.
        DebugMessage::from_raw(
            DebugMessageSeverity::from_bits_retain(message_severity as u32),
            DebugMessageCategory::from_bits_retain(message_category as u32),
            unsafe {
            &*callback_data
        });

    i32::from(panic::catch_unwind(AssertUnwindSafe(|| (callback.0)(message))).is_err())
}

/// Native structure with the `struct_type`, `struct_size`, and `struct_next` header that nvImageCodec checks.
pub trait NativeStruct {
    /// Returns the structure with its header filled in and the other fields at their defaults.
    fn init() -> Self;
}

macro_rules! impl_native_struct {
    ($($ty:ident => $kind:ident $({ $($field:ident: $value:expr),* $(,)? })?;)*) => {
        $(
            impl NativeStruct for sys::$ty {
                fn init() -> Self {
                    Self {
                        struct_type: sys::nvimgcodecStructureType_t::$kind,
                        struct_size: size_of::<Self>() as sys::size_t,
                        struct_next: ptr::null_mut(),
                        $($($field: $value,)*)?
                        ..Default::default()
                    }
                }
            }
        )*
    };
}

impl_native_struct! {
    nvimgcodecProperties_t => NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES;
    nvimgcodecInstanceCreateInfo_t => NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO { load_builtin_modules: 1, load_extension_modules: 1 };
    nvimgcodecExecutionParams_t => NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS { device_id: sys::NVIMGCODEC_DEVICE_CURRENT };
    nvimgcodecDecodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    nvimgcodecEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS { quality_type: sys::nvimgcodecQualityType_t::NVIMGCODEC_QUALITY_TYPE_DEFAULT };
    nvimgcodecJpegEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS;
    nvimgcodecJpeg2kEncodeParams_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS;
    nvimgcodecOrientation_t => NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION;
    nvimgcodecImagePlaneInfo_t => NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
    nvimgcodecImageInfo_t => NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO { orientation: sys::nvimgcodecOrientation_t::init() };
    nvimgcodecJpegImageInfo_t => NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO;
    nvimgcodecRegion_t => NVIMGCODEC_STRUCTURE_TYPE_REGION { out_of_bounds_policy: sys::nvimgcodecOutOfBoundsPolicy_t::NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT };
    nvimgcodecCodeStreamView_t => NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW { region: sys::nvimgcodecRegion_t::init() };
    nvimgcodecCodeStreamInfo_t => NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO;
    nvimgcodecMetadata_t => NVIMGCODEC_STRUCTURE_TYPE_METADATA;
    nvimgcodecDebugMessengerDesc_t => NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC;
}
