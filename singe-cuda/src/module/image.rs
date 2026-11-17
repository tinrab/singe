use std::borrow::Cow;

use crate::error::{Error, Result};

/// Borrowed or owned bytes containing CUDA device code.
///
/// Every image is nonempty and is either NUL-terminated PTX text or a self-describing binary image.
/// [`ModuleImage::from_ptx`] checks PTX text, compilers in this crate produce their images directly, and [`ModuleImage::from_bytes_unchecked`] takes binaries on the caller's word.
#[derive(Debug, Clone)]
pub struct ModuleImage<'a> {
    data: Cow<'a, [u8]>,
}

impl<'a> ModuleImage<'a> {
    /// Creates an image from PTX text, appending a NUL terminator if `ptx` does not end with one.
    ///
    /// Borrowed text that already ends with a NUL byte stays borrowed.
    ///
    /// # Errors
    ///
    /// Returns `Error::Empty` if `ptx` holds no text before its optional NUL terminator.
    /// Returns `Error::InteriorNul` if the text contains a NUL byte before its end.
    pub fn from_ptx(ptx: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let ptx = ptx.into();
        let text = ptx.strip_suffix(&[0]).unwrap_or(&ptx[..]);
        if text.is_empty() {
            return Err(Error::empty("module image"));
        }

        if text.contains(&0) {
            return Err(Error::InteriorNul);
        }

        let data = if ptx.last() == Some(&0) {
            ptx
        } else {
            let mut terminated = ptx.into_owned();
            terminated.push(0);
            Cow::Owned(terminated)
        };
        Ok(Self { data })
    }

    // NVRTC and nvJitLink return NUL-terminated PTX or self-describing binaries, so only emptiness needs checking.
    #[cfg(any(feature = "nvrtc_13_4", feature = "nvjitlink_13_4"))]
    pub(crate) fn from_compiler_output(data: impl Into<Cow<'a, [u8]>>) -> Result<Self> {
        let data = data.into();
        if data.is_empty() {
            return Err(Error::empty("module image"));
        }
        Ok(Self { data })
    }

    /// Creates an image from device-code bytes without checking them.
    ///
    /// # Safety
    ///
    /// `data` must be nonempty and must be a complete cubin, fatbin, ELF, or Tile IR image whose headers describe sizes and offsets within `data`, or PTX text ending in its only NUL byte.
    /// CUDA's loaders take no length and trust the image headers, so other bytes make them read out of bounds.
    #[must_use]
    pub unsafe fn from_bytes_unchecked(data: impl Into<Cow<'a, [u8]>>) -> Self {
        Self { data: data.into() }
    }

    /// Returns a pointer to the image bytes, valid while the image storage remains alive.
    #[must_use]
    pub fn as_ptr(&self) -> *const () {
        self.data.as_ptr().cast()
    }

    /// Borrows the complete image bytes, including the NUL terminator of PTX text.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_ref()
    }
}
