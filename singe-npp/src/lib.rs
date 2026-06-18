//! Safe NVIDIA Performance Primitives wrappers for image and signal APIs.
//!
//! This crate wraps NPP version queries, stream contexts, image and signal
//! device memory, image processing pipelines, signal processing pipelines, typed
//! image views, channel layouts, sizes, rectangles, and temporary workspaces.
//!
//! # Example
//!
//! ```no_run
//! use singe_cuda::context::Context;
//! use singe_npp::{
//!     C3, ImagePipeline, ImageView, StreamContext, Workspace,
//!     types::{ComparisonOperation, InterpolationMode, Size},
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let ctx = Context::create()?;
//! let stream = ctx.create_stream()?;
//! let stream_context = StreamContext::create(&stream)?;
//! let mut workspace = Workspace::create();
//!
//! let device_memory = todo!();
//! let size = Size::new(640, 480);
//! let source = ImageView::<u8, C3>::from_memory(&device_memory, size)?;
//!
//! let output = ImagePipeline::from_view(&stream_context, &mut workspace, source)
//!     .resize(Size::new(256, 256), InterpolationMode::Lanczos)?
//!     .rgb_to_gray()?
//!     .threshold(128, ComparisonOperation::Less)?
//!     .filter_sharpen()?
//!     .finish()?;
//! let image = output.copy_to_host_vec()?;
//!
//! println!("NPP version: {:?}", singe_npp::version()?);
//! # Ok(())
//! # }
//! ```

pub mod context;
pub mod error;
pub mod image;
pub mod pipeline;
pub mod signal;
pub mod types;
pub mod workspace;

pub(crate) mod utility;

#[cfg(feature = "testing")]
pub mod testing;

// TODO: refactor into mods
pub use context::StreamContext;
pub use error::{Error, Result};
pub use image::{
    AC4, AlphaIgnoredRgba, C1, C2, C3, C4, ChannelLayout, Channels2, Gray, Image, ImageView,
    ImageViewMut, MaskView, MaskViewMut, Rgb, Rgba, SupportedImage,
};
pub use pipeline::{ImageBacking, ImagePipeline, SignalBacking, SignalPipeline, Workspace};
pub use signal::{Signal, SignalView, SignalViewMut, SupportedSignal};

use std::ptr;

use singe_npp_sys as sys;

/// Returns the loaded NPP library version.
///
/// # Errors
///
/// Returns an error if NPP returns a null version pointer.
pub fn version() -> Result<Version> {
    let raw = unsafe { sys::nppGetLibVersion() };
    if raw.is_null() {
        return Err(Error::NullHandle);
    }

    let raw = unsafe { ptr::read(raw) };
    Ok(Version {
        major: raw.major,
        minor: raw.minor,
        build: raw.build,
    })
}

/// NPP library version components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Version {
    /// Major version number.
    pub major: i32,
    /// Minor version number.
    pub minor: i32,
    /// Build number.
    pub build: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() -> Result<()> {
        let version = version()?;
        assert_ne!(version.major, 0);
        Ok(())
    }
}
