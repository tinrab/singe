pub(crate) mod arithmetic;
mod arithmetic_validation;
pub(crate) mod color;
mod color_descriptors;
pub(crate) mod compare;
pub(crate) mod exchange;
mod exchange_validation;
pub(crate) mod filtering;
mod filtering_distance;
mod filtering_flood_fill;
mod filtering_hog;
mod filtering_labels;
mod filtering_separable;
mod filtering_validation;
pub(crate) mod geometry;
mod geometry_helpers;
pub(crate) mod linear;
/// Native NPP image allocation owners, typed constructors, and compact-copy helpers.
pub mod memory;
pub(crate) mod morphology;
pub(crate) mod statistics;
mod statistics_batch_metrics;
mod statistics_histograms;
mod statistics_integral;
mod statistics_pair_metrics;
mod statistics_template;
mod statistics_validation;
pub(crate) mod threshold;

/// Video pixel formats, validated plane storage, and color conversions.
pub mod video;

/// Typed image descriptors, channel layouts, planar views, masks, and windows.
pub mod view;

pub use memory::{Image, SupportedImage, create_uninit};
pub use view::{
    AC4, AllChannels, AlphaIgnoredRgba, C1, C2, C3, C4, ChannelLayout, Channels2, Gray,
    ImageOutputView, ImageView, ImageViewMut, ImageWindow, MaskOutputView, MaskView, MaskViewMut,
    PlanarImageOutputView, Rgb, Rgba,
};

/// Low-level image wrappers that keep upstream NPP operation names and shapes.
///
/// Prefer `Image`, `ImageView`, and `crate::pipeline::ImagePipeline` for the
/// typed Rust API. This module is for porting NPP examples and for operations
/// that do not have a typed pipeline surface yet.
///
/// Native enqueue operations require callers to retain allocations and every pointer argument through completion.
/// Order prior producers and conflicting accesses on the actual stream and CUDA context.
/// Read inputs and read-write arguments must contain valid initialized values when accessed.
/// Output views grant writable storage, without claiming that submission initialized the complete image.
/// Preserve unwritten alpha, masked pixels, and uncovered geometry explicitly before exposing initialized values.
/// Captured calls require retained resources on every replay and do not establish initialization during recording.
pub mod raw {
    /// Raw video-format conversions over image planes.
    pub mod video {
        pub use crate::image::video::raw::*;
    }

    /// Per-pixel arithmetic, constants, scaling, and logical operations.
    pub mod arithmetic {
        pub use crate::image::arithmetic::*;
    }

    /// Color-space, gamma, channel, and lookup-table conversions.
    pub mod color {
        pub use crate::image::color::*;
    }

    /// Image comparisons producing output masks.
    pub mod compare {
        pub use crate::image::compare::*;
    }

    /// Image copies, pixel fills, element conversions, and channel rearrangement.
    pub mod exchange {
        pub use crate::image::exchange::*;
    }

    /// Spatial filters, feature extraction, and region labeling.
    pub mod filtering {
        pub use crate::image::filtering::*;
    }

    /// Resize, remap, affine, perspective, and related geometric transforms.
    pub mod geometry {
        pub use crate::image::geometry::*;
    }

    /// Complex-image magnitude and squared-magnitude operations.
    pub mod linear {
        pub use crate::image::linear::*;
    }

    /// Image dilation, erosion, opening, closing, and composite morphology.
    pub mod morphology {
        pub use crate::image::morphology::*;
    }

    /// Image reductions, histograms, quality metrics, profiles, and integral images.
    pub mod statistics {
        pub use crate::image::statistics::*;
    }

    /// Pixel thresholds and fused difference-threshold operations.
    pub mod threshold {
        pub use crate::image::threshold::*;
    }
}
