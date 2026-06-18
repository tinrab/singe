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
pub mod view;

pub use memory::{Image, SupportedImage, create};
pub use view::{
    AC4, AlphaIgnoredRgba, C1, C2, C3, C4, ChannelLayout, Channels2, Gray, ImageView, ImageViewMut,
    MaskView, MaskViewMut, Rgb, Rgba,
};

/// Low-level image wrappers that keep upstream NPP operation names and shapes.
///
/// Prefer `Image`, `ImageView`, and `crate::pipeline::ImagePipeline` for the
/// typed Rust API. This module is for porting NPP examples and for operations
/// that do not have a typed pipeline surface yet.
pub mod raw {
    pub mod arithmetic {
        pub use super::super::arithmetic::*;
    }

    pub mod color {
        pub use super::super::color::*;
    }

    pub mod compare {
        pub use super::super::compare::*;
    }

    pub mod exchange {
        pub use super::super::exchange::*;
    }

    pub mod filtering {
        pub use super::super::filtering::*;
    }

    pub mod geometry {
        pub use super::super::geometry::*;
    }

    pub mod linear {
        pub use super::super::linear::*;
    }

    pub mod morphology {
        pub use super::super::morphology::*;
    }

    pub mod statistics {
        pub use super::super::statistics::*;
    }

    pub mod threshold {
        pub use super::super::threshold::*;
    }
}
