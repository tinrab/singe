//! Fluent NPP pipelines.

mod image;
mod memory_context;
mod signal;
mod workspace;

#[cfg(all(test, feature = "testing"))]
mod tests;

pub use image::{AlphaIgnoredRgba, Channels2};
pub use image::{
    ColorTwistBatchConstantsMatrix4, ColorTwistConstants4, ColorTwistMatrix, ColorTwistMatrix4,
    CompressedMarkerLabels, ContiguousImage, GradientVector, Gray, HistogramOfGradients,
    HoughLines, ImageBacking, ImageHistograms, ImageIndexedMinMax, ImageIndexedStatistic,
    ImageMeanStandardDeviation, ImageMinMax, ImagePipeline, ImageSquaredIntegral, ImageStatistic,
    PlanarImage, Remap, Resize, ResizeBatchAdvanced, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rgb,
    Rgba, Rotate, SemiplanarImage, SubsampledPlanarImage, WarpAffine, WarpAffineBatch,
    WarpPerspective, WarpPerspectiveBatch, WarpQuad,
};
pub use signal::{
    SignalBacking, SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax, SignalPipeline,
};
pub use workspace::Workspace;
pub(crate) use workspace::{ImageAllocator, SignalAllocator};
