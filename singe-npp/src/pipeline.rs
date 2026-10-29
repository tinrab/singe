//! Fluent NPP pipelines.

mod image;
mod signal;
mod workspace;

#[cfg(all(test, feature = "testing"))]
mod tests;

pub use image::{
    AlphaIgnoredRgba, Channels2, ColorTwistBatchConstantsMatrix4, ColorTwistConstants4,
    ColorTwistMatrix, ColorTwistMatrix4, CompressedMarkerLabels, ContiguousImage, ContourGeometry,
    ContourOrder, Contours, GradientVector, Gray, HistogramOfGradients, HoughLines,
    HoughLinesOutput, ImageBacking, ImageHistograms, ImageIndexedMinMax, ImageIndexedStatistic,
    ImageMeanStandardDeviation, ImageMinMax, ImagePipeline, ImageSquaredIntegral, ImageStatistic,
    InterpolatedContours, MarkerLabels, PlanarImage, Remap, Resize, ResizeBatchAdvanced,
    ResizeSqrPixel, ResizeSqrPixelAdvanced, Rgb, Rgba, Rotate, SemiplanarImage,
    SubsampledPlanarImage, UninitImagePipeline, WarpAffine, WarpAffineBatch, WarpPerspective,
    WarpPerspectiveBatch, WarpQuad,
};
pub use signal::{
    SignalBacking, SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax, SignalPipeline,
    UninitSignalPipeline,
};
pub(crate) use workspace::SignalAllocator;
pub use workspace::Workspace;
