//! Fluent NPP pipelines.

mod image;
mod signal;
mod workspace;

pub(crate) use self::workspace::SignalAllocator;
pub use self::{
    image::{
        AlphaIgnoredRgba, Channels2, ColorTwistBatchConstantsMatrix4, ColorTwistConstants4,
        ColorTwistMatrix, ColorTwistMatrix4, CompressedMarkerLabels, ContiguousImage,
        ContourGeometry, ContourOrder, Contours, GradientVector, Gray, HistogramOfGradients,
        HoughLines, HoughLinesOutput, ImageBacking, ImageHistograms, ImageIndexedMinMax,
        ImageIndexedStatistic, ImageMeanStandardDeviation, ImageMinMax, ImagePipeline,
        ImageSquaredIntegral, ImageStatistic, InterpolatedContours, MarkerLabels, PlanarImage,
        Remap, Resize, ResizeBatchAdvanced, ResizeSqrPixel, ResizeSqrPixelAdvanced, Rgb, Rgba,
        Rotate, SemiplanarImage, SubsampledPlanarImage, UninitImagePipeline, WarpAffine,
        WarpAffineBatch, WarpPerspective, WarpPerspectiveBatch, WarpQuad,
    },
    signal::{
        SignalBacking, SignalIndexedValue, SignalMeanStandardDeviation, SignalMinMax,
        SignalPipeline, UninitSignalPipeline,
    },
    workspace::Workspace,
};

#[cfg(all(test, feature = "testing"))]
mod tests;
