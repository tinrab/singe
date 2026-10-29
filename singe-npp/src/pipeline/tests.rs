use std::mem::size_of;

use singe_cuda::{
    memory::DeviceMemory,
    types::{Complex32, Complex64},
};

use crate::{
    error::Result,
    image::{
        geometry,
        view::{AC4, C1, C3, C4, ImageView},
    },
    pipeline::{ImageBacking, ImagePipeline, SignalBacking, SignalPipeline, Workspace},
    signal::{memory::Signal, view::SignalView},
    testing::create_stream_context,
    types::{
        AffineCoefficients, AlphaOperation, Axis, BayerGridPosition, BorderType, ColorSpace,
        ComparisonOperation, ComplexI16, ComplexI32, ComplexI64, ConnectedRegion,
        DifferentialKernel, HintAlgorithm, HistogramOfGradientsConfig, ImageNormalization,
        InterpolationMode, MaskSize, PerspectiveCoefficients, Point, Point32f, PointPolar,
        QuadrangleF64, Rectangle, RoundMode, Size, WatershedSegmentBoundaryType, ZeroCrossingType,
    },
};

mod tests_image_advanced;
mod tests_image_color_geometry;
mod tests_image_core;
mod tests_image_statistics_filtering;
mod tests_pipeline_backing;
mod tests_signal_core;
mod tests_signal_statistics;
