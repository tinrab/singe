use std::mem::size_of;

use singe_cuda::memory::DeviceMemory;
use singe_cuda::types::{Complex32, Complex64};

use crate::{
    error::Result,
    image::{
        geometry,
        memory::Image,
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

#[path = "tests/tests_image_advanced.rs"]
mod tests_image_advanced;
#[path = "tests/tests_image_color_geometry.rs"]
mod tests_image_color_geometry;
#[path = "tests/tests_image_core.rs"]
mod tests_image_core;
#[path = "tests/tests_image_statistics_filtering.rs"]
mod tests_image_statistics_filtering;
#[path = "tests/tests_pipeline_backing.rs"]
mod tests_pipeline_backing;
#[path = "tests/tests_signal_core.rs"]
mod tests_signal_core;
#[path = "tests/tests_signal_statistics.rs"]
mod tests_signal_statistics;
