use super::*;

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum InterpolationMode {
    Undefined = sys::NppiInterpolationMode::NPPI_INTER_UNDEFINED as _,
    Nearest = sys::NppiInterpolationMode::NPPI_INTER_NN as _,
    Linear = sys::NppiInterpolationMode::NPPI_INTER_LINEAR as _,
    Cubic = sys::NppiInterpolationMode::NPPI_INTER_CUBIC as _,
    CubicBspline = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_BSPLINE as _,
    CubicCatmullRom = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_CATMULLROM as _,
    CubicB05C03 = sys::NppiInterpolationMode::NPPI_INTER_CUBIC2P_B05C03 as _,
    Super = sys::NppiInterpolationMode::NPPI_INTER_SUPER as _,
    Lanczos = sys::NppiInterpolationMode::NPPI_INTER_LANCZOS as _,
    Lanczos3Advanced = sys::NppiInterpolationMode::NPPI_INTER_LANCZOS3_ADVANCED as _,
    SmoothEdge = sys::NppiInterpolationMode::NPPI_SMOOTH_EDGE as _,
}

impl_enum_conversion!(sys::NppiInterpolationMode, InterpolationMode);

impl From<InterpolationMode> for i32 {
    fn from(value: InterpolationMode) -> Self {
        let raw: u32 = value.into();
        raw as _
    }
}

impl_enum_display!(InterpolationMode, {
    InterpolationMode::Undefined => "NPPI_INTER_UNDEFINED",
    InterpolationMode::Nearest => "NPPI_INTER_NN",
    InterpolationMode::Linear => "NPPI_INTER_LINEAR",
    InterpolationMode::Cubic => "NPPI_INTER_CUBIC",
    InterpolationMode::CubicBspline => "NPPI_INTER_CUBIC2P_BSPLINE",
    InterpolationMode::CubicCatmullRom => "NPPI_INTER_CUBIC2P_CATMULLROM",
    InterpolationMode::CubicB05C03 => "NPPI_INTER_CUBIC2P_B05C03",
    InterpolationMode::Super => "NPPI_INTER_SUPER",
    InterpolationMode::Lanczos => "NPPI_INTER_LANCZOS",
    InterpolationMode::Lanczos3Advanced => "NPPI_INTER_LANCZOS3_ADVANCED",
    InterpolationMode::SmoothEdge => "NPPI_SMOOTH_EDGE",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum RoundMode {
    Near = sys::NppRoundMode::NPP_RND_NEAR as _,
    Financial = sys::NppRoundMode::NPP_RND_FINANCIAL as _,
    Zero = sys::NppRoundMode::NPP_RND_ZERO as _,
}

impl_enum_conversion!(sys::NppRoundMode, RoundMode);

impl_enum_display!(RoundMode, {
    RoundMode::Near => "NPP_RND_NEAR",
    RoundMode::Financial => "NPP_RND_FINANCIAL",
    RoundMode::Zero => "NPP_RND_ZERO",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum HintAlgorithm {
    None = sys::NppHintAlgorithm::NPP_ALG_HINT_NONE as _,
    Fast = sys::NppHintAlgorithm::NPP_ALG_HINT_FAST as _,
    Accurate = sys::NppHintAlgorithm::NPP_ALG_HINT_ACCURATE as _,
}

impl_enum_conversion!(sys::NppHintAlgorithm, HintAlgorithm);

impl_enum_display!(HintAlgorithm, {
    HintAlgorithm::None => "NPP_ALG_HINT_NONE",
    HintAlgorithm::Fast => "NPP_ALG_HINT_FAST",
    HintAlgorithm::Accurate => "NPP_ALG_HINT_ACCURATE",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ImageNormalization {
    Infinity = sys::NppiNorm::nppiNormInf as _,
    L1 = sys::NppiNorm::nppiNormL1 as _,
    L2 = sys::NppiNorm::nppiNormL2 as _,
}

impl_enum_conversion!(sys::NppiNorm, ImageNormalization);

impl_enum_display!(ImageNormalization, {
    ImageNormalization::Infinity => "nppiNormInf",
    ImageNormalization::L1 => "nppiNormL1",
    ImageNormalization::L2 => "nppiNormL2",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum WatershedSegmentBoundaryType {
    None = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE as _,
    Black = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK as _,
    White = sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE as _,
    Contrast =
        sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST as _,
    BoundariesOnly =
        sys::NppiWatershedSegmentBoundaryType::NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY as _,
}

impl_enum_conversion!(
    sys::NppiWatershedSegmentBoundaryType,
    WatershedSegmentBoundaryType
);

impl_enum_display!(WatershedSegmentBoundaryType, {
    WatershedSegmentBoundaryType::None => "NPP_WATERSHED_SEGMENT_BOUNDARIES_NONE",
    WatershedSegmentBoundaryType::Black => "NPP_WATERSHED_SEGMENT_BOUNDARIES_BLACK",
    WatershedSegmentBoundaryType::White => "NPP_WATERSHED_SEGMENT_BOUNDARIES_WHITE",
    WatershedSegmentBoundaryType::Contrast => "NPP_WATERSHED_SEGMENT_BOUNDARIES_CONTRAST",
    WatershedSegmentBoundaryType::BoundariesOnly => "NPP_WATERSHED_SEGMENT_BOUNDARIES_ONLY",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ComparisonOperation {
    Less = sys::NppCmpOp::NPP_CMP_LESS as _,
    LessEqual = sys::NppCmpOp::NPP_CMP_LESS_EQ as _,
    Equal = sys::NppCmpOp::NPP_CMP_EQ as _,
    GreaterEqual = sys::NppCmpOp::NPP_CMP_GREATER_EQ as _,
    Greater = sys::NppCmpOp::NPP_CMP_GREATER as _,
}

impl_enum_conversion!(sys::NppCmpOp, ComparisonOperation);

impl_enum_display!(ComparisonOperation, {
    ComparisonOperation::Less => "NPP_CMP_LESS",
    ComparisonOperation::LessEqual => "NPP_CMP_LESS_EQ",
    ComparisonOperation::Equal => "NPP_CMP_EQ",
    ComparisonOperation::GreaterEqual => "NPP_CMP_GREATER_EQ",
    ComparisonOperation::Greater => "NPP_CMP_GREATER",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BorderType {
    Undefined = sys::NppiBorderType::NPP_BORDER_UNDEFINED as _,
    Constant = sys::NppiBorderType::NPP_BORDER_CONSTANT as _,
    Replicate = sys::NppiBorderType::NPP_BORDER_REPLICATE as _,
    Wrap = sys::NppiBorderType::NPP_BORDER_WRAP as _,
    Mirror = sys::NppiBorderType::NPP_BORDER_MIRROR as _,
}

impl_enum_conversion!(sys::NppiBorderType, BorderType);

impl_enum_display!(BorderType, {
    BorderType::Undefined => "NPP_BORDER_UNDEFINED",
    BorderType::Constant => "NPP_BORDER_CONSTANT",
    BorderType::Replicate => "NPP_BORDER_REPLICATE",
    BorderType::Wrap => "NPP_BORDER_WRAP",
    BorderType::Mirror => "NPP_BORDER_MIRROR",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum MaskSize {
    Mask1x3 = sys::NppiMaskSize::NPP_MASK_SIZE_1_X_3 as _,
    Mask1x5 = sys::NppiMaskSize::NPP_MASK_SIZE_1_X_5 as _,
    Mask3x1 = sys::NppiMaskSize::NPP_MASK_SIZE_3_X_1 as _,
    Mask5x1 = sys::NppiMaskSize::NPP_MASK_SIZE_5_X_1 as _,
    Mask3x3 = sys::NppiMaskSize::NPP_MASK_SIZE_3_X_3 as _,
    Mask5x5 = sys::NppiMaskSize::NPP_MASK_SIZE_5_X_5 as _,
    Mask7x7 = sys::NppiMaskSize::NPP_MASK_SIZE_7_X_7 as _,
    Mask9x9 = sys::NppiMaskSize::NPP_MASK_SIZE_9_X_9 as _,
    Mask11x11 = sys::NppiMaskSize::NPP_MASK_SIZE_11_X_11 as _,
    Mask13x13 = sys::NppiMaskSize::NPP_MASK_SIZE_13_X_13 as _,
    Mask15x15 = sys::NppiMaskSize::NPP_MASK_SIZE_15_X_15 as _,
}

impl_enum_conversion!(sys::NppiMaskSize, MaskSize);

impl_enum_display!(MaskSize, {
    MaskSize::Mask1x3 => "NPP_MASK_SIZE_1_X_3",
    MaskSize::Mask1x5 => "NPP_MASK_SIZE_1_X_5",
    MaskSize::Mask3x1 => "NPP_MASK_SIZE_3_X_1",
    MaskSize::Mask5x1 => "NPP_MASK_SIZE_5_X_1",
    MaskSize::Mask3x3 => "NPP_MASK_SIZE_3_X_3",
    MaskSize::Mask5x5 => "NPP_MASK_SIZE_5_X_5",
    MaskSize::Mask7x7 => "NPP_MASK_SIZE_7_X_7",
    MaskSize::Mask9x9 => "NPP_MASK_SIZE_9_X_9",
    MaskSize::Mask11x11 => "NPP_MASK_SIZE_11_X_11",
    MaskSize::Mask13x13 => "NPP_MASK_SIZE_13_X_13",
    MaskSize::Mask15x15 => "NPP_MASK_SIZE_15_X_15",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum Axis {
    Horizontal = sys::NppiAxis::NPP_HORIZONTAL_AXIS as _,
    Vertical = sys::NppiAxis::NPP_VERTICAL_AXIS as _,
    Both = sys::NppiAxis::NPP_BOTH_AXIS as _,
}

impl_enum_conversion!(sys::NppiAxis, Axis);

impl_enum_display!(Axis, {
    Axis::Horizontal => "NPP_HORIZONTAL_AXIS",
    Axis::Vertical => "NPP_VERTICAL_AXIS",
    Axis::Both => "NPP_BOTH_AXIS",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum BayerGridPosition {
    Bggr = sys::NppiBayerGridPosition::NPPI_BAYER_BGGR as _,
    Rggb = sys::NppiBayerGridPosition::NPPI_BAYER_RGGB as _,
    Gbrg = sys::NppiBayerGridPosition::NPPI_BAYER_GBRG as _,
    Grbg = sys::NppiBayerGridPosition::NPPI_BAYER_GRBG as _,
}

impl_enum_conversion!(sys::NppiBayerGridPosition, BayerGridPosition);

impl_enum_display!(BayerGridPosition, {
    BayerGridPosition::Bggr => "NPPI_BAYER_BGGR",
    BayerGridPosition::Rggb => "NPPI_BAYER_RGGB",
    BayerGridPosition::Gbrg => "NPPI_BAYER_GBRG",
    BayerGridPosition::Grbg => "NPPI_BAYER_GRBG",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum DifferentialKernel {
    Sobel = sys::NppiDifferentialKernel::NPP_FILTER_SOBEL as _,
    Scharr = sys::NppiDifferentialKernel::NPP_FILTER_SCHARR as _,
}

impl_enum_conversion!(sys::NppiDifferentialKernel, DifferentialKernel);

impl_enum_display!(DifferentialKernel, {
    DifferentialKernel::Sobel => "NPP_FILTER_SOBEL",
    DifferentialKernel::Scharr => "NPP_FILTER_SCHARR",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum AlphaOperation {
    Over = sys::NppiAlphaOp::NPPI_OP_ALPHA_OVER as _,
    In = sys::NppiAlphaOp::NPPI_OP_ALPHA_IN as _,
    Out = sys::NppiAlphaOp::NPPI_OP_ALPHA_OUT as _,
    Atop = sys::NppiAlphaOp::NPPI_OP_ALPHA_ATOP as _,
    Xor = sys::NppiAlphaOp::NPPI_OP_ALPHA_XOR as _,
    Plus = sys::NppiAlphaOp::NPPI_OP_ALPHA_PLUS as _,
    OverPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_OVER_PREMUL as _,
    InPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_IN_PREMUL as _,
    OutPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_OUT_PREMUL as _,
    AtopPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_ATOP_PREMUL as _,
    XorPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_XOR_PREMUL as _,
    PlusPremultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_PLUS_PREMUL as _,
    Premultiplied = sys::NppiAlphaOp::NPPI_OP_ALPHA_PREMUL as _,
}

impl_enum_conversion!(sys::NppiAlphaOp, AlphaOperation);

impl_enum_display!(AlphaOperation, {
    AlphaOperation::Over => "NPPI_OP_ALPHA_OVER",
    AlphaOperation::In => "NPPI_OP_ALPHA_IN",
    AlphaOperation::Out => "NPPI_OP_ALPHA_OUT",
    AlphaOperation::Atop => "NPPI_OP_ALPHA_ATOP",
    AlphaOperation::Xor => "NPPI_OP_ALPHA_XOR",
    AlphaOperation::Plus => "NPPI_OP_ALPHA_PLUS",
    AlphaOperation::OverPremultiplied => "NPPI_OP_ALPHA_OVER_PREMUL",
    AlphaOperation::InPremultiplied => "NPPI_OP_ALPHA_IN_PREMUL",
    AlphaOperation::OutPremultiplied => "NPPI_OP_ALPHA_OUT_PREMUL",
    AlphaOperation::AtopPremultiplied => "NPPI_OP_ALPHA_ATOP_PREMUL",
    AlphaOperation::XorPremultiplied => "NPPI_OP_ALPHA_XOR_PREMUL",
    AlphaOperation::PlusPremultiplied => "NPPI_OP_ALPHA_PLUS_PREMUL",
    AlphaOperation::Premultiplied => "NPPI_OP_ALPHA_PREMUL",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum ZeroCrossingType {
    SignChange = sys::NppsZCType::nppZCR as _,
    SignChangeXor = sys::NppsZCType::nppZCXor as _,
    SignChangeCountZero = sys::NppsZCType::nppZCC as _,
}

impl_enum_conversion!(sys::NppsZCType, ZeroCrossingType);

impl_enum_display!(ZeroCrossingType, {
    ZeroCrossingType::SignChange => "nppZCR",
    ZeroCrossingType::SignChangeXor => "nppZCXor",
    ZeroCrossingType::SignChangeCountZero => "nppZCC",
});

#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, TryFromPrimitive, IntoPrimitive)]
#[repr(u32)]
pub enum HuffmanTableType {
    Dc = sys::NppiHuffmanTableType::nppiDCTable as _,
    Ac = sys::NppiHuffmanTableType::nppiACTable as _,
}

impl_enum_conversion!(sys::NppiHuffmanTableType, HuffmanTableType);

impl_enum_display!(HuffmanTableType, {
    HuffmanTableType::Dc => "nppiDCTable",
    HuffmanTableType::Ac => "nppiACTable",
});
