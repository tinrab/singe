mod bits;
mod macros;

use std::hash::{Hash, Hasher};

pub use self::bits::BitHash;
use crate::{
    backend::registry::attributes::macros::{
        attribute_struct, bit_hash_traits, operation_attributes,
    },
    data_type::DataType,
    device::Device,
    error::Result,
    layout::Layout,
    scalar::Scalar,
    shape::Shape,
};

operation_attributes! {
    /// Attributes of an operation, excluding its input and output values.
    ///
    /// Each operation accepts one variant, named by its attribute kind in the operation table.
    /// Variants with several fields hold a typed struct that kernels and tensor methods can borrow directly.
    pub enum OperationAttributes {
        /// No attributes.
        None,
        /// Graph input name.
        Input(String),
        /// Embedded graph constant.
        Constant,
        /// Scalar arithmetic operand.
        Scalar(Scalar),
        /// One tensor dimension.
        Dimension(i64),
        /// Reduction dimensions and keepdim flag.
        Reduction(ReductionAttributes),
        /// Top-k selection parameters.
        TopK(TopKAttributes),
        /// Sorted dimension and direction.
        Sort(SortAttributes),
        /// Pad widths and fill value.
        Pad(PadAttributes),
        /// Transposed dimensions.
        Transpose(TransposeAttributes),
        /// Output dimension order.
        Permute(Vec<i64>),
        /// Requested output shape.
        Shape(Shape),
        /// Slice range along one dimension.
        Slice(SliceAttributes),
        /// Concatenation dimension.
        Concat(i64),
        /// Dimension selected by an index tensor.
        IndexSelect(i64),
        /// First dimension `index_put` writes and whether it adds instead of replacing.
        IndexPut(IndexPutAttributes),
        /// Number of repeats per key or value head.
        RepeatKv(i64),
        // The length of a signal a transform produces.
        Length(i64),
        /// Tensor contraction modes and scaling.
        Contraction(ContractionAttributes),
        /// Two-dimensional convolution parameters.
        Conv2d(Conv2dAttributes),
        // Three-dimensional convolution parameters.
        Conv3d(Conv3dAttributes),
        /// Two-dimensional pooling parameters.
        Pool2d(Pool2dAttributes),
        /// Batch normalization parameters.
        BatchNormalization(BatchNormalizationAttributes),
        /// Layer or RMS normalization parameters.
        LayerNormalization(LayerNormalizationAttributes),
        /// Scaled dot-product attention parameters.
        ScaledDotProductAttention(ScaledDotProductAttentionAttributes),
        /// Flattened dimension range.
        Flatten(FlattenAttributes),
        /// Cast target data type.
        Cast(DataType),
        /// Quantization target and channel axis.
        Quantization(QuantizationAttributes),
        /// Clamp range.
        Clamp(ClampAttributes),
        /// Linear interpolation weight.
        Lerp(f32),
        /// Copy target device.
        CopyTo(Device),
        /// Strided addressing into a flat operand.
        Strided(StridedAttributes),
        /// Random sampling parameters.
        Random(RandomAttributes),
        /// Scalar attributes of a custom operation, in the order its kernels read them.
        Custom(Vec<Scalar>),
    }
}

bit_hash_traits!(OperationAttributes);

attribute_struct! {
    /// Reduction dimensions and keepdim flag.

    #[derive(Debug, Clone)]
    pub struct ReductionAttributes {
        /// Reduced dimensions.
        pub dimensions: Vec<i64>,
        /// Whether reduced dimensions are retained with size one.
        pub keepdim: bool,
    }
}

attribute_struct! {
    /// Top-k selection parameters.

    #[derive(Debug, Clone, Copy)]
    pub struct TopKAttributes {
        /// Number of selected elements.
        pub k: i64,
        /// Selected dimension.
        pub dim: i64,
        /// Whether to select the largest values instead of the smallest.
        pub largest: bool,
        /// Whether selected values are sorted.
        pub sorted: bool,
    }
}

attribute_struct! {
    /// Sorted dimension and direction.
    #[derive(Debug, Clone, Copy)]
    pub struct SortAttributes {
        /// Sorted dimension.
        pub dim: i64,
        /// Whether values sort from largest to smallest.
        pub descending: bool,
    }
}

attribute_struct! {
    /// Pad widths and fill value.

    #[derive(Debug, Clone)]
    pub struct PadAttributes {
        /// Low and high padding per dimension.
        pub padding: Vec<(i64, i64)>,
        /// Fill value.
        pub value: f32,
    }
}

attribute_struct! {
    /// Transposed dimensions.

    #[derive(Debug, Clone, Copy)]
    pub struct TransposeAttributes {
        /// First swapped dimension.
        pub dim0: i64,
        /// Second swapped dimension.
        pub dim1: i64,
    }
}

attribute_struct! {
    /// Slice range along one dimension.

    #[derive(Debug, Clone, Copy)]
    pub struct SliceAttributes {
        /// Sliced dimension.
        pub dim: i64,
        /// Start index.
        pub start: i64,
        /// Number of selected elements.
        pub length: i64,
        /// Distance between selected elements, at least one.
        pub step: i64,
    }
}

attribute_struct! {
    /// Tensor contraction modes and scaling.

    #[derive(Debug, Clone)]
    pub struct ContractionAttributes {
        /// Modes of the left operand.
        pub a_mode: Vec<i64>,
        /// Modes of the right operand.
        pub b_mode: Vec<i64>,
        /// Modes of the accumulator operand.
        pub c_mode: Vec<i64>,
        /// Modes of the output.
        pub d_mode: Vec<i64>,
        /// Scale of the contraction.
        pub alpha: f32,
        /// Scale of the accumulator.
        pub beta: f32,
    }
}

attribute_struct! {
    /// Two-dimensional convolution parameters.

    #[derive(Debug, Clone, Copy)]
    pub struct Conv2dAttributes {
        /// Whether the optional bias input is present.
        pub has_bias: bool,
        /// Stride height and width.
        pub stride: (i64, i64),
        /// Padding top, bottom, left, and right.
        pub padding: (i64, i64, i64, i64),
        /// Dilation height and width.
        pub dilation: (i64, i64),
        /// Group count.
        pub groups: i64,
    }
}

attribute_struct! {
    // Three-dimensional convolution parameters, each tuple in depth, height, and width order, with each axis padded equally on both sides, as PyTorch's `conv3d` takes it.
    #[derive(Debug, Clone, Copy)]
    pub struct Conv3dAttributes {
        pub has_bias: bool,
        pub stride: (i64, i64, i64),
        pub padding: (i64, i64, i64),
        pub dilation: (i64, i64, i64),
        pub groups: i64,
    }
}

attribute_struct! {
    /// Two-dimensional pooling parameters.

    #[derive(Debug, Clone, Copy)]
    pub struct Pool2dAttributes {
        /// Kernel height and width.
        pub kernel_size: (i64, i64),
        /// Stride height and width.
        pub stride: (i64, i64),
        /// Padding top, bottom, left, and right.
        pub padding: (i64, i64, i64, i64),
        /// Dilation height and width.
        pub dilation: (i64, i64),
        /// Whether the output shape rounds up.
        pub ceil_mode: bool,
        /// Whether average pooling counts padded elements, or `None` for max pooling.
        pub count_include_pad: Option<bool>,
    }
}

attribute_struct! {
    /// Batch normalization parameters.
    #[derive(Debug, Clone, Copy)]
    pub struct BatchNormalizationAttributes {
        /// Whether the optional weight input is present.
        pub has_weight: bool,
        /// Whether the optional bias input is present.
        pub has_bias: bool,
        /// Whether the optional running mean input is present.
        pub has_running_mean: bool,
        /// Whether the optional running variance input is present.
        pub has_running_var: bool,
        /// Whether statistics come from the current batch.
        pub training: bool,
        /// Running statistic momentum.
        pub momentum: f32,
        /// Numerical epsilon.
        pub eps: f32,
    }
}

attribute_struct! {
    /// Layer or RMS normalization parameters.

    #[derive(Debug, Clone)]
    pub struct LayerNormalizationAttributes {
        /// Normalized trailing shape.
        pub normalized_shape: Vec<i64>,
        /// Whether the optional weight input is present.
        pub has_weight: bool,
        /// Whether the optional bias input is present.
        pub has_bias: bool,
        /// Numerical epsilon.
        pub eps: f32,
    }
}

attribute_struct! {
    /// Scaled dot-product attention parameters.

    #[derive(Debug, Clone, Copy)]
    pub struct ScaledDotProductAttentionAttributes {
        /// Explicit attention scale, or `None` for the reciprocal square root of the head dimension.
        pub scale: Option<f32>,
        /// Whether causal masking is enabled.
        pub causal: bool,
        /// The keys each query sees before itself, itself included: key `j` is hidden from query `i` when `j + window <= i`, as sliding-window attention hides it.
        pub window: Option<i64>,
        /// The bound of `softcap · tanh(score / softcap)`, applied to the scaled scores before masking, as Gemma 2 caps its attention logits.
        pub softcap: Option<f32>,
        // Whether a fourth operand, of the query's type and broadcast to the scores' `[B, H, L, S]`, adds to the capped scores before the softmax, hiding keys where it is negative infinity.
        pub masked: bool,
        // Whether a further operand of the query's type holds one logit per query head that joins every row's softmax and weighs no value, as GPT-OSS's attention sinks do.
        pub sinks: bool,
        // Whether a last operand, `[B]` of `I32`, holds each batch's count of keys, hiding the keys past it from every query of that batch, as a padded batch's lengths do; a count of zero gives zeros.
        pub key_lengths: bool,
    }
}

attribute_struct! {
    /// Flattened dimension range.

    #[derive(Debug, Clone, Copy)]
    pub struct FlattenAttributes {
        /// First flattened dimension.
        pub start_dim: i64,
        /// Last flattened dimension.
        pub end_dim: i64,
    }
}

/// How a dequantized tensor packs two 4-bit values in each byte, the low nibble first, along its last axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedFormat {
    /// Integers: signed from -8 to 7 in an `I8` tensor, unsigned from 0 to 15 in a `U8` tensor.
    Int4,
    /// FP4 with two exponent bits and one mantissa bit: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives.
    Float4E2M1,
}

attribute_struct! {
    /// Quantization target and channel axis; scales and zero points are operands.

    #[derive(Debug, Clone, Copy)]
    pub struct QuantizationAttributes {
        /// Dimension that per-channel or per-block scales and zero points run along, or `None` for one scale and zero point.
        pub axis: Option<i64>,
        /// Number of consecutive positions along `axis` that share a scale and zero point, or `None` for one per position.
        pub block_size: Option<i64>,
        /// Quantized output data type for quantize, or `None` for dequantize.
        pub data_type: Option<DataType>,
        /// For dequantize, how the input packs two 4-bit values per byte along its last axis, which doubles that axis in the output, or `None` for one value per element.
        pub packed: Option<PackedFormat>,
    }
}

attribute_struct! {
    /// Strided addressing of the elements of a flat, rank-one operand.

    #[derive(Debug, Clone)]
    pub struct StridedAttributes {
        /// Addressed dimensions.
        pub dimensions: Vec<i64>,
        /// Element stride of each dimension.
        pub strides: Vec<i64>,
        /// Element offset of the first addressed element.
        pub offset: usize,
    }
}

impl StridedAttributes {
    /// Returns the strided addressing of `layout`, relative to the start of its allocation.
    #[must_use]
    pub fn from_layout(layout: &Layout) -> Self {
        Self {
            dimensions: layout.dimensions().to_vec(),
            strides: layout.strides().to_vec(),
            offset: layout.storage_offset(),
        }
    }

    /// Returns this addressing as a layout whose offset is shifted by `base_offset`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions or strides are invalid.
    pub fn layout(&self, base_offset: usize) -> Result<Layout> {
        Ok(Layout::with_strides(&self.dimensions, &self.strides)?
            .with_offset(base_offset + self.offset))
    }
}

attribute_struct! {
    /// Parameters of a random operation; the generator state is its last operand.

    #[derive(Debug, Clone)]
    pub struct RandomAttributes {
        /// Output dimensions of `uniform` and `normal`; empty for operations shaped by their input.
        pub dimensions: Vec<i64>,
        /// Output data type of `uniform` and `normal`.
        pub data_type: DataType,
        /// Lower bound or mean.
        pub first: f32,
        /// Upper bound or standard deviation.
        pub second: f32,
        /// Samples per row for `multinomial`.
        pub samples: i64,
        /// Philox blocks to skip past the generator state's offset.
        pub offset: u64,
    }
}

attribute_struct! {
    /// First dimension `index_put` writes and whether it adds instead of replacing.
    #[derive(Debug, Clone, Copy)]
    pub struct IndexPutAttributes {
        /// First indexed dimension.
        pub dim: i64,
        /// Whether values add to the selected elements instead of replacing them.
        pub accumulate: bool,
    }
}

attribute_struct! {
    /// Clamp range.

    #[derive(Debug, Clone, Copy)]
    pub struct ClampAttributes {
        /// Minimum value.
        pub min: f32,
        /// Maximum value.
        pub max: f32,
    }
}

macro_rules! attribute_conversions {
    ($($variant:ident($attributes:ident)),+ $(,)?) => {
        $(
            impl From<$attributes> for OperationAttributes {
                fn from(attributes: $attributes) -> Self {
                    Self::$variant(attributes)
                }
            }
        )+
    };
}

attribute_conversions!(
    Reduction(ReductionAttributes),
    TopK(TopKAttributes),
    Sort(SortAttributes),
    Pad(PadAttributes),
    Transpose(TransposeAttributes),
    Slice(SliceAttributes),
    Contraction(ContractionAttributes),
    Conv2d(Conv2dAttributes),
    Conv3d(Conv3dAttributes),
    Pool2d(Pool2dAttributes),
    BatchNormalization(BatchNormalizationAttributes),
    LayerNormalization(LayerNormalizationAttributes),
    ScaledDotProductAttention(ScaledDotProductAttentionAttributes),
    Flatten(FlattenAttributes),
    Quantization(QuantizationAttributes),
    Clamp(ClampAttributes),
    IndexPut(IndexPutAttributes),
    Strided(StridedAttributes),
    Random(RandomAttributes),
);

/// Generates accessors that borrow the attributes of one variant.
macro_rules! borrowed_accessors {
    ($($(#[doc = $doc:literal])+ $method:ident => $variant:ident: $type:ty),+ $(,)?) => {
        impl OperationAttributes {
            $(
                $(#[doc = $doc])+
                #[must_use]
                pub fn $method(&self) -> Option<&$type> {
                    match self {
                        Self::$variant(attributes) => Some(attributes),
                        _ => None,
                    }
                }
            )+
        }
    };
}

/// Generates accessors that copy the attribute value of one variant.
macro_rules! copied_accessors {
    ($($(#[doc = $doc:literal])+ $method:ident => $variant:ident: $type:ty),+ $(,)?) => {
        impl OperationAttributes {
            $(
                $(#[doc = $doc])+
                #[must_use]
                pub fn $method(&self) -> Option<$type> {
                    match self {
                        Self::$variant(value) => Some(*value),
                        _ => None,
                    }
                }
            )+
        }
    };
}

borrowed_accessors! {
    /// Returns the reduction attributes.
    reduction => Reduction: ReductionAttributes,
    /// Returns the top-k attributes.
    topk => TopK: TopKAttributes,
    /// Returns the pad attributes.
    pad => Pad: PadAttributes,
    /// Returns the transposed dimensions.
    transpose => Transpose: TransposeAttributes,
    /// Returns the permutation order.
    permutation_order => Permute: [i64],
    /// Returns the requested shape.
    shape => Shape: Shape,
    /// Returns the slice attributes.
    slice => Slice: SliceAttributes,
    /// Returns the contraction attributes.
    contraction => Contraction: ContractionAttributes,
    /// Returns the two-dimensional convolution attributes.
    conv2d => Conv2d: Conv2dAttributes,
    /// Returns the three-dimensional convolution attributes.
    conv3d => Conv3d: Conv3dAttributes,
    /// Returns the two-dimensional pooling attributes.
    pool2d => Pool2d: Pool2dAttributes,
    /// Returns the batch normalization attributes.
    batch_normalization => BatchNormalization: BatchNormalizationAttributes,
    /// Returns the layer or RMS normalization attributes.
    layer_normalization => LayerNormalization: LayerNormalizationAttributes,
    /// Returns the scaled dot-product attention attributes.
    scaled_dot_product_attention => ScaledDotProductAttention: ScaledDotProductAttentionAttributes,
    /// Returns the flattened dimension range.
    flatten => Flatten: FlattenAttributes,
    /// Returns the quantization attributes.
    quantization => Quantization: QuantizationAttributes,
    /// Returns the clamp range.
    clamp => Clamp: ClampAttributes,
    /// Returns the sorted dimension and direction.
    sort => Sort: SortAttributes,
    /// Returns the `index_put` dimension and accumulation mode.
    index_put => IndexPut: IndexPutAttributes,
    /// Returns the strided addressing.
    strided => Strided: StridedAttributes,
    /// Returns the random sampling parameters.
    random => Random: RandomAttributes,
    /// Returns the scalar attributes of a custom operation.
    custom => Custom: [Scalar],
}

copied_accessors! {
    /// Returns the scalar operand.
    scalar => Scalar: Scalar,
    /// Returns the selected dimension.
    dimension => Dimension: i64,
    /// Returns the concatenation dimension.
    concat_dimension => Concat: i64,
    /// Returns the dimension selected by an index tensor.
    index_select_dimension => IndexSelect: i64,
    /// Returns the number of repeats per key or value head.
    repeat_kv_repeats => RepeatKv: i64,
    /// Returns the length of the signal a transform produces.
    length => Length: i64,
    /// Returns the cast target data type.
    cast_data_type => Cast: DataType,
    /// Returns the linear interpolation weight.
    lerp_t => Lerp: f32,
    /// Returns the copy target device.
    copy_device => CopyTo: Device,
}
