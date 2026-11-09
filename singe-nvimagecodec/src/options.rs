use std::num::NonZeroUsize;

use crate::{
    error::Result,
    utility::{bool_option_value, options_to_string, validate_option_key, validate_option_value},
};

/// Options for creating a [`Decoder`](crate::decode::Decoder), in the order they were added.
#[derive(Debug, Clone, Default)]
pub struct DecoderOptions {
    entries: Vec<DecoderOption>,
}

/// Options for creating an [`Encoder`](crate::encode::Encoder), in the order they were added.
#[derive(Debug, Clone, Default)]
pub struct EncoderOptions {
    entries: Vec<EncoderOption>,
}

/// Option for creating a decoder.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DecoderOption {
    /// Option for nvImageCodec itself rather than one codec.
    Global(GlobalCodecOption),
    /// Option for every decoder that supports it.
    AnyDecoder(AnyDecoderOption),
    /// Configures the hybrid nvJPEG decoder.
    NvJpegCuda(NvJpegCudaDecoderOption),
    /// Configures the hardware nvJPEG decoder.
    NvJpegHardware(NvJpegHardwareDecoderOption),
    /// Configures the lossless nvJPEG decoder.
    NvJpegLossless(NvJpegLosslessDecoderOption),
    /// Configures the CUDA JPEG 2000 decoder.
    NvJpeg2kCuda(NvJpeg2kCudaDecoderOption),
    /// Configures the CPU libjpeg-turbo decoder.
    LibJpegTurbo(LibJpegTurboDecoderOption),
    /// Option given by name and value, passed to nvImageCodec unchanged.
    Custom(CustomCodecOption),
}

/// Option for creating an encoder.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum EncoderOption {
    /// Option for nvImageCodec itself rather than one codec.
    Global(GlobalCodecOption),
    /// Option given by name and value, passed to nvImageCodec unchanged.
    Custom(CustomCodecOption),
}

/// Options for nvImageCodec itself, for decoders and encoders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GlobalCodecOption {
    /// Number of CUDA streams that nvImageCodec uses.
    ///
    /// nvImageCodec accepts 1 up to the number of worker threads; only the lower bound is checked here.
    NumCudaStreams(NonZeroUsize),
}

/// Options for every decoder that supports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum AnyDecoderOption {
    /// Enables interpolated chroma upsampling to reduce color bleeding around edges.
    ///
    /// `false` selects faster nearest-neighbor upsampling.
    /// The JPEG decoders enable it by default.
    FancyUpsampling(bool),
}

/// Options for the `nvjpeg_cuda_decoder` CPU/GPU hybrid JPEG plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NvJpegCudaDecoderOption {
    /// Enables interpolated chroma upsampling to reduce color bleeding around edges.
    ///
    /// `false` selects faster nearest-neighbor upsampling.
    /// The JPEG decoders enable it by default.
    FancyUpsampling(bool),
    /// Pixel count above which Huffman decoding runs on the GPU; 1,000,000 by default.
    HybridHuffmanThreshold(usize),
    /// Extra bytes to add to each device allocation.
    DeviceMemoryPadding(usize),
    /// Extra bytes to add to each pinned host allocation.
    HostMemoryPadding(usize),
    /// Whether to allocate the decoder's internal buffers up front.
    PreallocateBuffers(bool),
    /// Experimental extra flags passed to nvJPEG when the decoder is created.
    ExtraFlags(usize),
}

/// Options for the `nvjpeg_hw_decoder` hardware JPEG plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NvJpegHardwareDecoderOption {
    /// Enables interpolated chroma upsampling to reduce color bleeding around edges.
    ///
    /// `false` selects faster nearest-neighbor upsampling.
    /// The JPEG decoders enable it by default.
    FancyUpsampling(bool),
    /// Expected image width in pixels, for allocating buffers up front.
    PreallocateWidthHint(usize),
    /// Expected image height in pixels, for allocating buffers up front.
    PreallocateHeightHint(usize),
    /// Expected batch size, for allocating buffers up front.
    PreallocateBatchSize(usize),
    /// Experimental extra flags passed to nvJPEG when the decoder is created.
    ExtraFlags(usize),
}

/// Options for the `nvjpeg_lossless_decoder` plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NvJpegLosslessDecoderOption {
    /// Enables interpolated chroma upsampling to reduce color bleeding around edges.
    ///
    /// `false` selects faster nearest-neighbor upsampling.
    /// The JPEG decoders enable it by default.
    FancyUpsampling(bool),
    /// Experimental extra flags passed to nvJPEG when the decoder is created.
    ExtraFlags(usize),
}

/// Options for the `nvjpeg2k_cuda_decoder` JPEG 2000 plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NvJpeg2kCudaDecoderOption {
    /// Number of JPEG 2000 tiles decoded in parallel, shared across all worker threads.
    ///
    /// The default is 16.
    NumParallelTiles(NonZeroUsize),
    /// Extra bytes to add to each device allocation.
    DeviceMemoryPadding(usize),
    /// Extra bytes to add to each pinned host allocation.
    HostMemoryPadding(usize),
}

/// Options for the `libjpeg_turbo_decoder` CPU JPEG plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LibJpegTurboDecoderOption {
    /// Enables interpolated chroma upsampling to reduce color bleeding around edges.
    ///
    /// `false` selects faster nearest-neighbor upsampling.
    /// The JPEG decoders enable it by default.
    FancyUpsampling(bool),
    /// Enables the faster, less accurate integer inverse discrete cosine transform.
    ///
    /// `false` by default, which uses the more accurate algorithm.
    FastIdct(bool),
}

/// Option given by name and value, for one codec or for all of them.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CustomCodecOption {
    codec_id: Option<String>,
    name: String,
    value: String,
}

impl DecoderOptions {
    /// Creates an empty option list, which leaves every option at its default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an option after the existing ones, without replacing any.
    pub fn push(&mut self, option: impl Into<DecoderOption>) -> &mut Self {
        self.entries.push(option.into());
        self
    }

    /// Adds an option after the existing ones and returns the list.
    #[must_use]
    pub fn with(mut self, option: impl Into<DecoderOption>) -> Self {
        self.push(option);
        self
    }

    /// Returns whether no options have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the options as nvImageCodec's option string, or `None` if there are none.
    ///
    /// Options are separated by spaces, in the order they were added, as `:name=value` for every codec or `codec_id:name=value` for one.
    /// Repeated options are kept, and nvImageCodec decides which one applies.
    pub fn render(&self) -> Option<String> {
        options_to_string(
            self.entries.iter().map(DecoderOption::entry),
            OptionEntry::write_to,
        )
    }
}

impl EncoderOptions {
    /// Creates an empty option list, which leaves every option at its default.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an option after the existing ones, without replacing any.
    pub fn push(&mut self, option: impl Into<EncoderOption>) -> &mut Self {
        self.entries.push(option.into());
        self
    }

    /// Adds an option after the existing ones and returns the list.
    #[must_use]
    pub fn with(mut self, option: impl Into<EncoderOption>) -> Self {
        self.push(option);
        self
    }

    /// Returns whether no options have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns the options as nvImageCodec's option string, or `None` if there are none.
    ///
    /// Options are separated by spaces, in the order they were added, as `:name=value` for every codec or `codec_id:name=value` for one.
    /// Repeated options are kept, and nvImageCodec decides which one applies.
    pub fn render(&self) -> Option<String> {
        options_to_string(
            self.entries.iter().map(EncoderOption::entry),
            OptionEntry::write_to,
        )
    }
}

#[expect(
    clippy::needless_pass_by_value,
    reason = "option constructors accept both owned values and borrowed values through ToString"
)]
impl CustomCodecOption {
    /// Creates an option that applies to every codec.
    ///
    /// # Errors
    ///
    /// Returns an error if a field is empty or contains whitespace, a NUL byte, `:`, or `=`.
    /// The codec checks the name and value when the decoder or encoder is created.
    pub fn global(name: impl ToString, value: impl ToString) -> Result<Self> {
        Self::create(None, name, value)
    }

    /// Creates an option that applies only to the codec with ID `codec_id`.
    ///
    /// # Errors
    ///
    /// Returns an error if a field is empty or contains whitespace, a NUL byte, `:`, or `=`.
    /// The codec checks the name and value when the decoder or encoder is created.
    pub fn for_codec(
        codec_id: impl ToString,
        name: impl ToString,
        value: impl ToString,
    ) -> Result<Self> {
        Self::create(Some(codec_id.to_string()), name, value)
    }

    fn create(codec_id: Option<String>, name: impl ToString, value: impl ToString) -> Result<Self> {
        let name = name.to_string();
        let value = value.to_string();
        if let Some(codec_id) = codec_id.as_deref() {
            validate_option_key(codec_id, "codec_id")?;
        }

        validate_option_key(&name, "name")?;
        validate_option_value(&value, "value")?;

        Ok(Self {
            codec_id,
            name,
            value,
        })
    }

    fn entry(&self) -> OptionEntry<'_> {
        OptionEntry {
            codec_id: self.codec_id.as_deref(),
            name: &self.name,
            value: OptionValue::Borrowed(&self.value),
        }
    }
}

impl DecoderOption {
    fn entry(&self) -> OptionEntry<'_> {
        match self {
            Self::Global(option) => option.entry(),
            Self::AnyDecoder(option) => option.entry(),
            Self::NvJpegCuda(option) => option.entry(),
            Self::NvJpegHardware(option) => option.entry(),
            Self::NvJpegLossless(option) => option.entry(),
            Self::NvJpeg2kCuda(option) => option.entry(),
            Self::LibJpegTurbo(option) => option.entry(),
            Self::Custom(option) => option.entry(),
        }
    }
}

impl EncoderOption {
    fn entry(&self) -> OptionEntry<'_> {
        match self {
            Self::Global(option) => option.entry(),
            Self::Custom(option) => option.entry(),
        }
    }
}

impl GlobalCodecOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::NumCudaStreams(value) => OptionEntry::global("num_cuda_streams", value.get()),
        }
    }
}

impl AnyDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::FancyUpsampling(value) => {
                OptionEntry::global("fancy_upsampling", bool_option_value(value))
            }
        }
    }
}

impl NvJpegCudaDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::FancyUpsampling(value) => OptionEntry::for_codec(
                NVJPEG_CUDA_DECODER,
                "fancy_upsampling",
                bool_option_value(value),
            ),
            Self::HybridHuffmanThreshold(value) => {
                OptionEntry::for_codec(NVJPEG_CUDA_DECODER, "hybrid_huffman_threshold", value)
            }
            Self::DeviceMemoryPadding(value) => {
                OptionEntry::for_codec(NVJPEG_CUDA_DECODER, "device_memory_padding", value)
            }
            Self::HostMemoryPadding(value) => {
                OptionEntry::for_codec(NVJPEG_CUDA_DECODER, "host_memory_padding", value)
            }
            Self::PreallocateBuffers(value) => OptionEntry::for_codec(
                NVJPEG_CUDA_DECODER,
                "preallocate_buffers",
                bool_option_value(value),
            ),
            Self::ExtraFlags(value) => {
                OptionEntry::for_codec(NVJPEG_CUDA_DECODER, "extra_flags", value)
            }
        }
    }
}

impl NvJpegHardwareDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::FancyUpsampling(value) => OptionEntry::for_codec(
                NVJPEG_HARDWARE_DECODER,
                "fancy_upsampling",
                bool_option_value(value),
            ),
            Self::PreallocateWidthHint(value) => {
                OptionEntry::for_codec(NVJPEG_HARDWARE_DECODER, "preallocate_width_hint", value)
            }
            Self::PreallocateHeightHint(value) => {
                OptionEntry::for_codec(NVJPEG_HARDWARE_DECODER, "preallocate_height_hint", value)
            }
            Self::PreallocateBatchSize(value) => {
                OptionEntry::for_codec(NVJPEG_HARDWARE_DECODER, "preallocate_batch_size", value)
            }
            Self::ExtraFlags(value) => {
                OptionEntry::for_codec(NVJPEG_HARDWARE_DECODER, "extra_flags", value)
            }
        }
    }
}

impl NvJpegLosslessDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::FancyUpsampling(value) => OptionEntry::for_codec(
                NVJPEG_LOSSLESS_DECODER,
                "fancy_upsampling",
                bool_option_value(value),
            ),
            Self::ExtraFlags(value) => {
                OptionEntry::for_codec(NVJPEG_LOSSLESS_DECODER, "extra_flags", value)
            }
        }
    }
}

impl NvJpeg2kCudaDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::NumParallelTiles(value) => {
                OptionEntry::for_codec(NVJPEG2K_CUDA_DECODER, "num_parallel_tiles", value.get())
            }
            Self::DeviceMemoryPadding(value) => {
                OptionEntry::for_codec(NVJPEG2K_CUDA_DECODER, "device_memory_padding", value)
            }
            Self::HostMemoryPadding(value) => {
                OptionEntry::for_codec(NVJPEG2K_CUDA_DECODER, "host_memory_padding", value)
            }
        }
    }
}

impl LibJpegTurboDecoderOption {
    fn entry(self) -> OptionEntry<'static> {
        match self {
            Self::FancyUpsampling(value) => OptionEntry::for_codec(
                LIBJPEG_TURBO_DECODER,
                "fancy_upsampling",
                bool_option_value(value),
            ),
            Self::FastIdct(value) => {
                OptionEntry::for_codec(LIBJPEG_TURBO_DECODER, "fast_idct", bool_option_value(value))
            }
        }
    }
}

impl From<GlobalCodecOption> for DecoderOption {
    fn from(option: GlobalCodecOption) -> Self {
        Self::Global(option)
    }
}

impl From<AnyDecoderOption> for DecoderOption {
    fn from(option: AnyDecoderOption) -> Self {
        Self::AnyDecoder(option)
    }
}

impl From<NvJpegCudaDecoderOption> for DecoderOption {
    fn from(option: NvJpegCudaDecoderOption) -> Self {
        Self::NvJpegCuda(option)
    }
}

impl From<NvJpegHardwareDecoderOption> for DecoderOption {
    fn from(option: NvJpegHardwareDecoderOption) -> Self {
        Self::NvJpegHardware(option)
    }
}

impl From<NvJpegLosslessDecoderOption> for DecoderOption {
    fn from(option: NvJpegLosslessDecoderOption) -> Self {
        Self::NvJpegLossless(option)
    }
}

impl From<NvJpeg2kCudaDecoderOption> for DecoderOption {
    fn from(option: NvJpeg2kCudaDecoderOption) -> Self {
        Self::NvJpeg2kCuda(option)
    }
}

impl From<LibJpegTurboDecoderOption> for DecoderOption {
    fn from(option: LibJpegTurboDecoderOption) -> Self {
        Self::LibJpegTurbo(option)
    }
}

impl From<CustomCodecOption> for DecoderOption {
    fn from(option: CustomCodecOption) -> Self {
        Self::Custom(option)
    }
}

impl From<GlobalCodecOption> for EncoderOption {
    fn from(option: GlobalCodecOption) -> Self {
        Self::Global(option)
    }
}

impl From<CustomCodecOption> for EncoderOption {
    fn from(option: CustomCodecOption) -> Self {
        Self::Custom(option)
    }
}

#[derive(Debug, Clone, Copy)]
struct OptionEntry<'a> {
    codec_id: Option<&'a str>,
    name: &'a str,
    value: OptionValue<'a>,
}

#[derive(Debug, Clone, Copy)]
enum OptionValue<'a> {
    Static(&'static str),
    Borrowed(&'a str),
    Usize(usize),
}

impl<'a> OptionEntry<'a> {
    fn global(name: &'a str, value: impl Into<OptionValue<'a>>) -> Self {
        Self {
            codec_id: None,
            name,
            value: value.into(),
        }
    }

    fn for_codec(codec_id: &'a str, name: &'a str, value: impl Into<OptionValue<'a>>) -> Self {
        Self {
            codec_id: Some(codec_id),
            name,
            value: value.into(),
        }
    }

    fn write_to(self, out: &mut String) {
        if let Some(codec_id) = self.codec_id {
            out.push_str(codec_id);
        }

        out.push(':');
        out.push_str(self.name);
        out.push('=');
        match self.value {
            OptionValue::Static(value) | OptionValue::Borrowed(value) => out.push_str(value),
            OptionValue::Usize(value) => out.push_str(&value.to_string()),
        }
    }
}

impl From<&'static str> for OptionValue<'_> {
    fn from(value: &'static str) -> Self {
        Self::Static(value)
    }
}

impl From<usize> for OptionValue<'_> {
    fn from(value: usize) -> Self {
        Self::Usize(value)
    }
}

const NVJPEG_CUDA_DECODER: &str = "nvjpeg_cuda_decoder";
const NVJPEG_HARDWARE_DECODER: &str = "nvjpeg_hw_decoder";
const NVJPEG_LOSSLESS_DECODER: &str = "nvjpeg_lossless_decoder";
const NVJPEG2K_CUDA_DECODER: &str = "nvjpeg2k_cuda_decoder";
const LIBJPEG_TURBO_DECODER: &str = "libjpeg_turbo_decoder";

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use crate::error::Error;

    use super::*;

    #[test]
    fn test_decoder_render_options() -> Result<()> {
        let options = DecoderOptions::new()
            .with(AnyDecoderOption::FancyUpsampling(false))
            .with(NvJpegCudaDecoderOption::HybridHuffmanThreshold(0))
            .with(NvJpegCudaDecoderOption::DeviceMemoryPadding(1024));

        let options = options.render().ok_or_else(|| Error::Empty {
            name: "options".into(),
        })?;

        assert_eq!(
            options,
            ":fancy_upsampling=0 nvjpeg_cuda_decoder:hybrid_huffman_threshold=0 nvjpeg_cuda_decoder:device_memory_padding=1024",
        );
        Ok(())
    }

    #[test]
    fn test_encoder_render_options() -> Result<()> {
        let streams =
            NonZeroUsize::new(4).ok_or_else(|| Error::out_of_range("num_cuda_streams"))?;
        let options = EncoderOptions::new().with(GlobalCodecOption::NumCudaStreams(streams));

        let options = options.render().ok_or_else(|| Error::Empty {
            name: "options".into(),
        })?;

        assert_eq!(options, ":num_cuda_streams=4");
        Ok(())
    }

    #[test]
    fn test_custom_options_validate_fields() {
        let result = CustomCodecOption::for_codec("bad id", "name", "1");

        assert!(matches!(result, Err(Error::InvalidOption { name }) if name == "codec_id"));

        let result = CustomCodecOption::global("name", "bad=value");

        assert!(matches!(result, Err(Error::InvalidOption { name }) if name == "value"));
    }
}
