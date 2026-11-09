//! JPEG 2000 parsing and decoding.

use std::{
    ffi::CString,
    mem::{self, MaybeUninit},
    path::Path,
    ptr,
    sync::Arc,
};

use singe_cuda::{
    context::Context as CudaContext, error::Error as CudaError, memory::DeviceMemory,
    stream::Stream as CudaStream, view::DeviceRegionProvider,
};
use singe_nvjpeg_sys::k2 as sys;

use crate::{
    allocator::AllocatorSet,
    error::{Error, Result},
    image::{PlaneMut, Size},
    k2::{
        Backend, ColorSpace, Context, Image, ImageComponentInfo, ImageFormat, ImageInfo,
        ImagePixel, ImageType, decode_pixel_type, image::validate_plane,
        validate_image_component_count,
    },
    try_ffi_k2,
    utility::{checked_mul, complete, to_non_zero_u64},
};

/// Region of a JPEG 2000 image or tile to decode, in pixels, with exclusive ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeArea {
    pub(crate) start_x: u32,
    pub(crate) end_x: u32,
    pub(crate) start_y: u32,
    pub(crate) end_y: u32,
}

/// Settings for a JPEG 2000 [`Decoder`]: output layout, region, and RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodeOptions {
    pub(crate) output_format: ImageFormat,
    pub(crate) area: Option<DecodeArea>,
    pub(crate) force_rgb: bool,
}

/// State that a JPEG 2000 decoder keeps across calls.
#[derive(Debug)]
pub struct DecodeState {
    pub(crate) raw: sys::nvjpeg2kDecodeState_t,
    pub(crate) handle: Context,
}

impl DecodeState {
    /// Takes ownership of native decoder state created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpeg2kDecodeState_t, handle: &Context) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Releases the native handle without destroying it.
    ///
    /// Destroy it, or pass it back to [`from_raw`](Self::from_raw), while its [`Context`] is alive.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvjpeg2kDecodeState_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }
}

/// JPEG 2000 decoder.
///
/// It owns its [`Context`] and decoder state, and reuses them across calls.
#[derive(Debug)]
pub struct Decoder {
    pub(crate) handle: Context,
    pub(crate) state: DecodeState,
    pub(crate) options: DecodeOptions,
}

/// Builder for a JPEG 2000 [`Decoder`] with a custom backend, options, or allocators.
#[derive(Debug, Clone, Copy)]
pub struct DecoderBuilder {
    pub(crate) backend: Backend,
    pub(crate) options: DecodeOptions,
    pub(crate) allocators: Option<AllocatorSet>,
}

/// Size and row pitch of the memory for one decoded component.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DecodedComponentSpec {
    pub(crate) info: ImageComponentInfo,
    pub(crate) pitch_in_bytes: usize,
    pub(crate) byte_len: usize,
}

/// One decoded component in device memory.
#[derive(Debug)]
pub struct DecodedComponent<T = u8> {
    pub(crate) data: DeviceMemory<T>,
    pub(crate) spec: DecodedComponentSpec,
}

/// Decoded JPEG 2000 image in device memory, with one plane per component.
#[derive(Debug)]
pub struct DecodedImage<T = u8> {
    pub(crate) pixel_type: ImageType,
    pub(crate) components: Vec<DecodedComponent<T>>,
}

/// Parsed JPEG 2000 data, from which image information can be read.
///
/// It keeps the data it parsed until it is destroyed.
#[derive(Debug)]
pub struct CodeStream {
    pub(crate) raw: sys::nvjpeg2kStream_t,
    pub(crate) handle: Context,
    backing: Vec<Box<[u8]>>,
    parsed: bool,
}

/// Decode settings for nvJPEG 2000: region, RGB conversion, and output layout.
#[derive(Debug)]
pub struct DecodeParams {
    pub(crate) raw: sys::nvjpeg2kDecodeParams_t,
    pub(crate) handle: Context,
}

impl DecodeArea {
    /// Creates a region from (`start_x`, `start_y`) up to, but not including, (`end_x`, `end_y`).
    ///
    /// The image and tile bounds are checked when the region is used.
    ///
    /// # Errors
    ///
    /// Returns an error if a start is not less than its end.
    pub fn create(start_x: u32, end_x: u32, start_y: u32, end_y: u32) -> Result<Self> {
        if start_x >= end_x {
            return Err(Error::OutOfRange {
                name: "decode area width".into(),
            });
        }

        if start_y >= end_y {
            return Err(Error::OutOfRange {
                name: "decode area height".into(),
            });
        }
        Ok(Self {
            start_x,
            end_x,
            start_y,
            end_y,
        })
    }

    /// Returns the first column.
    #[must_use]
    pub const fn start_x(self) -> u32 {
        self.start_x
    }

    /// Returns the column after the last one.
    #[must_use]
    pub const fn end_x(self) -> u32 {
        self.end_x
    }

    /// Returns the first row.
    #[must_use]
    pub const fn start_y(self) -> u32 {
        self.start_y
    }

    /// Returns the row after the last one.
    #[must_use]
    pub const fn end_y(self) -> u32 {
        self.end_y
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.end_x - self.start_x
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.end_y - self.start_y
    }
}

impl DecodeOptions {
    /// Creates options with nvJPEG 2000's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            output_format: ImageFormat::Planar,
            area: None,
            force_rgb: false,
        }
    }

    /// Sets whether the output is planar or interleaved.
    #[must_use]
    pub const fn with_output_format(mut self, output_format: ImageFormat) -> Self {
        self.output_format = output_format;
        self
    }

    /// Decodes only the region `area` of the image or tile.
    #[must_use]
    pub const fn with_area(mut self, area: DecodeArea) -> Self {
        self.area = Some(area);
        self
    }

    /// Sets whether to convert images with 4:2:2 or 4:2:0 chroma subsampling to RGB.
    #[must_use]
    pub const fn with_force_rgb(mut self, force_rgb: bool) -> Self {
        self.force_rgb = force_rgb;
        self
    }

    /// Returns whether the output is planar or interleaved.
    #[must_use]
    pub const fn output_format(self) -> ImageFormat {
        self.output_format
    }

    /// Returns the region to decode, or `None` for the whole image.
    #[must_use]
    pub const fn area(self) -> Option<DecodeArea> {
        self.area
    }

    /// Returns whether subsampled images are converted to RGB.
    #[must_use]
    pub const fn force_rgb(self) -> bool {
        self.force_rgb
    }

    fn apply_to(self, params: &mut DecodeParams) -> Result<()> {
        params.set_output_format(self.output_format)?;
        if let Some(area) = self.area {
            params.set_area(area)?;
        }

        if self.force_rgb {
            params.set_rgb_output(true)?;
        }
        Ok(())
    }

    /// Returns the size each output component must cover for the region and layout.
    fn output_sizes(self, code_stream: &CodeStream) -> Result<Vec<Size>> {
        let image = code_stream.image_info()?;
        validate_image_component_count(image.component_count() as usize)?;
        let full = image.size();
        if full.is_empty() {
            return Err(Error::OutOfRange {
                name: "image dimensions".into(),
            });
        }

        let infos = code_stream.component_infos()?;
        let (x0, y0, x1, y1) = if let Some(area) = self.area {
            if area.end_x > full.width() || area.end_y > full.height() {
                return Err(Error::OutOfRange {
                    name: "decode area".into(),
                });
            }
            (area.start_x, area.start_y, area.end_x, area.end_y)
        } else {
            (0, 0, full.width(), full.height())
        };
        if self.output_format == ImageFormat::Interleaved {
            return Ok(vec![Size::new(x1 - x0, y1 - y0)]);
        }
        infos
            .iter()
            .enumerate()
            .map(|(index, info)| {
                if self.force_rgb && index < 3 && infos.len() >= 3 {
                    return Ok(Size::new(x1 - x0, y1 - y0));
                }

                let component = info.size();
                if component.is_empty() {
                    return Err(Error::OutOfRange {
                        name: "component dimensions".into(),
                    });
                }

                if self.area.is_none() {
                    return Ok(component);
                }

                // Sample coverage includes a boundary sample for an unaligned crop.
                // Output views remain write-only; this does not promote untouched samples.
                let dx = full.width().div_ceil(component.width());
                let dy = full.height().div_ceil(component.height());
                Ok(Size::new(
                    x1.div_ceil(dx) - x0 / dx,
                    y1.div_ceil(dy) - y0 / dy,
                ))
            })
            .collect()
    }

    pub(crate) fn validate_device_allocation(self) -> Result<()> {
        if self.output_format != ImageFormat::Planar {
            return Err(Error::UnsupportedDecodeConfiguration {
                reason: "decode_to_device supports only planar JPEG 2000 output".into(),
            });
        }

        if self.area.is_some() {
            return Err(Error::UnsupportedDecodeConfiguration {
                reason: "decode_to_device does not infer JPEG 2000 ROI output dimensions".into(),
            });
        }

        if self.force_rgb {
            return Err(Error::UnsupportedDecodeConfiguration {
                reason: "decode_to_device does not infer forced RGB JPEG 2000 output components"
                    .into(),
            });
        }
        Ok(())
    }
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl DecodedComponentSpec {
    /// Returns the size and pitch of tightly packed memory for a component described by `info`, with samples of `pixel_type`.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero or the size overflows `usize`.
    pub fn for_component(info: ImageComponentInfo, pixel_type: ImageType) -> Result<Self> {
        let size = info.size();
        if size.is_empty() {
            return Err(Error::Empty {
                name: "component".into(),
            });
        }

        let element_size = pixel_type.element_size();
        let pitch_in_bytes =
            (size.width as usize)
                .checked_mul(element_size)
                .ok_or(Error::OutOfRange {
                    name: "pitch_in_bytes".into(),
                })?;
        let byte_len =
            pitch_in_bytes
                .checked_mul(size.height as usize)
                .ok_or(Error::OutOfRange {
                    name: "byte_len".into(),
                })?;
        Ok(Self {
            info,
            pitch_in_bytes,
            byte_len,
        })
    }

    /// Returns the component information.
    #[must_use]
    pub const fn info(self) -> ImageComponentInfo {
        self.info
    }

    /// Returns the row pitch in bytes.
    #[must_use]
    pub const fn pitch_in_bytes(self) -> usize {
        self.pitch_in_bytes
    }

    /// Returns the size of the memory in bytes.
    #[must_use]
    pub const fn byte_len(self) -> usize {
        self.byte_len
    }
}

impl<T> DecodedComponent<T> {
    /// Returns the decoded samples.
    #[must_use]
    pub const fn data(&self) -> &DeviceMemory<T> {
        &self.data
    }

    /// Returns the decoded samples mutably.
    pub const fn data_mut(&mut self) -> &mut DeviceMemory<T> {
        &mut self.data
    }

    /// Returns the size and layout of the memory.
    #[must_use]
    pub const fn spec(&self) -> DecodedComponentSpec {
        self.spec
    }

    /// Returns the decoded samples and their layout.
    #[must_use]
    pub fn into_parts(self) -> (DeviceMemory<T>, DecodedComponentSpec) {
        (self.data, self.spec)
    }
}

impl DecodedImage {
    /// Allocates uninitialized device memory in `context` for every component in `component_infos`.
    ///
    /// The sample type is chosen from the components' bit depth and signedness.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no components or more than four, a dimension is zero, a size overflows, an allocation fails, the bit depth is not supported, or the components need different sample types.
    pub fn create_uninit(
        context: &Arc<CudaContext>,
        component_infos: &[ImageComponentInfo],
    ) -> Result<DecodedImage<MaybeUninit<u8>>> {
        let pixel_type = decode_pixel_type(component_infos)?;
        Self::create_uninit_with_pixel_type(context, component_infos, pixel_type)
    }

    /// Allocates uninitialized device memory in `context` for every component in `component_infos`, with samples of `pixel_type`.
    ///
    /// It does not check that `pixel_type` suits the components' bit depth.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no components or more than four, a dimension is zero, a size overflows, or an allocation fails.
    pub fn create_uninit_with_pixel_type(
        context: &Arc<CudaContext>,
        component_infos: &[ImageComponentInfo],
        pixel_type: ImageType,
    ) -> Result<DecodedImage<MaybeUninit<u8>>> {
        if component_infos.is_empty() {
            return Err(Error::Empty {
                name: "component_infos".into(),
            });
        }

        validate_image_component_count(component_infos.len())?;
        let components = component_infos
            .iter()
            .copied()
            .map(|info| {
                let spec = DecodedComponentSpec::for_component(info, pixel_type)?;
                Ok(DecodedComponent {
                    data: DeviceMemory::<u8>::create_uninit(context, spec.byte_len)?,
                    spec,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(DecodedImage {
            pixel_type,
            components,
        })
    }
}

impl<T> DecodedImage<T> {
    /// Returns the sample type.
    #[must_use]
    pub const fn pixel_type(&self) -> ImageType {
        self.pixel_type
    }

    /// Returns the decoded components.
    #[must_use]
    pub fn components(&self) -> &[DecodedComponent<T>] {
        &self.components
    }

    /// Returns the decoded components mutably.
    pub fn components_mut(&mut self) -> &mut [DecodedComponent<T>] {
        &mut self.components
    }

    /// Returns the sample type and the decoded components.
    #[must_use]
    pub fn into_parts(self) -> (ImageType, Vec<DecodedComponent<T>>) {
        (self.pixel_type, self.components)
    }

    pub(crate) fn descriptor(&mut self) -> Result<Image> {
        if self.components.is_empty() {
            return Err(Error::Empty {
                name: "components".into(),
            });
        }

        validate_image_component_count(self.components.len())?;
        let pixel_data = self
            .components
            .iter_mut()
            .map(|component| component.data.as_mut_ptr().cast())
            .collect::<Vec<_>>();
        let pitch_in_bytes = self
            .components
            .iter()
            .map(|component| component.spec.pitch_in_bytes)
            .collect::<Vec<_>>();
        // SAFETY: the plane pointers and pitches describe buffers that stay borrowed for the image lifetime.
        unsafe {
            Image::from_raw_parts(
                self.components.len(),
                pixel_data,
                pitch_in_bytes,
                self.pixel_type,
            )
        }
    }
}

impl Decoder {
    /// Returns a builder for a decoder with a custom backend, options, or allocators.
    #[must_use]
    pub const fn builder() -> DecoderBuilder {
        DecoderBuilder::new()
    }

    /// Creates a decoder in `cuda_context` that uses `backend` and the default options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the handle or state.
    pub fn create(cuda_context: &Arc<CudaContext>, backend: Backend) -> Result<Self> {
        Self::builder().backend(backend).create(cuda_context)
    }

    /// Creates a decoder in `cuda_context` that uses `backend` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the handle or state.
    pub fn create_with_options(
        cuda_context: &Arc<CudaContext>,
        backend: Backend,
        options: DecodeOptions,
    ) -> Result<Self> {
        Self::builder()
            .backend(backend)
            .options(options)
            .create(cuda_context)
    }

    /// Creates a decoder that uses an existing `handle` and `options`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the state.
    pub fn from_handle(handle: Context, options: DecodeOptions) -> Result<Self> {
        let state = handle.create_decode_state()?;
        Ok(Self {
            handle,
            state,
            options,
        })
    }

    /// Returns the nvJPEG 2000 handle.
    #[must_use]
    pub const fn handle(&self) -> &Context {
        &self.handle
    }

    /// Returns the decode options.
    #[must_use]
    pub const fn options(&self) -> DecodeOptions {
        self.options
    }

    /// Replaces the decode options.
    pub fn set_options(&mut self, options: DecodeOptions) {
        self.options = options;
    }

    /// Parses the JPEG 2000 data in `data`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created, the CUDA context cannot be bound, `data` is empty or too long for nvJPEG 2000, or nvJPEG 2000 cannot parse it.
    pub fn parse(&self, data: &[u8]) -> Result<CodeStream> {
        let mut code_stream = CodeStream::create(&self.handle)?;
        code_stream.parse(data)?;
        Ok(code_stream)
    }

    /// Parses the JPEG 2000 file at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream cannot be created, the CUDA context cannot be bound, `path` contains a NUL byte, or nvJPEG 2000 cannot read or parse the file.
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<CodeStream> {
        let mut code_stream = CodeStream::create(&self.handle)?;
        code_stream.parse_file(path)?;
        Ok(code_stream)
    }

    /// Creates [`DecodeParams`] that match the current options.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG 2000 fails to create the settings or rejects an option.
    pub fn create_params(&self) -> Result<DecodeParams> {
        let mut params = DecodeParams::create(&self.handle)?;
        self.options.apply_to(&mut params)?;
        Ok(params)
    }

    /// Decodes the parsed image in `code_stream` into `planes`, and waits for `stream`.
    ///
    /// Planar output needs one plane per component; interleaved output needs one plane holding every component.
    /// With a region, subsampled components that do not line up with it also need the samples on its edges.
    /// The call waits for the GPU work even if it fails, and writes only the image's samples, not row padding or unused memory.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured, `code_stream` or a plane belongs to another CUDA context, the CUDA context cannot be bound, `code_stream` has not been parsed successfully, nvJPEG 2000 fails a query, the region or plane sizes are invalid, the planes are too few or too small, a size overflows, nvJPEG 2000 rejects the settings, or decoding or waiting for `stream` fails.
    pub fn decode_code_stream_into<T: ImagePixel, const C: usize>(
        &mut self,
        code_stream: &CodeStream,
        planes: &mut [PlaneMut<'_, T>; C],
        stream: &CudaStream,
    ) -> Result<()> {
        stream.ensure_not_capturing()?;
        self.handle.ensure_stream(stream)?;
        self.handle.ensure_context(code_stream.cuda_context())?;
        let sizes = self.options.output_sizes(code_stream)?;
        if C != sizes.len() {
            return Err(Error::LengthMismatch {
                name: "output planes".into(),
                expected: sizes.len(),
                actual: C,
            });
        }

        let components = code_stream.image_info()?.component_count() as usize;
        for (plane, size) in planes.iter().zip(&sizes) {
            if plane.context() != stream.context() {
                return Err(CudaError::MemoryContextMismatch.into());
            }

            if plane.size().width() < size.width() || plane.size().height() < size.height() {
                return Err(Error::OutOfRange {
                    name: "decode output dimensions".into(),
                });
            }

            let samples = if self.options.output_format == ImageFormat::Interleaved {
                components
            } else {
                1
            };
            let row_bytes = checked_mul(
                checked_mul(size.width() as usize, samples, "output samples")?,
                T::IMAGE_TYPE.element_size(),
                "output row bytes",
            )?;
            validate_plane(plane.region(), *size, plane.pitch(), row_bytes)?;
        }

        let pointers = planes
            .iter_mut()
            .map(|plane| plane.as_mut_ptr().cast())
            .collect();
        let pitches = planes.iter().map(PlaneMut::pitch).collect();
        // SAFETY: output views and selected layout establish capacities and exclusive borrows.
        let mut output =
            unsafe { Image::from_raw_parts(components, pointers, pitches, T::IMAGE_TYPE)? };
        let regions = planes
            .iter()
            .map(|plane| plane.region())
            .collect::<Vec<_>>();
        let params = self.create_params()?;
        complete(stream, || {
            stream.join_regions(&regions)?;
            // SAFETY: every owner and descriptor remains live through checked completion.
            unsafe {
                self.handle.decode_image_async(
                    &mut self.state,
                    code_stream,
                    &params,
                    &mut output,
                    stream,
                )
            }
        })
    }

    /// Allocates planar output, decodes the parsed image in `code_stream` into it, and waits for `stream`.
    ///
    /// It supports only planar output of the whole image without RGB conversion; use [`decode_code_stream_into`](Self::decode_code_stream_into) for interleaved output, a region, or custom pitches.
    /// The output holds samples of the type [`pixel_type`](DecodedImage::pixel_type) reports, in tightly packed planes.
    ///
    /// # Errors
    ///
    /// Returns an error if `stream` is being captured, the options ask for interleaved output, a region, or RGB conversion, nvJPEG 2000 fails a query, the components have an unsupported bit depth or need different sample types, an allocation fails, the contexts differ, nvJPEG 2000 rejects the settings, or decoding or waiting for `stream` fails.
    pub fn decode_code_stream_to_device(
        &mut self,
        code_stream: &CodeStream,
        stream: &CudaStream,
    ) -> Result<DecodedImage> {
        stream.ensure_not_capturing()?;
        self.options.validate_device_allocation()?;
        let component_infos = code_stream.component_infos()?;
        let mut output = DecodedImage::create_uninit(stream.context(), &component_infos)?;
        let params = self.create_params()?;
        let mut image = output.descriptor()?;
        let regions = output
            .components
            .iter()
            .map(|component| component.data.region())
            .collect::<Vec<_>>();
        complete(stream, || {
            stream.join_regions(&regions)?;
            // SAFETY: `complete` drains the stream before returning, so every borrowed buffer outlives the queued work.
            unsafe {
                self.handle.decode_image_async(
                    &mut self.state,
                    code_stream,
                    &params,
                    &mut image,
                    stream,
                )
            }
        })?;
        // SAFETY: the completed planar decode writes every tightly packed component byte.
        Ok(unsafe { output.assume_init() })
    }

    /// Parses `data` and decodes it into `planes`, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`parse`](Self::parse) or [`decode_code_stream_into`](Self::decode_code_stream_into) fails.
    pub fn decode_into<T: ImagePixel, const C: usize>(
        &mut self,
        data: &[u8],
        planes: &mut [PlaneMut<'_, T>; C],
        stream: &CudaStream,
    ) -> Result<()> {
        stream.ensure_not_capturing()?;
        let code_stream = self.parse(data)?;
        self.decode_code_stream_into(&code_stream, planes, stream)
    }

    /// Parses the JPEG 2000 file at `path` and decodes it into `planes`, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`parse_file`](Self::parse_file) or [`decode_code_stream_into`](Self::decode_code_stream_into) fails.
    pub fn decode_file_into<T: ImagePixel, const C: usize>(
        &mut self,
        path: impl AsRef<Path>,
        planes: &mut [PlaneMut<'_, T>; C],
        stream: &CudaStream,
    ) -> Result<()> {
        let code_stream = self.parse_file(path)?;
        self.decode_code_stream_into(&code_stream, planes, stream)
    }

    /// Parses `data`, allocates planar output, and decodes into it, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`parse`](Self::parse) or [`decode_code_stream_to_device`](Self::decode_code_stream_to_device) fails.
    pub fn decode_to_device(&mut self, data: &[u8], stream: &CudaStream) -> Result<DecodedImage> {
        stream.ensure_not_capturing()?;
        let code_stream = self.parse(data)?;
        self.decode_code_stream_to_device(&code_stream, stream)
    }

    /// Parses the JPEG 2000 file at `path`, allocates planar output, and decodes into it, waiting for `stream`.
    ///
    /// # Errors
    ///
    /// Returns an error if [`parse_file`](Self::parse_file) or [`decode_code_stream_to_device`](Self::decode_code_stream_to_device) fails.
    pub fn decode_file_to_device(
        &mut self,
        path: impl AsRef<Path>,
        stream: &CudaStream,
    ) -> Result<DecodedImage> {
        let code_stream = self.parse_file(path)?;
        self.decode_code_stream_to_device(&code_stream, stream)
    }
}

impl DecoderBuilder {
    /// Creates a builder with nvJPEG 2000's defaults.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            backend: Backend::Default,
            options: DecodeOptions::new(),
            allocators: None,
        }
    }

    /// Sets the backend.
    #[must_use]
    pub const fn backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Sets the decode options.
    #[must_use]
    pub const fn options(mut self, options: DecodeOptions) -> Self {
        self.options = options;
        self
    }

    /// Sets custom device and pinned host allocators.
    #[must_use]
    pub const fn allocators(mut self, allocators: AllocatorSet) -> Self {
        self.allocators = Some(allocators);
        self
    }

    /// Creates the decoder in `cuda_context`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, or nvJPEG 2000 rejects the settings or fails to create the handle or state.
    pub fn create(self, cuda_context: &Arc<CudaContext>) -> Result<Decoder> {
        let mut builder = Context::builder().backend(self.backend);
        if let Some(allocators) = self.allocators {
            builder = builder.allocators(allocators);
        }

        let handle = builder.create(cuda_context)?;
        Decoder::from_handle(handle, self.options)
    }
}

impl Default for DecoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeStream {
    /// Returns the CUDA context of the stream.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Returns whether the last parse succeeded.
    #[must_use]
    pub fn is_parsed(&self) -> bool {
        self.parsed
    }

    fn ensure_parsed(&self) -> Result<()> {
        if !self.parsed {
            return Err(Error::BitstreamNotParsed);
        }
        Ok(())
    }

    /// Creates an empty stream for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create the stream.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kStreamCreate` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
            backing: Vec::new(),
            parsed: false,
        })
    }

    /// Parses the header of the JPEG 2000 data in `data`, the first step of decoding.
    ///
    /// The image information can be read afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `data` is empty or too long for nvJPEG 2000, or nvJPEG 2000 cannot parse it.
    pub fn parse(&mut self, data: &[u8]) -> Result<()> {
        self.handle.bind()?;
        let length = to_non_zero_u64(data.len(), "data length")?;
        let data = data.to_vec().into_boxed_slice();
        let ptr = data.as_ptr();
        // Register backing storage before the native parser can retain it or fail.
        self.backing.push(data);
        self.parsed = false;
        // SAFETY: `nvjpeg2kStreamParse` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamParse(
                self.handle.as_raw(),
                ptr,
                length,
                0,
                0,
                self.raw
            ))?;
        }

        // Successful parsing replaces the previous bitstream. Failed parsing keeps both.
        let previous = self.backing.len() - 1;
        self.backing.drain(..previous);
        self.parsed = true;
        Ok(())
    }

    /// Parses the header of the JPEG 2000 file at `path`.
    ///
    /// The image information can be read afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, `path` contains a NUL byte, or nvJPEG 2000 cannot read or parse the file.
    pub fn parse_file(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.handle.bind()?;

        let path = CString::new(path.as_ref().as_os_str().as_encoded_bytes())
            .map_err(|_| Error::InteriorNul)?;
        self.parsed = false;
        // SAFETY: `nvjpeg2kStreamParseFile` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamParseFile(
                self.handle.as_raw(),
                path.as_ptr(),
                self.raw
            ))
        }?;
        self.backing.clear();
        self.parsed = true;
        Ok(())
    }

    /// Returns the size, tiling, and component count of the image, which help with allocating the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query.
    pub fn image_info(&self) -> Result<ImageInfo> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut info = sys::nvjpeg2kImageInfo_t::default();
        // SAFETY: `nvjpeg2kStreamGetImageInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetImageInfo(self.raw, &raw mut info))?;
        }
        Ok(ImageInfo::from_raw(info))
    }

    /// Returns the size, bit depth, and signedness of component `component`.
    ///
    /// Use it when components differ in size.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query, including for an invalid component index.
    pub fn component_info(&self, component: u32) -> Result<ImageComponentInfo> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut info = sys::nvjpeg2kImageComponentInfo_t::default();
        // SAFETY: `nvjpeg2kStreamGetImageComponentInfo` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetImageComponentInfo(
                self.raw,
                &raw mut info,
                component
            ))?;
        }
        Ok(ImageComponentInfo::from_raw(info))
    }

    /// Returns the information of every component.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query.
    pub fn component_infos(&self) -> Result<Vec<ImageComponentInfo>> {
        let info = self.image_info()?;
        (0..info.component_count())
            .map(|component| self.component_info(component))
            .collect()
    }

    /// Returns the color space of a JP2 file.
    ///
    /// A bare codestream has no color space, so it reports [`ColorSpace::Unknown`].
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query.
    pub fn color_space(&self) -> Result<ColorSpace> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut color_space = sys::nvjpeg2kColorSpace_t::NVJPEG2K_COLORSPACE_UNKNOWN;
        // SAFETY: `nvjpeg2kStreamGetColorSpace` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetColorSpace(
                self.raw,
                &raw mut color_space
            ))?;
        }
        Ok(color_space.into())
    }

    /// Returns the number of resolution levels in tile `tile_id`: the number of wavelet decomposition levels plus one.
    ///
    /// Tiles are numbered in raster order, so the tile at (`x`, `y`) has index `y * tiles_across + x`; [`image_info`](Self::image_info) reports the tile counts.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query, including for an invalid tile index.
    pub fn resolutions_in_tile(&self, tile_id: u32) -> Result<u32> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut resolutions = 0;
        // SAFETY: `nvjpeg2kStreamGetResolutionsInTile` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetResolutionsInTile(
                self.raw,
                tile_id,
                &raw mut resolutions
            ))?;
        }
        Ok(resolutions)
    }

    /// Returns the size of component `component` in tile `tile_id`.
    ///
    /// Use it when the image size is not a multiple of the tile size.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query, including for an invalid component or tile index.
    pub fn tile_component_dimensions(&self, component: u32, tile_id: u32) -> Result<Size> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut width = 0;
        let mut height = 0;
        // SAFETY: `nvjpeg2kStreamGetTileComponentDim` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetTileComponentDim(
                self.raw,
                component,
                tile_id,
                &raw mut width,
                &raw mut height
            ))?;
        }
        Ok(Size::new(width, height))
    }

    /// Returns the size of component `component` in tile `tile_id` at resolution level `resolution_level`.
    ///
    /// With `scale = 1 << (resolutions - resolution_level - 1)`, the width is `(tile_width + scale - 1) >> scale` and the height is computed the same way.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream has not been parsed successfully, the CUDA context cannot be bound, or nvJPEG 2000 fails the query, including for an invalid component, tile, or resolution index.
    pub fn resolution_component_dimensions(
        &self,
        component: u32,
        tile_id: u32,
        resolution_level: u32,
    ) -> Result<Size> {
        self.ensure_parsed()?;
        self.handle.bind()?;

        let mut width = 0;
        let mut height = 0;
        // SAFETY: `nvjpeg2kStreamGetResolutionComponentDim` receives live handles owned by the wrappers and valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kStreamGetResolutionComponentDim(
                self.raw,
                component,
                tile_id,
                resolution_level,
                &raw mut width,
                &raw mut height
            ))?;
        }
        Ok(Size::new(width, height))
    }

    /// Returns the native handle, which this stream still owns.
    ///
    /// If code outside this crate parses through it, use the raw decode functions afterwards, because the parse state tracked here no longer matches.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpeg2kStream_t {
        self.raw
    }
}

impl DecodeParams {
    /// Takes ownership of native decode settings created with `handle`.
    ///
    /// # Safety
    ///
    /// `raw` must be valid, created with `handle`, and owned by nobody else.
    /// It is destroyed when the returned value drops, so all work that used it must have finished.
    ///
    /// # Errors
    ///
    /// Returns an error if `raw` is null.
    pub unsafe fn from_raw(raw: sys::nvjpeg2kDecodeParams_t, handle: &Context) -> Result<Self> {
        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Releases the native handle without destroying it.
    ///
    /// Destroy it, or pass it back to [`from_raw`](Self::from_raw), while its [`Context`] is alive.
    #[must_use]
    pub fn into_raw(mut self) -> sys::nvjpeg2kDecodeParams_t {
        mem::replace(&mut self.raw, ptr::null_mut())
    }

    /// Returns the CUDA context of the decode settings.
    #[must_use]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        self.handle.cuda_context()
    }

    /// Creates decode settings for `handle`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 fails to create them.
    pub fn create(handle: &Context) -> Result<Self> {
        handle.bind()?;

        let mut raw = ptr::null_mut();
        // SAFETY: `nvjpeg2kDecodeParamsCreate` receives valid output locations.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeParamsCreate(&raw mut raw))?;
        }

        if raw.is_null() {
            return Err(Error::NullHandle);
        }
        Ok(Self {
            raw,
            handle: handle.clone(),
        })
    }

    /// Decodes only the region from (`start_x`, `start_y`) up to, but not including, (`end_x`, `end_y`).
    ///
    /// The coordinates are relative to the image and must lie in the tile being decoded.
    /// All zeros decodes the whole tile.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound, a nonzero region has a start that is not less than its end, or nvJPEG 2000 rejects the region.
    pub fn set_decode_area(
        &mut self,
        start_x: u32,
        end_x: u32,
        start_y: u32,
        end_y: u32,
    ) -> Result<&mut Self> {
        self.handle.bind()?;

        if !(start_x == 0 && end_x == 0 && start_y == 0 && end_y == 0) {
            DecodeArea::create(start_x, end_x, start_y, end_y)?;
        }

        // SAFETY: `nvjpeg2kDecodeParamsSetDecodeArea` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeParamsSetDecodeArea(
                self.raw, start_x, end_x, start_y, end_y
            ))?;
        }
        Ok(self)
    }

    /// Decodes only the region `area`.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the region.
    pub fn set_area(&mut self, area: DecodeArea) -> Result<&mut Self> {
        self.set_decode_area(area.start_x, area.end_x, area.start_y, area.end_y)
    }

    /// Sets whether to convert images with 4:2:2 or 4:2:0 chroma subsampling to RGB; other images are not affected.
    ///
    /// The conversion follows sYCC (IEC 61966-2-1 Amendment 1).
    /// It is applied per tile, so chroma may be slightly off for tiles with odd sizes.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the setting.
    pub fn set_rgb_output(&mut self, force_rgb: bool) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpeg2kDecodeParamsSetRGBOutput` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeParamsSetRGBOutput(
                self.raw,
                i32::from(force_rgb)
            ))?;
        }
        Ok(self)
    }

    /// Sets whether the output is planar or interleaved.
    ///
    /// Interleaved output of an image with 4:2:2 or 4:2:0 subsampling also needs [`set_rgb_output`](Self::set_rgb_output).
    /// An interleaved output plane needs a pitch of at least `width * components * bytes_per_sample` bytes and that pitch times the height in total, with 1 byte per sample for [`ImageType::U8`] and 2 otherwise.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA context cannot be bound or nvJPEG 2000 rejects the layout.
    pub fn set_output_format(&mut self, format: ImageFormat) -> Result<&mut Self> {
        self.handle.bind()?;

        // SAFETY: `nvjpeg2kDecodeParamsSetOutputFormat` receives live handles owned by the wrappers.
        unsafe {
            try_ffi_k2!(sys::nvjpeg2kDecodeParamsSetOutputFormat(
                self.raw,
                format.into()
            ))?;
        }
        Ok(self)
    }

    /// Returns the native handle, which this value still owns.
    #[must_use]
    pub const fn as_raw(&self) -> sys::nvjpeg2kDecodeParams_t {
        self.raw
    }
}

impl Drop for DecodeState {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kDecodeStateDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kDecodeStateDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k decode state: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for CodeStream {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kStreamDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kStreamDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k stream: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl Drop for DecodeParams {
    fn drop(&mut self) {
        if self.handle.bind().is_err() {
            return;
        }
        if !self.raw.is_null() {
            // SAFETY: `nvjpeg2kDecodeParamsDestroy` receives live handles owned by the wrappers.
            unsafe {
                if let Err(error) = try_ffi_k2!(sys::nvjpeg2kDecodeParamsDestroy(self.raw)) {
                    eprintln!(
                        "failed to destroy nvjpeg2k decode params: {}",
                        singe_core::ErrorChain(&error)
                    );
                }
            }
        }
    }
}

impl DecodedComponent<MaybeUninit<u8>> {
    /// Marks the component as initialized.
    ///
    /// # Safety
    ///
    /// Every byte must be initialized, and the work that wrote it must have finished before it is read.
    #[must_use]
    pub unsafe fn assume_init(self) -> DecodedComponent {
        DecodedComponent {
            // SAFETY: the caller upholds this function's safety contract.
            data: unsafe { self.data.assume_init() },
            spec: self.spec,
        }
    }
}

impl DecodedImage<MaybeUninit<u8>> {
    /// Marks every component as initialized.
    ///
    /// # Safety
    ///
    /// Every byte of every component must be initialized, and the work that wrote them must have finished before they are read.
    #[must_use]
    pub unsafe fn assume_init(self) -> DecodedImage {
        DecodedImage {
            pixel_type: self.pixel_type,
            components: self
                .components
                .into_iter()
                // SAFETY: the caller upholds this function's safety contract.
                .map(|component| unsafe { component.assume_init() })
                .collect(),
        }
    }
}
