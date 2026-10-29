use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, exchange,
        video::{
            VideoFormat, VideoOutputView, VideoView, invalid, prepare_conversion,
            validate_conversion,
        },
        view::{C1, C3, C4, ImageOutputView},
    },
    try_ffi,
    types::{ColorSpace, DataTypeLike, HintAlgorithm, Point, Size},
};

// NPP 13.2's newer semiplanar decoders mix luma/chroma samples and depend on
// initialized row padding. Its color-twist kernels use the proper NV12 footprint.
// These matrices use full-range Y'CbCr and the non-constant-luminance coefficients
// in ITU-T H.273, Table 4. P010 stores its ten code bits in the high bits of u16.
fn matrix(
    color_space: ColorSpace,
    maximum: f32,
    neutral: f32,
    output_maximum: f32,
    format: VideoFormat,
) -> Result<[[f32; 4]; 3]> {
    let (kr, kb) = match color_space {
        ColorSpace::Bt601Jpeg => (0.299, 0.114),
        ColorSpace::Bt709Hdtv => (0.2126, 0.0722),
        ColorSpace::Bt2020Hdtv => (0.2627, 0.0593),
        ColorSpace::TranscodeOnly => return Err(invalid("RGB color space")),
    };
    let kg = 1.0 - kr - kb;
    let scale = output_maximum / maximum;
    let red_v = 2.0 * (1.0 - kr) * scale;
    let blue_u = 2.0 * (1.0 - kb) * scale;
    let green_u = -2.0 * kb * (1.0 - kb) / kg * scale;
    let green_v = -2.0 * kr * (1.0 - kr) / kg * scale;
    let mut rows = [
        [scale, 0.0, red_v, -red_v * neutral],
        [scale, green_u, green_v, -(green_u + green_v) * neutral],
        [scale, blue_u, 0.0, -blue_u * neutral],
    ];
    if matches!(format, VideoFormat::Bgr | VideoFormat::Bgra) {
        rows.swap(0, 2);
    }
    Ok(rows)
}

fn output_samples<'a, T: DataTypeLike>(
    destination: &'a mut VideoOutputView<'_, T>,
    channels: i32,
) -> Result<ImageOutputView<'a, T, C1>> {
    let size = destination.size();
    let samples = Size::new(
        size.width
            .checked_mul(channels)
            .ok_or_else(|| invalid("video output width"))?,
        size.height,
    );
    let plane = &mut destination.planes[0];
    // SAFETY: the checked frame format covers this exact scalar ROI; the exclusive
    // reborrow preserves the original allocation, context, pitch and capacity.
    unsafe { ImageOutputView::from_raw_parts(plane.region(), samples, plane.step()) }
}

fn write_u8(
    context: &StreamContext,
    source: &Image<u8, C3>,
    destination: &mut VideoOutputView<'_, u8>,
    alpha: u8,
) -> Result<()> {
    match destination.format() {
        VideoFormat::Rgb | VideoFormat::Bgr => {
            output_samples(destination, 3)?.copy_from(&source.view()?.into_samples()?)
        }
        VideoFormat::Rgba | VideoFormat::Bgra => {
            let source = source.view()?;
            let size = destination.size();
            let plane = &mut destination.planes[0];
            context.sync_scope(|| {
                // SAFETY: the checked four-channel ROI is exclusively writable;
                // the channel order produces every color channel and the supplied alpha.
                unsafe {
                    let mut output = ImageOutputView::<u8, C4>::from_raw_parts(
                        plane.region(),
                        size,
                        plane.step(),
                    )?;
                    exchange::swap_channels_c3_to_c4(
                        context,
                        &source,
                        &mut output,
                        [0, 1, 2, 3],
                        alpha,
                    )
                }
            })
        }
        _ => Err(invalid("RGB destination format")),
    }
}

pub fn decode_u8(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, u8>,
    format: VideoFormat,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    validate_conversion(
        source,
        VideoFormat::Nv12,
        destination,
        format,
        Point::ZERO,
        false,
        true,
    )?;
    prepare_conversion(context, source, destination)?;
    let twist = matrix(color_space, 255.0, 128.0, 255.0, format)?;
    let mut colors = Image::<u8, C3>::create_uninit(context.cuda_context(), destination.size())?;
    {
        let pointers = [source.planes()[0].as_ptr(), source.planes()[1].as_ptr()];
        let mut steps = [source.planes()[0].step(), source.planes()[1].step()];
        let mut output = colors.output()?;
        context.sync_scope(|| {
            // SAFETY: validated NV12 planes cover the full subsampled footprint;
            // pointer tables, matrix, output and owners survive native completion.
            unsafe {
                try_ffi!(sys::nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(
                    pointers.as_ptr(),
                    steps.as_mut_ptr(),
                    output.as_mut_ptr(),
                    output.step(),
                    output.size().into(),
                    twist.as_ptr(),
                    context.as_raw()
                ))
            }
        })?;
    }
    // SAFETY: the completed color-twist conversion writes every packed color sample.
    let colors = unsafe { colors.assume_init() };
    write_u8(context, &colors, destination, alpha)
}

fn decode_integer(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    size: Size,
    format: VideoFormat,
    color_space: ColorSpace,
) -> Result<Image<u16, C3>> {
    let maximum = match source.format() {
        VideoFormat::P010 => 1023.0 * 64.0,
        VideoFormat::P016 => 65535.0,
        _ => return Err(invalid("16-bit semiplanar format")),
    };
    let twist = matrix(color_space, maximum, 32768.0, 65535.0, format)?;
    let mut colors = Image::<u16, C3>::create_uninit(context.cuda_context(), size)?;
    {
        let pointers = [source.planes()[0].as_ptr(), source.planes()[1].as_ptr()];
        let mut steps = [source.planes()[0].step(), source.planes()[1].step()];
        let mut output = colors.output()?;
        context.sync_scope(|| {
            // SAFETY: scalar u16 planes preserve the P010/P016 row geometry, with
            // independently checked strides and full-color output retained through completion.
            unsafe {
                try_ffi!(sys::nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(
                    pointers.as_ptr(),
                    steps.as_mut_ptr(),
                    output.as_mut_ptr(),
                    output.step(),
                    size.into(),
                    twist.as_ptr(),
                    context.as_raw()
                ))
            }
        })?;
    }
    // SAFETY: the completed conversion initialized all three color channels.
    Ok(unsafe { colors.assume_init() })
}

pub fn decode_u16_to_u8(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_format: VideoFormat,
    format: VideoFormat,
    color_space: ColorSpace,
    alpha: u8,
) -> Result<()> {
    validate_conversion(
        source,
        source_format,
        destination,
        format,
        Point::ZERO,
        false,
        true,
    )?;
    prepare_conversion(context, source, destination)?;
    let colors = decode_integer(context, source, destination.size(), format, color_space)?;
    let mut reduced = Image::<u8, C3>::create_uninit(context.cuda_context(), destination.size())?;
    {
        let source = colors.view()?.into_samples()?;
        let mut output = reduced.output()?.into_samples()?;
        context.sync_scope(|| {
            // SAFETY: the initialized colors and complete scalar output remain alive through conversion.
            unsafe {
                exchange::scale_u16_to_u8_c1(context, &source, &mut output, HintAlgorithm::Accurate)
            }
        })?;
    }
    // SAFETY: completed scaling writes every packed output color sample.
    let reduced = unsafe { reduced.assume_init() };
    write_u8(context, &reduced, destination, alpha)
}

pub fn decode_u16(
    context: &StreamContext,
    source: &VideoView<'_, u16>,
    destination: &mut VideoOutputView<'_, u16>,
    source_format: VideoFormat,
    format: VideoFormat,
    color_space: ColorSpace,
    alpha: u16,
) -> Result<()> {
    validate_conversion(
        source,
        source_format,
        destination,
        format,
        Point::ZERO,
        false,
        true,
    )?;
    prepare_conversion(context, source, destination)?;
    let colors = decode_integer(context, source, destination.size(), format, color_space)?;
    match format {
        VideoFormat::Rgb => {
            output_samples(destination, 3)?.copy_from(&colors.view()?.into_samples()?)
        }
        VideoFormat::Rgba => {
            let source = colors.view()?;
            let size = destination.size();
            let plane = &mut destination.planes[0];
            context.sync_scope(|| {
                // SAFETY: all four destination channels are written; the alpha is copied exactly.
                unsafe {
                    let mut output = ImageOutputView::<u16, C4>::from_raw_parts(
                        plane.region(),
                        size,
                        plane.step(),
                    )?;
                    exchange::swap_channels_c3_to_c4(
                        context,
                        &source,
                        &mut output,
                        [0, 1, 2, 3],
                        alpha,
                    )
                }
            })
        }
        _ => Err(invalid("16-bit RGB destination format")),
    }
}
