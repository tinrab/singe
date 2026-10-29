use singe_cuda::types::f16;
use singe_npp_sys as sys;

use crate::{
    context::StreamContext,
    error::Result,
    image::{
        Image, exchange,
        video::{
            Frame, VideoFormat, VideoOutputView, VideoView, invalid, prepare_conversion,
            validate_conversion,
        },
        view::{C1, C3, C4, ImageOutputView, ImageView},
    },
    try_ffi,
    types::{ColorSpace, Point, RoundMode, Size},
};

// NPP 13.2's half UYVP decoders return success without producing pixels, and its
// half encoders produce zero bytes for the exercised input ranges and color spaces.
// Compose its integer codecs and scalar conversions so successful safe calls have
// a complete output contract.
// Half RGB channels use the normalized range [0, 1].

fn scaled_float<S>(
    context: &StreamContext,
    source: &ImageView<'_, S, C1>,
    scale: f32,
    convert: ScaledFloatCallback<S>,
) -> Result<Image<f32, C1>> {
    let mut floats = Image::<f32, C1>::create_uninit(context.cuda_context(), source.size())?;
    {
        let mut output = floats.output()?;
        context.sync_scope(|| {
            // SAFETY: both kernels share this stream, and conversion produces every
            // scalar before the multiply reads it. No initialized Rust view is
            // constructed for the intermediate until both kernels have completed.
            unsafe {
                convert(context, source, &mut output)?;
                try_ffi!(sys::nppiMulC_32f_C1IR_Ctx(
                    scale,
                    output.as_mut_ptr(),
                    output.step(),
                    output.size().into(),
                    context.as_raw(),
                ))?;
            }
            Ok(())
        })?;
    }
    // SAFETY: completed conversion and multiplication produce every logical scalar.
    Ok(unsafe { floats.assume_init() })
}

fn to_integer(context: &StreamContext, source: &ImageView<'_, f16, C1>) -> Result<Image<u16, C1>> {
    let floats = scaled_float(context, source, 65535.0, exchange::convert_f16_to_f32_c1)?;
    let mut integers = Image::<u16, C1>::create_uninit(context.cuda_context(), source.size())?;
    {
        let source = floats.view()?;
        let mut output = integers.output()?;
        context.sync_scope(|| {
            // SAFETY: initialized float scalars and exclusive output remain alive until completion.
            unsafe {
                exchange::convert_f32_to_u16_c1(context, &source, &mut output, RoundMode::Near)
            }
        })?;
    }
    // SAFETY: the completed scalar conversion writes every logical integer sample.
    Ok(unsafe { integers.assume_init() })
}

fn to_half(
    context: &StreamContext,
    source: &ImageView<'_, u16, C1>,
    destination: &mut ImageOutputView<'_, f16, C1>,
) -> Result<()> {
    let floats = scaled_float(
        context,
        source,
        1.0 / 65535.0,
        exchange::convert_u16_to_f32_c1,
    )?;
    let source = floats.view()?;
    context.sync_scope(|| {
        // SAFETY: the scalar producer completes before its owned float input is released.
        unsafe { exchange::convert_f32_to_f16_c1(context, &source, destination, RoundMode::Near) }
    })
}

pub fn encode(
    context: &StreamContext,
    source: &VideoView<'_, f16>,
    destination: &mut VideoOutputView<'_, u8>,
    source_format: VideoFormat,
    offset: Point,
    color_space: ColorSpace,
) -> Result<()> {
    validate_conversion(
        source,
        source_format,
        destination,
        VideoFormat::Uyvp10,
        offset,
        true,
        true,
    )?;
    prepare_conversion(context, source, destination)?;
    let mut planes = Vec::with_capacity(source.planes().len());
    for plane in source.planes() {
        planes.push(to_integer(context, plane)?);
    }
    let integers = Frame::from_planes(source_format, source.size(), planes)?;
    let source = integers.view()?;
    if source
        .planes()
        .iter()
        .any(|plane| plane.step() != source.planes()[0].step())
    {
        return Err(invalid("converted RGB pitch"));
    }
    context.sync_scope(|| {
        // SAFETY: integer conversion initialized every stored source sample, and this
        // scope retains its owners and the packed output through native completion.
        unsafe {
            match source_format {
                VideoFormat::Rgb => try_ffi!(sys::nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx(
                    source.planes()[0].as_ptr(),
                    source.planes()[0].step(),
                    offset.into(),
                    destination.planes[0].as_mut_ptr(),
                    destination.planes[0].step(),
                    destination.size().into(),
                    color_space.into(),
                    context.as_raw(),
                )),
                VideoFormat::Rgba => try_ffi!(sys::nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx(
                    source.planes()[0].as_ptr(),
                    source.planes()[0].step(),
                    offset.into(),
                    destination.planes[0].as_mut_ptr(),
                    destination.planes[0].step(),
                    destination.size().into(),
                    color_space.into(),
                    context.as_raw(),
                )),
                VideoFormat::RgbPlanar => {
                    let pointers = [
                        source.planes()[0].as_ptr(),
                        source.planes()[1].as_ptr(),
                        source.planes()[2].as_ptr(),
                    ];
                    try_ffi!(sys::nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx(
                        pointers.as_ptr(),
                        source.planes()[0].step(),
                        offset.into(),
                        destination.planes[0].as_mut_ptr(),
                        destination.planes[0].step(),
                        destination.size().into(),
                        color_space.into(),
                        context.as_raw(),
                    ))
                }
                _ => Err(invalid("half RGB source format")),
            }
        }
    })
}

#[expect(
    clippy::too_many_lines,
    reason = "half-precision decoding keeps all format-specific native conversion paths together"
)]
pub fn decode(
    context: &StreamContext,
    source: &VideoView<'_, u8>,
    destination: &mut VideoOutputView<'_, f16>,
    destination_format: VideoFormat,
    offset: Point,
    color_space: ColorSpace,
    alpha: f16,
) -> Result<()> {
    validate_conversion(
        source,
        VideoFormat::Uyvp10,
        destination,
        destination_format,
        offset,
        true,
        true,
    )?;
    prepare_conversion(context, source, destination)?;
    let size = destination.size();
    let mut integers = Image::<u16, C3>::create_uninit(context.cuda_context(), size)?;
    {
        let mut output = integers.output()?;
        context.sync_scope(|| {
            // SAFETY: the integer decoder produces all RGB channels and completes before promotion.
            unsafe {
                try_ffi!(sys::nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx(
                    source.planes()[0].as_ptr(),
                    source.planes()[0].step(),
                    offset.into(),
                    output.as_mut_ptr(),
                    output.step(),
                    size.into(),
                    color_space.into(),
                    context.as_raw(),
                ))
            }
        })?;
    }
    // SAFETY: the completed integer decoder initialized the entire RGB image.
    let integers = unsafe { integers.assume_init() };
    match destination_format {
        VideoFormat::Rgb => {
            let plane = &mut destination.planes[0];
            // SAFETY: the checked video frame grants this exact packed RGB sample extent.
            let mut output = unsafe {
                ImageOutputView::from_raw_parts(
                    plane.region(),
                    Size::new(
                        size.width
                            .checked_mul(3)
                            .ok_or_else(|| invalid("half RGB width"))?,
                        size.height,
                    ),
                    plane.step(),
                )?
            };
            to_half(context, &integers.view()?.into_samples()?, &mut output)
        }
        VideoFormat::Rgba => {
            let mut rgb = Image::<f16, C3>::create_uninit(context.cuda_context(), size)?;
            to_half(
                context,
                &integers.view()?.into_samples()?,
                &mut rgb.output()?.into_samples()?,
            )?;
            // SAFETY: completed conversion initialized every packed RGB half sample.
            let rgb = unsafe { rgb.assume_init() };
            let source = rgb.view()?;
            let plane = &mut destination.planes[0];
            context.sync_scope(|| {
                // SAFETY: half values and u16 have identical size and alignment; this
                // operation copies their bits without numeric interpretation. The order
                // writes all four channels and inserts the exact half alpha.
                unsafe {
                    let bits =
                        ImageView::<u16, C3>::from_raw_parts(source.region(), size, source.step())?;
                    let mut output = ImageOutputView::<u16, C4>::from_raw_parts(
                        plane.region(),
                        size,
                        plane.step(),
                    )?;
                    exchange::swap_channels_c3_to_c4(
                        context,
                        &bits,
                        &mut output,
                        [0, 1, 2, 3],
                        alpha.to_bits(),
                    )
                }
            })
        }
        VideoFormat::RgbPlanar => {
            let mut red = Image::<u16, C1>::create_uninit(context.cuda_context(), size)?;
            let mut green = Image::<u16, C1>::create_uninit(context.cuda_context(), size)?;
            let mut blue = Image::<u16, C1>::create_uninit(context.cuda_context(), size)?;
            {
                let source = integers.view()?;
                let mut red_out = red.output()?;
                let mut green_out = green.output()?;
                let mut blue_out = blue.output()?;
                let pointers = [
                    red_out.as_mut_ptr(),
                    green_out.as_mut_ptr(),
                    blue_out.as_mut_ptr(),
                ];
                context.sync_scope(|| {
                    // SAFETY: all three same-sized native allocations have identical checked
                    // pitches, and their owners survive the completed packed-to-planar copy.
                    if red_out.step() != green_out.step() || red_out.step() != blue_out.step() {
                        return Err(invalid("half planar pitch"));
                    }
                    // SAFETY: `nppiCopy_16u_C3P3R_Ctx` receives live handles owned by the wrappers and pointers into buffers borrowed for the call.
                    unsafe {
                        try_ffi!(sys::nppiCopy_16u_C3P3R_Ctx(
                            source.as_ptr(),
                            source.step(),
                            pointers.as_ptr(),
                            red_out.step(),
                            size.into(),
                            context.as_raw()
                        ))
                    }
                })?;
            }
            // SAFETY: the completed planar copy writes every sample in all three planes.
            let planes = unsafe { [red.assume_init(), green.assume_init(), blue.assume_init()] };
            for (source, destination) in planes.iter().zip(destination.planes.iter_mut()) {
                // SAFETY: the frame's checked planar geometry covers this scalar output ROI.
                let mut output = unsafe {
                    ImageOutputView::from_raw_parts(destination.region(), size, destination.step())?
                };
                to_half(context, &source.view()?, &mut output)?;
            }
            Ok(())
        }
        _ => Err(invalid("half RGB destination format")),
    }
}

type ScaledFloatCallback<S> = unsafe fn(
    &StreamContext,
    &ImageView<'_, S, C1>,
    &mut ImageOutputView<'_, f32, C1>,
) -> Result<()>;
