//! Device-to-device dtype conversion.

use singe_cuda::{
    stream::Stream,
    view::{DeviceSlice, DeviceSliceMut},
};

#[cfg(feature = "dtype-bf16")]
use singe_cuda::types::bf16;
#[cfg(feature = "dtype-f16")]
use singe_cuda::types::f16;

#[cfg(feature = "cutile")]
use crate::cuda::cutile;
use crate::{
    cuda::interop::{borrowed_stream, input_pointer, output_pointer},
    error::Result,
    utility::ensure_len,
};

macro_rules! cast_fn {
    ($name:ident, $out_ty:ty, $in_ty:ty) => {
        pub fn $name(
            stream: &Stream,
            out: &mut impl DeviceSliceMut<$out_ty>,
            input: &impl DeviceSlice<$in_ty>,
        ) -> Result<()> {
            let len = out.len();
            ensure_len(input.len(), len)?;
            let stream = borrowed_stream(stream)?;
            cutile::cast::$name(&stream, output_pointer(out), input_pointer(input), len)
        }
    };
}

#[cfg(feature = "dtype-f32")]
#[cfg(feature = "dtype-f16")]
cast_fn!(f16_to_f32, f32, f16);
#[cfg(feature = "dtype-f32")]
#[cfg(feature = "dtype-bf16")]
cast_fn!(bf16_to_f32, f32, bf16);
#[cfg(feature = "dtype-f64")]
#[cfg(feature = "dtype-f16")]
cast_fn!(f16_to_f64, f64, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(f16_to_u8, u8, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
cast_fn!(f16_to_i8, i8, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
cast_fn!(f16_to_u32, u32, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
cast_fn!(f16_to_i32, i32, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u64"))]
cast_fn!(f16_to_u64, u64, f16);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i64"))]
cast_fn!(f16_to_i64, i64, f16);
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_f16, f16, f32);
#[cfg(feature = "dtype-bf16")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_bf16, bf16, f32);
#[cfg(feature = "dtype-f64")]
#[cfg(feature = "dtype-f32")]
cast_fn!(f32_to_f64, f64, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(f32_to_u8, u8, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
cast_fn!(f32_to_i8, i8, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
cast_fn!(f32_to_u32, u32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
cast_fn!(f32_to_i32, i32, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u64"))]
cast_fn!(f32_to_u64, u64, f32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i64"))]
cast_fn!(f32_to_i64, i64, f32);
#[cfg(feature = "dtype-f16")]
#[cfg(feature = "dtype-f64")]
cast_fn!(f64_to_f16, f16, f64);
#[cfg(feature = "dtype-f32")]
#[cfg(feature = "dtype-f64")]
cast_fn!(f64_to_f32, f32, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(f64_to_u8, u8, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
cast_fn!(f64_to_i8, i8, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
cast_fn!(f64_to_u32, u32, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
cast_fn!(f64_to_i32, i32, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u64"))]
cast_fn!(f64_to_u64, u64, f64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i64"))]
cast_fn!(f64_to_i64, i64, f64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(bool_to_f16, f16, u8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(bool_to_f32, f32, u8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(bool_to_f64, f64, u8);
#[cfg(feature = "dtype-u8")]
cast_fn!(bool_to_u8, u8, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
cast_fn!(bool_to_i8, i8, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u32"))]
cast_fn!(bool_to_u32, u32, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
cast_fn!(bool_to_i32, i32, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u64"))]
cast_fn!(bool_to_u64, u64, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i64"))]
cast_fn!(bool_to_i64, i64, u8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(u8_to_f16, f16, u8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(u8_to_f32, f32, u8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(u8_to_f64, f64, u8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i8"))]
cast_fn!(i8_to_f16, f16, i8);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i8"))]
cast_fn!(i8_to_f32, f32, i8);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i8"))]
cast_fn!(i8_to_f64, f64, i8);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u32"))]
cast_fn!(u32_to_f16, f16, u32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u32"))]
cast_fn!(u32_to_f32, f32, u32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u32"))]
cast_fn!(u32_to_f64, f64, u32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-u64"))]
cast_fn!(u64_to_f16, f16, u64);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u64"))]
cast_fn!(u64_to_f32, f32, u64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u64"))]
cast_fn!(u64_to_f64, f64, u64);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i32"))]
cast_fn!(i32_to_f16, f16, i32);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i32"))]
cast_fn!(i32_to_f32, f32, i32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i32"))]
cast_fn!(i32_to_f64, f64, i32);
#[cfg(all(feature = "dtype-f16", feature = "dtype-i64"))]
cast_fn!(i64_to_f16, f16, i64);
#[cfg(all(feature = "dtype-f32", feature = "dtype-i64"))]
cast_fn!(i64_to_f32, f32, i64);
#[cfg(all(feature = "dtype-f64", feature = "dtype-i64"))]
cast_fn!(i64_to_f64, f64, i64);

#[cfg(all(feature = "dtype-f16", feature = "dtype-u8"))]
cast_fn!(f16_to_bool, u8, f16);
#[cfg(all(feature = "dtype-f32", feature = "dtype-u8"))]
cast_fn!(f32_to_bool, u8, f32);
#[cfg(all(feature = "dtype-f64", feature = "dtype-u8"))]
cast_fn!(f64_to_bool, u8, f64);
#[cfg(feature = "dtype-u8")]
cast_fn!(u8_to_bool, u8, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
cast_fn!(i8_to_bool, u8, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
cast_fn!(u32_to_bool, u8, u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
cast_fn!(i32_to_bool, u8, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
cast_fn!(u64_to_bool, u8, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
cast_fn!(i64_to_bool, u8, i64);
#[cfg(feature = "dtype-u8")]
cast_fn!(u8_to_u8, u8, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i8"))]
cast_fn!(u8_to_i8, i8, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u32"))]
cast_fn!(u8_to_u32, u32, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i32"))]
cast_fn!(u8_to_i32, i32, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-u64"))]
cast_fn!(u8_to_u64, u64, u8);
#[cfg(all(feature = "dtype-u8", feature = "dtype-i64"))]
cast_fn!(u8_to_i64, i64, u8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u8"))]
cast_fn!(i8_to_u8, u8, i8);
#[cfg(feature = "dtype-i8")]
cast_fn!(i8_to_i8, i8, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u32"))]
cast_fn!(i8_to_u32, u32, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i32"))]
cast_fn!(i8_to_i32, i32, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-u64"))]
cast_fn!(i8_to_u64, u64, i8);
#[cfg(all(feature = "dtype-i8", feature = "dtype-i64"))]
cast_fn!(i8_to_i64, i64, i8);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u8"))]
cast_fn!(u32_to_u8, u8, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i8"))]
cast_fn!(u32_to_i8, i8, u32);
#[cfg(feature = "dtype-u32")]
cast_fn!(u32_to_u32, u32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i32"))]
cast_fn!(u32_to_i32, i32, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-u64"))]
cast_fn!(u32_to_u64, u64, u32);
#[cfg(all(feature = "dtype-u32", feature = "dtype-i64"))]
cast_fn!(u32_to_i64, i64, u32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u8"))]
cast_fn!(i32_to_u8, u8, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-i8"))]
cast_fn!(i32_to_i8, i8, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u32"))]
cast_fn!(i32_to_u32, u32, i32);
#[cfg(feature = "dtype-i32")]
cast_fn!(i32_to_i32, i32, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-u64"))]
cast_fn!(i32_to_u64, u64, i32);
#[cfg(all(feature = "dtype-i32", feature = "dtype-i64"))]
cast_fn!(i32_to_i64, i64, i32);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u8"))]
cast_fn!(u64_to_u8, u8, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i8"))]
cast_fn!(u64_to_i8, i8, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-u32"))]
cast_fn!(u64_to_u32, u32, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i32"))]
cast_fn!(u64_to_i32, i32, u64);
#[cfg(feature = "dtype-u64")]
cast_fn!(u64_to_u64, u64, u64);
#[cfg(all(feature = "dtype-u64", feature = "dtype-i64"))]
cast_fn!(u64_to_i64, i64, u64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u8"))]
cast_fn!(i64_to_u8, u8, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i8"))]
cast_fn!(i64_to_i8, i8, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u32"))]
cast_fn!(i64_to_u32, u32, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-i32"))]
cast_fn!(i64_to_i32, i32, i64);
#[cfg(all(feature = "dtype-i64", feature = "dtype-u64"))]
cast_fn!(i64_to_u64, u64, i64);
#[cfg(feature = "dtype-i64")]
cast_fn!(i64_to_i64, i64, i64);

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(feature = "dtype-f32", feature = "dtype-bf16"))]
    #[test]
    fn f32_to_bf16_matches_host_conversion() -> Result<()> {
        use singe_cuda::{
            context::Context as CudaContext, device::Device, memory::DeviceMemory,
            stream::StreamFlags, types::bf16,
        };

        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let input_host = vec![0.0f32, 1.0, -0.5, 3.25, -7.75, 0.001, 1024.0];
        let input = DeviceMemory::from_slice(&input_host)?;
        let mut out = DeviceMemory::<bf16>::create(input_host.len())?;

        f32_to_bf16(&stream, &mut out, &input)?;
        stream.synchronize()?;

        let expected = input_host
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect::<Vec<_>>();
        let actual = out
            .copy_to_host_vec()?
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        Ok(())
    }

    #[cfg(all(feature = "dtype-f32", feature = "dtype-bf16"))]
    #[test]
    fn bf16_to_f32_matches_host_conversion() -> Result<()> {
        use singe_cuda::{
            context::Context as CudaContext, device::Device, memory::DeviceMemory,
            stream::StreamFlags, types::bf16,
        };

        let Ok(context) = CudaContext::retain_primary_for_device(Device::new(0)) else {
            return Ok(());
        };
        let stream = context.create_stream_with_flags(StreamFlags::NON_BLOCKING)?;
        let input_host = [0.0f32, 1.0, -0.5, 3.25, -7.75, 0.001, 1024.0]
            .into_iter()
            .map(bf16::from_f32)
            .collect::<Vec<_>>();
        let input = DeviceMemory::from_slice(&input_host)?;
        let mut out = DeviceMemory::<f32>::create(input_host.len())?;

        bf16_to_f32(&stream, &mut out, &input)?;
        stream.synchronize()?;

        let expected = input_host
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let actual = out.copy_to_host_vec()?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
