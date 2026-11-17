# singe-nvimagecodec

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvimagecodec.svg)](https://crates.io/crates/singe-nvimagecodec)
[![Documentation](https://docs.rs/singe-nvimagecodec/badge.svg)](https://docs.rs/singe-nvimagecodec)
![License](https://img.shields.io/crates/l/singe-nvimagecodec.svg)

Safe Rust wrappers for nvImageCodec, which decodes and encodes JPEG, JPEG 2000, TIFF, PNG, BMP, and other formats through codec modules that run on the CPU or the GPU.

This crate provides:

- **Code streams:** `CodeStream` holds compressed data in a file or in host memory, reports its image information, and selects one image or a region of it.
- **Images:** `Image` and `ImageInfo` describe host or device pixel buffers, with plane layouts, sample types, color specs, and chroma subsampling.
- **Decoding and encoding:** `Decoder` and `Encoder` process one image or a batch, report a status per image, and answer `can_decode` and `can_encode` queries.
- **Execution:** `ExecutionParams` picks the CPU or a CUDA context, and the `unsafe` `decode_async` and `encode_async` methods return a `Future` without waiting.

Codec plugins, executors, and allocators have no safe API; write them against [`singe-nvimagecodec-sys`](../singe-nvimagecodec-sys/README.md).

## Example

Decode a JPEG file into a grayscale image in host memory:

```rust,no_run
use std::mem::MaybeUninit;

use singe_cuda::context::Context;
use singe_nvimagecodec::{
    code_stream::CodeStream,
    codec::ExecutionParams,
    decode::{DecodeParams, Decoder},
    image::{Image, ImageInfo, ImagePlaneInfo},
    instance::{Instance, InstanceOptions},
    types::{ChromaSubsampling, ColorSpec, SampleDataType, SampleFormat},
};

fn main() -> anyhow::Result<()> {
    let instance = Instance::create(&InstanceOptions::default())?;
    let input = CodeStream::from_file(&instance, "image.jpg", None)?;
    let input_info = input.image_info()?;
    let plane = input_info
        .plane(0)
        .ok_or_else(|| anyhow::anyhow!("image has no planes"))?;

    let (width, height) = (plane.width(), plane.height());
    let row_stride = usize::try_from(width)?;
    let mut output = vec![MaybeUninit::<u8>::uninit(); row_stride * usize::try_from(height)?];
    let planes = [ImagePlaneInfo::create(width, height, row_stride, 1, SampleDataType::U8)?];
    let mut output_info = ImageInfo::from_host_output(&mut output, SampleFormat::PlanarY, &planes)?;
    output_info
        .set_color_spec(ColorSpec::Gray)
        .set_chroma_subsampling(ChromaSubsampling::Gray);
    let mut output_image = Image::create(&instance, output_info)?;

    let context = Context::create()?;
    let decoder = Decoder::create(&instance, &ExecutionParams::new(&context), None)?;
    decoder.decode_one(&input, &mut output_image, &DecodeParams::new())?;
    println!("first row: {:?}", output_image.host_row(0, 0)?);
    Ok(())
}
```

The safe `decode` and `encode` methods, and their `_one` forms, wait for all CPU and GPU work and check each image's status before returning.
`ExecutionParams::cpu_only()` runs the codecs that support it on the CPU.
The default `InstanceOptions` load nvImageCodec's codec extension modules from its default install location.

## More examples

- [Decode](./examples/decode.rs): check that the decoder supports a bundled JPEG, size the planes from its metadata, decode it into planar RGB in host memory, and print a checksum.
