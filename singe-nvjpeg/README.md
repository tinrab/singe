# singe-nvjpeg

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvjpeg.svg)](https://crates.io/crates/singe-nvjpeg)
[![Documentation](https://docs.rs/singe-nvjpeg/badge.svg)](https://docs.rs/singe-nvjpeg)
![License](https://img.shields.io/crates/l/singe-nvjpeg.svg)

Safe Rust wrappers for nvJPEG, which decodes and encodes JPEG images on the GPU, and for nvJPEG 2000 behind the `k2` feature.
The crate works on bytes in memory; read and write files with `std::fs`.

This crate provides:

- **Decoding:** `Decoder` for one image at a time, with output formats such as interleaved or planar RGB, planar YUV, and luma only, and options for a region, scaling, and EXIF orientation.
- **Batched decoding:** `BatchedDecoder` decodes several JPEGs in one call.
- **Encoding:** `Encoder` writes device images as JPEG, with a quality setting, chroma subsampling, and interleaved or planar input.
- **Buffers and allocators:** reusable pinned and device buffers, and custom allocators.
- **JPEG 2000:** decoding and encoding through nvJPEG 2000, in the `k2` module.

## Example

Decode a JPEG file into interleaved RGB pixels in device memory and copy them to the host:

```rust,no_run
use singe_cuda::{context::Context as CudaContext, stream::StreamOptions};
use singe_nvjpeg::{
    context::{Context, ContextOptions},
    decode::{DecodeOptions, Decoder},
    types::OutputFormat,
};

fn main() -> anyhow::Result<()> {
    let cuda_context = CudaContext::create()?;
    let stream = cuda_context.create_stream(StreamOptions::default())?;
    let context = Context::create(&cuda_context, &ContextOptions::default())?;

    let options = DecodeOptions::new().with_output_format(OutputFormat::RgbInterleaved);
    let mut decoder = Decoder::create(&context, options)?;
    let data = std::fs::read("input.jpg")?;
    let decoded = decoder.decode_to_device(&data, &stream)?;

    let plane = decoded.single_plane()?;
    let size = plane.spec().size();
    let pixels = plane.data().copy_to_host_vec()?;
    println!("decoded {}x{} into {} RGB bytes", size.width(), size.height(), pixels.len());
    Ok(())
}
```

`decode_to_device` allocates the output planes, decodes into them, and waits for the GPU before returning.
The `unsafe` `*_async` functions enqueue the same work without waiting, for stream capture and for callers that manage their own buffers.

## Features

- `k2`: adds the `k2` module for JPEG 2000, which loads nvJPEG 2000 at run time.

## More examples

The examples write their output to the system temporary directory.

- [JPEG decode](./examples/decode.rs): decode a JPEG file and write a PPM image.
- [Batch decode](./examples/batch_decode.rs): decode several JPEG files in one batch and write PPM images.
- [JPEG encode](./examples/encode.rs): encode a generated RGB gradient as JPEG.
- [JPEG 2000 decode](./examples/jpeg2k_decode.rs): decode a JPEG 2000 file and write a PGM image, with `--features k2`.
- [JPEG 2000 encode](./examples/jpeg2k_encode.rs): losslessly encode a generated RGB gradient as a JP2 file, with `--features k2`.
