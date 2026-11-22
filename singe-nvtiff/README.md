# singe-nvtiff

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvtiff.svg)](https://crates.io/crates/singe-nvtiff)
[![Documentation](https://docs.rs/singe-nvtiff/badge.svg)](https://docs.rs/singe-nvtiff)
![License](https://img.shields.io/crates/l/singe-nvtiff.svg)

Safe Rust wrappers for NVIDIA nvTIFF, for decoding and encoding TIFF images on the GPU.

This crate provides:

- **Parsing:** `TiffStream` opens a TIFF file from a path or from bytes and reads image information, strip and tile geometry, tag values, and `GeoTIFF` keys.
- **Decoding:** `Decoder` decodes selected images, or regions of them, into device memory.
- **Encoding:** `Encoder` encodes device images into an `EncodedTiff` in host memory, with image geometry and `GeoTIFF` keys.
- **Allocators:** custom device and pinned host allocators for the decoder and encoder.

## Example

Decode the first image of a TIFF file and encode it into a new file:

```rust,no_run
use singe_cuda::{context::Context, stream::StreamOptions};
use singe_nvtiff::{
    allocator::AllocatorSet,
    decode::Decoder,
    encode::{EncodeParams, Encoder},
    image::IfdOffset,
    stream::TiffStream,
};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let stream = context.create_stream(StreamOptions::default())?;

    let source = TiffStream::open_file("input.tif")?;
    let info = source.image_info(IfdOffset::new(0))?;
    let mut decoder = Decoder::create(AllocatorSet::new(), &stream)?;
    let pixels = decoder.decode_to_device(&source)?;

    let mut encoder = Encoder::create(AllocatorSet::new(), &stream)?;
    let mut params = EncodeParams::create(&stream)?;
    params.set_image_info(&info.to_encode_info()?)?;
    encoder.encode(&mut params, [&pixels])?.write_file("output.tif")?;
    Ok(())
}
```

A decoder or encoder runs its work on the stream it was created with.
`decode_to_device` allocates tightly packed device memory, decodes into it, and waits for the GPU before returning.
`encode` also waits, and the `EncodedTiff` it returns holds the file in host memory.
The `unsafe` `decode_async`, `encode_async`, and `finalize_async` functions enqueue the work without waiting.

## More examples

- [Decode and encode](./examples/decode_encode.rs): decode a range of images, print their metadata, write them as PNM files, and optionally encode them into a new TIFF file.
- [Region decode](./examples/decode_image_roi.rs): decode a region of one image and write it as a PNM file.
- [`GeoTIFF` decode](./examples/geotiff_decode.rs): print TIFF tags and `GeoTIFF` keys, decode a range of images, and write them as PNM files.
