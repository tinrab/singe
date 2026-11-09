# singe-nvjpeg

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvjpeg.svg)](https://crates.io/crates/singe-nvjpeg)
[![Documentation](https://docs.rs/singe-nvjpeg/badge.svg)](https://docs.rs/singe-nvjpeg)
![License](https://img.shields.io/crates/l/singe-nvjpeg.svg)

Safe Rust wrappers for the NVIDIA nvJPEG and nvJPEG 2000 libraries.

This crate provides:

- Single-image and batched JPEG decoding.
- JPEG encoding with configurable quality and input layouts.
- Device output, planar/interleaved pixel views, and decode options.
- JPEG 2000 encoding and decoding through the optional `k2` feature.

## Example

Decode a JPEG into interleaved RGB device memory:

```rust
use singe_cuda::context::Context;
use singe_nvjpeg::{
	decode::{DecodeOptions, Decoder},
	types::OutputFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let stream = context.create_stream()?;
	let mut decoder = Decoder::builder()
		.options(DecodeOptions::new().with_output_format(OutputFormat::RgbInterleaved))
		.create(&context)?;
	let decoded = decoder.decode_file_to_device("input.jpg", &stream)?;
	let pixels = decoded.single_plane()?.data().copy_to_host_vec()?;
	println!("decoded {} RGB bytes", pixels.len());
	Ok(())
}
```

The decoder keeps its library handle and CUDA context alive, and returns once the pixels are written.
Padding bytes at the end of each row are not written.

Enable the `k2` feature for the JPEG 2000 API, which loads nvJPEG 2000 at run time.

## More examples

- [JPEG decode](examples/decode.rs): decode a file and write a PPM image.
- [Batch decode](examples/batch_decode.rs): decode several JPEGs together.
- [JPEG encode](examples/encode.rs): encode pixel data.
- [JPEG 2000 decode](examples/jpeg2k_decode.rs): decode with `--features k2`.
- [JPEG 2000 encode](examples/jpeg2k_encode.rs): encode with `--features k2`.

See the [API reference](https://docs.rs/singe-nvjpeg).
