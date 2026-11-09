# singe-nvtiff

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvtiff.svg)](https://crates.io/crates/singe-nvtiff)
[![Documentation](https://docs.rs/singe-nvtiff/badge.svg)](https://docs.rs/singe-nvtiff)
![License](https://img.shields.io/crates/l/singe-nvtiff.svg)

Safe Rust wrappers for the NVIDIA nvTIFF library.

This crate provides:

- TIFF parsing and image metadata queries.
- GPU decoding with selected images and regions of interest.
- TIFF encoding to owned host bytes or files.
- Typed image layouts, decode parameters, and GeoTIFF metadata support.

## Example

Decode the first TIFF image and encode it to a new file:

```rust
use singe_cuda::context::Context;
use singe_nvtiff::{
	decode::Decoder,
	encode::{EncodeParams, Encoder},
	image::IfdOffset,
	stream::TiffStream,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
	let context = Context::create()?;
	let stream = context.create_stream()?;
	let source = TiffStream::open_file("input.tif")?;
	let info = source.image_info(IfdOffset::new(0))?;
	let mut decoder = Decoder::create(&stream)?;
	let pixels = decoder.decode_to_device(&source)?;
	let mut encoder = Encoder::create(&stream)?;
	let mut params = EncodeParams::create(&stream)?;
	params.set_image_info(&info.to_encode_info()?)?;
	encoder.encode(&mut params, [&pixels])?.write_file("output.tif")?;
	Ok(())
}
```

`TiffStream` holds the parsed file, and decoding returns once the pixels are written.
`EncodedTiff` holds the encoded file in host memory, independent of the CUDA resources that produced it.

## More examples

- [Decode and encode](examples/decode_encode.rs): select frames, inspect metadata, and write output.
- [Region decode](examples/decode_image_roi.rs): decode a selected image region.
- [GeoTIFF](examples/geotiff_decode.rs): inspect geographic metadata and decode imagery.

See the [API reference](https://docs.rs/singe-nvtiff).
