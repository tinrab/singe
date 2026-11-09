# singe-nvimagecodec

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-nvimagecodec.svg)](https://crates.io/crates/singe-nvimagecodec)
[![Documentation](https://docs.rs/singe-nvimagecodec/badge.svg)](https://docs.rs/singe-nvimagecodec)
![License](https://img.shields.io/crates/l/singe-nvimagecodec.svg)

Safe Rust wrappers for the NVIDIA nvImageCodec library.

This crate provides:

- Image decoding and encoding through nvImageCodec's codec modules.
- Host and device image buffers with explicit plane layouts and sample types.
- Batch processing, capability queries, and per-image status results.
- Execution on the CPU or on a chosen CUDA context.

## Example

Decode a JPEG into an `Image` with the file's dimensions and sample format.

```rust,ignore
use singe_nvimagecodec::{
	code_stream::CodeStream,
	codec::ExecutionParams,
	decode::{DecodeParams, Decoder},
	types::ProcessingStatus,
};

let input = CodeStream::from_file(&instance, "input.jpg", None)?;
let execution = ExecutionParams::new(&context);
let mut decoder = Decoder::create(&instance, &execution, None)?;
let params = DecodeParams::new();
let status = decoder.decode_one(&input, &mut image, &params)?;
assert_eq!(status, ProcessingStatus::SUCCESS);
```

`CodeStream` holds encoded input, while `Image` describes the destination pixels.
Check processing status before reading decoded samples.
The [complete example](examples/decode.rs) also reads the image metadata, allocates the planes, and copies the pixels back to the host.
Use `ExecutionParams::cpu_only()` to run on the CPU, for codecs that support it.

See the [API reference](https://docs.rs/singe-nvimagecodec).
