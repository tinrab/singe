# singe-cufile

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufile.svg)](https://crates.io/crates/singe-cufile)
[![Documentation](https://docs.rs/singe-cufile/badge.svg)](https://docs.rs/singe-cufile)
![License](https://img.shields.io/crates/l/singe-cufile.svg)

Safe Rust wrappers for the NVIDIA cuFile library, which moves data directly between files and GPU memory with GPUDirect Storage.

This crate provides:

- Reads and writes between files and GPU memory.
- File-handle and device-buffer registration.
- Batch, vectored, and stream I/O.
- Driver configuration, statistics, and capability queries.

## Example

Read a file into GPU memory, then write the bytes that were read to another file.
Assume `input_file` and `output_file` are files opened for direct I/O, and `context` is a CUDA context:

```rust,ignore
use singe_cuda::memory::DeviceMemory;
use singe_cufile::{driver::Driver, file::FileHandle};

let driver = Driver::create()?;
let input = FileHandle::create(&driver, &input_file)?;
let output = FileHandle::create(&driver, &output_file)?;
let mut buffer = DeviceMemory::<u8>::create_uninit(&context, 4096)?;
let read = input.read(&mut buffer, 4096, 0, 0)?;
// SAFETY: the completed read initialized exactly this prefix of byte storage.
let buffer = unsafe { buffer.assume_init_prefix(read)? };
let written = output.write(&buffer, read, 0, 0)?;
assert_eq!(written, read);
```

`Driver` initializes cuFile, and `FileHandle` registers an open file with it.
The two offsets are positions in the file and in the buffer.

GPUDirect Storage requires Linux and a supported file system.

## More examples

- [Registered write](examples/registered_write.rs): register a device buffer before writing.
- [Batch write](examples/batch_write.rs): submit several I/O operations together.

See the [API reference](https://docs.rs/singe-cufile).
