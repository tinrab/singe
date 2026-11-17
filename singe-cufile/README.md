# singe-cufile

<!-- [![Rust](https://github.com/tinrab/singe/workflows/Rust/badge.svg)](https://github.com/tinrab/singe/actions) -->

[![Latest version](https://img.shields.io/crates/v/singe-cufile.svg)](https://crates.io/crates/singe-cufile)
[![Documentation](https://docs.rs/singe-cufile/badge.svg)](https://docs.rs/singe-cufile)
![License](https://img.shields.io/crates/l/singe-cufile.svg)

Safe Rust wrappers for cuFile, the NVIDIA `GPUDirect` Storage library, which reads and writes files directly to and from GPU memory.

This crate provides:

- **Driver session:** `Driver`, a shared lease that opens cuFile on first use and closes it after the last lease, with driver properties and limits.
- **File I/O:** `FileHandle` reads and writes that wait for the transfer, vectored forms, stream-ordered `*_scoped` forms, and `unsafe` `read_async` and `write_async`.
- **Registration:** registered device buffers and streams.
- **Batch I/O:** `Batch` submits several requests at once and reports an event for each.
- **Configuration and statistics:** process-wide parameters, buffer-pool settings, operation counters, and per-GPU statistics.

## Example

Write 4096 bytes from GPU memory to a file, read them back into new GPU memory, and compare the two buffers:

```rust,no_run
use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};

use singe_cuda::{context::Context, memory::DeviceMemory};
use singe_cufile::{
    buffer::{IoSlice, IoSliceMut},
    driver::Driver,
    file::FileHandle,
};

fn main() -> anyhow::Result<()> {
    let context = Context::create()?;
    let driver = Driver::create()?;

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open("cufile-example.bin")?;
    let handle = FileHandle::create(&driver, &file)?;

    let source = DeviceMemory::<u8>::filled(&context, 4096, 0xab)?;
    let written = handle.write(IoSlice::create(&source, ..)?, 0)?;

    let mut destination = DeviceMemory::<u8>::create_uninit(&context, written)?;
    let read = handle.read(IoSliceMut::create(&mut destination, ..)?, 0)?;
    // SAFETY: the completed read initialized exactly the first `read` bytes.
    let destination = unsafe { destination.assume_init_prefix(read)? };
    assert_eq!(destination.copy_to_host_vec()?, source.copy_to_host_vec()?);
    Ok(())
}
```

`write` and `read` wait for the transfer and return the number of bytes moved, which can be less than requested.
The second argument is the offset in the file.
`IoSlice::create` takes an initialized device or host buffer, and `IoSliceMut::create` also takes uninitialized device memory; `..` selects every byte.

The direct path needs a file opened with `O_DIRECT` on a file system `GPUDirect` Storage supports, such as ext4 or XFS on local `NVMe` drives.
When its configuration allows it, cuFile's compatibility mode runs the same calls through host bounce buffers instead, and `Driver::properties` reports which modes are available.

## More examples

- [Read and write](./examples/basic_rw_integrity.rs): read a file into GPU memory, write it to another file, and compare the files.
- [Registered write](./examples/registered_write.rs): register a device buffer with cuFile before writing it.
- [Batch write](./examples/batch_write.rs): submit several writes as one batch.
