use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
    process,
};

use singe_cuda::{context::Context, device::Device, memory::DeviceMemory};
use singe_cufile::{
    buffer::{IoSlice, IoSliceMut},
    driver::Driver,
    file::FileHandle,
};

const SIZE: usize = 1024 * 1024;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <input-file> <output-file> <gpu-id>", args[0]);
        process::exit(2);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let gpu_id = args[3].parse()?;

    if input_path == output_path {
        anyhow::bail!("input and output files must be different");
    }

    let context = Context::create_for_device(Device::new(gpu_id))?;
    context.bind()?;
    let driver = Driver::create()?;

    create_pattern_file(input_path, SIZE)?;

    let input = open_direct_read(input_path)?;
    let output = open_direct_write(output_path)?;
    let input = FileHandle::create(&driver, &input)?;
    let output = FileHandle::create(&driver, &output)?;

    let mut device = DeviceMemory::<u8>::create_uninit(&context, SIZE)?;

    // Read directly from storage into GPU memory, then write the same GPU buffer
    // back out to another file. The wrapper keeps the cuFile handle lifecycle
    // scoped to ordinary Rust values.
    let read = input.read(IoSliceMut::create(&mut device, ..)?, 0)?;
    // SAFETY: synchronous file I/O initialized exactly the returned byte count.
    let device = unsafe { device.assume_init_prefix(read)? };
    let written = output.write(IoSlice::create(&device, ..)?, 0)?;
    drop(output);

    if written != read {
        anyhow::bail!("short write: read {read} bytes, wrote {written} bytes");
    }

    let expected = fs::read(input_path)?;
    let actual = fs::read(output_path)?;
    if expected != actual {
        anyhow::bail!("input and output file contents differ");
    }

    println!("read {read} bytes into GPU memory and wrote {written} bytes back");
    println!("file contents match");
    Ok(())
}

fn create_pattern_file(path: &Path, size: usize) -> io::Result<()> {
    let mut file = File::create(path)?;
    let mut pattern = vec![0u8; size];
    for (byte, value) in pattern.iter_mut().zip((0..251_u8).cycle()) {
        *byte = value;
    }
    file.write_all(&pattern)
}

fn open_direct_read(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}

fn open_direct_write(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
}
