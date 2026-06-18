use std::{
    env,
    fs::{self, File, OpenOptions},
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

use singe_cuda::{device::Device, memory::DeviceMemory};
use singe_cufile::{
    buffer::RegisteredBuffer, driver::Driver, file::FileHandle, types::BufferRegisterFlags,
};

const SIZE: usize = 128 * 1024;
const O_DIRECT: i32 = 0o40000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <output-file> <gpu-id>", args[0]);
        std::process::exit(2);
    }

    let output_path = Path::new(&args[1]);
    let gpu_id = args[2].parse()?;

    Device::new(gpu_id).set_current()?;
    let _driver = Driver::create()?;

    let output = open_direct_write(output_path)?;
    let output = FileHandle::create(&output)?;

    let mut device = DeviceMemory::<u8>::create(SIZE)?;
    device.set_value(0xab)?;

    // Explicit buffer registration is optional for many cuFile paths, but it is
    // useful when the same GPU allocation will be used repeatedly.
    let registered = RegisteredBuffer::register_memory(&device, BufferRegisterFlags::empty())?;

    let written = output.write(&registered, SIZE, 0, 0)?;
    drop(output);

    if written != SIZE {
        return Err(format!("short write: wrote {written} of {SIZE} bytes").into());
    }

    let contents = fs::read(output_path)?;
    if contents.len() != SIZE || contents.iter().any(|byte| *byte != 0xab) {
        return Err("output file does not contain the expected 0xab pattern".into());
    }

    println!("wrote {written} bytes from registered GPU memory");
    Ok(())
}

fn open_direct_write(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT)
        .open(path)
}
