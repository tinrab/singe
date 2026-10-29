use std::{
    env,
    error::Error,
    fs::{self, File, OpenOptions},
    io::Result as IoResult,
    os::unix::fs::OpenOptionsExt,
    path::Path,
    process,
};

use singe_cuda::{context::Context, device::Device, memory::DeviceMemory};
use singe_cufile::{
    buffer::RegisteredBuffer, driver::Driver, file::FileHandle, types::BufferRegisterFlags,
};

const SIZE: usize = 128 * 1024;
const O_DIRECT: i32 = 0o40000;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: {} <output-file> <gpu-id>", args[0]);
        process::exit(2);
    }

    let output_path = Path::new(&args[1]);
    let gpu_id = args[2].parse()?;

    let context = Context::retain_primary_for_device(Device::new(gpu_id))?;
    context.bind()?;
    let driver = Driver::create()?;

    let output = open_direct_write(output_path)?;
    let output = FileHandle::create(&driver, &output)?;

    let device = DeviceMemory::<u8>::filled(&context, SIZE, 0xab)?;

    // Explicit buffer registration is optional for many cuFile paths, but it is
    // useful when the same GPU allocation will be used repeatedly.
    // SAFETY: the device allocation outlives this registration, which is dropped after synchronous I/O completes.
    let registered =
        unsafe { RegisteredBuffer::register(&driver, &device, BufferRegisterFlags::empty())? };

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

fn open_direct_write(path: &Path) -> IoResult<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT)
        .open(path)
}
