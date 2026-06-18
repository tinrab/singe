use std::{
    env,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::Path,
};

use singe_cuda::{device::Device, memory::DeviceMemory};
use singe_cufile::{driver::Driver, file::FileHandle};

const SIZE: usize = 1024 * 1024;
const O_DIRECT: i32 = 0o40000;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <input-file> <output-file> <gpu-id>", args[0]);
        std::process::exit(2);
    }

    let input_path = Path::new(&args[1]);
    let output_path = Path::new(&args[2]);
    let gpu_id = args[3].parse()?;

    if input_path == output_path {
        return Err("input and output files must be different".into());
    }

    Device::new(gpu_id).set_current()?;
    let _driver = Driver::create()?;

    create_pattern_file(input_path, SIZE)?;

    let input = open_direct_read(input_path)?;
    let output = open_direct_write(output_path)?;
    let input = FileHandle::create(&input)?;
    let output = FileHandle::create(&output)?;

    let mut device = DeviceMemory::<u8>::zeroes(SIZE)?;

    // Read directly from storage into GPU memory, then write the same GPU buffer
    // back out to another file. The wrapper keeps the cuFile handle lifecycle
    // scoped to ordinary Rust values.
    let read = input.read(&mut device, SIZE, 0, 0)?;
    let written = output.write(&device, read, 0, 0)?;
    drop(output);

    if written != read {
        return Err(format!("short write: read {read} bytes, wrote {written} bytes").into());
    }

    let expected = fs::read(input_path)?;
    let actual = fs::read(output_path)?;
    if expected != actual {
        return Err("input and output file contents differ".into());
    }

    println!("read {read} bytes into GPU memory and wrote {written} bytes back");
    println!("file contents match");
    Ok(())
}

fn create_pattern_file(path: &Path, size: usize) -> io::Result<()> {
    let mut file = File::create(path)?;
    let mut pattern = vec![0u8; size];
    for (index, byte) in pattern.iter_mut().enumerate() {
        *byte = (index % 251) as u8;
    }
    file.write_all(&pattern)
}

fn open_direct_read(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)
}

fn open_direct_write(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .custom_flags(O_DIRECT)
        .open(path)
}
