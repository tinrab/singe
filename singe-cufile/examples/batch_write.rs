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
    batch::{Batch, BatchCookie, BatchRequest},
    buffer::RegisteredBuffer,
    driver::Driver,
    file::FileHandle,
    types::BufferRegisterFlags,
};

const CHUNK_SIZE: usize = 4096;
const MAX_BATCH: usize = 128;
const O_DIRECT: i32 = 0o40000;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        eprintln!("usage: {} <output-file> <gpu-id> <batch-size>", args[0]);
        process::exit(2);
    }

    let output_path = Path::new(&args[1]);
    let gpu_id = args[2].parse()?;
    let batch_size: usize = args[3].parse()?;
    if !(1..=MAX_BATCH).contains(&batch_size) {
        return Err(format!("batch size must be in 1..={MAX_BATCH}").into());
    }

    let context = Context::retain_primary_for_device(Device::new(gpu_id))?;
    context.bind()?;
    let driver = Driver::create()?;

    let output = open_direct_write(output_path)?;
    let output = FileHandle::create(&driver, &output)?;

    let mut device_buffers = Vec::with_capacity(batch_size);
    for index in 0..batch_size {
        let buffer =
            DeviceMemory::<u8>::filled(&context, CHUNK_SIZE, 0xef_u8.wrapping_add(index as u8))?;
        device_buffers.push(buffer);
    }

    let registered_buffers = device_buffers
        .iter()
        .map(|buffer| {
            // SAFETY: device_buffers outlives these registrations, and execute completes all I/O before they drop.
            unsafe { RegisteredBuffer::register(&driver, buffer, BufferRegisterFlags::empty()) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    let requests = registered_buffers
        .iter()
        .enumerate()
        .map(|(index, buffer)| {
            BatchRequest::write(
                &output,
                buffer,
                CHUNK_SIZE,
                (index * CHUNK_SIZE) as u64,
                0,
                BatchCookie::new(index),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    // The completed batch keeps the file, registered storage, and native request array alive through every completion.
    let mut batch = Batch::create(&driver, batch_size)?;
    let events = batch.execute(requests)?;
    for event in events {
        let index = event.cookie.get();
        let bytes = event.bytes_transferred()?;
        println!("batch entry {index} completed with {bytes} bytes");
    }
    drop(output);

    verify_pattern_file(output_path, batch_size)?;
    println!("verified {batch_size} batch writes of {CHUNK_SIZE} bytes each");
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

fn verify_pattern_file(path: &Path, batch_size: usize) -> Result<(), Box<dyn Error>> {
    let contents = fs::read(path)?;
    let expected_len = batch_size * CHUNK_SIZE;
    if contents.len() != expected_len {
        return Err(format!(
            "unexpected output length: got {}, expected {expected_len}",
            contents.len()
        )
        .into());
    }

    for index in 0..batch_size {
        let expected = 0xef_u8.wrapping_add(index as u8);
        let range = index * CHUNK_SIZE..(index + 1) * CHUNK_SIZE;
        if contents[range].iter().any(|byte| *byte != expected) {
            return Err(format!("batch chunk {index} does not match expected pattern").into());
        }
    }
    Ok(())
}
