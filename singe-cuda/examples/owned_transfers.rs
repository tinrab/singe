//! Compose an owned upload and readback, then await the host result.

use std::error::Error;

use singe_cuda::{
    context::Context,
    submission::{
        Operation,
        transfer::{DeviceValue, Upload},
    },
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let stream = context.create_stream()?;
    let input: Vec<u32> = (0..4096).collect();

    // Upload copies the input into owned pinned staging. The original host
    // allocation is no longer needed, even before submission begins.
    let upload = Upload::create(&context, &input)?;
    drop(input);
    let round_trip = upload.then(DeviceValue::readback);
    let output = round_trip.submit_on(&stream)?.await?;

    // Resolution exposes initialized host values only after the transfers complete.
    assert_eq!(output, (0..4096).collect::<Vec<u32>>());
    println!("uploaded and read back {} elements", output.len());
    Ok(())
}
