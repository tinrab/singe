//! Distribute independent transfer jobs over a pool of CUDA streams.

use std::error::Error;

use singe_cuda::{
    context::Context,
    submission::{
        Operation, StreamPool,
        transfer::{DeviceValue, Upload},
    },
};

fn main() -> Result<(), Box<dyn Error>> {
    let context = Context::create()?;
    let pool = StreamPool::create(&context, 2)?;
    let mut submissions = Vec::new();

    for job in 0..8_u32 {
        let input = vec![job; 4096];
        let operation = Upload::create(&context, &input)?.then(DeviceValue::readback);
        // Round-robin submission selects a stream. Each job owns separate
        // buffers, so different streams can make progress independently.
        submissions.push((job, pool.submit(operation)?));
    }

    // Submit every job before waiting, allowing CUDA to overlap independent work.
    for (job, submission) in submissions {
        assert_eq!(submission.wait()?, vec![job; 4096]);
    }
    println!(
        "verified 8 jobs submitted across {} streams",
        pool.streams().len()
    );
    Ok(())
}
