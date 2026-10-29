use std::{io, mem::size_of, sync::Arc};

use singe_cudnn::{
    error::Result,
    frontend::logging::{self, Logger, TensorDumpDisplay, TensorDumpFormat},
};

fn main() -> Result<()> {
    // This writer also receives preparation, plan-selection and execution diagnostics.
    // Omit explicit configuration to use CUDNN_FRONTEND_LOG_INFO and CUDNN_FRONTEND_LOG_FILE.
    let logger = Arc::new(Logger::new(io::stderr()));
    logging::set_logger(Some(Arc::clone(&logger)))?;
    logging::message(format_args!("application frontend logger installed"))?;

    let lengths = [0_i32, 32, 64, -1];
    let bytes = lengths
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    for format in [
        TensorDumpFormat::Decimal,
        TensorDumpFormat::Hexadecimal,
        TensorDumpFormat::Base64,
    ] {
        let values = TensorDumpDisplay::create(&bytes, size_of::<i32>(), format)?;
        logger.message(format_args!("sequence metadata {format:?}: {values}"))?;
    }

    // Disable automatic metadata readback while keeping ordinary messages enabled.
    logger.set_tensor_dump_enabled(false);
    logging::message(format_args!("tensor readback disabled"))?;
    logger.set_tensor_dump_enabled(true);
    // Reload the environment, or use set_logger(None) to disable all frontend messages.
    logging::configure_from_environment()?;
    Ok(())
}
