//! Rust take on the CUDA "deviceQuery" sample: print the driver and runtime
//! versions and the main properties of every visible device.

use std::error::Error;

use singe_cuda::{device::Device, driver, runtime};

fn main() -> Result<(), Box<dyn Error>> {
    println!("CUDA driver version: {}", driver::version()?);
    println!("CUDA runtime version: {}", runtime::version()?);

    let count = Device::count()?;
    println!("detected {count} CUDA capable device(s)");

    for ordinal in 0..count {
        let device = Device::new(ordinal);
        let properties = device.properties()?;
        println!();
        println!("device {ordinal}: {}", properties.name);
        println!(
            "  compute capability:       {}.{}",
            properties.major, properties.minor
        );
        println!(
            "  global memory:            {} MiB",
            properties.total_global_mem / (1024 * 1024)
        );
        println!(
            "  multiprocessors:          {}",
            properties.multi_processor_count
        );
        println!("  warp size:                {}", properties.warp_size);
        println!(
            "  max threads per block:    {}",
            properties.max_threads_per_block
        );
        println!(
            "  max block dimensions:     {:?}",
            properties.max_threads_dim
        );
        println!("  max grid dimensions:      {:?}", properties.max_grid_size);
        println!(
            "  shared memory per block:  {} KiB",
            properties.shared_mem_per_block / 1024
        );
        println!(
            "  L2 cache size:            {} KiB",
            properties.l2_cache_size / 1024
        );
        println!(
            "  memory bus width:         {} bits",
            properties.memory_bus_width
        );
        println!("  PCI bus id:               {}", device.pci_bus_id()?);
    }
    Ok(())
}
