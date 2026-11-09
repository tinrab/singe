mod clear;
mod stats;

use super::*;

fn clear_allocator_cache(device: Device) {
    cuda::clear_cache(device).unwrap();
}

fn byte_shape(byte_size: usize) -> Shape {
    Shape::create([i64::try_from(byte_size).unwrap()]).unwrap()
}
