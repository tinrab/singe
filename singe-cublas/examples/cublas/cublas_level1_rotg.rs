use singe_cublas::{blas::level1::drotg, context::Context, error::Result};
use singe_cuda::{context::Context as CudaContext, device::Device};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut a = 2.1_f64;
    let mut b = 1.2_f64;
    let mut c = 2.1_f64;
    let mut s = 1.2_f64;

    drotg(&cublas, &mut a, &mut b, &mut c, &mut s)?;

    assert!((a - 2.4186773244895647).abs() < 1.0e-12);
    assert!((b - 0.49613893835683387).abs() < 1.0e-12);

    println!("rotg a: {a}");
    println!("rotg b: {b}");
    Ok(())
}
