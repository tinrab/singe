use singe_cublas::{blas::level1::drotmg, context::Context, error::Result};
use singe_cuda::{context::Context as CudaContext, device::Device};

fn main() -> Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut d1 = 1.0_f64;
    let mut d2 = 5.0_f64;
    let mut x1 = 2.1_f64;
    let y1 = 1.2_f64;
    let mut param = [1.0_f64, 5.0, 6.0, 7.0, 8.0];

    drotmg(&cublas, &mut d1, &mut d2, &mut x1, &y1, &mut param)?;

    assert!((d1 - 3.098901098901099).abs() < 1.0e-12);
    assert!((d2 - 0.6197802197802198).abs() < 1.0e-12);
    assert!((x1 - 1.9363636363636365).abs() < 1.0e-12);

    println!("rotmg d1: {d1}");
    println!("rotmg d2: {d2}");
    println!("rotmg x1: {x1}");
    println!("rotmg param: {param:?}");
    Ok(())
}
