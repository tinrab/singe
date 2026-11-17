use singe_cublas::{
    blas::level1::rotmg,
    context::Context,
    scalar::{Scalar, ScalarMut},
};
use singe_cuda::{context::Context as CudaContext, device::Device};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut d1 = 1.0_f64;
    let mut d2 = 5.0_f64;
    let mut x1 = 2.1_f64;
    let y1 = 1.2_f64;
    let mut param = [1.0_f64, 5.0, 6.0, 7.0, 8.0];

    rotmg(
        &cublas,
        ScalarMut::host(&mut d1),
        ScalarMut::host(&mut d2),
        ScalarMut::host(&mut x1),
        Scalar::host(&y1),
        ScalarMut::host(&mut param),
    )?;

    assert!((d1 - 3.098_901_098_901_099).abs() < 1.0e-12);
    assert!((d2 - 0.619_780_219_780_219_8).abs() < 1.0e-12);
    assert!((x1 - 1.936_363_636_363_636_5).abs() < 1.0e-12);

    println!("rotmg d1: {d1}");
    println!("rotmg d2: {d2}");
    println!("rotmg x1: {x1}");
    println!("rotmg param: {param:?}");
    Ok(())
}
