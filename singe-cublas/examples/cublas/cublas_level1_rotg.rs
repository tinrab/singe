use singe_cublas::{blas::level1::rotg, context::Context, scalar::ScalarMut};
use singe_cuda::{context::Context as CudaContext, device::Device};

fn main() -> anyhow::Result<()> {
    let cuda = CudaContext::create_for_device(Device::new(0))?;
    let cublas = Context::create(&cuda)?;
    let mut a = 2.1_f64;
    let mut b = 1.2_f64;
    let mut c = 2.1_f64;
    let mut s = 1.2_f64;

    rotg(
        &cublas,
        ScalarMut::host(&mut a),
        ScalarMut::host(&mut b),
        ScalarMut::host(&mut c),
        ScalarMut::host(&mut s),
    )?;

    assert!((a - 2.418_677_324_489_564_7).abs() < 1.0e-12);
    assert!((b - 0.496_138_938_356_833_87).abs() < 1.0e-12);

    println!("rotg a: {a}");
    println!("rotg b: {b}");
    Ok(())
}
