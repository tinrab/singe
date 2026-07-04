use singe_cublas::{
    error::Result,
    types::Operation,
    xt::{blas::sgemm, context::Context},
};

fn main() -> Result<()> {
    let ctx = Context::create()?;
    ctx.set_block_dim(64)?;

    let alpha = 1.0_f32;
    let beta = 0.0_f32;

    // Column-major 2x2 matrices:
    // A = [1 3; 2 4], B = [5 7; 6 8].
    let a = [1.0_f32, 2.0, 3.0, 4.0];
    let b = [5.0_f32, 6.0, 7.0, 8.0];
    let mut c = [0.0_f32; 4];

    sgemm(
        &ctx,
        Operation::NonTranspose,
        Operation::NonTranspose,
        2,
        2,
        2,
        &alpha,
        &a,
        2,
        &b,
        2,
        &beta,
        &mut c,
        2,
    )?;

    singe_core::assert_close!(&c, &[23.0, 34.0, 31.0, 46.0], 1.0e-4);

    Ok(())
}
