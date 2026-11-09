use std::panic::{self, AssertUnwindSafe};

use super::*;
use crate::{
    backend::cuda,
    device::Device,
    error::Error,
    graph::{Determinism, NumericalPolicy, NumericalScope},
};

#[test]
fn test_cublas_modes_restore_after_errors_and_unwinding() {
    let cuda = cuda::execution_context(Device::Cuda(0)).unwrap();
    let cublas = CublasContext::create(cuda.runtime().context()).unwrap();
    cublas.set_math_mode(MathMode::TF32_TENSOR_OP).unwrap();
    cublas.set_atomics_mode(AtomicsMode::Allowed).unwrap();
    let _scope = NumericalScope::enter(NumericalPolicy {
        determinism: Determinism::Require,
        ..Default::default()
    });
    let check = |context: &CublasContext| {
        assert_eq!(
            context.math_mode().unwrap(),
            MathMode::PEDANTIC | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION
        );
        assert_eq!(context.atomics_mode().unwrap(), AtomicsMode::NotAllowed);
    };
    assert!(
        with_cublas_numerics::<()>(&cublas, |context| {
            check(context);
            Err(Error::invalid_argument("test"))
        })
        .is_err()
    );
    assert_eq!(cublas.math_mode().unwrap(), MathMode::TF32_TENSOR_OP);
    assert_eq!(cublas.atomics_mode().unwrap(), AtomicsMode::Allowed);
    assert!(
        panic::catch_unwind(AssertUnwindSafe(|| {
            let _ = with_cublas_numerics::<()>(&cublas, |context| {
                check(context);
                panic!("exercise unwinding")
            });
        }))
        .is_err()
    );
    assert_eq!(cublas.math_mode().unwrap(), MathMode::TF32_TENSOR_OP);
    assert_eq!(cublas.atomics_mode().unwrap(), AtomicsMode::Allowed);
}
