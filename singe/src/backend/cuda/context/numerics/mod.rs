use singe_core::ErrorChain;
use singe_cublas::{
    context::Context as CublasContext,
    types::{AtomicsMode, MathMode},
};

use crate::{
    error::Result,
    graph::{ComputePrecision, Determinism, NumericalPolicy},
};

struct CublasNumericalGuard<'a> {
    context: &'a CublasContext,
    math: MathMode,
    atomics: AtomicsMode,
    active: bool,
}

impl<'a> CublasNumericalGuard<'a> {
    fn create(context: &'a CublasContext) -> Result<Self> {
        let guard = Self {
            context,
            math: context.math_mode()?,
            atomics: context.atomics_mode()?,
            active: true,
        };
        let policy = NumericalPolicy::current();
        let math = match policy.precision {
            ComputePrecision::Full => MathMode::PEDANTIC,
            ComputePrecision::AllowTf32 | ComputePrecision::Relaxed => MathMode::TF32_TENSOR_OP,
        } | MathMode::DISALLOW_REDUCED_PRECISION_REDUCTION;
        context.set_math_mode(math)?;
        if policy.determinism == Determinism::Require {
            context.set_atomics_mode(AtomicsMode::NotAllowed)?;
        }
        Ok(guard)
    }

    fn restore(&mut self) -> Result<()> {
        self.active = false;
        let math = self.context.set_math_mode(self.math);
        let atomics = self.context.set_atomics_mode(self.atomics);
        math?;
        atomics?;
        Ok(())
    }
}

impl Drop for CublasNumericalGuard<'_> {
    fn drop(&mut self) {
        if self.active
            && let Err(error) = self.restore()
        {
            eprintln!("failed to restore cublas math mode: {}", ErrorChain(&error));
        }
    }
}

pub fn with_cublas_numerics<T>(
    context: &CublasContext,
    f: impl FnOnce(&CublasContext) -> Result<T>,
) -> Result<T> {
    let mut guard = CublasNumericalGuard::create(context)?;
    let result = f(context);
    let reset = guard.restore();
    match (result, reset) {
        (Ok(value), Ok(())) => Ok(value),
        (Ok(_), Err(error)) | (Err(error), _) => Err(error),
    }
}

#[cfg(test)]
mod tests;
