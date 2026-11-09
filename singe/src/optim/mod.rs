//! Optimizers that update parameters from the gradients [`Tensor::backward`] accumulates.
//!
//! Parameters are leaf tensors that require gradients.
//! Each `step` runs without recording gradients and writes the new values into the parameters' storage, so tensors that share a parameter see the update.

use crate::{autograd::no_grad, error::Result, tensor::Tensor};

/// Stochastic gradient descent with optional momentum and L2 weight decay, as `torch.optim.SGD` computes it.
#[derive(Debug)]
pub struct Sgd {
    parameters: Vec<Tensor>,
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
    velocity: Vec<Option<Tensor>>,
}

impl Sgd {
    /// Creates an optimizer for `parameters` with `learning_rate`, no momentum, and no weight decay.
    #[must_use]
    pub fn new(parameters: Vec<Tensor>, learning_rate: f64) -> Self {
        let velocity = vec![None; parameters.len()];
        Self {
            parameters,
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            velocity,
        }
    }

    /// Accumulates updates with `momentum`, as heavy-ball momentum does.
    #[must_use]
    pub const fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Adds `weight_decay` times each parameter to its gradient.
    #[must_use]
    pub const fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Updates every parameter that has a gradient.
    ///
    /// # Errors
    ///
    /// Returns an error if an update operation fails.
    pub fn step(&mut self) -> Result<()> {
        let _no_grad = no_grad();
        for (parameter, velocity) in self.parameters.iter().zip(&mut self.velocity) {
            let Some(mut gradient) = parameter.grad() else {
                continue;
            };
            if self.weight_decay != 0.0 {
                gradient = gradient.add(&parameter.mul_scalar(self.weight_decay)?)?;
            }

            if self.momentum != 0.0 {
                gradient = match velocity.take() {
                    Some(previous) => previous.mul_scalar(self.momentum)?.add(&gradient)?,
                    None => gradient,
                };
                *velocity = Some(gradient.clone());
            }

            parameter.assign(&parameter.sub(&gradient.mul_scalar(self.learning_rate)?)?)?;
        }
        Ok(())
    }

    /// Clears the gradients of every parameter.
    pub fn zero_grad(&self) {
        zero_grad(&self.parameters);
    }
}

/// Adam with decoupled weight decay, as `torch.optim.AdamW` computes it.
#[derive(Debug)]
pub struct AdamW {
    parameters: Vec<Tensor>,
    learning_rate: f64,
    betas: (f64, f64),
    eps: f64,
    weight_decay: f64,
    step: i32,
    moments: Vec<Option<(Tensor, Tensor)>>,
}

impl AdamW {
    /// Creates an optimizer for `parameters` with `learning_rate` and `PyTorch`'s defaults: betas `(0.9, 0.999)`, epsilon `1e-8`, and weight decay `0.01`.
    #[must_use]
    pub fn new(parameters: Vec<Tensor>, learning_rate: f64) -> Self {
        let moments = vec![None; parameters.len()];
        Self {
            parameters,
            learning_rate,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            step: 0,
            moments,
        }
    }

    /// Sets the decay rates of the first and second moment estimates.
    #[must_use]
    pub const fn with_betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    /// Sets the term added to the denominator for numerical stability.
    #[must_use]
    pub const fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Sets the decoupled weight decay, applied to the parameters rather than the gradients.
    #[must_use]
    pub const fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Updates every parameter that has a gradient.
    ///
    /// # Errors
    ///
    /// Returns an error if an update operation fails.
    pub fn step(&mut self) -> Result<()> {
        let _no_grad = no_grad();
        self.step = self.step.saturating_add(1);
        let (beta1, beta2) = self.betas;
        let correction1 = 1.0 - beta1.powi(self.step);
        let correction2 = 1.0 - beta2.powi(self.step);
        for (parameter, moments) in self.parameters.iter().zip(&mut self.moments) {
            let Some(gradient) = parameter.grad() else {
                continue;
            };
            let (first, second) = match moments.take() {
                Some((first, second)) => (
                    first
                        .mul_scalar(beta1)?
                        .add(&gradient.mul_scalar(1.0 - beta1)?)?,
                    second
                        .mul_scalar(beta2)?
                        .add(&gradient.square()?.mul_scalar(1.0 - beta2)?)?,
                ),
                None => (
                    gradient.mul_scalar(1.0 - beta1)?,
                    gradient.square()?.mul_scalar(1.0 - beta2)?,
                ),
            };
            let denominator = second
                .mul_scalar(1.0 / correction2)?
                .sqrt()?
                .add_scalar(self.eps)?;
            let update = first.mul_scalar(1.0 / correction1)?.div(&denominator)?;
            let decayed =
                parameter.mul_scalar(self.learning_rate.mul_add(-self.weight_decay, 1.0))?;
            parameter.assign(&decayed.sub(&update.mul_scalar(self.learning_rate)?)?)?;
            *moments = Some((first, second));
        }
        Ok(())
    }

    /// Clears the gradients of every parameter.
    pub fn zero_grad(&self) {
        zero_grad(&self.parameters);
    }
}

fn zero_grad(parameters: &[Tensor]) {
    for parameter in parameters {
        parameter.zero_grad();
    }
}

#[cfg(test)]
mod tests;
