//! Drawing on the host: tokens from their logits, the likeliest or a draw as `transformers` samples, through `TemperatureLogitsWarper` and `TopKLogitsWarper` and then a multinomial draw, and values from Student-T distributions.
//! The draws come from `singe`'s Philox generator, so a seed repeats them, but they are not `PyTorch`'s draws.

use std::f32::consts::TAU;

use serde::{Deserialize, Serialize};
use singe::random::philox;

use crate::{
    error::{Error, Result},
    head::argmax,
};

/// How a generator samples, as a generation config states it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Sampling {
    pub do_sample: bool,
    pub temperature: f32,
    pub top_k: Option<usize>,
}

impl Default for Sampling {
    /// `GenerationConfig`'s defaults: greedy, or when sampling, temperature 1 and the 50 likeliest tokens.
    fn default() -> Self {
        Self {
            do_sample: false,
            temperature: 1.0,
            top_k: Some(50),
        }
    }
}

/// A generation config's sampling fields, `GenerationConfig`'s defaults where it leaves them out.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct SamplingFields {
    pub do_sample: bool,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
}

impl SamplingFields {
    #[must_use]
    pub fn sampling(&self) -> Sampling {
        let defaults = Sampling::default();
        Sampling {
            do_sample: self.do_sample,
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_k: self.top_k.or(defaults.top_k),
        }
    }
}

/// A request's sampling, each field the model's own where `None`, and the seed of its draws.
#[derive(Debug, Clone, Copy, PartialEq, Default, Deserialize)]
#[serde(default)]
pub struct SamplingRequest {
    pub do_sample: Option<bool>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: u64,
}

impl SamplingRequest {
    /// Returns a sampler of this request over the model's own `sampling`.
    #[must_use]
    pub fn sampler(&self, sampling: Sampling) -> Sampler {
        Sampler {
            sampling: Sampling {
                do_sample: self.do_sample.unwrap_or(sampling.do_sample),
                temperature: self.temperature.unwrap_or(sampling.temperature),
                top_k: self.top_k.or(sampling.top_k),
            },
            uniforms: Uniforms::new(self.seed),
        }
    }
}

/// Uniform draws in `(0, 1)`, one Philox block's first word each, from a seed.
#[derive(Debug, Clone)]
pub struct Uniforms {
    seed: u64,
    counter: u64,
}

impl Uniforms {
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    /// Returns the next draw, 24 random bits centred in their interval so it is never 0 or 1.
    pub fn uniform(&mut self) -> f32 {
        let bits = philox(self.seed, self.counter)[0];
        self.counter += 1;
        ((bits >> 8) as f32 + 0.5) / 16_777_216.0
    }

    /// Returns a standard normal draw, by Box–Muller.
    pub fn normal(&mut self) -> f32 {
        let (first, second) = (self.uniform(), self.uniform());
        (-2.0 * first.ln()).sqrt() * (TAU * second).cos()
    }

    /// Returns a draw from the gamma distribution of `shape` and unit scale, by Marsaglia and Tsang's method, which a shape below 1 reaches through `shape + 1`.
    pub fn gamma(&mut self, shape: f32) -> f32 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.uniform().powf(1.0 / shape);
        }

        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            if self.uniform().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    /// Returns a draw from the standard Student-T distribution of `df` degrees of freedom: a normal draw over the root of a chi-squared draw over `df`.
    pub fn student_t(&mut self, df: f32) -> f32 {
        self.normal() / (2.0 * self.gamma(df / 2.0) / df).sqrt()
    }
}

/// Chooses tokens one after another, drawing from a counter that each draw advances.
#[derive(Debug, Clone)]
pub struct Sampler {
    sampling: Sampling,
    uniforms: Uniforms,
}

impl Sampler {
    /// Returns the token chosen from `logits`: the greatest, or a draw from the softmax of the logits over the temperature, kept to the `top_k` greatest.
    ///
    /// # Errors
    ///
    /// Returns an error if there are no logits.
    pub fn choose(&mut self, logits: &[f32]) -> Result<usize> {
        let greatest = || {
            argmax(logits.iter().copied().enumerate())
                .map(|(token, _)| token)
                .ok_or_else(|| Error::invalid("no logits to choose from"))
        };
        let Sampling {
            do_sample,
            temperature,
            top_k,
        } = self.sampling;
        if !do_sample {
            return greatest();
        }

        let mut ranked = logits
            .iter()
            .map(|&logit| logit / temperature)
            .enumerate()
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
        if let Some(kept) = top_k.filter(|&kept| kept > 0) {
            // `TopKLogitsWarper` keeps every token tied with the last of the `k`.
            if let Some(&(_, threshold)) = ranked.get(kept - 1) {
                ranked.retain(|&(_, logit)| logit >= threshold);
            }
        }

        let Some(&(_, highest)) = ranked.first() else {
            return greatest();
        };
        let weights = ranked
            .iter()
            .map(|&(_, logit)| (logit - highest).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f32>();
        let draw = self.uniforms.uniform() * total;
        let mut reached = 0.0;
        for (&(token, _), weight) in ranked.iter().zip(&weights) {
            reached += weight;
            if draw < reached {
                return Ok(token);
            }
        }
        Ok(ranked[ranked.len() - 1].0)
    }
}
