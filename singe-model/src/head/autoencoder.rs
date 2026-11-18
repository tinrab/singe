use singe::{
    Result as SingeResult, Tensor,
    nn::{Model, Sequential},
};

// An image autoencoder's two halves: an encoder from images, `[B, C, H, W]` in [-1, 1], to latents, and a decoder from latents back to images.
// Each half has its own state, so each captures, tunes, and exports alone.
#[derive(Debug, Clone)]
pub struct Autoencoder {
    encoder: Model<Sequential>,
    decoder: Model<Sequential>,
    // The latent channels of an encoder to a distribution, whose output is the means and then the log-variances.
    distribution: Option<i64>,
}

impl Autoencoder {
    #[must_use]
    pub const fn new(encoder: Model<Sequential>, decoder: Model<Sequential>) -> Self {
        Self {
            encoder,
            decoder,
            distribution: None,
        }
    }

    // The same autoencoder, its encoder giving a distribution over `latent` channels.
    #[must_use]
    pub const fn with_distribution(mut self, latent: i64) -> Self {
        self.distribution = Some(latent);
        self
    }

    #[must_use]
    pub const fn encoder(&self) -> &Model<Sequential> {
        &self.encoder
    }

    #[must_use]
    pub const fn decoder(&self) -> &Model<Sequential> {
        &self.decoder
    }

    // Returns the latents `diffusers`' `encode` returns, a distribution's mode, before any pipeline's scaling factor.
    pub fn encode(&self, images: &Tensor) -> SingeResult<Tensor> {
        let encoded = self.encoder.forward(images)?;
        match self.distribution {
            Some(latent) => encoded.narrow(1, 0, latent),
            None => Ok(encoded),
        }
    }

    // Returns the latent distribution's means and deviations, as `DiagonalGaussianDistribution` reads them, with the log-variances clamped to [-30, 20]; an encoder without a distribution gives its latents and no deviation.
    pub fn encode_distribution(&self, images: &Tensor) -> SingeResult<(Tensor, Option<Tensor>)> {
        let encoded = self.encoder.forward(images)?;
        let Some(latent) = self.distribution else {
            return Ok((encoded, None));
        };
        let deviation = encoded
            .narrow(1, latent, latent)?
            .clamp(-30.0, 20.0)?
            .mul_scalar(0.5)?
            .exp()?;
        Ok((encoded.narrow(1, 0, latent)?, Some(deviation)))
    }

    pub fn decode(&self, latents: &Tensor) -> SingeResult<Tensor> {
        self.decoder.forward(latents)
    }
}
