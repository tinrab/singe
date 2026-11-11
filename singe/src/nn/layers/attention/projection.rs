use crate::{
    error::Result,
    nn::{Layer, Linear, ModelState, OffsetRmsNorm, RmsNorm},
    tensor::Tensor,
};

// The norm a [`HeadProjection`] applies over each head.
#[derive(Debug, Clone)]
pub enum HeadNorm {
    Rms(RmsNorm),
    // Scaled by one plus its weight, as Gemma and Qwen3.5 write it.
    OffsetRms(OffsetRmsNorm),
}

impl From<RmsNorm> for HeadNorm {
    fn from(norm: RmsNorm) -> Self {
        Self::Rms(norm)
    }
}

impl From<OffsetRmsNorm> for HeadNorm {
    fn from(norm: OffsetRmsNorm) -> Self {
        Self::OffsetRms(norm)
    }
}

impl Layer for HeadNorm {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        match self {
            Self::Rms(norm) => norm.forward(input, state),
            Self::OffsetRms(norm) => norm.forward(input, state),
        }
    }
}

// A linear layer, a reshape from `[..., W]` to heads `[..., H, D]`, and an optional norm over each head.
//
// With an output gate, the linear layer gives each head `2 · D` features: the head's own, then a gate that attention multiplies its output by, through a sigmoid, as Qwen3.5's gated attention does.
#[derive(Debug, Clone)]
pub struct HeadProjection {
    linear: Linear,
    norm: Option<HeadNorm>,
    heads: i64,
    head_dim: i64,
    output_gate: bool,
}

impl HeadProjection {
    #[must_use]
    pub const fn new(linear: Linear, heads: i64, head_dim: i64) -> Self {
        Self {
            linear,
            norm: None,
            heads,
            head_dim,
            output_gate: false,
        }
    }

    // Norms each head after the reshape.
    #[must_use]
    pub fn with_norm(mut self, norm: impl Into<HeadNorm>) -> Self {
        self.norm = Some(norm.into());
        self
    }

    // Reads a gate for each head beside its features, which `project` returns apart.
    #[must_use]
    pub const fn with_output_gate(mut self) -> Self {
        self.output_gate = true;
        self
    }

    #[must_use]
    pub const fn heads(&self) -> i64 {
        self.heads
    }

    #[must_use]
    pub const fn head_dim(&self) -> i64 {
        self.head_dim
    }

    // Projects `input` to heads `[..., H, D]`, and to the gate `[..., H, D]` if this projection reads one.
    pub fn project(&self, input: &Tensor, state: &ModelState) -> Result<(Tensor, Option<Tensor>)> {
        let rank = input.shape().dimensions().len();
        let mut shape = (0..rank.saturating_sub(1))
            .map(|axis| input.dimension(axis))
            .collect::<Result<Vec<_>>>()?;
        let per_head = if self.output_gate {
            2 * self.head_dim
        } else {
            self.head_dim
        };
        shape.extend([self.heads.into(), per_head.into()]);
        let projected = input.apply(&self.linear, state)?.reshape_with(&shape)?;
        let (heads, gate) = if self.output_gate {
            (
                projected.narrow(-1, 0, self.head_dim)?,
                Some(projected.narrow(-1, self.head_dim, self.head_dim)?),
            )
        } else {
            (projected, None)
        };
        let heads = match &self.norm {
            Some(norm) => norm.forward(&heads, state)?,
            None => heads,
        };
        Ok((heads, gate))
    }
}

impl Layer for HeadProjection {
    // Projects `input` to heads, dropping any gate.
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        self.project(input, state).map(|(heads, _)| heads)
    }
}
