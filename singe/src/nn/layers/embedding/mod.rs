// Lookups of learned vectors by index.

use crate::{
    backend::registry::{Operation, OperationAttributes},
    dispatch::call_one,
    error::{Error, Result},
    nn::{Layer, LayerNorm, ModelState, PatchEmbedding, StateKey, TokenBatch},
    shape::Shape,
    tensor::Tensor,
};

/// A layer selecting rows from a named embedding weight table using input indices.
#[derive(Debug, Clone)]
pub struct Embedding {
    weight: StateKey,
}

impl Embedding {
    /// Stores the embedding weight key without resolving its table or validating index metadata.
    #[must_use]
    pub fn new(weight: StateKey) -> Self {
        Self { weight }
    }
}

impl Layer for Embedding {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        call_one(
            Operation::Embedding,
            OperationAttributes::None,
            &[state.tensor(&self.weight)?, input],
        )
    }
}

// A token's embedding, plus its absolute position's and its token type's when a model has them, then normed if it norms them, as BERT-family encoders embed.
#[derive(Debug, Clone)]
pub struct TokenEmbeddings {
    words: Embedding,
    positions: Option<StateKey>,
    token_types: Option<Embedding>,
    norm: Option<LayerNorm>,
}

impl TokenEmbeddings {
    #[must_use]
    pub const fn new(words: Embedding) -> Self {
        Self {
            words,
            positions: None,
            token_types: None,
            norm: None,
        }
    }

    // Adds learned absolute positions from a table `[P, W]`, of which a batch of `T` tokens reads the first `T` rows.
    #[must_use]
    pub fn with_positions(mut self, table: StateKey) -> Self {
        self.positions = Some(table);
        self
    }

    #[must_use]
    pub fn with_token_types(mut self, token_types: Embedding) -> Self {
        self.token_types = Some(token_types);
        self
    }

    #[must_use]
    pub fn with_norm(mut self, norm: LayerNorm) -> Self {
        self.norm = Some(norm);
        self
    }
}

impl Layer<TokenBatch> for TokenEmbeddings {
    fn forward(&self, input: &TokenBatch, state: &ModelState) -> Result<Tensor> {
        let mut hidden = input.tokens().apply(&self.words, state)?;
        if let Some(table) = &self.positions {
            let length = input.tokens().shape().dimensions()[1];
            hidden = hidden.add(&state.tensor(table)?.narrow(0, 0, length)?)?;
        }

        if let Some(token_types) = &self.token_types {
            hidden = hidden.add(&input.token_types().apply(token_types, state)?)?;
        }

        match &self.norm {
            Some(norm) => hidden.apply(norm, state),
            None => Ok(hidden),
        }
    }
}

// An image's tokens as a vision transformer starts from them: its patches' embeddings, after a learned class token if the model has one, plus learned positions if it has them.
#[derive(Debug, Clone)]
pub struct PatchTokens {
    patches: PatchEmbedding,
    class_token: Option<StateKey>,
    positions: Option<StateKey>,
}

impl PatchTokens {
    #[must_use]
    pub const fn new(patches: PatchEmbedding) -> Self {
        Self {
            patches,
            class_token: None,
            positions: None,
        }
    }

    // Puts a learned token, `[1, 1, D]`, before each image's patches.
    #[must_use]
    pub fn with_class_token(mut self, class_token: StateKey) -> Self {
        self.class_token = Some(class_token);
        self
    }

    // Adds learned positions, `[1, N, D]`, one per token, so an image must have the number of patches they were learned for.
    #[must_use]
    pub fn with_positions(mut self, positions: StateKey) -> Self {
        self.positions = Some(positions);
        self
    }
}

impl Layer for PatchTokens {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let mut tokens = input.apply(&self.patches, state)?;
        let &[batch, _, width] = tokens.shape().dimensions() else {
            return Err(Error::invalid_argument("patch tokens"));
        };
        if let Some(class_token) = &self.class_token {
            let class_token = state
                .tensor(class_token)?
                .expand(Shape::create([batch, 1, width])?)?;
            tokens = Tensor::concat(&[class_token, tokens], 1)?;
        }

        let Some(positions) = &self.positions else {
            return Ok(tokens);
        };
        let positions = state.tensor(positions)?;
        // Interpolating positions to another grid is not here yet, so another image size is an error rather than a different result.
        if positions.shape().dimensions().get(1) != tokens.shape().dimensions().get(1) {
            return Err(Error::invalid_argument(
                "an image size whose patch count differs from the learned positions",
            ));
        }
        tokens.add(positions)
    }
}
