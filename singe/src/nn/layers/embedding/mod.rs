// Lookups of learned vectors by index.

use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    nn::{Layer, LayerNorm, ModelState, PatchEmbedding, StateKey, TokenBatch},
    shape::Shape,
    tensor::{InterpolateMode, Tensor},
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
    // The table row of a sequence's first position.
    first_position: i64,
    token_types: Option<Embedding>,
    norm: Option<LayerNorm>,
}

impl TokenEmbeddings {
    #[must_use]
    pub const fn new(words: Embedding) -> Self {
        Self {
            words,
            positions: None,
            first_position: 0,
            token_types: None,
            norm: None,
        }
    }

    // Adds learned absolute positions from a table `[P, W]`, of which a batch of `T` tokens reads `T` rows from its first position.
    #[must_use]
    pub fn with_positions(mut self, table: StateKey) -> Self {
        self.positions = Some(table);
        self
    }

    // Numbers a sequence's positions from `first`, as RoBERTa numbers them after its padding id.
    #[must_use]
    pub const fn with_first_position(mut self, first: i64) -> Self {
        self.first_position = first;
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
            hidden = hidden.add(
                &state
                    .tensor(table)?
                    .narrow(0, self.first_position, length)?,
            )?;
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
    // The patch side, when positions learned for one square grid resize to another.
    resized_positions: Option<i64>,
}

impl PatchTokens {
    #[must_use]
    pub const fn new(patches: PatchEmbedding) -> Self {
        Self {
            patches,
            class_token: None,
            positions: None,
            resized_positions: None,
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

    // Resizes the positions of a square grid of patches to an image's grid of `patch`-pixel patches bicubically, as DINOv2 and other vision transformers read images of other sizes; a class token's position stays as it is.
    #[must_use]
    pub const fn with_resized_positions(mut self, patch: i64) -> Self {
        self.resized_positions = Some(patch);
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
        let mut positions = state.tensor(positions)?.clone();
        if let Some(patch) = self.resized_positions {
            let leading = i64::from(self.class_token.is_some());
            positions = resized_positions(&positions, input, patch, leading)?;
        }

        if positions.shape().dimensions().get(1) != tokens.shape().dimensions().get(1) {
            return Err(Error::invalid_argument(
                "an image size whose patch count differs from the learned positions",
            ));
        }
        tokens.add(&positions)
    }
}

// Positions `[1, leading + S · S, D]` resized to `images`' grid of `patch`-pixel patches, unless the grid is square and of the same size, as `transformers`' `interpolate_pos_encoding` computes them in `F32`.
fn resized_positions(
    positions: &Tensor,
    images: &Tensor,
    patch: i64,
    leading: i64,
) -> Result<Tensor> {
    let &[_, count, width] = positions.shape().dimensions() else {
        return Err(Error::invalid_argument(
            "positions of other than three axes",
        ));
    };
    let &[.., height, image_width] = images.shape().dimensions() else {
        return Err(Error::invalid_argument("images of fewer than two axes"));
    };
    let (rows, columns) = (height / patch, image_width / patch);
    let stored = count - leading;
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        reason = "grids are far below 2^53 patches"
    )]
    let side = (stored as f64).sqrt() as i64;
    if side * side != stored {
        return Err(Error::invalid_argument(
            "positions of other than a square grid",
        ));
    }

    if rows == columns && rows * columns == stored {
        return Ok(positions.clone());
    }

    let grid = positions
        .narrow(1, leading, stored)?
        .reshape([1, side, side, width])?
        .permute(&[0, 3, 1, 2])?
        .cast(DataType::F32)?
        .interpolate((rows, columns), InterpolateMode::Bicubic)?
        .cast(positions.data_type())?
        .permute(&[0, 2, 3, 1])?
        .reshape([1, rows * columns, width])?;
    Tensor::concat(&[positions.narrow(1, 0, leading)?, grid], 1)
}
