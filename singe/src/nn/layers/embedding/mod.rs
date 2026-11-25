// Lookups of learned vectors by index.

use crate::{
    backend::registry::{Operation, OperationAttributes},
    data_type::DataType,
    dispatch::call_one,
    error::{Error, Result},
    graph::capture,
    nn::{Layer, LayerNorm, ModelState, PatchEmbedding, StateKey, TokenBatch},
    shape::spatial::Spatial2d,
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
        let tokens = input.tokens();
        let mut hidden = tokens.apply(&self.words, state)?;
        if let Some(table) = &self.positions {
            // Each token's position counts the tokens before it, from tensors alone, so a program built with a run-time length reads as many rows as it has tokens.
            let ones = tokens.equal(tokens)?.cast(DataType::I64)?;
            let positions = ones.cumsum(1)?.sub(&ones)?;
            let table = state.tensor(table)?;
            let rows = table
                .shape()
                .dimensions()
                .first()
                .copied()
                .ok_or_else(|| Error::invalid_argument("position table"))?;
            let table = table.narrow(0, self.first_position, rows - self.first_position)?;
            hidden = hidden.add(&call_one(
                Operation::Embedding,
                OperationAttributes::None,
                &[&table, &positions],
            )?)?;
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

// An image's tokens as a vision transformer starts from them: its patches' embeddings, after the learned tokens the model puts first, such as a class token, plus learned positions if it has them.
#[derive(Debug, Clone)]
pub struct PatchTokens {
    patches: PatchEmbedding,
    leading: Vec<StateKey>,
    positions: Option<StateKey>,
    // The patch side, when positions learned for one square grid resize to another.
    resized_positions: Option<i64>,
}

impl PatchTokens {
    #[must_use]
    pub const fn new(patches: PatchEmbedding) -> Self {
        Self {
            patches,
            leading: Vec::new(),
            positions: None,
            resized_positions: None,
        }
    }

    // Puts a learned token, `[1, 1, D]`, after the leading tokens added before it and before each image's patches, as a class or a distillation token.
    #[must_use]
    pub fn with_leading_token(mut self, token: StateKey) -> Self {
        self.leading.push(token);
        self
    }

    // Adds learned positions, `[1, N, D]`, one per token, so an image must have the number of patches they were learned for.
    #[must_use]
    pub fn with_positions(mut self, positions: StateKey) -> Self {
        self.positions = Some(positions);
        self
    }

    // Resizes the positions of a square grid of patches to an image's grid of `patch`-pixel patches bicubically, as DINOv2 and other vision transformers read images of other sizes; the leading tokens' positions stay as they are.
    #[must_use]
    pub const fn with_resized_positions(mut self, patch: i64) -> Self {
        self.resized_positions = Some(patch);
        self
    }
}

impl Layer for PatchTokens {
    fn forward(&self, input: &Tensor, state: &ModelState) -> Result<Tensor> {
        let mut tokens = input.apply(&self.patches, state)?;
        if tokens.shape().dimensions().len() != 3 {
            return Err(Error::invalid_argument("patch tokens"));
        }

        if !self.leading.is_empty() {
            // Zeros of one token per image, `[B, 1, W]`, broadcast each learned token to the batch from tensors alone, so a program built with a run-time batch follows it.
            let zeros = tokens.narrow(1, 0, 1)?.mul_scalar(0.0)?;
            let mut parts = self
                .leading
                .iter()
                .map(|token| zeros.add(state.tensor(token)?))
                .collect::<Result<Vec<_>>>()?;
            parts.push(tokens);
            tokens = Tensor::concat(&parts, 1)?;
        }

        let Some(positions) = &self.positions else {
            return Ok(tokens);
        };
        let mut positions = state.tensor(positions)?.clone();
        if let Some(patch) = self.resized_positions {
            let leading = i64::try_from(self.leading.len())
                .map_err(|_| Error::out_of_range("leading tokens"))?;
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
// A program with run-time axes resizes them at every size, as a traced `transformers` model does, which leaves a grid of the same size as it is.
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
    let rank = images.shape().dimensions().len();
    let size = Spatial2d::new(
        images.dimension(rank - 2)? / patch,
        images.dimension(rank - 1)? / patch,
    );
    let stored = count - leading;
    let side = (stored as f64).sqrt() as i64;
    if side * side != stored {
        return Err(Error::invalid_argument(
            "positions of other than a square grid",
        ));
    }

    if rows == columns && rows * columns == stored && !capture::has_run_time_axes()? {
        return Ok(positions.clone());
    }

    let grid = positions
        .narrow(1, leading, stored)?
        .reshape([1, side, side, width])?
        .permute(&[0, 3, 1, 2])?
        .cast(DataType::F32)?
        .interpolate(size, InterpolateMode::Bicubic)?
        .cast(positions.data_type())?
        .permute(&[0, 2, 3, 1])?
        .flatten(1, 2)?;
    Tensor::concat(&[positions.narrow(1, 0, leading)?, grid], 1)
}
