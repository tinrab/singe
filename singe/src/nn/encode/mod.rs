// What an encoder reads: a batch of token sequences padded at their ends to one length, with each sequence's count of real tokens.

use std::iter::repeat_n;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    graph::DimensionExpr,
    nn::decode::input_group,
    shape::Shape,
    tensor::Tensor,
};

input_group! {
    // Token ids `[B, T]`, their token types `[B, T]`, and each row's count of real tokens, `[B]` of `I64`, which come first in the row.
    pub struct TokenBatch {
        tokens,
        token_types,
        lengths,
    }
}

#[derive(Debug, Clone)]
pub struct Placement {
    destinations: Tensor,
}

impl TokenBatch {
    // A batch of `tokens` and `token_types`, each `[B, T]`, whose rows start with `lengths`, `[B]`, real tokens.
    //
    // # Errors
    //
    // Returns an error if the shapes differ or the tensors are not `I64`.
    pub fn create(tokens: Tensor, token_types: Tensor, lengths: Tensor) -> Result<Self> {
        check_tokens(&tokens, &token_types)?;
        if lengths.data_type() != DataType::I64
            || lengths.shape().dimensions() != &tokens.shape().dimensions()[..1]
        {
            return Err(Error::invalid_argument(
                "token batch lengths must be [batch] of i64",
            ));
        }
        Ok(Self {
            tokens,
            token_types,
            lengths,
        })
    }

    pub fn from_mask(
        tokens: &Tensor,
        token_types: &Tensor,
        mask: &Tensor,
    ) -> Result<(Self, Placement)> {
        check_tokens(tokens, token_types)?;
        if mask.data_type() != DataType::Bool || mask.shape() != tokens.shape() {
            return Err(Error::invalid_argument(
                "a token batch's mask must be bool of the tokens' shape",
            ));
        }

        let real = mask.cast(DataType::I64)?;
        let padded = real.rsub_scalar(1)?;
        let lengths = real.sum(&[1], true)?;
        let real_before = real.cumsum(1)?.sub(&real)?;
        let padded_before = padded.cumsum(1)?.sub(&padded)?;
        let columns = Tensor::where_select(mask, &real_before, &lengths.add(&padded_before)?)?;

        let (batch, length) = (tokens.dimension(0)?, tokens.dimension(1)?);
        let flat = Tensor::range(tokens.device(), 0, batch.clone() * length.clone())?;
        let starts = flat
            .reshape_with(&[batch, length.clone()])?
            .sub(&Tensor::range(tokens.device(), 0, length)?.unsqueeze(0)?)?;
        let destinations = columns
            .add(&starts)?
            .reshape_with(&[DimensionExpr::constant(-1)])?;
        let sources = flat.mul_scalar(0)?.index_copy(0, &destinations, &flat)?;
        let moved = |tensor: &Tensor| {
            tensor
                .reshape_with(&[DimensionExpr::constant(-1)])?
                .index_select(0, &sources)?
                .reshape_with(&[tensor.dimension(0)?, tensor.dimension(1)?])
        };
        let batch = Self::create(moved(tokens)?, moved(token_types)?, lengths.squeeze(1)?)?;
        Ok((batch, Placement { destinations }))
    }

    // A batch of `sequences` on `device`, padded with id zero to the longest, all of token type zero.
    //
    // # Errors
    //
    // Returns an error if there is no sequence, every sequence is empty, or creating a tensor fails.
    pub fn from_sequences(device: Device, sequences: &[&[i64]]) -> Result<Self> {
        let length = sequences
            .iter()
            .map(|sequence| sequence.len())
            .max()
            .unwrap_or(0);
        if length == 0 {
            return Err(Error::invalid_argument(
                "a token batch holds at least one token",
            ));
        }

        let mut tokens = Vec::with_capacity(sequences.len() * length);
        let mut lengths = Vec::with_capacity(sequences.len());
        for sequence in sequences {
            tokens.extend_from_slice(sequence);
            tokens.extend(repeat_n(0, length - sequence.len()));
            lengths.push(
                i64::try_from(sequence.len()).map_err(|_| Error::out_of_range("token count"))?,
            );
        }

        let rows =
            i64::try_from(sequences.len()).map_err(|_| Error::out_of_range("token batch size"))?;
        let shape = Shape::create([
            rows,
            i64::try_from(length).map_err(|_| Error::out_of_range("token batch length"))?,
        ])?;
        let token_types = Tensor::zeroed_on(device, DataType::I64, shape.clone())?;
        Self::create(
            Tensor::from_vec(device, shape, tokens)?,
            token_types,
            Tensor::from_vec(device, Shape::create([rows])?, lengths)?,
        )
    }

    // This batch with `token_types` in place of its own.
    //
    // # Errors
    //
    // Returns an error if `token_types` is not `I64` of the batch's shape.
    pub fn with_token_types(self, token_types: Tensor) -> Result<Self> {
        Self::create(self.tokens, token_types, self.lengths)
    }

    #[must_use]
    pub const fn tokens(&self) -> &Tensor {
        &self.tokens
    }

    #[must_use]
    pub const fn token_types(&self) -> &Tensor {
        &self.token_types
    }

    #[must_use]
    pub const fn lengths(&self) -> &Tensor {
        &self.lengths
    }

    // Each row's count of real tokens as attention reads it, `[B]` of `I32`.
    pub fn key_lengths(&self) -> Result<Tensor> {
        self.lengths.cast(DataType::I32)
    }

    pub fn mask(&self) -> Result<Tensor> {
        Tensor::range(self.tokens.device(), 0, self.tokens.dimension(1)?)?
            .unsqueeze(0)?
            .less(&self.lengths.unsqueeze(1)?)
    }
}

impl Placement {
    pub fn restore(&self, values: &Tensor) -> Result<Tensor> {
        let rank = values.shape().dimensions().len();
        if rank < 2 {
            return Err(Error::invalid_argument(
                "restored values are [batch, tokens, ...]",
            ));
        }

        let dimensions = (0..rank)
            .map(|axis| values.dimension(axis))
            .collect::<Result<Vec<_>>>()?;
        let mut flat = vec![DimensionExpr::constant(-1)];
        flat.extend_from_slice(&dimensions[2..]);
        values
            .reshape_with(&flat)?
            .index_select(0, &self.destinations)?
            .reshape_with(&dimensions)
    }
}

fn check_tokens(tokens: &Tensor, token_types: &Tensor) -> Result<()> {
    if tokens.shape().dimensions().len() != 2 || token_types.shape() != tokens.shape() {
        return Err(Error::invalid_argument(
            "token batch ids must be [batch, tokens]",
        ));
    }

    if tokens.data_type() != DataType::I64 || token_types.data_type() != DataType::I64 {
        return Err(Error::invalid_argument("token batch ids are i64"));
    }
    Ok(())
}
