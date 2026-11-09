// What an encoder reads: a batch of token sequences padded to one length, with which tokens are real.

use std::iter::repeat_n;

use crate::{
    data_type::DataType,
    device::Device,
    error::{Error, Result},
    nn::decode::input_group,
    shape::Shape,
    tensor::Tensor,
};

input_group! {
    // Token ids `[B, T]`, their token types `[B, T]`, and `padding`, `[B, T]` of `Bool`, true where a token is real, as a batch padded at its end has it: each sequence's real tokens come first.
    pub struct TokenBatch {
        tokens,
        token_types,
        padding,
    }
}

impl TokenBatch {
    // A batch of `tokens`, `token_types`, and `padding`, each `[B, T]`.
    //
    // # Errors
    //
    // Returns an error if their shapes differ, the ids are not `I64`, or `padding` is not `Bool`.
    pub fn create(tokens: Tensor, token_types: Tensor, padding: Tensor) -> Result<Self> {
        if tokens.shape().dimensions().len() != 2
            || token_types.shape() != tokens.shape()
            || padding.shape() != tokens.shape()
        {
            return Err(Error::invalid_argument(
                "token batch tensors must all be [batch, tokens]",
            ));
        }

        if tokens.data_type() != DataType::I64
            || token_types.data_type() != DataType::I64
            || padding.data_type() != DataType::Bool
        {
            return Err(Error::invalid_argument(
                "token batch ids are i64 and its padding is bool",
            ));
        }
        Ok(Self {
            tokens,
            token_types,
            padding,
        })
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
        let mut padding = Vec::with_capacity(sequences.len() * length);
        for sequence in sequences {
            tokens.extend_from_slice(sequence);
            tokens.extend(repeat_n(0, length - sequence.len()));
            padding.extend((0..length).map(|position| position < sequence.len()));
        }

        let shape = Shape::create([
            i64::try_from(sequences.len()).map_err(|_| Error::out_of_range("token batch size"))?,
            i64::try_from(length).map_err(|_| Error::out_of_range("token batch length"))?,
        ])?;
        let token_types = Tensor::zeroed_on(device, DataType::I64, shape.clone())?;
        Self::create(
            Tensor::from_vec(device, shape.clone(), tokens)?,
            token_types,
            Tensor::from_vec(device, shape, padding)?,
        )
    }

    // This batch with `token_types` in place of its own.
    //
    // # Errors
    //
    // Returns an error if `token_types` is not `I64` of the batch's shape.
    pub fn with_token_types(self, token_types: Tensor) -> Result<Self> {
        Self::create(self.tokens, token_types, self.padding)
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
    pub const fn padding(&self) -> &Tensor {
        &self.padding
    }

    // Each sequence's count of real tokens, `[B]` of `I32`, which attention reads in place of a mask.
    pub fn key_lengths(&self) -> Result<Tensor> {
        self.padding
            .cast(DataType::I32)?
            .sum(&[1], false)?
            .cast(DataType::I32)
    }
}
