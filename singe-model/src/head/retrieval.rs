//! A multi-vector retriever, as `ColBERT`-style models are: each token of a query and of a document becomes a unit embedding, and a document scores against a query by late interaction.

use std::fmt::Debug;

use singe::{DataType, Error as SingeError, Tensor};

use crate::{
    error::Result,
    processor::{image::Image, text::TextTokenizer},
};

pub trait DocumentRetriever: Debug {
    /// Returns each of a query's tokens' embedding, `[T, D]`.
    fn query(&self, tokenizer: &TextTokenizer, query: &str) -> Result<Tensor>;

    /// Returns each of a document image's tokens' embedding, `[T, D]`.
    fn document(&self, tokenizer: &TextTokenizer, image: &Image) -> Result<Tensor>;
}

/// Returns the late interaction score of a query's and a document's token embeddings, `[T, D]` each, as `ColBERT`'s `MaxSim` scores them: the sum over the query's tokens of each one's largest dot product with a document token.
///
/// # Errors
///
/// Returns an error if `document` is not two-dimensional.
pub fn late_interaction(query: &Tensor, document: &Tensor) -> Result<f32> {
    let &[columns, _] = document.shape().dimensions() else {
        return Err(SingeError::invalid_argument("document embeddings, [T, D]").into());
    };
    let columns =
        usize::try_from(columns).map_err(|_| SingeError::out_of_range("document tokens"))?;
    let products = query
        .cast(DataType::F32)?
        .matmul(&document.cast(DataType::F32)?.transpose(0, 1)?)?
        .to_vec::<f32>()?;
    Ok(products
        .chunks(columns.max(1))
        .map(|row| row.iter().copied().fold(f32::NEG_INFINITY, f32::max))
        .sum())
}
