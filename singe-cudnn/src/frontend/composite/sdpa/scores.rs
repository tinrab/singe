use crate::{
    backend::{
        pointwise::PointwiseMode,
        tensor::{Shape, TensorId},
    },
    data_type::DataType,
    error::{Error, Result},
    frontend::{
        composite::sdpa::support::SDPA_RANK,
        error::Error as FrontendError,
        graph::Graph,
        operation::{SdpaScoreModifiers, SdpaScoreSubgraph},
    },
};

impl Graph {
    /// Applies scale, the custom graph, built-in bias, `ALiBi` and masks in C++ order.
    pub(crate) fn sdpa_scores_with_modifiers(
        &mut self,
        scores: TensorId,
        scale: Option<TensorId>,
        modifiers: &SdpaScoreModifiers,
        score_subgraph: Option<&SdpaScoreSubgraph>,
        score_subgraph_error_name: &str,
        compute_type: DataType,
        output_type: DataType,
    ) -> Result<TensorId> {
        let mut scores = match scale {
            Some(scale) => {
                self.sdpa_binary(PointwiseMode::Mul, scores, scale, output_type, compute_type)?
            }
            None => scores,
        };
        if let Some(score_subgraph) = score_subgraph {
            scores = self.import_score_subgraph(
                score_subgraph,
                scores,
                score_subgraph_error_name,
                None,
            )?;
        }
        scores = self.sdpa_optional_add(scores, modifiers.bias(), compute_type, output_type)?;
        let alibi_bias = self.sdpa_optional_alibi_bias(scores, modifiers)?;
        scores = self.sdpa_optional_add(scores, alibi_bias, compute_type, output_type)?;
        scores =
            self.sdpa_optional_add(scores, modifiers.additive_mask(), compute_type, output_type)?;
        if modifiers.padding_mask()
            && let Some((sequence_length_query, sequence_length_key_value)) =
                modifiers.sequence_lengths()
        {
            scores =
                self.padding_mask_select(scores, sequence_length_query, sequence_length_key_value)?;
        }
        if let Some(band) = modifiers.diagonal_band() {
            scores = self.sdpa_diagonal_band_mask(scores, band, modifiers.sequence_inputs())?;
        }
        Ok(scores)
    }
}

/// Attention requires positive head counts and an integral query-to-cache grouping ratio.
pub fn attention_heads_compatible(query_heads: i64, cache_heads: i64) -> bool {
    query_heads > 0 && cache_heads > 0 && query_heads % cache_heads == 0
}

pub fn infer_sdpa_scores_shape(q: &Shape, k: &Shape) -> Result<Shape> {
    if q.rank() != 4 || k.rank() != 4 {
        return Err(Error::LengthMismatch {
            name: SDPA_RANK.into(),
            expected: 4,
            actual: q.rank().min(k.rank()),
        });
    }

    let q_dims = q.dimensions();
    let k_dims = k.dimensions();
    if q_dims[0] != k_dims[0]
        || !attention_heads_compatible(q_dims[1], k_dims[1])
        || q_dims[3] != k_dims[3]
    {
        return Err(FrontendError::SdpaQueryKeyShapeMismatch {
            query: q_dims.to_vec(),
            key: k_dims.to_vec(),
        }
        .into());
    }

    Shape::contiguous([q_dims[0], q_dims[1], q_dims[2], k_dims[2]])
}

pub fn infer_sdpa_output_shape(scores: &Shape, v: &Shape) -> Result<Shape> {
    if scores.rank() != 4 || v.rank() != 4 {
        return Err(Error::LengthMismatch {
            name: SDPA_RANK.into(),
            expected: 4,
            actual: scores.rank().min(v.rank()),
        });
    }

    let scores_dims = scores.dimensions();
    let v_dims = v.dimensions();
    if scores_dims[0] != v_dims[0]
        || !attention_heads_compatible(scores_dims[1], v_dims[1])
        || scores_dims[3] != v_dims[2]
    {
        return Err(FrontendError::SdpaScoresValueShapeMismatch {
            scores: scores_dims.to_vec(),
            value: v_dims.to_vec(),
        }
        .into());
    }

    Shape::contiguous(vec![
        scores_dims[0],
        scores_dims[1],
        scores_dims[2],
        v_dims[3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heads_must_divide_query_heads() {
        assert!(attention_heads_compatible(8, 2));
        assert!(!attention_heads_compatible(8, 3));
        assert!(!attention_heads_compatible(0, 1));
        assert!(!attention_heads_compatible(4, 0));
    }

    #[test]
    fn test_scores_and_output_shapes() -> Result<()> {
        let query = Shape::contiguous([2, 8, 16, 64])?;
        let key = Shape::contiguous([2, 2, 32, 64])?;
        let scores = infer_sdpa_scores_shape(&query, &key)?;
        assert_eq!(scores.dimensions(), [2, 8, 16, 32]);
        let value = Shape::contiguous([2, 2, 32, 128])?;
        let output = infer_sdpa_output_shape(&scores, &value)?;
        assert_eq!(output.dimensions(), [2, 8, 16, 128]);
        Ok(())
    }

    #[test]
    fn test_scores_reject_mismatched_hidden_dimensions() -> Result<()> {
        let query = Shape::contiguous([2, 8, 16, 64])?;
        let key = Shape::contiguous([2, 8, 32, 32])?;
        assert!(infer_sdpa_scores_shape(&query, &key).is_err());
        Ok(())
    }
}
