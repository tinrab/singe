// `fill-mask`: text with mask tokens to the likeliest tokens for each mask, as `transformers`' pipeline fills them.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{MaskedLm, ScoreFunction},
    host, model,
    processor::text::{EncodeOptions, TextInput, TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, run_alone},
};

#[derive(Debug)]
pub struct FillMaskPipeline {
    model: MaskedLm,
    tokenizer: TextTokenizer,
    mask: i64,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct FillMaskParameters {
    pub top_k: usize,
    // Scores only these tokens, each a token of the vocabulary or, if not, the first token its text encodes to.
    pub targets: Option<Vec<String>>,
}

impl Default for FillMaskParameters {
    fn default() -> Self {
        Self {
            top_k: 5,
            targets: None,
        }
    }
}

// A token for a mask, its probability, and the text with it in place.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fill {
    pub score: f32,
    pub token: i64,
    pub token_str: String,
    pub sequence: String,
}

// The fills of a text's one mask, or of each of its masks.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum FillMaskOutput {
    One(Vec<Fill>),
    Each(Vec<Vec<Fill>>),
}

impl Pipeline for FillMaskPipeline {
    type Input = String;

    type Parameters = FillMaskParameters;

    type Output = FillMaskOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let tokenizer = TextTokenizer::load(source)?;
        let mask = tokenizer
            .mask_id()
            .ok_or_else(|| Error::unsupported("a tokenizer without a mask token"))?;
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer,
            mask,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        run_alone(self, input, parameters)
    }

    fn run(
        &self,
        inputs: &[Self::Input],
        parameters: &Self::Parameters,
    ) -> Result<Vec<Self::Output>> {
        let targets = parameters
            .targets
            .as_ref()
            .map(|targets| self.target_ids(targets))
            .transpose()?;
        let texts = inputs
            .iter()
            .map(|text| TextInput::Single(text))
            .collect::<Vec<_>>();
        let encoded =
            self.tokenizer
                .encode_inputs(&texts, &EncodeOptions::default(), self.options.device)?;
        let positions = encoded
            .rows
            .iter()
            .enumerate()
            .flat_map(|(row, encoded)| {
                encoded
                    .ids
                    .iter()
                    .enumerate()
                    .filter(|&(_, &id)| id == self.mask)
                    .map(move |(token, _)| (row, token))
            })
            .collect::<Vec<_>>();
        let scores = self.model.scores_at(&encoded.batch, &positions)?;
        encoded
            .rows
            .iter()
            .enumerate()
            .map(|(row, encoded)| {
                let masks = positions
                    .iter()
                    .zip(&scores)
                    .filter(|((mask_row, _), _)| *mask_row == row)
                    .map(|((_, token), scores)| (*token, scores))
                    .collect::<Vec<_>>();
                if masks.is_empty() {
                    return Err(Error::invalid("a text without the mask token"));
                }

                // A text of one mask decodes without its special tokens, and one of several with them, as `transformers` does.
                let single = masks.len() == 1;
                let fills = masks
                    .iter()
                    .map(|&(token, scores)| {
                        self.fills(
                            &encoded.ids,
                            token,
                            scores,
                            targets.as_deref(),
                            single,
                            parameters.top_k,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(if single {
                    FillMaskOutput::One(fills.into_iter().next().unwrap_or_default())
                } else {
                    FillMaskOutput::Each(fills)
                })
            })
            .collect()
    }
}

impl FillMaskPipeline {
    // The ids of `targets`: each one's own id in the vocabulary, or the first id its text encodes to.
    fn target_ids(&self, targets: &[String]) -> Result<Vec<i64>> {
        let mut ids = Vec::new();
        for target in targets {
            let id = match self.tokenizer.token_id(target) {
                Ok(id) => Some(id),
                Err(_) => self.tokenizer.encode(target, false)?.first().copied(),
            };
            if let Some(id) = id.filter(|id| !ids.contains(id)) {
                ids.push(id);
            }
        }

        if ids.is_empty() {
            return Err(Error::invalid("fill-mask targets that encode to no token"));
        }

        Ok(ids)
    }

    // The likeliest `top_k` tokens for the mask at `mask` of a row of `ids`, among `targets` if given, with their probabilities over the whole vocabulary.
    fn fills(
        &self,
        ids: &[i64],
        mask: usize,
        scores: &[f32],
        targets: Option<&[i64]>,
        single: bool,
        top_k: usize,
    ) -> Result<Vec<Fill>> {
        let probabilities = ScoreFunction::Softmax.apply(scores);
        let mut candidates = match targets {
            Some(targets) => targets
                .iter()
                .map(|&id| {
                    let probability = usize::try_from(id)
                        .ok()
                        .and_then(|index| probabilities.get(index))
                        .ok_or_else(|| {
                            Error::unsupported(format!(
                                "target token {id} outside the model's vocabulary"
                            ))
                        })?;
                    Ok((id, *probability))
                })
                .collect::<Result<Vec<_>>>()?,
            None => (0_i64..)
                .zip(probabilities.iter().copied())
                .collect::<Vec<_>>(),
        };
        host::best_first(&mut candidates, top_k, |&(_, score)| score);
        candidates
            .into_iter()
            .map(|(token, score)| {
                let mut filled = ids.to_vec();
                filled[mask] = token;
                Ok(Fill {
                    score,
                    token,
                    token_str: self.tokenizer.decode(&[token], false)?,
                    sequence: self.tokenizer.decode(&filled, single)?,
                })
            })
            .collect()
    }
}
