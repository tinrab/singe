//! `token-classification`: text to its tokens' labels, or to the entities grouped from them, as `transformers`' pipeline finds them.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{ScoreFunction, TokenClassifier, argmax},
    model,
    processor::text::{EncodeOptions, EncodedRow, TextInput, TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, run_alone},
};

#[derive(Debug)]
pub struct TokenClassificationPipeline {
    model: TokenClassifier,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

/// How tokens become entities: each token alone (`none`), adjacent tokens of one entity grouped (`simple`), or, for `first`, `average`, and `max`, each word's tokens first agreed on one label: its first token's, the average of their scores, or its most confident token's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AggregationStrategy {
    #[default]
    None,
    Simple,
    First,
    Average,
    Max,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Deserialize)]
#[serde(default)]
pub struct TokenClassificationParameters {
    pub aggregation_strategy: AggregationStrategy,
    /// The labels left out of the output, `["O"]` when `None`.
    pub ignore_labels: Option<Vec<String>>,
    /// The tokens each text is truncated to, keeping its special tokens, with no truncation for `None`.
    pub max_length: Option<usize>,
}

/// A token and its label, or a group of tokens and the entity they share, with its characters in the text.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Entity {
    Token {
        entity: String,
        score: f32,
        index: usize,
        word: String,
        start: usize,
        end: usize,
    },
    Group {
        entity_group: String,
        score: f32,
        word: String,
        start: usize,
        end: usize,
    },
}

/// A token before aggregation: its text, its labels' probabilities, and whether it continues the word before it.
#[derive(Debug, Clone)]
struct Piece {
    word: String,
    scores: Vec<f32>,
    start: usize,
    end: usize,
    index: usize,
    subword: bool,
}

/// A token or a word with the label it takes.
#[derive(Debug, Clone)]
struct Labeled {
    entity: String,
    score: f32,
    index: Option<usize>,
    word: String,
    start: usize,
    end: usize,
}

impl Pipeline for TokenClassificationPipeline {
    type Input = String;

    type Parameters = TokenClassificationParameters;

    type Output = Vec<Entity>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load_served(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
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
        let texts = inputs
            .iter()
            .map(|text| TextInput::Single(text))
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            max_length: parameters.max_length,
            ..EncodeOptions::default()
        };
        let encoded = self
            .tokenizer
            .encode_inputs(&texts, &options, self.options.device)?;
        let scores = self.model.scores(&encoded.batch)?;
        let ignored = parameters
            .ignore_labels
            .clone()
            .unwrap_or_else(|| vec!["O".to_owned()]);
        encoded
            .rows
            .iter()
            .zip(&scores)
            .map(|(row, scores)| {
                let pieces = self.pieces(&inputs[row.input], row, scores);
                let entities = self.entities(pieces, parameters.aggregation_strategy)?;
                Ok(entities
                    .into_iter()
                    .filter(|entity| {
                        let label = match entity {
                            Entity::Token { entity, .. } => entity,
                            Entity::Group { entity_group, .. } => entity_group,
                        };
                        !ignored.contains(label)
                    })
                    .collect())
            })
            .collect()
    }
}

impl TokenClassificationPipeline {
    /// Returns a text's tokens but its special ones, each with its labels' probabilities.
    /// A token continues a word when its text differs from its characters in the text, as `WordPiece`'s `##` piece does, or, for a tokenizer that marks no pieces, when no space comes before it.
    fn pieces(&self, text: &str, row: &EncodedRow, scores: &[Vec<f32>]) -> Vec<Piece> {
        let characters = text.chars().collect::<Vec<_>>();
        let marked = self.tokenizer.subword_prefix().is_some();
        let unknown = self.tokenizer.unknown_id();
        row.ids
            .iter()
            .enumerate()
            .filter(|&(index, _)| row.sequences[index].is_some())
            .map(|(index, &id)| {
                let (start, end) = row.offsets[index];
                let reference = characters[start.min(end)..end].iter().collect::<String>();
                let token = self.tokenizer.token(id).unwrap_or_default();
                let (word, subword) = if Some(id) == unknown {
                    (reference, false)
                } else if marked {
                    let subword = token.chars().count() != reference.chars().count();
                    (token, subword)
                } else {
                    let subword = start > 0
                        && !characters[start - 1..=start.min(characters.len() - 1)].contains(&' ');
                    (token, subword)
                };
                Piece {
                    word,
                    scores: ScoreFunction::Softmax.apply(&scores[index]),
                    start,
                    end,
                    index,
                    subword,
                }
            })
            .collect()
    }

    fn entities(&self, pieces: Vec<Piece>, strategy: AggregationStrategy) -> Result<Vec<Entity>> {
        let labeled = match strategy {
            AggregationStrategy::None | AggregationStrategy::Simple => pieces
                .into_iter()
                .map(|piece| {
                    let (label, score) = best(&piece.scores)?;
                    Ok(Labeled {
                        entity: self.label(label),
                        score,
                        index: Some(piece.index),
                        word: piece.word,
                        start: piece.start,
                        end: piece.end,
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            AggregationStrategy::First
            | AggregationStrategy::Average
            | AggregationStrategy::Max => self.words(pieces, strategy)?,
        };
        if strategy == AggregationStrategy::None {
            return Ok(labeled
                .into_iter()
                .map(|token| Entity::Token {
                    entity: token.entity,
                    score: token.score,
                    index: token.index.unwrap_or_default(),
                    word: token.word,
                    start: token.start,
                    end: token.end,
                })
                .collect());
        }

        self.groups(labeled)
    }

    fn label(&self, index: usize) -> String {
        self.model.config().labels()[index].clone()
    }

    /// Gives each word one label, from its pieces' scores as `strategy` combines them.
    fn words(&self, pieces: Vec<Piece>, strategy: AggregationStrategy) -> Result<Vec<Labeled>> {
        let mut words = Vec::<Vec<Piece>>::new();
        for piece in pieces {
            match words.last_mut() {
                Some(word) if piece.subword => word.push(piece),
                _ => words.push(vec![piece]),
            }
        }

        words
            .into_iter()
            .map(|pieces| {
                let scores = match strategy {
                    // The first of the most confident pieces, as Python's `max` keeps.
                    AggregationStrategy::Max => {
                        let top = |scores: &[f32]| {
                            argmax(scores.iter().copied().enumerate()).map(|(_, score)| score)
                        };
                        pieces
                            .iter()
                            .map(|piece| &piece.scores)
                            .fold(None::<&Vec<f32>>, |kept, scores| match kept {
                                Some(kept) if top(kept) >= top(scores) => Some(kept),
                                _ => Some(scores),
                            })
                            .cloned()
                            .unwrap_or_default()
                    }
                    AggregationStrategy::Average => {
                        let count = pieces.len() as f32;
                        (0..pieces[0].scores.len())
                            .map(|label| {
                                pieces.iter().map(|piece| piece.scores[label]).sum::<f32>() / count
                            })
                            .collect()
                    }
                    _ => pieces[0].scores.clone(),
                };
                let (label, score) = best(&scores)?;
                Ok(Labeled {
                    entity: self.label(label),
                    score,
                    index: None,
                    word: self
                        .tokenizer
                        .join_tokens(pieces.iter().map(|piece| piece.word.clone()).collect())?,
                    start: pieces[0].start,
                    end: pieces[pieces.len() - 1].end,
                })
            })
            .collect()
    }

    /// Groups adjacent tokens or words of one entity, which a `B-` label begins and `I-` labels continue.
    fn groups(&self, labeled: Vec<Labeled>) -> Result<Vec<Entity>> {
        let mut groups = Vec::<Vec<Labeled>>::new();
        for item in labeled {
            match groups.last_mut() {
                Some(group) => {
                    let (begins, tag) = tag(&item.entity);
                    let (_, last) = tag_of(group);
                    if tag == last && !begins {
                        group.push(item);
                    } else {
                        groups.push(vec![item]);
                    }
                }
                None => groups.push(vec![item]),
            }
        }

        groups
            .into_iter()
            .map(|group| {
                let entity_group = group[0]
                    .entity
                    .split_once('-')
                    .map_or(group[0].entity.as_str(), |(_, tag)| tag)
                    .to_owned();
                let score = group.iter().map(|item| item.score).sum::<f32>() / group.len() as f32;
                Ok(Entity::Group {
                    entity_group,
                    score,
                    word: self
                        .tokenizer
                        .join_tokens(group.iter().map(|item| item.word.clone()).collect())?,
                    start: group[0].start,
                    end: group[group.len() - 1].end,
                })
            })
            .collect()
    }
}

/// Returns the best label of `scores` and its score.
fn best(scores: &[f32]) -> Result<(usize, f32)> {
    argmax(scores.iter().copied().enumerate())
        .ok_or_else(|| Error::unsupported("a classifier of no labels"))
}

/// Returns whether a label begins an entity, and the entity's tag: `B-PER` begins `PER`, `I-PER` continues it, and a label without a prefix continues its own tag.
fn tag(label: &str) -> (bool, &str) {
    label
        .strip_prefix("B-")
        .map(|tag| (true, tag))
        .or_else(|| label.strip_prefix("I-").map(|tag| (false, tag)))
        .unwrap_or((false, label))
}

fn tag_of(group: &[Labeled]) -> (bool, &str) {
    group.last().map_or((false, ""), |last| tag(&last.entity))
}
