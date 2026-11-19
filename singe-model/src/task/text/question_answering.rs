// `question-answering`: a question and a context to the spans of the context that answer it, as `transformers` 4's pipeline found them; `transformers` 5 removed the pipeline but keeps the heads.
// A context longer than a window is read in windows that overlap, and the same answer found in several windows sums its scores.

use serde::{Deserialize, Serialize};

use crate::{
    error::{Error, Result},
    head::{QuestionAnswerer, ScoreFunction},
    host, model,
    processor::text::{EncodeOptions, EncodedRow, TextInput, TextTokenizer, Truncation},
    source::{LoadOptions, Source},
    task::{Pipeline, run_alone},
};

// The longest window when a tokenizer does not limit it to less.
const MAX_WINDOW: usize = 384;

// What a token outside the context scores before the softmax, as `transformers` masks it.
const MASKED: f32 = -10_000.0;

#[derive(Debug)]
pub struct QuestionAnsweringPipeline {
    model: QuestionAnswerer,
    tokenizer: TextTokenizer,
    classifier: Option<i64>,
    options: LoadOptions,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct QuestionAnsweringInput {
    pub question: String,
    pub context: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct QuestionAnsweringParameters {
    pub top_k: usize,
    // The most tokens of an answer.
    pub max_answer_len: usize,
    // Also offers the empty answer, scored as the model scores the classifier token.
    pub handle_impossible_answer: bool,
    // The tokens of a window, question included; 384, or the tokenizer's limit if less, when absent.
    pub max_seq_len: Option<usize>,
    // The tokens each window repeats from the one before; half a window, at most 128, when absent.
    pub doc_stride: Option<usize>,
    // Widens an answer to the whole words its first and last tokens are in.
    pub align_to_words: bool,
}

impl Default for QuestionAnsweringParameters {
    fn default() -> Self {
        Self {
            top_k: 1,
            max_answer_len: 15,
            handle_impossible_answer: false,
            max_seq_len: None,
            doc_stride: None,
            align_to_words: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Answer {
    pub score: f32,
    pub start: usize,
    pub end: usize,
    pub answer: String,
}

// One answer, or several from the best when there are more, as `transformers` returns them.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum QuestionAnsweringOutput {
    One(Answer),
    Many(Vec<Answer>),
}

impl Pipeline for QuestionAnsweringPipeline {
    type Input = QuestionAnsweringInput;

    type Parameters = QuestionAnsweringParameters;

    type Output = QuestionAnsweringOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        let tokenizer = TextTokenizer::load(source)?;
        Ok(Self {
            model: model::load(source, options)?,
            classifier: tokenizer.classifier_id()?,
            tokenizer,
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
        if parameters.top_k == 0 {
            return Err(Error::invalid("question answering with `top_k` of zero"));
        }

        let window = parameters.max_seq_len.unwrap_or_else(|| {
            self.tokenizer
                .max_length()
                .map_or(MAX_WINDOW, |length| length.min(MAX_WINDOW))
        });
        let stride = parameters
            .doc_stride
            .unwrap_or_else(|| (window / 2).min(128));
        if stride > window {
            return Err(Error::invalid("a stride longer than its window"));
        }

        let pairs = inputs
            .iter()
            .map(|input| TextInput::Pair(&input.question, &input.context))
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            max_length: Some(window),
            truncation: Truncation::OnlySecond,
            stride: Some(stride),
        };
        let encoded = self
            .tokenizer
            .encode_inputs(&pairs, &options, self.options.device)?;
        let scores = self.model.scores(&encoded.batch)?;
        inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let windows = encoded
                    .rows
                    .iter()
                    .zip(&scores)
                    .filter(|(row, _)| row.input == index);
                Ok(self.answers(input, windows, parameters))
            })
            .collect()
    }
}

impl QuestionAnsweringPipeline {
    fn answers<'a>(
        &self,
        input: &QuestionAnsweringInput,
        windows: impl Iterator<Item = (&'a EncodedRow, &'a Vec<[f32; 2]>)>,
        parameters: &QuestionAnsweringParameters,
    ) -> QuestionAnsweringOutput {
        let context = input.context.chars().collect::<Vec<_>>();
        let mut null = f32::INFINITY;
        let mut answers = Vec::<Answer>::new();
        for (row, scores) in windows {
            null = null.min(self.window_answers(&context, row, scores, parameters, &mut answers));
        }

        if parameters.handle_impossible_answer {
            answers.push(Answer {
                score: null,
                start: 0,
                end: 0,
                answer: String::new(),
            });
        }

        host::best_first(&mut answers, parameters.top_k, |answer| answer.score);
        if answers.len() == 1 {
            QuestionAnsweringOutput::One(answers.remove(0))
        } else {
            QuestionAnsweringOutput::Many(answers)
        }
    }

    fn window_answers(
        &self,
        context: &[char],
        row: &EncodedRow,
        scores: &[[f32; 2]],
        parameters: &QuestionAnsweringParameters,
        answers: &mut Vec<Answer>,
    ) -> f32 {
        // A window's context tokens may answer, and so may its classifier token, whose scores the softmax counts.
        let answerable = (0..row.ids.len())
            .map(|token| row.sequences[token] == Some(1) || Some(row.ids[token]) == self.classifier)
            .collect::<Vec<_>>();
        let normalized = |side: usize| {
            let logits = (0..row.ids.len())
                .map(|token| {
                    if answerable[token] {
                        scores[token][side]
                    } else {
                        MASKED
                    }
                })
                .collect::<Vec<_>>();
            ScoreFunction::Softmax.apply(&logits)
        };
        let (mut starts, mut ends) = (normalized(0), normalized(1));
        let null = starts[0] * ends[0];
        starts[0] = 0.0;
        ends[0] = 0.0;
        for (start, end, score) in spans(&starts, &ends, parameters) {
            // The first token stays a candidate whatever it is, as `transformers`' filter lets it.
            if !(start == 0 || answerable[start]) || !(end == 0 || answerable[end]) {
                continue;
            }

            let (first, last) = if parameters.align_to_words {
                (word_chars(row, start).0, word_chars(row, end).1)
            } else {
                (row.offsets[start].0, row.offsets[end].1)
            };
            let text = context[first.min(last)..last].iter().collect::<String>();
            match answers
                .iter_mut()
                .find(|answer| answer.answer.to_lowercase() == text.to_lowercase())
            {
                Some(answer) => answer.score += score,
                None => answers.push(Answer {
                    score,
                    start: first,
                    end: last,
                    answer: text,
                }),
            }
        }
        null
    }
}

// The likeliest spans of at most `max_answer_len` tokens, best first, as many as `transformers` keeps before it drops those outside the context: twice `top_k` and ten more when it aligns to words.
fn spans(
    starts: &[f32],
    ends: &[f32],
    parameters: &QuestionAnsweringParameters,
) -> Vec<(usize, usize, f32)> {
    let kept = if parameters.align_to_words {
        parameters.top_k * 2 + 10
    } else {
        parameters.top_k
    };
    let mut candidates = Vec::new();
    for (start, &start_score) in starts.iter().enumerate() {
        let last = (start + parameters.max_answer_len).min(ends.len());
        for (end, &end_score) in ends.iter().enumerate().take(last).skip(start) {
            candidates.push((start, end, start_score * end_score));
        }
    }

    // Ties keep their order, start then end, as the first of equal scores comes first.
    host::best_first(&mut candidates, kept, |&(_, _, score)| score);
    candidates
}

// The characters of the context word token `token` is in, or the token's own characters for a token of no context word.
fn word_chars(row: &EncodedRow, token: usize) -> (usize, usize) {
    let Some(word) = row.words[token].filter(|_| row.sequences[token] == Some(1)) else {
        return row.offsets[token];
    };
    (0..row.ids.len())
        .filter(|&other| row.words[other] == Some(word) && row.sequences[other] == Some(1))
        .fold((usize::MAX, 0), |(first, last), other| {
            (
                first.min(row.offsets[other].0),
                last.max(row.offsets[other].1),
            )
        })
}
