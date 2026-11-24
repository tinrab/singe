//! `decision`: a state and typed questions about it to a probability for each option, from one forward pass of a language model per question, as `transformers`' decision pipeline asks a model without a decision head: the generic prompt in the model's chat template, and the logits of the options' labels after it.

use std::result::Result as StdResult;

use minijinja::{Value as JinjaValue, context, value::Serde};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Map, Value};
use singe::{DataType, Error as SingeError, generation::DecoderOptions};

use crate::{
    error::{Error, Result},
    head::{CausalLm, ScoreFunction, argmax},
    model,
    processor::{
        chat::{ChatTemplate, Message, environment},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::Pipeline,
};

/// The prompt of `transformers`' `DECISION_PROMPT`: the state, the instructions, each option's label and description, and how to answer.
const DECISION_PROMPT: &str = r#"
{%- macro text(x) %}{{ "" if x is none else x if x is string else x|tojson }}{% endmacro -%}
{%- set ns = namespace(lines=[text(question.instructions)]) -%}
{%- for option in question.options -%}
    {%- set label, answer = question.labels[loop.index0], text(question.descriptions[loop.index0]) -%}
    {%- if question.type == "noul" -%}
        {%- set answer = answer or (option if question.labels != ["Yes", "No"] else "") -%}
        {%- set ns.lines = ns.lines + ([label ~ ": " ~ answer] if answer else []) -%}
    {%- else -%}
        {%- set answer = (option ~ ": " ~ answer if answer else option) if question.type == "choice" else answer -%}
        {%- set ns.lines = ns.lines + [label ~ ". " ~ answer] -%}
    {%- endif -%}
{%- endfor -%}
{{- text(state) ~ "\n\n" ~ ns.lines|select|join("\n") ~ "\n" -}}
{%- if question.type == "noul" -%}
    {{- "Answer " ~ question.labels[0] ~ " or " ~ question.labels[1] ~ " only." -}}
{%- else -%}
    {%- set kind = "letter" if question.labels == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: question.labels|length]|list else "label" -%}
    {{- "Answer with the " ~ kind ~ " of the " ~ ("option" if question.type == "choice" else "level") -}}
    {{- " that fits best (" ~ question.labels[0] ~ " to " ~ question.labels[-1] ~ ") only." -}}
{%- endif -%}"#;

#[derive(Debug)]
pub struct DecisionPipeline {
    model: Box<dyn CausalLm>,
    tokenizer: TextTokenizer,
    template: ChatTemplate,
}

/// A request: the state questions ask about, and each question by its name.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum QuestionKind {
    Noul,
    Choice,
    Score,
}

/// A question as the prompt reads it: its kind, its instructions, and its options with their labels and descriptions.
#[derive(Debug, Serialize)]
struct Question {
    name: String,
    #[serde(rename = "type")]
    kind: QuestionKind,
    instructions: Value,
    options: Vec<String>,
    labels: Vec<String>,
    descriptions: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        confidence: f32,
        probabilities: Map<String, Value>,
    },
    Score {
        score: f32,
        legend: Map<String, Value>,
        confidence: f32,
        probabilities: Map<String, Value>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Answers(pub Vec<(String, Answer)>);

impl Serialize for Answers {
    fn serialize<S: Serializer>(&self, serializer: S) -> StdResult<S::Ok, S::Error> {
        serializer.collect_map(self.0.iter().map(|(name, answer)| (name, answer)))
    }
}

impl Question {
    /// Reads the request's question `name`, as `Question.from_request` reads it with the default decision config.
    fn parse(name: &str, question: &Value) -> Result<Self> {
        let invalid = |what: &str| Error::invalid(format!("question {name}: {what}"));
        let kind = match question["type"].as_str() {
            Some("noul") => QuestionKind::Noul,
            Some("choice") => QuestionKind::Choice,
            Some("score") => QuestionKind::Score,
            Some(other) => return Err(invalid(&format!("type {other}"))),
            None => return Err(invalid("no type")),
        };
        let criteria = &question["criteria"];
        let (options, descriptions): (Vec<String>, Vec<Value>) = match kind {
            QuestionKind::Noul => ["true", "false"]
                .iter()
                .map(|&option| {
                    (
                        option.to_owned(),
                        criteria.get(option).cloned().unwrap_or(Value::Null),
                    )
                })
                .unzip(),
            QuestionKind::Choice => criteria
                .as_object()
                .filter(|criteria| !criteria.is_empty())
                .ok_or_else(|| invalid("a choice needs criteria"))?
                .iter()
                .map(|(option, description)| (option.clone(), description.clone()))
                .unzip(),
            QuestionKind::Score => criteria
                .as_array()
                .filter(|criteria| !criteria.is_empty())
                .ok_or_else(|| invalid("a score needs criteria"))?
                .iter()
                .enumerate()
                .map(|(level, description)| (level.to_string(), description.clone()))
                .unzip(),
        };
        let labels = if kind == QuestionKind::Noul {
            vec!["Yes".to_owned(), "No".to_owned()]
        } else {
            ('A'..='Z').map(String::from).collect()
        };
        if labels.len() < options.len() {
            return Err(invalid("more options than labels"));
        }
        Ok(Self {
            name: name.to_owned(),
            labels: labels[..options.len()].to_vec(),
            kind,
            instructions: question.get("instructions").cloned().unwrap_or(Value::Null),
            options,
            descriptions,
        })
    }

    fn prompt(&self, state: &Value) -> Result<String> {
        let environment = environment()?;
        let template = environment
            .template_from_str(DECISION_PROMPT)
            .map_err(Error::Template)?;
        template
            .render(context! {
                state => JinjaValue::from(Serde(state)),
                question => JinjaValue::from(Serde(self)),
            })
            .map_err(Error::Template)
    }

    /// Returns the answer from each option's probability, as `Question.answer` writes it.
    fn answer(&self, probabilities: &[f32]) -> Answer {
        let by_option = self
            .options
            .iter()
            .zip(probabilities)
            .map(|(option, &probability)| (option.clone(), Value::from(probability)))
            .collect::<Map<_, _>>();
        let count = probabilities.len() as f32;
        let best = argmax(probabilities.iter().copied().enumerate()).map_or(0, |(index, _)| index);
        match self.kind {
            QuestionKind::Noul => Answer::Noul {
                noul: probabilities[0],
            },
            QuestionKind::Choice => Answer::Choice {
                choice: self.options[best].clone(),
                confidence: if probabilities.len() == 1 {
                    1.0
                } else {
                    ((probabilities[best] - 1.0 / count) / (1.0 - 1.0 / count)).max(0.0)
                },
                probabilities: by_option,
            },
            QuestionKind::Score => {
                let distance = probabilities
                    .iter()
                    .enumerate()
                    .map(|(level, probability)| probability * (level as f32 - best as f32).abs())
                    .sum::<f32>();
                let uniform = (0..probabilities.len())
                    .map(|level| (level as f32 - (count - 1.0) / 2.0).abs())
                    .sum::<f32>()
                    / count;
                Answer::Score {
                    score: probabilities
                        .iter()
                        .enumerate()
                        .map(|(level, probability)| level as f32 * probability)
                        .sum::<f32>(),
                    legend: self
                        .options
                        .iter()
                        .zip(&self.descriptions)
                        .map(|(option, description)| (option.clone(), description.clone()))
                        .collect(),
                    confidence: if probabilities.len() == 1 {
                        1.0
                    } else {
                        (1.0 - distance / uniform).max(0.0)
                    },
                    probabilities: by_option,
                }
            }
        }
    }
}

impl DecisionPipeline {
    fn ask(
        &self,
        state: &Value,
        question: &Question,
        arguments: &Map<String, Value>,
    ) -> Result<Answer> {
        let text = question.prompt(state)?;
        let rendered = self
            .template
            .render(&[Message::user(&text)], true, arguments)?;
        let tokens = self.tokenizer.encode(&rendered, false)?;
        let labels = question
            .labels
            .iter()
            .map(
                |label| match self.tokenizer.encode(label, false)?.as_slice() {
                    [token] => Ok(*token),
                    _ => Err(Error::unsupported(format!(
                        "label {label} is not one token"
                    ))),
                },
            )
            .collect::<Result<Vec<_>>>()?;
        let capacity = i64::try_from(tokens.len() + 1)
            .map_err(|_| SingeError::out_of_range("prompt tokens"))?;
        let mut decoder = self
            .model
            .decoder(DecoderOptions::new(1, capacity), false)?;
        let logits = decoder
            .step(&[&tokens], None)?
            .cast(DataType::F32)?
            .to_vec::<f32>()?;
        let scores = labels
            .iter()
            .map(|&label| {
                usize::try_from(label)
                    .ok()
                    .and_then(|label| logits.get(label).copied())
                    .ok_or_else(|| Error::unsupported(format!("label token {label}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(question.answer(&ScoreFunction::Softmax.apply(&scores)))
    }
}

impl Pipeline for DecisionPipeline {
    type Input = DecisionRequest;

    type Parameters = ();

    type Output = Answers;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source)?,
            template: ChatTemplate::load(source)?
                .ok_or_else(|| Error::unsupported("a decision model without a chat template"))?,
        })
    }

    fn run_one(&self, request: &Self::Input, (): &Self::Parameters) -> Result<Self::Output> {
        let mut arguments = Map::new();
        arguments.insert("enable_thinking".to_owned(), Value::Bool(false));
        request
            .questions
            .iter()
            .map(|(name, question)| {
                let question = Question::parse(name, question)?;
                let answer = self.ask(&request.state, &question, &arguments)?;
                Ok((question.name, answer))
            })
            .collect::<Result<Vec<_>>>()
            .map(Answers)
    }
}
