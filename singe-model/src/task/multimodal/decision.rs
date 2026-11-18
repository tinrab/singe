// `decision`: a state and typed questions about it to a probability for each option, from one forward pass of a language model per question, as `transformers`' decision pipeline asks a model without a decision head: the generic prompt in the model's chat template, and the logits of the options' labels after it.

use minijinja::syntax::SyntaxConfig;
use minijinja::value::Serde;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use singe::{DataType, generation::DecoderOptions};

use crate::{
    error::{Error, Result},
    head::{CausalLm, ScoreFunction, argmax},
    model,
    processing::{
        chat::{ChatTemplate, Message},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

// `transformers`' `DECISION_PROMPT`: the state, the instructions, each option's label and description, and how to answer.
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

// A request: the state questions ask about, and each question by its name.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionRequest {
    pub state: Value,
    pub questions: Map<String, Value>,
}

// A question as the prompt reads it: its kind, its instructions, and its options with their labels and descriptions.
struct Question {
    name: String,
    kind: String,
    instructions: Value,
    options: Vec<String>,
    labels: Vec<String>,
    descriptions: Vec<Value>,
}

impl Question {
    // A question of the request, as `Question.from_request` reads it with the default decision config.
    fn parse(name: &str, question: &Value) -> Result<Self> {
        let invalid = |what: &str| Error::UnsupportedConfig(format!("question {name}: {what}"));
        let kind = question["type"]
            .as_str()
            .ok_or_else(|| invalid("no type"))?
            .to_owned();
        let criteria = &question["criteria"];
        let (options, descriptions): (Vec<String>, Vec<Value>) = match kind.as_str() {
            "noul" => ["true", "false"]
                .iter()
                .map(|&option| {
                    (
                        (*option).to_owned(),
                        criteria.get(option).cloned().unwrap_or(Value::Null),
                    )
                })
                .unzip(),
            "choice" => criteria
                .as_object()
                .filter(|criteria| !criteria.is_empty())
                .ok_or_else(|| invalid("a choice needs criteria"))?
                .iter()
                .map(|(option, description)| (option.clone(), description.clone()))
                .unzip(),
            "score" => criteria
                .as_array()
                .filter(|criteria| !criteria.is_empty())
                .ok_or_else(|| invalid("a score needs criteria"))?
                .iter()
                .enumerate()
                .map(|(level, description)| (level.to_string(), description.clone()))
                .unzip(),
            other => return Err(invalid(&format!("type {other}"))),
        };
        let labels = if kind == "noul" {
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

    // The answer from each option's probability, as `Question.answer` writes it.
    #[expect(clippy::cast_precision_loss, reason = "option counts are small")]
    fn answer(&self, probabilities: &[f32]) -> Value {
        let by_option = self
            .options
            .iter()
            .zip(probabilities)
            .map(|(option, &probability)| (option.clone(), json!(probability)))
            .collect::<Map<_, _>>();
        let count = probabilities.len() as f32;
        if self.kind == "noul" {
            return json!({"type": "noul", "noul": probabilities[0]});
        }

        let best = argmax(probabilities).map_or(0, |(index, _)| index);
        if self.kind == "choice" {
            let confidence = if probabilities.len() == 1 {
                1.0
            } else {
                ((probabilities[best] - 1.0 / count) / (1.0 - 1.0 / count)).max(0.0)
            };
            return json!({"type": "choice", "choice": self.options[best], "confidence": confidence, "probabilities": by_option});
        }

        let distance = probabilities
            .iter()
            .enumerate()
            .map(|(level, probability)| probability * (level as f32 - best as f32).abs())
            .sum::<f32>();
        let uniform = (0..probabilities.len())
            .map(|level| (level as f32 - (count - 1.0) / 2.0).abs())
            .sum::<f32>()
            / count;
        let confidence = if probabilities.len() == 1 {
            1.0
        } else {
            (1.0 - distance / uniform).max(0.0)
        };
        let score = probabilities
            .iter()
            .enumerate()
            .map(|(level, probability)| level as f32 * probability)
            .sum::<f32>();
        let legend = self
            .options
            .iter()
            .zip(&self.descriptions)
            .map(|(option, description)| (option.clone(), description.clone()))
            .collect::<Map<_, _>>();
        json!({"type": "score", "score": score, "legend": legend, "confidence": confidence, "probabilities": by_option})
    }
}

impl DecisionPipeline {
    // The generic prompt for `question` about `state`.
    fn prompt(state: &Value, question: &Question) -> Result<String> {
        let mut environment = minijinja::Environment::new();
        environment.set_syntax(
            SyntaxConfig::builder()
                .trim_blocks(true)
                .lstrip_blocks(true)
                .build()
                .map_err(Error::Template)?,
        );
        minijinja_contrib::add_to_environment(&mut environment);
        let template = environment
            .template_from_str(DECISION_PROMPT)
            .map_err(Error::Template)?;
        template
            .render(minijinja::context! {
                state => minijinja::Value::from(Serde(state)),
                question => minijinja::Value::from(Serde(json!({
                    "name": question.name,
                    "type": question.kind,
                    "instructions": question.instructions,
                    "options": question.options,
                    "labels": question.labels,
                    "descriptions": question.descriptions,
                }))),
            })
            .map_err(Error::Template)
    }
}

impl Pipeline for DecisionPipeline {
    const TASK: Task = Task::Decision;

    type Input = DecisionRequest;

    type Parameters = ();

    type Output = Map<String, Value>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
            template: ChatTemplate::load(source)?
                .ok_or_else(|| Error::unsupported("a decision model without a chat template"))?,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        let mut arguments = Map::new();
        arguments.insert("enable_thinking".to_owned(), Value::Bool(false));
        inputs
            .iter()
            .map(|request| {
                let mut answers = Map::new();
                for (name, question) in &request.questions {
                    let question = Question::parse(name, question)?;
                    let text = Self::prompt(&request.state, &question)?;
                    let rendered =
                        self.template
                            .render(&[Message::user(&text)], true, &arguments)?;
                    let tokens = self.tokenizer.encode(&rendered, false)?;
                    let labels = question
                        .labels
                        .iter()
                        .map(
                            |label| match self.tokenizer.encode(label, false)?.as_slice() {
                                [token] => Ok(*token),
                                _ => Err(Error::UnsupportedConfig(format!(
                                    "label {label} is not one token"
                                ))),
                            },
                        )
                        .collect::<Result<Vec<_>>>()?;
                    let capacity = i64::try_from(tokens.len() + 1)
                        .map_err(|_| Error::unsupported("a prompt this long"))?;
                    let mut decoder = self
                        .model
                        .decoder(DecoderOptions::new(1, capacity), false)?;
                    let logits = decoder
                        .step(&[&tokens], None)?
                        .cast(DataType::F32)?
                        .to_vec::<f32>()?;
                    let vocabulary = logits.len();
                    let scores = labels
                        .iter()
                        .map(|&label| {
                            usize::try_from(label)
                                .ok()
                                .and_then(|label| (label < vocabulary).then(|| logits[label]))
                                .ok_or_else(|| {
                                    Error::UnsupportedConfig(format!("label token {label}"))
                                })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    answers.insert(
                        question.name.clone(),
                        question.answer(&ScoreFunction::Softmax.apply(&scores)),
                    );
                }
                Ok(answers)
            })
            .collect()
    }
}
