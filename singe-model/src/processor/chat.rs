// Chat messages to the prompt a chat model was trained on, through its snapshot's Jinja chat template, as `transformers`' `apply_chat_template` renders it.

use std::{fs, result::Result as StdResult};

use minijinja::{
    Environment, Error as JinjaError, ErrorKind, Value, context, syntax::SyntaxConfig, value::Serde,
};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as JsonValue};

use crate::{
    error::{Error, Result},
    processor::conversion::{special_or, tokenizer_config},
    source::Source,
};

// One message of a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Content,
}

impl Message {
    #[must_use]
    pub fn new(role: &str, content: Content) -> Self {
        Self {
            role: role.to_owned(),
            content,
        }
    }

    #[must_use]
    pub fn user(text: &str) -> Self {
        Self::new("user", Content::Text(text.to_owned()))
    }
}

// What a message says: text alone, or parts of text, images, and audio, as `transformers`' multimodal messages hold them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

// A part of a message; an image or a clip names the file it is read from, which the template sees as a placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: String },
    Image { image: String },
    Audio { audio: String },
    // A clip: the directory of its decoded frames, and the frames per second they were decoded at.
    Video { video: String, fps: f64 },
}

impl Content {
    #[must_use]
    pub fn parts(&self) -> &[Part] {
        match self {
            Self::Text(_) => &[],
            Self::Parts(parts) => parts,
        }
    }
}

pub(crate) fn environment<'source>() -> Result<Environment<'source>> {
    let mut environment = Environment::new();
    environment.set_syntax(
        SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(Error::Template)?,
    );
    minijinja_contrib::add_to_environment(&mut environment);
    Ok(environment)
}

// A snapshot's chat template, with the special tokens it may write.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    // Reads `chat_template.jinja`, or `tokenizer_config.json`'s `chat_template`, choosing the template named `default` among several; a snapshot without one has none.
    pub fn load(source: &Source) -> Result<Option<Self>> {
        let config = tokenizer_config(source)?;
        let path = source.directory().join("chat_template.jinja");
        let template = if path.exists() {
            Some(fs::read_to_string(&path).map_err(Error::io(&path))?)
        } else {
            match &config["chat_template"] {
                JsonValue::String(template) => Some(template.clone()),
                JsonValue::Array(templates) => templates
                    .iter()
                    .find(|template| template["name"] == "default")
                    .and_then(|template| template["template"].as_str())
                    .map(str::to_owned),
                _ => None,
            }
        };
        Ok(template.map(|template| Self {
            template,
            bos_token: special_or(&config, "bos_token", ""),
            eos_token: special_or(&config, "eos_token", ""),
        }))
    }

    // The same special tokens under another template, as a processor that renders some conversations through its own uses them.
    #[must_use]
    pub fn with_template(&self, template: &str) -> Self {
        Self {
            template: template.to_owned(),
            ..self.clone()
        }
    }

    // Renders `messages`, ending with the start of the assistant's turn when `generation_prompt` is set, with `arguments` such as `enable_thinking` passed to the template.
    pub fn render(
        &self,
        messages: &[Message],
        generation_prompt: bool,
        arguments: &Map<String, JsonValue>,
    ) -> Result<String> {
        // `transformers` renders with these settings and Python's string methods.
        let mut environment = environment()?;
        environment.set_unknown_method_callback(unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> StdResult<Value, JinjaError> {
                Err(JinjaError::new(ErrorKind::InvalidOperation, message))
            },
        );
        let template = environment
            .template_from_str(&self.template)
            .map_err(Error::Template)?;
        let base = context! {
            messages => Value::from(Serde(messages)),
            add_generation_prompt => generation_prompt,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
        };
        let arguments = Value::from(Serde(arguments));
        template
            .render(context! { ..arguments, ..base })
            .map_err(Error::Template)
    }
}
