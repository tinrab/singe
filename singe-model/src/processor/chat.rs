//! Chat messages rendered to the prompt a chat model was trained on, through its snapshot's Jinja chat template, as `transformers`' `apply_chat_template` renders it.

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

/// One message of a conversation: its role, such as `user` or `assistant`, and its content.
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

/// What a message says: text alone, or parts of text, images, audio, and video, as `transformers`' multimodal messages hold them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

/// A part of a message: text, or an image, audio, or video part that carries its data, for which a chat template writes the model's placeholder.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text {
        text: String,
    },
    /// An image, as a `data:` URL or bare base64 of its file.
    Image {
        image: String,
    },
    Audio {
        audio: String,
    },
    /// A clip: its decoded frames, each as an image part carries one, and the frames per second they were decoded at.
    Video {
        video: Vec<String>,
        fps: f64,
    },
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

/// A snapshot's chat template, with the special tokens it may write.
#[derive(Debug, Clone)]
pub struct ChatTemplate {
    template: String,
    bos_token: String,
    eos_token: String,
}

impl ChatTemplate {
    /// Loads the chat template from `chat_template.jinja`, or from `tokenizer_config.json`'s `chat_template`, choosing the template named `default` among several.
    /// Returns `None` for a snapshot without one.
    ///
    /// # Errors
    ///
    /// Returns an error if `tokenizer_config.json` or `chat_template.jinja` is there but cannot be read or parsed.
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

    /// Returns these special tokens under `template` instead, for a processor that renders some conversations through a template of its own.
    #[must_use]
    pub fn with_template(&self, template: &str) -> Self {
        Self {
            template: template.to_owned(),
            ..self.clone()
        }
    }

    /// Renders `messages`, ending with the start of the assistant's turn when `generation_prompt` is set, with `arguments` such as `enable_thinking` passed to the template.
    ///
    /// # Errors
    ///
    /// Returns an error if the template does not parse, calls `raise_exception`, or fails to render.
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
