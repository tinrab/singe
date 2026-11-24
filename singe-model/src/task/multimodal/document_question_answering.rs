//! `document-question-answering`: an image of a document and a question to the answer's text, here from Donut, which reads the image without OCR and writes its answer after a prompt naming the question, as `transformers`' pipeline asks it.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::{Seq2SeqInput, Seq2SeqLm},
    model,
    processor::{
        image::{Image, ImageTransforms},
        text::TextTokenizer,
    },
    source::{LoadOptions, Source},
    task::{Pipeline, seq2seq::Seq2SeqParameters},
};

#[derive(Debug)]
pub struct DocumentQuestionAnsweringPipeline {
    model: Seq2SeqLm,
    processor: ImageTransforms,
    tokenizer: TextTokenizer,
    options: LoadOptions,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentQuestion {
    pub image: Image,
    pub question: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DocumentAnswer {
    pub answer: Option<String>,
}

/// Returns the answer in a generated sequence: after its first tag, the text between `<s_answer>` and the last `</s_answer>` on its line, as the pipeline's `<s_answer>(.*)</s_answer>` finds it.
fn answer(sequence: &str) -> Option<String> {
    // The first tag, `<.*?>`, which ends before any line break.
    let mut text = sequence.to_owned();
    let mut from = 0;
    while let Some(start) = text[from..].find('<').map(|at| at + from) {
        let rest = &text[start..];
        let end = rest.find('>');
        let line = rest.find('\n');
        match end {
            Some(end) if line.is_none_or(|line| end < line) => {
                text.replace_range(start..=start + end, "");
                break;
            }
            _ => from = start + 1,
        }
    }

    let text = text.trim();
    let mut search = 0;
    while let Some(start) = text[search..].find("<s_answer>").map(|at| at + search) {
        let rest = &text[start + "<s_answer>".len()..];
        let line = rest.split('\n').next().unwrap_or_default();
        if let Some(end) = line.rfind("</s_answer>") {
            return Some(line[..end].trim().to_owned());
        }

        search = start + 1;
    }
    None
}

impl Pipeline for DocumentQuestionAnsweringPipeline {
    type Input = DocumentQuestion;

    type Parameters = Seq2SeqParameters;

    type Output = Vec<DocumentAnswer>;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            processor: ImageTransforms::load(source)?,
            tokenizer: TextTokenizer::load(source)?,
            options,
        })
    }

    fn run_one(&self, input: &Self::Input, parameters: &Self::Parameters) -> Result<Self::Output> {
        let pixels =
            self.processor
                .preprocess(&input.image, self.options.device, self.options.data_type)?;
        let encoded = self.model.encode(Seq2SeqInput::Pixels(&pixels))?;
        let prompt = format!(
            "<s_docvqa><s_question>{}</s_question><s_answer>",
            input.question
        );
        let mut tokens = self.tokenizer.encode(&prompt, false)?;
        tokens.extend(self.model.generate(
            &encoded,
            &tokens,
            parameters.max_new_tokens,
            parameters.compile,
        )?);
        // The sequence with its special tokens, but its end and padding tokens.
        let sequence = self
            .tokenizer
            .decode(&tokens, false)?
            .replace("</s>", "")
            .replace("<pad>", "");
        Ok(vec![DocumentAnswer {
            answer: answer(&sequence),
        }])
    }
}
