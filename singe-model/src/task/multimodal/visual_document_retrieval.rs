// `visual-document-retrieval`: text queries and images of document pages to each page's score for each query, by late interaction of their tokens' embeddings, as `sentence-transformers`' multi-vector encoders score them; `transformers` has no pipeline for it.

use serde::{Deserialize, Serialize};

use crate::{
    error::Result,
    head::{DocumentRetriever, late_interaction},
    model,
    processing::{image::Image, text::TextTokenizer},
    source::{LoadOptions, Source},
    task::{Pipeline, Task},
};

#[derive(Debug)]
pub struct VisualDocumentRetrievalPipeline {
    model: Box<dyn DocumentRetriever>,
    tokenizer: TextTokenizer,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RetrievalInput {
    pub queries: Vec<String>,
    pub documents: Vec<Image>,
}

// Each query's score for each document, in the inputs' order.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RetrievalOutput {
    pub scores: Vec<Vec<f32>>,
}

impl Pipeline for VisualDocumentRetrievalPipeline {
    const TASK: Task = Task::VisualDocumentRetrieval;

    type Input = RetrievalInput;

    type Parameters = ();

    type Output = RetrievalOutput;

    fn load(source: &Source, options: LoadOptions) -> Result<Self> {
        Ok(Self {
            model: model::load(source, options)?,
            tokenizer: TextTokenizer::load(source.directory())?,
        })
    }

    fn run(&self, inputs: &[Self::Input], (): &Self::Parameters) -> Result<Vec<Self::Output>> {
        inputs
            .iter()
            .map(|input| {
                let documents = input
                    .documents
                    .iter()
                    .map(|image| self.model.document(&self.tokenizer, image))
                    .collect::<Result<Vec<_>>>()?;
                let scores = input
                    .queries
                    .iter()
                    .map(|query| {
                        let query = self.model.query(&self.tokenizer, query)?;
                        documents
                            .iter()
                            .map(|document| late_interaction(&query, document))
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(RetrievalOutput { scores })
            })
            .collect()
    }
}
