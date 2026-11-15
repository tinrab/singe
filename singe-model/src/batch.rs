use singe::{
    Result,
    generation::{CachedDecoder, DecoderOptions, GenerationOptions},
    nn::{DecoderModel, Model},
};

#[derive(Debug, Clone, Default)]
pub struct BatchOptions {
    pub generation: GenerationOptions,
    pub rows: usize,
    pub compile: bool,
}

impl BatchOptions {
    #[must_use]
    pub const fn new(generation: GenerationOptions) -> Self {
        Self {
            generation,
            rows: 1,
            compile: false,
        }
    }

    #[must_use]
    pub const fn with_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }

    #[must_use]
    pub const fn with_compile(mut self, compile: bool) -> Self {
        self.compile = compile;
        self
    }

    pub fn decoder<L: DecoderModel>(
        &self,
        model: &Model<L>,
        prompts: impl Iterator<Item = usize>,
    ) -> Result<CachedDecoder<L>> {
        // Each row holds the longest prompt and the tokens generated after it.
        let longest = prompts.max().unwrap_or(0) + self.generation.max_new_tokens;
        let capacity =
            i64::try_from(longest).map_err(|_| singe::Error::out_of_range("batch capacity"))?;
        let mut decoder = CachedDecoder::create(model, DecoderOptions::new(self.rows, capacity))?;
        if self.compile {
            decoder.compile()?;
        }

        Ok(decoder)
    }
}
