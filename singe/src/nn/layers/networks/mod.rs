// Whole networks composed of blocks, such as a causal language model.

mod causal_lm;

pub use self::causal_lm::{CausalLm, CausalLmParts, Decoder};
