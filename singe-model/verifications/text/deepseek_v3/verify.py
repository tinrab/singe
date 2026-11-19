"""DeepSeek-V3, through a tiny random checkpoint of a dense layer and a mixture-of-experts layer (256 experts in 8 groups, multi-head latent attention, and the stored multi-token-prediction layer the import skips), generating greedily after two prompts with `transformers`' `generate`, as `mod.rs` generates in Singe.
Random weights test the architecture whole; DeepSeek-V3's trained weights need FP8 loading."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("story", text_generation)
verification.finish()
