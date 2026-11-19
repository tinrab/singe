"""Llama, through SmolLM2-135M, generating greedily after two prompts with `transformers`' `generate`, as `mod.rs` generates in Singe."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("story", text_generation)
verification.finish()
