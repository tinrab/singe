"""Gemma 2, through a tiny random checkpoint whose one layer slides its window by the default layout and caps its scores, generating greedily after two prompts with `transformers`' `generate`, as `mod.rs` generates in Singe.
Random weights test the architecture whole; Gemma 2's smallest trained snapshot (2.6B) does not fit 8 GB in F32."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("story", text_generation)
verification.finish()
