"""Granite-4.0-H, through granite-4.0-h-350m, generating greedily after a short prompt and after one longer than the reference's scan chunk with `transformers`' `generate`, as `mod.rs` generates in Singe."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("long", text_generation)
verification.finish()
