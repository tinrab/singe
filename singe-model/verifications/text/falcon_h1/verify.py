"""Falcon-H1, generating greedily through Falcon-H1-Tiny-90M after a short prompt and through Falcon-H1-0.5B after one longer than the reference's scan chunk with `transformers`' `generate`, as `mod.rs` generates in Singe."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("tiny", text_generation)
verification.check("long", text_generation)
verification.finish()
