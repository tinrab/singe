"""RecurrentGemma 2B in BF16, generating greedily after a function header and after a count to one hundred, with `transformers`' `generate`, as `mod.rs` generates in Singe.
Its 2.7B parameters are stored in F32 and do not fit 8 GB in F32, and no smaller RecurrentGemma exists.
The prompts start with `<bos>`, without which the model's likeliest tokens are near ties that BF16 rounding decides: on the factory paragraph the F32 model's top two first tokens are 0.07 apart, and Singe followed F32 where the BF16 reference did not."""

from support import Verification, text_generation

verification = Verification(__file__)
verification.check("prompt", text_generation)
verification.check("long", text_generation)
verification.finish()
