"""BitNet b1.58 2B in BF16, generating greedily after a count and after a function header with `transformers`' `generate`, as `mod.rs` generates in Singe.
`transformers` unpacks the ternary weights into BF16 at load, 4.8 GB, so F32 does not fit 8 GB, and its F32 path fails anyway, multiplying F32 inputs by the BF16 weights it unpacks.
The prompts are ones whose first eight greedy tokens lead the runner-up by at least 0.4: after "The most important resource in Factorio is", " the" and " your" are one BF16 step apart on the GPU and tied on the CPU.
Its weight unpacking and activation quantization are `torch.compile`d, which needs `/sbin/ldconfig` that this development shell lacks, so they run eagerly: the same F32 operations, unfused."""

import torch._dynamo

from support import Verification, text_generation

torch._dynamo.config.disable = True

verification = Verification(__file__)
verification.check("count", text_generation)
verification.check("code", text_generation)
verification.finish()
