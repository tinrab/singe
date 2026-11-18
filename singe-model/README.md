# singe-model

Models built on `singe`, and the pipelines that run them for Hugging Face's tasks.
Models live under `model/<category>/<kind>`, the heads tasks ask models for under `head`, and one pipeline or placeholder per task under `task/<category>`.

## Adding a model

1. Write the model's Singe config and layers in `src/model/<category>/<kind>/mod.rs`, from `assembly`'s blocks where they fit, with Singe's tensor names.
   The category is the section of `transformers`' docs its `model_doc` page is in, or `diffusion` for a `diffusers` model.
2. Write the importer, `transformers.rs` or `diffusers.rs` beside it: the config read from the snapshot's into Singe's, and `format::Rules` mapping every stored tensor to a Singe tensor.
   A load refuses a stored tensor no rule maps and a Singe tensor no rule covers.
3. Declare the model's `MODEL` entry, with its names in each format and a loader for each head it serves, and list it in its category's `MODELS`.
   A model built on `assembly::PostNormEncoder` implements `head::EncoderModel` instead of writing layers and loaders, and its entry lists `head::encoder_heads::<Model>()`, as `model::text::distilbert` does.
4. A head no existing one fits gets a module in `head` and a line in `heads!`, and a task without a pipeline gets one in `src/task/<category>/<task>.rs`.
5. Name the snapshots its cases read in `verifications/snapshots.toml`, and download them with `just snapshot NAME`.
6. Add its verification in `verifications/<category>/<kind>/`:
    - `cases.json`: each case's name, snapshot, task, inputs, parameters, and the reference's outputs;
    - `verify.py`: the reference code, written as `transformers`, `sentence-transformers`, or `diffusers` is used, which records the outputs with `--write`:
      `uv run --locked python singe-model/verifications/<category>/<kind>/verify.py --write`;
    - `verify.rs`: the same cases through Singe's pipelines, each as `session.check::<Pipeline>("case", DataType::F32)`, in the type its reference loads.
7. List `verify.rs` in its category's `verifications/<category>/mod.rs`, and run both sides with `just verify FILTER`, which runs the cases whose `model/case` labels contain `FILTER`, such as `text/bert` or `ranking`.

Both sides report each case's load, first run, and median run, split into the GPU's work and the host's.
`just verify` writes the media cases read under `verifications/media/` on first use.
