Singe is an inference-focused machine learning framework written in Rust, with safe wrappers for the CUDA ecosystem.

This is a greenfield project with no users to keep compatible with.
There is no backwards compatibility, no deprecated aliases or shims, and no legacy code paths.
When an API changes, update every caller in the same change and delete the old form.

Explore the relevant code, and the closest existing implementation, before changing or adding anything.
Private notes for this checkout are in `wip/AGENTS.md` when it exists; read it before larger work.

## Working order

- Write every change the task needs first, then compile, lint, and test once at the end.
- Run only the most specific check for the change, such as one test, one crate's lint, or one model's verification.
- Never run several cargo commands at once; they share `target/`.

## Environment and commands

The toolchain, CUDA, and the NVIDIA libraries come from `flake.nix` through direnv.
Use `uv` for Python.
The root `justfile` has the recipes for formatting, checking, linting, testing, verifying models, generating code, and benchmarks; read it for the details.

- `just lint` is the authority on warnings, and its recipe owns the list of allowed lints; don't suppress those per site.
- GPU tests depend on hardware and run serialized.

## Workspace

- `singe` is the framework; `singe-model` holds the models built on it.
- The NVIDIA wrapper crates are safe wrappers over `*-sys` crates of generated bindings, loaded at run time through `singe-loader`.
- `xtask` holds code generation and the `no-slop` checker.
- Dependency versions, lints, and profiles live only in the root `Cargo.toml`.

## Tests

- Keep tests few: they cover `singe`'s core components for correctness, and no new ones are added without need.
- Models are checked by verifications against their reference libraries, not by tests; a change to a model, or to what models share, is done when the touched models' verifications pass.
- References are generated, never edited by hand.
- Nothing in a test times anything; timing belongs in benchmarks and verifications.

## Generated code

Never hand-edit generated bindings, kernels, or injected documentation; change the generator or its inputs and regenerate.

## Wrapper crates

Follow the nearest existing wrapper and the style guide: typed errors and enums, checked conversions, narrow `unsafe` blocks with `// SAFETY:` comments, and GPU work through `singe-cuda`'s stream machinery.

## The no-slop checker

`cargo xtask no-slop` must report no findings.
Its rules cover naming, error messages, imports, blank lines, module layout, visibility, and lint suppressions; read them in `xtask/src/no_slop/rules/`.

## This repo is public

Before committing, check the staged diff:

```sh
git diff --cached | rg -n '/home/|/tmp/|/run/user/|/nix/store/|\.cache/huggingface|\.ts\.net|ses_[A-Za-z0-9]{20}'
```

Keep out of commits anything private or local: secrets, machine paths and names, hardware identifiers, snapshot paths, session IDs, private quotes and people's names, notes, plans, logs, benchmark results, and copies of third-party source.
These belong in `wip/`, which is gitignored.
A commit message is a single line.

## Constraints

- [docs/RUST_STYLE_GUIDE.md](docs/RUST_STYLE_GUIDE.md) is the source of truth for Rust style.
- Don't edit this file or the style guide unless asked.
- Use typed errors with lowercase messages.
- Don't write new doc comments (`///`, `//!`) unless a task asks for them; existing ones stay, updated only when their code changes.
- Don't write regular line comments in place of doc comments.
- Write an inline comment only where the code cannot say something itself.
- Write each sentence of prose and comments on its own line.
- Don't edit `uv.lock` by hand.
- Temporary files, notes, and todo lists go in `wip/`.
