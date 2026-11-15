Singe is an inference-focused machine learning framework written in Rust, with safe wrappers for the CUDA ecosystem.

This is a greenfield project with no users to keep compatible with:

- No backwards compatibility.
- No deprecated aliases, migration layers, or temporary shims.
- No legacy code paths.

When an API changes, update every caller in the same change and delete the old form.
Public APIs may change whenever that produces a cleaner design.

Explore the relevant crates before changing them, and inspect the closest existing implementation before adding an API, especially in a wrapper crate.

Private notes for this checkout, such as the current plan and where reference material lives, are in `wip/AGENTS.md` when it exists; read it before starting larger work.

## Environment

The toolchain comes from `flake.nix`, loaded by direnv (`.envrc`: `use flake`), which pins CUDA and the NVIDIA libraries and exports `CUDA_PATH` and related variables.
Find CUDA headers through `CUDA_PATH`.
Use `uv` for Python helpers.

## Commands

| Task                                            | Command                                                        |
| ----------------------------------------------- | -------------------------------------------------------------- |
| Format                                          | `just format`                                                  |
| Check libraries, tests, and examples            | `just check`                                                   |
| Clippy for the workspace                        | `just lint` (`just lint --fix` applies fixes)                  |
| Clippy for one crate                            | `just lint --package <crate> --features <features>`            |
| Full verification, including doctests           | `just test`                                                    |
| Framework tests, CPU feature sets then CUDA     | `just test-singe` (`just test-singe --test nn attention`)      |
| Write the PyTorch references the tests read     | `just prepare-tests`                                           |
| Write or check other generated files            | `just references` (`--check`, suite names)                     |
| Benchmarks, saved and compared by name          | `just bench` (`--save NAME`, `--compare NAME`, filters)        |
| No NVIDIA library linked at build time          | `just check-linking`                                           |
| Everything a change must pass                   | `just gate`                                                    |
| API documentation                               | `just docs` or `just docs-open`                                |
| Regenerate bindings, kernels, and injected docs | `just generate`                                                |
| Repository syntax rules                         | `cargo xtask no-slop` (`--fix`, `--fix-dry-run`, `-p <crate>`) |

- The root `justfile` exports `RUSTFLAGS=-Awarnings`, so plain builds hide warnings; `just lint` is the authority on warnings and Clippy.
- `just lint --package` uses `--no-default-features`, so name the crate's version feature too; feature names differ per crate, so check its `Cargo.toml`.
- A focused CUDA test runs as `RUST_TEST_THREADS=1 cargo test -p <crate> --features testing,cuda_13_4 -- --nocapture`; GPU tests must run serialized.
- Do not run several cargo commands at once against the shared `target/` directory; concurrent builds delete each other's artifacts.
- Batch changes before compiling or testing; the workspace is large, and every build and test run is slow.

## Workspace layout

- `singe`: the framework: tensors, graphs, ONNX import, the CPU and CUDA backends, layers, caches, generation, and exported CUDA C++ programs.
- `singe-model`: model configs and architectures built on `singe`; its README gives the steps for adding a model.
- `singe-kernel`, `singe-codegen`: reusable kernels, and the kernel IR with its emitters.
- `singe-testing`: comparing with PyTorch's references, and the benchmark runner; see its README.
- `singe-onnx`, `singe-ptx`: ONNX utilities, and a PTX parser and CPU emulator.
- NVIDIA wrapper crates (`singe-cuda`, `singe-cudnn`, `singe-cublas`, and the rest), each over a raw `*-sys` crate whose bindings are generated and checked in.
- `singe-loader` loads every NVIDIA library at run time, so no `*-sys` crate links one at build time.
- `xtask`: documentation injection, kernel generation, and the `no-slop` checker.
- Dependency versions are declared only in the root `[workspace.dependencies]`, and members use `x.workspace = true`; lints and profiles also live at the root.

## Tests and references

- Tests cover `singe`'s core components, operations, blocks, and modules, for correctness, as few as that takes.
  Models have no tests; `singe-model`'s examples show them working, and `scripts/check_generation.py` compares a text model's greedy tokens with `transformers`.
- A test that compares with PyTorch reads a generated reference through `singe_testing::reference!`; references are written by `scripts/references` through `just prepare-tests` and never edited by hand.
- Examples and benchmarks of real models read `SINGE_SNAPSHOT_<MODEL>`, a local Hugging Face snapshot directory.
- Nothing in a test times anything; timing belongs in the `benches/` targets.

## Generated data

- Do not hand-edit generated FFI declarations, generated kernels, or generated rustdoc unless the task targets them; change the recipe or its inputs and regenerate.
- Bindgen recipes live in the `*-sys/justfile`s; injected NVIDIA documentation comes from `xtask/docs/` through `xtask/src/document.rs`.

## Wrapper crate conventions

Follow the nearest existing wrapper; the style guide covers the details.

- Each wrapper has `error.rs` with a `#[non_exhaustive]` `Error`, a `Result` alias, a `Status` declared with `singe_core::status_enum!`, and a `try_ffi!` macro.
- Use typed enums and `bitflags` wrappers, checked conversions for C integer types, and narrow `unsafe` blocks with `// SAFETY:` comments.
- Plain settings values are `*Options`; wrappers of native parameter structs or handles are `*Params`, with `set_*` methods.
- Safe calls wait for their GPU work; `unsafe` functions that only enqueue work are named `<operation>_async` and take wrapper types, never raw `sys` handles.
- GPU work goes through `singe-cuda`'s stream machinery.

## The no-slop checker

`cargo xtask no-slop` must report no findings; its rules live in `xtask/src/no_slop/rules/`.
Among them:

- test functions start with `test_`, and fallible constructors are named `create`;
- error messages are lowercase, do not end in punctuation, and preserve their sources;
- imports are grouped into standard, external, and local roots;
- module trees use `name/mod.rs`, never a sibling `name.rs`, and modules are named for a responsibility, never `utils` or `helpers`;
- visibility is private, `pub(crate)`, or `pub`, never `pub(super)`;
- a multi-line block is followed by a blank line, unless the next statement returns the final value.

Exceptions take the form `// no-slop-allow: <rule> -- <reason>` on the line before the target.

## This repo is public

Everything pushed stays public in history.
Before committing, check the staged diff:

```sh
git diff --cached | rg -n '/home/|/tmp/|/run/user/|/nix/store/|\.cache/huggingface|\.ts\.net|ses_[A-Za-z0-9]{20}'
```

Keep these out of commits:

- secrets, tokens, and the contents of credential files;
- anything from the development machine: real paths, user, machine, and host names; write repo-relative paths, and use made-up names in tests;
- anything that identifies a real GPU or machine, such as device UUIDs or the output of NVML, `nvidia-smi`, or profilers;
- local model snapshot paths;
- session IDs, chat links, and quotes from private conversations, and people's names attached to their plans or opinions;
- plans, notes, logs, benchmark results, profiler reports, and other temporary files, which belong in `wip/` (gitignored);
- copies of third-party source; code ported from one says where it came from by project and path.

## Constraints

- **Style guide:** [docs/RUST_STYLE_GUIDE.md](docs/RUST_STYLE_GUIDE.md) is the source of truth for Rust engineering and style; read the relevant section before changing a crate, and don't edit it.
- **Errors:** typed errors (thiserror) with lowercase messages, not `String` errors.
- **No doc comments unless asked:** don't write new doc comments (`///`, `//!`); existing ones stay, updated only when their code changes.
  Write an inline `//` comment only where the code cannot say something itself, such as why a non-obvious rule holds.
- **One sentence per line** in Markdown prose and comments.
- **Lock files:** don't edit `uv.lock` by hand; `Cargo.lock` is gitignored.
- **Work files:** temporary files, notes, and todo lists go in `wip/`.
