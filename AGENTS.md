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

The toolchain comes from `flake.nix`, loaded by direnv (`.envrc`: `use flake`).
That shell pins CUDA and the NVIDIA library versions and exports `CUDA_PATH`, `CUDA_CCBIN`, `LD_LIBRARY_PATH`, and related variables; the justfile assumes its tools are on `PATH`.
Find CUDA headers through `CUDA_PATH`, for example `grep CUDA_VERSION "$CUDA_PATH/include/cuda.h"`.
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
| Model tests (`--include-ignored` for real ones) | `just test-models`                                             |
| Write the PyTorch references the tests read     | `just prepare-tests` (`--real` for the real-snapshot ones)     |
| Write or check other generated files            | `just references` (`--check`, suite names)                     |
| Benchmarks, saved and compared by name          | `just bench` (`--save NAME`, `--compare NAME`, filters)        |
| No NVIDIA library linked at build time          | `just check-linking`                                           |
| Everything a change must pass                   | `just gate`                                                    |
| API documentation                               | `just docs` or `just docs-open`                                |
| Regenerate bindings, kernels, and injected docs | `just generate`                                                |
| cuDNN frontend examples                         | `just frontend-examples`                                       |
| Repository syntax rules                         | `cargo xtask no-slop` (`--fix`, `--fix-dry-run`, `-p <crate>`) |

- The root `justfile` exports `RUSTFLAGS=-Awarnings`, so `just check` and plain cargo builds hide warnings; `just lint` is the authority on warnings and Clippy.
- `just lint --package` uses `--no-default-features` and all targets, so name the crate's version feature too.
  Feature names differ per crate, such as `nvjpeg_13_2,k2,testing`, `nvtiff_0_8,testing`, or `testing`; check the crate's `Cargo.toml`.
- The `justfile` defines the recipe feature sets in `TEST_FEATURES`, `LINT_FEATURES`, and `SINGE_TEST_FEATURES`.
- A focused CUDA test runs as `RUST_TEST_THREADS=1 cargo test -p <crate> --features testing,cuda_13_4 -- --nocapture`, adding the features the tested path needs.
  GPU tests depend on hardware and driver state and must run serialized.
- Build one crate's docs with warnings denied using `RUSTDOCFLAGS="-D warnings" cargo doc -p <crate> --no-deps --all-features`, and again with default features when a module is feature-gated.
- Do not run several cargo commands at once against the shared `target/` directory; concurrent builds delete each other's artifacts.
- Batch changes before compiling or testing; the workspace is large, and every build and test run is slow.

## Workspace layout

- `singe`: the framework, with tensors, shapes, layouts, storage, graphs, ONNX import, and the CPU and CUDA backends.
- `singe-core`: shared low-level macros and helpers, such as `status_enum!`, `impl_enum_conversion!`, `checked_conversions!`, and `LibraryVersion`.
- `singe-kernel`: reusable CPU and CUDA kernels; generated cuTile kernels live under `src/-generated/`.
- `singe-codegen`: the kernel IR, its schedules, an interpreter, and the CUDA C and Tile emitters.
- `singe-model`: model configs and architecture definitions built on `singe`, whose layers, caches, checkpoints, and generation loop live in `singe`.
- `singe-testing`: how tests compare with PyTorch's references, and the benchmark runner; see its README.
- `singe-onnx`: ONNX model utilities.
- `singe-ptx`: a PTX parser and a CPU emulator that runs PTX kernels without a GPU.
- NVIDIA wrapper crates: `singe-cuda`, `singe-cublas`, `singe-cudnn`, `singe-cudss`, `singe-cufft`, `singe-cufile`, `singe-cupti`, `singe-curand`, `singe-cusolver`, `singe-cusparse`, `singe-cutensor`, `singe-nccl`, `singe-npp`, `singe-nvcomp`, `singe-nvimagecodec`, `singe-nvjpeg`, `singe-nvml`, and `singe-nvtiff`.
- Raw FFI crates are `*-sys`; generated bindings are checked in, while `src/lib.rs` and each crate's `justfile` are handwritten.
- `singe-loader` loads every NVIDIA library at run time, so no `*-sys` crate links one at build time, except nvCOMP's C++ manager bridge.
- Proc-macro crates: `singe-macros` and `singe-cuda-macros`; `singe-cuda-find` discovers CUDA installations.
- `xtask`: documentation injection, cuTile kernel generation, and the `no-slop` checker.
- Dependencies: versions are declared only in the root `[workspace.dependencies]`, and members use `x.workspace = true`.
  Lints and profiles also live at the root.

## Architecture pointers

- Tensor semantics: `singe/src/tensor/`, `shape/`, `layout/`, `storage/`, `data_type/`, and `device.rs`.
- Operations and dispatch: operation metadata and inference in `singe/src/backend/registry/`, dispatch keys, kernels, autocast, and custom operations in `singe/src/dispatch/`, and backend kernel registration in `singe/src/backend/kernels.rs`.
- Backends: `singe/src/backend/`, with `cpu/` and `cuda/` implementations; graphs live in `singe/src/graph/` and ONNX import in `singe/src/onnx/`.
- CUDA wrappers: `singe-cuda/src/context.rs`, `device.rs`, `stream/`, `event.rs`, `memory/`, `module/`, `kernel/`, `graph/`, `submission/`, `nvrtc/`, and `jit.rs`.
- cuDNN: the frontend port in `singe-cudnn/src/frontend/`, backend descriptors, engines, and execution in `singe-cudnn/src/backend/`, and legacy descriptor APIs in `singe-cudnn/src/legacy/`.
- PTX: parsing in `singe-ptx/src/parser/`; the emulator defines supported instruction forms in `emulator/lowering/`, one module per family, and executes them in `emulator/engine/`.

## Tests and references

- `singe/tests` has five targets, `tensor`, `graph`, `kernels`, `nn`, and `onnx`, each a `main.rs` with a module per subject; `singe-model/tests/models` has one, by task.
- A test that compares with PyTorch reads a reference with `singe_testing::reference!`, compares with `assert_matches` under a named `Tolerance`, and runs captured graphs on every path with `check_paths`.
- References live under each crate's `tests/references`, as `tensors.safetensors` and `reference.json`, and are generated, never edited by hand.
  `scripts/references` writes them with PyTorch, one module per suite, through `just prepare-tests`, which `check`, `lint`, and the test recipes run first, since `singe`'s tests and examples embed some of them when they compile.
  It skips the work when `target/references` records a run with the same scripts and `uv.lock`; the real-snapshot references need `just prepare-tests --real` and the Hugging Face cache.
- Baselines of Singe's own results, which tests compare with to catch a change either way, live under `tests/baselines` and are checked in; `SINGE_UPDATE_CONFORMANCE=1` and `SINGE_UPDATE_ONNX_CONFORMANCE=1` rewrite them.
- Real-weight tests are ignored with one reason and read `SINGE_SNAPSHOT_<MODEL>`, a local Hugging Face snapshot directory.
- Nothing in a test times anything; timing belongs in the `benches/` targets, through `singe_testing::Benches`.

## Generated data

- Bindgen recipes live in the `*-sys/justfile`s, and each ends by rewriting the bindings' `unsafe extern "C"` blocks into `singe_loader::functions!` blocks, whose functions keep their names and load their symbols on first call.
- A `*-sys` crate declares each library it loads through `singe_loader::Library`'s constructor for it, such as `Library::cusparse()`, which names its files per platform in `singe-loader/src/libraries.rs`; it adds a `load` function per library, and `singe_loader::missing_status!` for each status type: what a function returns when its library or symbol is missing.
  Its safe crate returns `Error::Load` where a status says the library is missing and it would not load.
- Do not hand-edit generated FFI declarations, generated kernels, or generated rustdoc unless the task targets checked-in generated output.
- Injected NVIDIA documentation comes from normalized machine inputs in `xtask/docs/` through `xtask/src/document.rs`; fix the inputs or the injection logic, then regenerate, and use `jq` for the large JSON files.
- Generated binding filenames do not always match documentation versions; check the crate's `justfile`, `src/lib.rs`, and the generated filename together.

## Wrapper crate conventions

Follow the nearest existing wrapper; the style guide covers the details.

- Each wrapper has `error.rs` with a `#[non_exhaustive]` `Error`, a `Result` alias, a `Status` declared with `singe_core::status_enum!`, and a `try_ffi!` macro.
  Libraries without an error-string function write `Variant = CONSTANT => "description"` so the macro generates `description()`.
- Use typed enums and `bitflags` wrappers, checked conversions for C integer types, and narrow `unsafe` blocks with `// SAFETY:` comments.
- Plain settings values are `*Options`, with public fields when every combination is valid and `with_*` setters otherwise; wrappers of native parameter structs or handles are `*Params`, with `set_*` methods.
- Safe calls wait for their GPU work; `unsafe` functions that only enqueue work are named `<operation>_async` and take wrapper types, never raw `sys` handles.
- A stream in another CUDA context is the crate's `Error::StreamContextMismatch`; memory in another context is `CudaError::MemoryContextMismatch`.
- GPU work goes through `singe-cuda`'s stream machinery:
    - `Stream::sync_scope` runs eager work and waits for it even on failure;
    - `Stream::ensure_not_capturing` rejects graph capture before eager work;
    - `Stream::defer_drop_until_capture_end` keeps native owners alive through capture.
      An owner retained that way holds a `BorrowedStream` instead of the stream wrapper, to avoid a reference cycle.
- Keep ported examples close to upstream sample names while making the Rust APIs idiomatic.

## The no-slop checker

`cargo xtask no-slop` must report no findings.
Its rules live in `xtask/src/no_slop/rules/`, one module per rule, registered in `rules/mod.rs`; rule levels and exclusions can be set in a `[workspace.metadata.no-slop]` table in the root `Cargo.toml`.
Fixes may only change whitespace, comments, import grouping, or line breaks inside doc comments; `fix::validate` rejects anything else.
`no-dense-statement-runs` is off, and `xtask/src/no_slop/UPSTREAM.md` records why, along with where this copy differs from upstream.

The rules include:

- test functions start with `test_`;
- fallible constructors are named `create`;
- error messages are lowercase, do not end in punctuation, preserve their sources, and avoid vague text;
- imports are grouped into standard, external, and local roots;
- sibling declarations are separated by blank lines;
- module trees use `name/mod.rs`, never a sibling `name.rs`;
- a module's `mod` declarations come before its imports, and `mod tests` comes last;
- visibility is private, `pub(crate)`, or `pub`, never `pub(super)` or `pub(in …)`;
- each sentence of a doc comment is on its own line;
- lint suppressions name specific lints instead of `warnings` or `clippy::all`;
- a multi-line block is followed by a blank line, unless the next statement returns the final value;
- an error that exposes a source through `source()` does not also print it, so a wrapper that adds nothing is `#[error(transparent)]`;
- error contracts name a type instead of `String` or `Box<dyn Error>`, which only `main` and trait implementations may return;
- modules are named for a responsibility, never `utils`, `helpers`, or `common`.

Inside a private module, Clippy's `redundant_pub_crate` wants plain `pub` rather than `pub(crate)`.
Re-export a private child module's items as `pub use self::child::…`, so rustfmt and no-slop agree on import grouping.
Exceptions take the form `// no-slop-allow: <rule> -- <reason>` on the line before the target.

## This repo is public

Everything pushed stays public in history.
Before committing, check the staged diff:

```sh
git diff --cached | rg -n '/home/|/tmp/|/run/user/|/nix/store/|\.cache/huggingface|\.ts\.net|ses_[A-Za-z0-9]{20}'
```

Keep these out of commits:

- secrets, tokens, and the contents of credential files;
- anything from the development machine's filesystem: real paths, mount points, user and group names, machine names, and hostnames.
  Write repo-relative paths, and use made-up names in tests and fixtures;
- anything that identifies a real GPU or machine: device UUIDs, serial numbers, PCI bus IDs, and the output of NVML, `nvidia-smi`, or profilers that carries them;
- local model snapshot paths; tests read them from `SINGE_SNAPSHOT_<MODEL>`, and references must not record them;
- session IDs, chat or issue-tracker links, and quotes from private sessions, chats, or calls;
- people's names attached to their plans, opinions, or private discussions, and private repos or branches;
- agent handoff notes, plans, research notes, raw evidence, logs, benchmark results, profiler reports, and other temporary files.
  Keep them in `wip/`, which is gitignored, as is `benchmarks/`, where `just bench` saves its baselines;
- copies of third-party source kept for reference; they belong in `wip/` too, and code that ports from one says where it came from by project and path.

## Constraints

- **Style guide:** [docs/RUST_STYLE_GUIDE.md](docs/RUST_STYLE_GUIDE.md) is the source of truth for Rust engineering and style; read the relevant section before changing a crate, and don't edit it.
    - Rust tests are named `test_*`.
    - Use `pub(crate)`, never `pub(super)`.
    - Use typed errors (thiserror) with lowercase messages, not `String` errors.
- **No doc comments:** don't write new doc comments (`///`, `//!`) for now; they come at the end, before the crates are released.
  Existing doc comments stay as they are; update one only when the code it describes changes, and move it with its code.
  Write an inline `//` comment only where the code cannot say something itself, such as why a non-obvious rule holds.
- **One sentence per line:** write Markdown prose and comments with each sentence on its own line, as the style guide describes.
- **Lock files:** don't edit `uv.lock` by hand; let uv manage it.
  `Cargo.lock` is gitignored.
- **Work files:** temporary work files, notes, and todo lists go in `wip/`.

## Documentation

These rules apply when doc comments are written, before release; until then, see the constraints above.

- Seed a safe crate's doc comments from its `*-sys` bindings, then rewrite the result in Rust terms instead of keeping C names, `# Parameters`, or return-value sections.
- Summaries start with a third-person verb, `# Errors` sentences start with "Returns an error if", and `# Safety` comes before `# Errors`.
- Crates with `unsafe` `*_async` functions state their shared rules once, in a `# Raw execution` section of the crate docs.
- Module documentation goes at the top of the module file as `//!`.
- State observable behavior instead of vague claims, framing, or hedges.
