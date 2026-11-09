# Upstream provenance

This scanner is maintained in [no-slop-rs](https://github.com/tinrab/no-slop-rs).

## Design

- Rules are copied into the target repository, which owns them from then on, and local edits are policy to preserve.
- One agent skill covers fresh installs and reviewed updates, with separate procedures.
- Canonical source is synchronized into the skill's assets, and a check mode reports stale assets.
- Provenance is recorded in this file beside the code it describes.
- Agent tooling directories are excluded from scans by default.
- Rules run from `cargo xtask no-slop` on `syn` and `ra-ap-rustc_lexer`.
- Agents copy the runtime directly; there is no installer script.

## Updating and verification

This section applies to the `no-slop-rs` repository; installed copies follow the skill's update procedure instead.

After changing the scanner, run `cargo xtask sync-no-slop-skill-assets` and `cargo test --workspace --locked`.

## Installed copy

Source: the `no_slop` assets of the `install-no-slop-rs` skill, as installed on 2026-11-06.
Its assets state no revision, so that bundle is the only identity available.

Base: unknown.
The earlier copy recorded no source, so this was a two-way comparison, and every difference was read and either adopted or kept below.

Adopted from the bundle:

- The rules `no-source-in-error-display`, `no-log-and-return`, `no-eager-error-context`, `no-vague-module-names`, and `require-blank-after-block`.
- `no-string-errors` now covers `Box<dyn Error>`, and `require-lowercase-error-messages` now reports trailing punctuation and reads more error sites.
- The refactored `discovery`, `readable_spacing`, `error_support`, and `fix` modules, and `FixMode` in place of two option flags.

Disabled:

- `no-dense-statement-runs` is `off` in the root manifest.
  It found 3,655 runs after `require-blank-after-block`'s fix, 1,989 of them in `singe-kernel`, and it has no fix, so satisfying it would have meant blank lines chosen by a script.
  At a threshold of 20 statements in `DENSE_RUN` it would find 97, if it is wanted later for long runs only.

Local deviations to keep:

- `[workspace.metadata.no-slop.scopes]` in `config.rs`, read in `check_file`, which limits a rule to path prefixes.
- The rules `require-tests-module-last`, `no-literal-to-string-into`, and `require-mods-before-imports`, which now cover every crate, so no scope is set.
- `pub` in place of `pub(crate)` in `mod.rs`, since Clippy's `redundant_pub_crate` applies inside this private module.
- A test of `sentences_of` in `rules/doc_sentences.rs`.
- `LICENSE` beside the scanner, which predates this record.

Dropped local changes, which the bundle made unnecessary:

- `imports_parent`, which the bundle names `type_only`.
- An extra case in `fix.rs`'s edit selection, which `edit.start == last.start` already covers.
- `clippy::struct_excessive_bools` on `cli::Options`, since `FixMode` replaced those flags.
