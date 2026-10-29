TEST_FEATURES := "testing,cuda_13_4,cutile"
export RUSTFLAGS := "-Awarnings"
export CUDNN_LOGLEVEL_DBG := "0"
export CUDNN_LOGDEST_DBG := "stdout"

format:
    just --fmt --unstable

    cargo fmt

    find ./ -iname *.proto | xargs clang-format -style=Google -i

prepare:
    #!/usr/bin/env bash
    set -euxo pipefail

    crates=(
        singe-cublas
        singe-cublas-sys
        singe-cuda
        singe-cuda-sys
        singe-curand
        singe-curand-sys
        singe-cufile
        singe-cufile-sys
        singe-cupti
        singe-cupti-sys
        singe-cudnn
        singe-cudnn-sys
        singe-cudss
        singe-cudss-sys
        singe-cufft
        singe-cufft-sys
        singe-cusolver
        singe-cusolver-sys
        singe-cusparse
        singe-cusparse-sys
        singe-cutensor
        singe-cutensor-sys
        singe-kernel
        singe-nccl
        singe-nccl-sys
        singe-nvml
        singe-nvml-sys
        singe-npp
        singe-npp-sys
    )
    for crate in "${crates[@]}"; do
        mkdir -p "./${crate}/dist"
        cp ./dist/katex_header.html "./${crate}/dist/"
    done

generate:
    #!/usr/bin/env bash
    set -euxo pipefail

    crates=(
        singe-cublas-sys
        singe-cuda-sys
        singe-curand-sys
        singe-cufile-sys
        singe-cupti-sys
        singe-cudnn-sys
        singe-cudss-sys
        singe-cufft-sys
        singe-cusolver-sys
        singe-cusparse-sys
        singe-cutensor-sys
        singe-nccl-sys
        singe-nvml-sys
        singe-npp-sys
    )
    for crate in "${crates[@]}"; do
        (cd "./${crate}" && just generate)
    done

    cargo xtask document
    cargo xtask gen-cutile-dtype-kernels
    cargo xtask gen-cutile-fused

    cargo fmt

check:
    #!/usr/bin/env bash
    set -euxo pipefail

    cargo check --workspace --features {{ TEST_FEATURES }}
    cargo check --workspace --features {{ TEST_FEATURES }} --tests
    cargo check --workspace --features {{ TEST_FEATURES }} --examples

[arg("features", long="features")]
[arg("fix", long="fix", value="true")]
[arg("package", long="package")]
lint fix="false" package="" features=TEST_FEATURES:
    #!/usr/bin/env bash
    set -euxo pipefail

    # Skip cuTile-generated kernel APIs while keeping other workspace crates strict.
    scope=(--workspace --exclude singe-kernel)
    if [[ -n "{{ package }}" ]]; then
        scope=(--package "{{ package }}" --all-targets --no-default-features)
    fi

    disallow=(
        warnings
        deprecated
        # missing_docs
        # clippy::missing_errors_doc
        # clippy::missing_panics_doc
        unused_extern_crates
        unused_import_braces
        unused_qualifications
        clippy::clone_on_ref_ptr
        clippy::all
        clippy::correctness
        clippy::suspicious
        clippy::complexity
        clippy::perf
        clippy::style
        clippy::absolute_paths
        clippy::pedantic
        clippy::nursery
    )
    allow=(
        missing_docs
        clippy::missing_errors_doc
        clippy::missing_panics_doc
        trivial_casts
        trivial_numeric_casts
        clippy::cast_sign_loss
        clippy::cast_precision_loss
        clippy::assert_is_empty
        clippy::too_many_lines
        clippy::float_cmp
        clippy::too_long_first_doc_paragraph
        clippy::too_many_arguments
        clippy::missing_const_for_fn
    )


    if [ "{{ fix }}" = "true" ]; then
        cargo clippy --no-deps "${scope[@]}" --features "{{ features }}" --fix --allow-dirty \
            -- ${disallow[@]/#/-D } ${allow[@]/#/-A }
    else
        cargo clippy --no-deps "${scope[@]}" --features "{{ features }}" \
            -- ${disallow[@]/#/-D } ${allow[@]/#/-A }
    fi

test-prepare:
    #!/usr/bin/env bash
    set -euxo pipefail

    cd singe-onnx/tests && uv run simple.py

test:
    #!/usr/bin/env bash
    set -euxo pipefail

    RUST_TEST_THREADS=1 cargo test --workspace --lib --bins --tests --features {{ TEST_FEATURES }} -- --nocapture
    RUST_TEST_THREADS=1 cargo test --workspace --doc --features {{ TEST_FEATURES }} -- --nocapture

docs:
    cargo doc --workspace --all-features --no-deps

docs-open:
    cargo doc --workspace --all-features --no-deps --open

docs-json:
    #!/usr/bin/env bash
    set -euo pipefail

    packages=(
        singe-cublas-sys
        singe-cuda-sys
        singe-cudnn-sys
        singe-cudss-sys
        singe-cufft-sys
        singe-cufile-sys
        singe-curand-sys
        singe-cusolver-sys
        singe-cusparse-sys
        singe-cutensor-sys
        singe-nccl-sys
        singe-npp-sys
        singe-nvml-sys
    )

    for package in "${packages[@]}"; do
        cargo +nightly rustdoc -p "$package" --all-features -- -Z unstable-options --output-format json
    done

clean:
    #!/usr/bin/env bash
    set -euxo pipefail

    cargo clean

    crates=(
        singe-cublas
        singe-cublas-sys
        singe-cuda
        singe-cuda-sys
        singe-cufile
        singe-cufile-sys
        singe-cudnn
        singe-cudnn-sys
        singe-cudss
        singe-cudss-sys
        singe-cusolver
        singe-cusolver-sys
        singe-cupti
        singe-cupti-sys
        singe-cutensor
        singe-cutensor-sys
        singe-kernel
        singe-nccl
        singe-nccl-sys
        singe-npp
        singe-npp-sys
        singe-nvml
        singe-nvml-sys
        singe-cusolver
        singe-cusolver-sys
        singe-cusparse
        singe-cusparse-sys
    )
    for crate in "${crates[@]}"; do
        (cd "./${crate}" && just -f justfile clean)
    done
