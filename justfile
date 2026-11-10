TEST_FEATURES := "testing,cuda_13_4,onnx,cutile,k2,nvtiff_0_8"
LINT_FEATURES := "testing,cuda_13_4,onnx,k2,nvtiff_0_8"
SINGE_TEST_FEATURES := "testing,cuda_13_4,onnx"
export RUSTFLAGS := "-Awarnings"
export CUDNN_LOGLEVEL_DBG := "0"
export CUDNN_LOGDEST_DBG := "stdout"

format:
    just --fmt --unstable

    cargo fmt

    oxfmt .

    uvx ruff format .

    find ./ -iname *.proto | xargs clang-format -style=Google -i

    # find . -regex '.*\.\(cpp\|hpp\|h\|cu\)' -not -regex '.*\(target\|tests\).*' -exec clang-format -style=file -i {} \;

prepare:
    #!/usr/bin/env bash
    set -euxo pipefail

    for manifest in $(grep -l 'dist/katex_header.html' */Cargo.toml); do
        crate=$(dirname "$manifest")
        mkdir -p "./${crate}/dist"
        cp ./dist/katex_header.html "./${crate}/dist/"
    done

    cargo fmt

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
        singe-nvcomp-sys
        singe-nvjpeg-sys
        singe-nvtiff-sys
        singe-nvimagecodec-sys
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

check: prepare-tests
    #!/usr/bin/env bash
    set -euxo pipefail

    just check-singe-cpu
    cargo check --workspace --features {{ TEST_FEATURES }}
    cargo check --workspace --features {{ TEST_FEATURES }} --tests
    cargo check --workspace --features {{ TEST_FEATURES }} --examples

# Keep the framework's CPU builds independent of optional CUDA features.
check-singe-cpu: prepare-tests
    #!/usr/bin/env bash
    set -euo pipefail
    command -v rg >/dev/null
    for feature_set in "" testing onnx testing,onnx; do
        feature_args=()
        if [[ -n "$feature_set" ]]; then
            feature_args=(--features "$feature_set")
        fi
        dependency_tree="$(cargo tree -p singe --no-default-features --edges normal,build --prefix none "${feature_args[@]}")"
        if rg -q '^(cuda-|cutile|singe-(cuda|cublas|cudnn|cutensor)(-sys)? v)' <<< "$dependency_tree"; then
            echo "CPU framework features must not depend on CUDA tooling" >&2
            exit 1
        fi
        cargo check -p singe --no-default-features --all-targets "${feature_args[@]}"
    done

# Tests the framework in its four CPU feature sets, then with CUDA; arguments filter the CUDA run, as `just test-singe --test nn attention`.
test-singe *arguments: prepare-tests
    #!/usr/bin/env bash
    set -euo pipefail
    for feature_set in "" testing onnx testing,onnx; do
        feature_args=()
        if [[ -n "$feature_set" ]]; then
            feature_args=(--features "$feature_set")
        fi
        cargo test -p singe --no-default-features "${feature_args[@]}"
    done
    RUST_TEST_THREADS=1 cargo test -p singe --features {{ SINGE_TEST_FEATURES }} {{ arguments }}

# Tests the models, including the real-weight tests when their `SINGE_SNAPSHOT_<MODEL>` variables are set: `just test-models --include-ignored`.
test-models *arguments: prepare-tests
    RUST_TEST_THREADS=1 cargo test -p singe-model --features cuda_13_4 -- {{ arguments }}

# Writes the PyTorch references the tests read, which `singe`'s tests and examples embed when they compile, unless the last run wrote them with the same scripts and `uv.lock`; `--real` writes the real-snapshot ones from the Hugging Face cache.
prepare-tests *arguments:
    cd scripts && uv run --locked python -m references --prepare {{ arguments }}

# Writes or, with `--check`, checks every generated file, or the suites named: `just references --check onnx_schemas`, `just references qwen3 --real`.
references *arguments:
    cd scripts && uv run --locked python -m references {{ arguments }}

# Times single operations at the models' shapes, then the real models whose `SINGE_SNAPSHOT_<MODEL>` is set; arguments filter by name and save or compare baselines: `just bench --save main`, `just bench qwen3 --compare main`.
bench *arguments:
    cargo bench -p singe --features {{ SINGE_TEST_FEATURES }} --bench operations -- {{ arguments }}
    cargo bench -p singe-model --features cuda_13_4 --bench models -- {{ arguments }}

[arg("features", long="features")]
[arg("fix", long="fix", value="true")]
[arg("package", long="package")]
lint fix="false" package="" features=LINT_FEATURES: prepare-tests
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
        clippy::missing_safety_doc
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

test: prepare-tests
    #!/usr/bin/env bash
    set -euxo pipefail

    cargo test -p singe-macros -- --nocapture
    RUST_TEST_THREADS=1 cargo test --workspace --lib --bins --tests --features {{ TEST_FEATURES }} -- --nocapture
    RUST_TEST_THREADS=1 cargo test --workspace --doc --features {{ TEST_FEATURES }} -- --nocapture

check-linking: prepare-tests
    #!/usr/bin/env bash
    set -euo pipefail
    binaries=$(
        cargo test -p singe --features {{ SINGE_TEST_FEATURES }} --no-run --message-format=json
        cargo test -p singe-model --features cuda_13_4 --no-run --message-format=json
    )
    found=0
    for binary in $(jq -r "select(.executable != null) | .executable" <<< "$binaries"); do
        if linked=$(readelf -d "$binary" | grep NEEDED | grep -E "lib(cuda|cudart|nvidia|nvrtc|nvJitLink|nvvm|nvtx|cublas|cudnn|cufft|cufile|cupti|curand|cusolver|cusparse|cutensor|cudss|nccl|npp|nvcomp|nvimgcodec|nvjpeg|nvtiff)"); then
            echo "$binary links $linked"
            found=1
        fi
    done
    exit "$found"

gate:
    #!/usr/bin/env bash
    set -euxo pipefail

    just references --check onnx_schemas
    cargo xtask no-slop
    just check
    just lint
    just test-singe
    just prepare-tests --real
    just test-models --include-ignored
    just check-linking

frontend-examples:
    #!/usr/bin/env bash
    set -euo pipefail

    while IFS= read -r path; do
        name="$(basename "$path" .rs)"
        if [[ "$name" == "common" ]]; then
            continue
        fi

        example="frontend_${name}"
        echo "=== $example ==="
        cargo run -q -p singe-cudnn --example "$example"
    done < <(find singe-cudnn/examples/frontend -maxdepth 1 -name '*.rs' | sort)

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
        singe-nvcomp-sys
        singe-nvimagecodec-sys
        singe-nvtiff-sys
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
        singe
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
        singe-nvcomp
        singe-nvcomp-sys
        singe-nvtiff
        singe-nvtiff-sys
        singe-nvimagecodec
        singe-nvimagecodec-sys
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
