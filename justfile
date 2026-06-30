TEST_FEATURES := "testing,cuda_13_3"
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

    cp -r ./dist ./singe/
    cp -r ./dist ./singe-cublas/
    cp -r ./dist ./singe-cublas-sys/
    cp -r ./dist ./singe-cuda/
    cp -r ./dist ./singe-cuda-sys/
    cp -r ./dist ./singe-curand/
    cp -r ./dist ./singe-curand-sys/
    cp -r ./dist ./singe-cufile/
    cp -r ./dist ./singe-cufile-sys/
    cp -r ./dist ./singe-cupti/
    cp -r ./dist ./singe-cupti-sys/
    cp -r ./dist ./singe-cudnn/
    cp -r ./dist ./singe-cudnn-sys/
    cp -r ./dist ./singe-cudss/
    cp -r ./dist ./singe-cudss-sys/
    cp -r ./dist ./singe-cufft/
    cp -r ./dist ./singe-cufft-sys/
    cp -r ./dist ./singe-cusolver/
    cp -r ./dist ./singe-cusolver-sys/
    cp -r ./dist ./singe-cusparse/
    cp -r ./dist ./singe-cusparse-sys/
    cp -r ./dist ./singe-cutensor/
    cp -r ./dist ./singe-cutensor-sys/
    cp -r ./dist ./singe-nccl/
    cp -r ./dist ./singe-nccl-sys/
    cp -r ./dist ./singe-nvml/
    cp -r ./dist ./singe-nvml-sys/
    cp -r ./dist ./singe-cusparse/
    cp -r ./dist ./singe-cusparse-sys/
    cp -r ./dist ./singe-npp/
    cp -r ./dist ./singe-npp-sys/

generate:
    #!/usr/bin/env bash
    set -euxo pipefail

    (cd ./singe-cublas-sys && just generate)
    (cd ./singe-cuda-sys && just generate)
    (cd ./singe-curand-sys && just generate)
    (cd ./singe-cufile-sys && just generate)
    (cd ./singe-cupti-sys && just generate)
    (cd ./singe-cudnn-sys && just generate)
    (cd ./singe-cudss-sys && just generate)
    (cd ./singe-cufft-sys && just generate)
    (cd ./singe-cusolver-sys && just generate)
    (cd ./singe-cusparse-sys && just generate)
    (cd ./singe-cutensor-sys && just generate)
    (cd ./singe-nccl-sys && just generate)
    (cd ./singe-nvml-sys && just generate)
    (cd ./singe-cusparse-sys && just generate)
    (cd ./singe-npp-sys && just generate)

    docs_dir="./cuda-docs-scraper/docs"
    target_dir="./xtask/docs"

    cp $docs_dir/cublas/v13.3.0/cublas-api.json $target_dir/cublas-13.3-api.json

    cp $docs_dir/cuda-driver-api/v13.3.0/cuda-driver-api.json $target_dir/cuda-driver-13.3-api.json
    cp $docs_dir/cuda-runtime-api/v13.3.0/cuda-runtime-api.json $target_dir/cuda-runtime-13.3-api.json
    cp $docs_dir/nvtx/v3/nvtx-api.json $target_dir/nvtx-3-api.json
    cp $docs_dir/nvrtc/v13.3.0/nvrtc-api.json $target_dir/nvrtc-13.3.json
    cp $docs_dir/libnvvm/v13.3.0/libnvvm-api.json $target_dir/libnvvm-13.3-api.json

    cp $docs_dir/cudnn/v9.22.0/cudnn-api.json $target_dir/cudnn-9.22-api.json

    cp $docs_dir/cudss/v0.8.0/cudss-api.json $target_dir/cudss-0.8-api.json

    cp $docs_dir/cufft/v13.3.0/cufft-api.json $target_dir/cufft-13.3-api.json

    cp $docs_dir/cufile/v1.18/cufile-api.json $target_dir/cufile-1.18-api.json

    cp $docs_dir/cupti/v2026.2.0/cupti-api.json $target_dir/cupti-13.3-api.json

    cp $docs_dir/nvml/v13.3.0/nvml-api.json $target_dir/nvml-13.3-api.json

    cp $docs_dir/curand/v13.3.0/curand-api.json $target_dir/curand-13.3-api.json

    cp $docs_dir/cusolver/v13.3.0/cusolver-api.json $target_dir/cusolver-12.2-api.json
    cp $docs_dir/cusparse/v13.3.0/cusparse-api.json $target_dir/cusparse-12.8-api.json

    cp $docs_dir/cutensor/v2.6.0/cutensor-api.json $target_dir/cutensor-2.6-api.json

    cp $docs_dir/nccl/v2.28.7/nccl-api.json $target_dir/nccl-2.28-api.json

    cp $docs_dir/npp/v13.3.0/npp-api.json $target_dir/npp-13.3-api.json

    cp $docs_dir/ptx/v9.3/instruction-set.json $target_dir/ptx-9.3-instructions.json

    cargo xtask gen-ptx-instructions
    cargo xtask document

    cargo fmt

check:
    #!/usr/bin/env bash
    set -euxo pipefail

    # cargo check --workspace --no-default-features
    cargo check --workspace --features {{ TEST_FEATURES }}
    cargo check --workspace --features {{ TEST_FEATURES }} --tests
    cargo check --workspace --features {{ TEST_FEATURES }} --examples

[arg("fix", long="fix", value="true")]
lint fix="false":
    #!/usr/bin/env bash
    set -euxo pipefail

    # cargo fmt --all -- --check

    disallow=(
        warnings
        deprecated
        # unsafe_code
        # trivial_casts
        # trivial_numeric_casts
        # missing_docs
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
        # clippy::pedantic
        # clippy::nursery
        # clippy::missing_errors_doc
        # clippy::missing_panics_doc
    )
    allow=(
        missing_docs
        clippy::missing_safety_doc
        clippy::missing_panics_doc
        clippy::missing_errors_doc
        clippy::too_many_arguments
        clippy::missing_const_for_fn
        clippy::similar_names
        clippy::type_complexity
    )


    if [ "{{ fix }}" = "true" ]; then
        cargo clippy --workspace --features {{ TEST_FEATURES }} --fix --allow-dirty \
            -- ${disallow[@]/#/-D } ${allow[@]/#/-A }
    else
        cargo clippy --workspace --features {{ TEST_FEATURES }} \
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

    for crate in singe-cublas singe-cublas-sys singe-cuda singe-cuda-sys singe-cufile singe-cufile-sys singe-cudnn singe-cudnn-sys singe-cudss singe-cudss-sys singe-cusolver singe-cusolver-sys singe-cupti singe-cupti-sys singe-cutensor singe-cutensor-sys singe-nccl singe-nccl-sys singe-npp singe-npp-sys singe-nvml singe-nvml-sys singe-cusolver singe-cusolver-sys singe-cusparse singe-cusparse-sys; do
        (cd ./$crate && just -f justfile clean)
    done
