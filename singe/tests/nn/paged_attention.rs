// Paging values come from tensors, so every backend refuses ones that would address outside the queries or the caches.

#[cfg(all(feature = "testing", feature = "cuda"))]
use std::collections::HashMap;

#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::graph::{Graph, GraphPrepareOptions};
#[cfg(all(feature = "testing", feature = "cuda"))]
use singe::testing::maybe_lock_device;
use singe::{
    DataType, Device, Error, Result, Shape, Tensor,
    tensor::{AttentionOptions, Paging as TensorPaging},
};

const TOKENS: i64 = 2;
const HEADS: i64 = 2;
const DIM: i64 = 16;
const SLOTS: i64 = 4;
const PAGES: i64 = 2;

// The fused kernel takes latent rows of a multiple of 32 values.
const MLA_DIM: i64 = 32;

// One sequence of two queries over two keys in page zero.
const VALID: Paging = Paging {
    query_offsets: [0, 2],
    page_offsets: [0, 1],
    pages: [0],
    last_page_length: [2],
};

// Each of these reads or writes outside some buffer.
const INVALID: [Paging; 9] = [
    Paging {
        query_offsets: [0, 3],
        ..VALID
    },
    Paging {
        query_offsets: [-1, 2],
        ..VALID
    },
    Paging {
        query_offsets: [2, 0],
        ..VALID
    },
    Paging {
        page_offsets: [0, 2],
        ..VALID
    },
    Paging {
        page_offsets: [-1, 1],
        ..VALID
    },
    Paging {
        pages: [PAGES],
        ..VALID
    },
    Paging {
        pages: [-1],
        ..VALID
    },
    Paging {
        last_page_length: [SLOTS + 1],
        ..VALID
    },
    Paging {
        last_page_length: [-1],
        ..VALID
    },
];

#[derive(Debug, Clone, Copy)]
struct Paging {
    query_offsets: [i64; 2],
    page_offsets: [i64; 2],
    pages: [i64; 1],
    last_page_length: [i64; 1],
}

impl Paging {
    fn tensors(self, device: Device) -> [Tensor; 4] {
        let vector = |values: &[i64]| {
            Tensor::from_vec(
                device,
                Shape::create([i64::try_from(values.len()).unwrap()]).unwrap(),
                values.to_vec(),
            )
            .unwrap()
        };
        [
            vector(&self.query_offsets),
            vector(&self.page_offsets),
            vector(&self.pages),
            vector(&self.last_page_length),
        ]
    }
}

fn zeros(device: Device, data_type: DataType, dimensions: &[i64]) -> Tensor {
    Tensor::zeroed_on(device, data_type, Shape::create(dimensions).unwrap()).unwrap()
}

fn options() -> AttentionOptions {
    AttentionOptions {
        causal: true,
        ..AttentionOptions::default()
    }
}

fn attend(device: Device, data_type: DataType, paging: Paging) -> Result<(Tensor, Tensor)> {
    let queries = zeros(device, data_type, &[TOKENS, HEADS, DIM]);
    let cache = zeros(device, data_type, &[PAGES, SLOTS, HEADS, DIM]);
    let [query_offsets, page_offsets, pages, last] = paging.tensors(device);
    queries.paged_attention(
        &cache,
        &cache,
        TensorPaging {
            query_offsets: &query_offsets,
            page_offsets: &page_offsets,
            pages: &pages,
            last_page_lengths: &last,
        },
        options(),
    )
}

fn attend_mla(device: Device, data_type: DataType, paging: Paging) -> Result<(Tensor, Tensor)> {
    let queries = zeros(device, data_type, &[TOKENS, HEADS, MLA_DIM]);
    let cache = zeros(device, data_type, &[PAGES, SLOTS, MLA_DIM]);
    let [query_offsets, page_offsets, pages, last] = paging.tensors(device);
    queries.paged_mla_attention(
        &queries,
        &cache,
        &cache,
        TensorPaging {
            query_offsets: &query_offsets,
            page_offsets: &page_offsets,
            pages: &pages,
            last_page_lengths: &last,
        },
        options(),
    )
}

fn check_refusals(device: Device, data_type: DataType) {
    attend(device, data_type, VALID).unwrap();
    attend_mla(device, data_type, VALID).unwrap();
    for paging in INVALID {
        assert!(
            matches!(
                attend(device, data_type, paging),
                Err(Error::IndexOutOfBounds)
            ),
            "{paging:?}"
        );
        assert!(
            matches!(
                attend_mla(device, data_type, paging),
                Err(Error::IndexOutOfBounds)
            ),
            "mla {paging:?}"
        );
    }
    // A last page holding every slot is in range, and an empty sequence reads no pages.
    attend(
        device,
        data_type,
        Paging {
            last_page_length: [SLOTS],
            ..VALID
        },
    )
    .unwrap();
    attend(
        device,
        data_type,
        Paging {
            query_offsets: [0, 0],
            page_offsets: [0, 0],
            pages: [PAGES + 7],
            last_page_length: [-1],
        },
    )
    .unwrap();
}

#[test]
fn test_cpu_refuses_paging_values_outside_its_buffers() {
    check_refusals(Device::Cpu, DataType::F32);
}

#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_cuda_refuses_paging_values_outside_its_buffers() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    check_refusals(Device::Cuda(0), DataType::Bf16);
}

// A compiled graph computes paging values at run time, so its status word reports them after the replay, and a later replay recovers.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[test]
fn test_compiled_graphs_refuse_paging_values_at_run_time() {
    let Some(_lock) = maybe_lock_device() else {
        return;
    };
    let device = Device::Cuda(0);
    let queries = zeros(device, DataType::Bf16, &[TOKENS, HEADS, DIM]);
    let cache = zeros(device, DataType::Bf16, &[PAGES, SLOTS, HEADS, DIM]);
    let paging_inputs = [
        "query_offsets",
        "page_offsets",
        "pages",
        "last_page_lengths",
    ];
    let named = |paging: Paging| {
        paging_inputs
            .into_iter()
            .zip(paging.tensors(device))
            .map(|(name, tensor)| (name.to_owned(), tensor))
    };
    let inputs = [
        ("queries".to_owned(), queries.clone()),
        ("cache".to_owned(), cache.clone()),
    ]
    .into_iter()
    .chain(named(VALID));
    let (graph, _) = Graph::capture(inputs, |inputs| {
        Ok(vec![
            inputs[0]
                .paged_attention(
                    &inputs[1],
                    &inputs[1],
                    TensorPaging {
                        query_offsets: &inputs[2],
                        page_offsets: &inputs[3],
                        pages: &inputs[4],
                        last_page_lengths: &inputs[5],
                    },
                    options(),
                )?
                .0,
        ])
    })
    .unwrap();
    let mut executor = graph.prepare(GraphPrepareOptions::new(device)).unwrap();
    let run_inputs = |paging: Paging| {
        [
            ("queries".to_owned(), queries.clone()),
            ("cache".to_owned(), cache.clone()),
        ]
        .into_iter()
        .chain(named(paging))
        .collect::<HashMap<_, _>>()
    };
    executor.run(&run_inputs(VALID)).unwrap();
    for paging in INVALID {
        assert!(
            matches!(
                executor.run(&run_inputs(paging)),
                Err(Error::IndexOutOfBounds)
            ),
            "{paging:?}"
        );
        executor.run(&run_inputs(VALID)).unwrap();
    }
}
