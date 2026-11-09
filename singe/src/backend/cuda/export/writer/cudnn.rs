use std::iter::repeat_n;

use singe_cudnn::backend::execution::EngineChoice;

use crate::{
    backend::cuda::export::{
        program::{Activation, BufferView, CudnnAttentionCall, CudnnLengths, CudnnPagedCall},
        writer::{CUDNN_BASE, LibraryCode, Renderer, alignment, cudnn_type, integers},
    },
    data_type::DataType,
    error::Result,
    graph::{ComputePrecision, Determinism},
    utility::{emit, to_i64},
};

// A cuDNN tensor descriptor named `tensor_<name>`.
fn tensor_code(
    name: &str,
    uid: usize,
    shape: (&[i64], &[i64]),
    data_type: DataType,
    alignment: i64,
    is_virtual: bool,
) -> Result<String> {
    tensor_code_with(name, uid, shape, data_type, alignment, is_virtual, "")
}

// A cuDNN tensor descriptor, with `extra` passed on to the helper after `is_virtual`.
fn tensor_code_with(
    name: &str,
    uid: usize,
    shape: (&[i64], &[i64]),
    data_type: DataType,
    alignment: i64,
    is_virtual: bool,
    extra: &str,
) -> Result<String> {
    let (dimensions, strides) = shape;
    Ok(format!(
        "        const int64_t dimensions_{name}[] = {{{}}};\n        const int64_t strides_{name}[] = {{{}}};\n        cudnnBackendDescriptor_t tensor_{name};\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_tensor(&tensor_{name}, {uid}, {}, {}, dimensions_{name}, strides_{name}, {alignment}, {is_virtual}{extra}));\n",
        integers(dimensions.iter().copied()),
        integers(strides.iter().copied()),
        cudnn_type(data_type)?,
        dimensions.len(),
    ))
}

// What holds, runs, and releases a planned cuDNN graph whose `create` plans it and binds it to `pointers`.
fn plan_code(index: usize, create: String, pointers: &[String], bound_count: usize) -> LibraryCode {
    let uids = (0..bound_count)
        .map(|uid| uid.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    LibraryCode {
        members: format!(
            "    cudnnBackendDescriptor_t dnn_plan_{index};\n    cudnnBackendDescriptor_t dnn_pack_{index};\n    void* dnn_workspace_{index};\n    int64_t dnn_workspace_bytes_{index};\n"
        ),
        create,
        run: format!(
            "    {{\n        // A variant pack finalizes once, so each run has its own, which lives until the next run has no use for the last.\n        if (s->dnn_pack_{index} != nullptr) cudnnBackendDestroyDescriptor(s->dnn_pack_{index});\n        s->dnn_pack_{index} = nullptr;\n        const int64_t uids[] = {{{uids}}};\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &s->dnn_pack_{index}));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(s->dnn_pack_{index}, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, {bound_count}, uids));\n        void* pointers[] = {{{pointers}}};\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(s->dnn_pack_{index}, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, {bound_count}, pointers));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(s->dnn_pack_{index}, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &s->dnn_workspace_{index}));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendFinalize(s->dnn_pack_{index}));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnSetStream(s->cudnn, stream));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendExecute(s->cudnn, s->dnn_plan_{index}, s->dnn_pack_{index}));\n    }}\n",
            pointers = pointers.join(", "),
        ),
        destroy: format!(
            "    if (s->dnn_workspace_{index} != nullptr) cudaFree(s->dnn_workspace_{index});\n    if (s->dnn_pack_{index} != nullptr) cudnnBackendDestroyDescriptor(s->dnn_pack_{index});\n    if (s->dnn_plan_{index} != nullptr) cudnnBackendDestroyDescriptor(s->dnn_plan_{index});\n"
        ),
    }
}

// How many knobs a record may state and still be read from a table.
const TABLE_KNOBS: usize = 8;

// Reads the knobs `forced_engine_arguments` puts in a table: their count, then each knob's type and value.
pub const TABLE_KNOBS_HELPER: &str = "static std::vector<std::pair<int64_t, int64_t>> singe_table_knobs(int at) {\n    std::vector<std::pair<int64_t, int64_t>> knobs;\n    for (int64_t knob = 0; knob < singe_library_table[at]; ++knob) knobs.emplace_back(singe_library_table[at + 1 + 2 * knob], singe_library_table[at + 2 + 2 * knob]);\n    return knobs;\n}\n";

impl Renderer<'_> {
    // A cuDNN call: its first operation, which `first` writes, then the epilogue that leads to `output`.
    pub fn cudnn_call(
        &self,
        index: usize,
        operands: &[(&str, &BufferView)],
        bias: Option<&(&BufferView, Vec<i64>, Vec<i64>)>,
        activation: Option<Activation>,
        output: &BufferView,
        no_default_engine: bool,
        engine: Option<&EngineChoice>,
        rank: usize,
        first: impl FnOnce(&str, &str) -> Result<(String, Vec<&'static str>)>,
    ) -> Result<LibraryCode> {
        let compute = if output.data_type == DataType::F64 {
            DataType::F64
        } else {
            DataType::F32
        };
        let shape_of =
            |view: &BufferView| promoted(view.layout.dimensions(), view.layout.strides(), rank);
        let bound = |name: &str, uid: usize, view: &BufferView| {
            let (dimensions, strides) = shape_of(view);
            tensor_code(
                name,
                uid,
                (&dimensions, &strides),
                view.data_type,
                alignment(view),
                false,
            )
        };
        let mut create = String::from("    {\n");
        let mut tensors = Vec::new();
        let mut pointers = Vec::new();
        for (name, view) in operands {
            create.push_str(&bound(name, tensors.len(), view)?);
            tensors.push(format!("tensor_{name}"));
            pointers.push(self.address(view)?);
        }

        create.push_str(&bound("y", tensors.len(), output)?);
        tensors.push("tensor_y".to_owned());
        pointers.push(self.address(output)?);
        if let Some((view, dimensions, strides)) = bias {
            create.push_str(&tensor_code(
                "b",
                tensors.len(),
                (dimensions, strides),
                view.data_type,
                alignment(view),
                false,
            )?);
            tensors.push("tensor_b".to_owned());
            pointers.push(self.address(view)?);
        }

        let bound_count = tensors.len();
        let size = to_i64(compute.size_bytes(), "cudnn compute type size")?;
        // Values between the operations hold the compute type, laid out as the result.
        let (result_dimensions, result_strides) = shape_of(output);
        let intermediate = |name: &str, uid: usize| {
            tensor_code(
                name,
                uid,
                (&result_dimensions, &result_strides),
                compute,
                size,
                true,
            )
        };
        let (mut virtuals, mut next) = (String::new(), bound_count);
        let (first_output, biased_output) = match (bias.is_some(), activation.is_some()) {
            (false, false) => ("tensor_y", "tensor_y"),
            (true, false) => {
                virtuals.push_str(&intermediate("v", next)?);
                ("tensor_v", "tensor_y")
            }
            (false, true) => {
                virtuals.push_str(&intermediate("v", next)?);
                ("tensor_v", "tensor_v")
            }
            (true, true) => {
                virtuals.push_str(&intermediate("v", next)?);
                next += 1;
                virtuals.push_str(&intermediate("w_v", next)?);
                ("tensor_v", "tensor_w_v")
            }
        };
        create.push_str(&virtuals);
        let compute_name = cudnn_type(compute)?;
        emit!(
            create,
            "        const cudnnDataType_t compute = {compute_name};"
        );
        let (forward, mut retained) = first(first_output, compute_name)?;
        create.push_str(&forward);
        retained.push("forward");
        let mut operations = vec!["forward"];
        if bias.is_some() {
            emit!(
                create,
                "        cudnnBackendDescriptor_t biased;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_pointwise(&s->cudnn_retained, CUDNN_POINTWISE_ADD, compute, {first_output}, tensor_b, {biased_output}, &biased));"
            );
            operations.push("biased");
        }

        if let Some(activation) = activation {
            let mode = match activation {
                Activation::Relu => "CUDNN_POINTWISE_RELU_FWD",
                Activation::Gelu => "CUDNN_POINTWISE_GELU_APPROX_TANH_FWD",
            };
            emit!(
                create,
                "        cudnnBackendDescriptor_t activated;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_pointwise(&s->cudnn_retained, {mode}, compute, {biased_output}, nullptr, tensor_y, &activated));"
            );
            operations.push("activated");
        }

        let tensor_core = output.data_type != DataType::F32
            || self.program.numerics.precision() != ComputePrecision::Full;
        let deterministic = self.program.numerics.determinism() == Determinism::Require;
        // Each descriptor is retained once, since it is destroyed once.
        let mut intermediates = vec![first_output];
        if biased_output != first_output {
            intermediates.push(biased_output);
        }

        intermediates.retain(|name| *name != "tensor_y");
        let retained = tensors
            .iter()
            .map(String::as_str)
            .chain(retained)
            .chain(intermediates)
            .collect::<Vec<_>>()
            .join(", ");
        let (forced_engine, forced_knobs) = self.forced_engine_arguments(engine);
        emit!(
            create,
            "        cudnnBackendDescriptor_t operations[] = {{{operations}}};\n        for (cudnnBackendDescriptor_t retained : {{{retained}}}) s->cudnn_retained.push_back(retained);\n        SINGE_CUDNN_PLAN(singe_cudnn_plan(s->cudnn, operations, sizeof(operations) / sizeof(operations[0]), {tensor_core}, {deterministic}, {no_default_engine}, {forced_engine}, {forced_knobs}, &s->cudnn_retained, &s->dnn_plan_{index}, &s->dnn_workspace_bytes_{index}));\n        if (s->dnn_workspace_bytes_{index} > 0) SINGE_CHECK(cudaMalloc(&s->dnn_workspace_{index}, s->dnn_workspace_bytes_{index}));\n    }}",
            operations = operations.join(", ")
        );
        Ok(plan_code(index, create, &pointers, bound_count))
    }

    // The engine a program's `singe_cudnn_plan` tries first, and its knobs, as C++ arguments: -1 and nothing to try the heuristics alone.
    fn forced_engine_arguments(&self, engine: Option<&EngineChoice>) -> (String, String) {
        let Some(choice) = engine else {
            return ("-1".to_owned(), "{}".to_owned());
        };

        self.forced.borrow_mut().count += 1;
        let forced_engine = self.forced_value(choice.engine_index.as_i64());
        if !self.program.library_table || choice.knob_choices.len() > TABLE_KNOBS {
            let knobs = choice
                .knob_choices
                .iter()
                .map(|knob| format!("{{{}, {}}}", knob.knob_type.raw(), knob.value))
                .collect::<Vec<_>>()
                .join(", ");
            return (forced_engine, format!("{{{knobs}}}"));
        }

        // A fixed room for knobs, so choices with different counts share one source.
        let at = self.forced.borrow().table.len();
        let count = i64::try_from(choice.knob_choices.len()).unwrap_or(0);
        self.forced_value(count);
        for index in 0..TABLE_KNOBS {
            let knob = choice.knob_choices.get(index);
            self.forced_value(knob.map_or(0, |knob| knob.knob_type.raw().into()));
            self.forced_value(knob.map_or(0, |knob| knob.value));
        }
        (forced_engine, format!("singe_table_knobs({at})"))
    }

    // Attention through cuDNN: bound tensors first, then each mask's scores, then the attention that holds them.
    pub fn cudnn_attention(&self, index: usize, call: &CudnnAttentionCall) -> Result<LibraryCode> {
        let CudnnAttentionCall {
            query,
            key,
            value,
            output,
            scale,
            causal,
            window,
            paged,
            lengths,
            engine,
        } = call;
        let mut create = String::from("    {\n");
        let mut members = String::new();
        let mut names = Vec::new();
        let mut pointers = Vec::new();
        let packed_extra = if paged.is_some() {
            ", tensor_offsets"
        } else {
            ""
        };
        if let Some(paged) = paged {
            let CudnnPagedCall {
                paging,
                offsets,
                query_lengths,
                key_lengths,
                pages,
                stats,
            } = paged;
            let mut index_tensor = |name: &'static str,
                                    view: &BufferView,
                                    shape: ([i64; 4], [i64; 4])|
             -> Result<()> {
                create.push_str(&tensor_code(
                    name,
                    names.len(),
                    (&shape.0, &shape.1),
                    view.data_type,
                    alignment(view),
                    false,
                )?);
                names.push(format!("tensor_{name}"));
                pointers.push(self.address(view)?);
                Ok(())
            };
            index_tensor("offsets", offsets, paging.offsets_shape())?;
            index_tensor("query_lengths", query_lengths, paging.lengths_shape())?;
            index_tensor("key_lengths", key_lengths, paging.lengths_shape())?;
            index_tensor("key_pages", pages, paging.table_shape())?;
            index_tensor("value_pages", pages, paging.table_shape())?;
            index_tensor("stats", stats, paging.stats_shape())?;
        }

        // Dense attention's lengths are `[B]` to Singe, and `[B, 1, 1, 1]` to cuDNN.
        if let Some(CudnnLengths {
            query_lengths,
            key_lengths,
        }) = lengths
        {
            let shape = ([query.layout.dimensions()[0], 1, 1, 1], [1, 1, 1, 1]);
            for (name, view) in [
                ("query_lengths", query_lengths),
                ("key_lengths", key_lengths),
            ] {
                create.push_str(&tensor_code(
                    name,
                    names.len(),
                    (&shape.0, &shape.1),
                    view.data_type,
                    alignment(view),
                    false,
                )?);
                names.push(format!("tensor_{name}"));
                pointers.push(self.address(view)?);
            }
        }

        let packed_shape = paged.as_ref().map(|paged| paged.paging.packed_shape());
        let cache_shape = paged.as_ref().map(|paged| paged.paging.cache_shape());
        let mut operand = |name: &'static str,
                           view: &BufferView,
                           shape: Option<([i64; 4], [i64; 4])>,
                           extra: &str|
         -> Result<()> {
            let dimensions =
                shape.map_or_else(|| view.layout.dimensions().to_vec(), |s| s.0.to_vec());
            let strides = shape.map_or_else(|| view.layout.strides().to_vec(), |s| s.1.to_vec());
            create.push_str(&tensor_code_with(
                name,
                names.len(),
                (&dimensions, &strides),
                view.data_type,
                alignment(view),
                false,
                extra,
            )?);
            names.push(format!("tensor_{name}"));
            pointers.push(self.address(view)?);
            Ok(())
        };
        operand("q", query, packed_shape, packed_extra)?;
        operand("k", key, cache_shape, "")?;
        operand("v", value, cache_shape, "")?;
        operand("o", output, packed_shape, packed_extra)?;
        emit!(members, "    float dnn_scale_{index};");
        emit!(create, "        s->dnn_scale_{index} = {scale:?}f;");
        let by_value = ", nullptr, true";
        let scalar = |name: &str, uid: usize, data_type: DataType, rank: usize| {
            let unit = vec![1; rank];
            tensor_code_with(name, uid, (&unit, &unit), data_type, 4, false, by_value)
        };
        create.push_str(&scalar("scale", names.len(), DataType::F32, 1)?);
        names.push("tensor_scale".to_owned());
        pointers.push(format!("&s->dnn_scale_{index}"));
        let masked = *causal || window.is_some();
        let mut masks = Vec::new();
        let mut subgraph = ("nullptr".to_owned(), 0, 0);
        if masked {
            emit!(members, "    float dnn_infinity_{index};");
            emit!(
                create,
                "        s->dnn_infinity_{index} = -std::numeric_limits<float>::infinity();"
            );
            create.push_str(&scalar("infinity", names.len(), DataType::F32, 4)?);
            names.push("tensor_infinity".to_owned());
            pointers.push(format!("&s->dnn_infinity_{index}"));
            if let Some(window) = window {
                // A mask aligned to each batch's lengths counts in integers, and one aligned to the top left in floats.
                let (member, value, data_type) = if paged.is_some() {
                    ("int32_t", format!("{window}"), DataType::I32)
                } else {
                    ("float", format!("{window}.0f"), DataType::F32)
                };
                emit!(members, "    {member} dnn_window_{index};");
                emit!(create, "        s->dnn_window_{index} = {value};");
                create.push_str(&scalar("window", names.len(), data_type, 4)?);
                names.push("tensor_window".to_owned());
                pointers.push(format!("&s->dnn_window_{index}"));
            }
        }

        let bound_count = names.len();
        if masked {
            let scores = paged.as_ref().map_or_else(
                || {
                    [
                        query.layout.dimensions()[0],
                        query.layout.dimensions()[1],
                        query.layout.dimensions()[2],
                        key.layout.dimensions()[2],
                    ]
                },
                |paged| paged.paging.scores(),
            );
            let strides = [
                scores[1] * scores[2] * scores[3],
                scores[2] * scores[3],
                scores[3],
                1,
            ];
            let score = |name: &str, uid: usize| {
                tensor_code(name, uid, (&scores, &strides), DataType::F32, 4, true)
            };
            let first = names.len();
            create.push_str(&score("scores", first)?);
            names.push("tensor_scores".to_owned());
            let lengths = if paged.is_some() {
                "tensor_query_lengths, tensor_key_lengths"
            } else {
                "nullptr, nullptr"
            };
            let mut current = "tensor_scores";
            if *causal {
                let uid = names.len();
                create.push_str(&score("causal", uid)?);
                names.push("tensor_causal".to_owned());
                emit!(
                    create,
                    "        cudnnBackendDescriptor_t causal;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_band(&s->cudnn_retained, CUDNN_POINTWISE_CMP_GE, {current}, tensor_infinity, nullptr, {lengths}, tensor_causal, &causal));"
                );
                masks.push("causal");
                current = "tensor_causal";
            }

            if window.is_some() {
                let uid = names.len();
                create.push_str(&score("windowed", uid)?);
                names.push("tensor_windowed".to_owned());
                emit!(
                    create,
                    "        cudnnBackendDescriptor_t windowed;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_band(&s->cudnn_retained, CUDNN_POINTWISE_CMP_GT, {current}, tensor_infinity, tensor_window, {lengths}, tensor_windowed, &windowed));"
                );
                masks.push("windowed");
            }

            emit!(
                create,
                "        cudnnBackendDescriptor_t masks[] = {{{masks}}};\n        cudnnBackendDescriptor_t mask_graph;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_operation_graph(s->cudnn, masks, sizeof(masks) / sizeof(masks[0]), &s->cudnn_retained, &mask_graph));",
                masks = masks.join(", ")
            );
            subgraph = ("mask_graph".to_owned(), first, names.len() - 1);
        }

        let (mask_graph, scores_in, scores_out) = subgraph;
        let paged_arguments = if paged.is_some() {
            "tensor_stats, tensor_query_lengths, tensor_key_lengths, tensor_key_pages, tensor_value_pages"
        } else if lengths.is_some() {
            "nullptr, tensor_query_lengths, tensor_key_lengths, nullptr, nullptr"
        } else {
            "nullptr, nullptr, nullptr, nullptr, nullptr"
        };
        emit!(
            create,
            "        cudnnBackendDescriptor_t attention;\n        SINGE_LIBRARY({CUDNN_BASE}, singe_cudnn_attention(&s->cudnn_retained, tensor_q, tensor_k, tensor_v, tensor_o, tensor_scale, {paged_arguments}, {mask_graph}, {scores_in}, {scores_out}, &attention));"
        );
        let tensor_core = output.data_type != DataType::F32
            || self.program.numerics.precision() != ComputePrecision::Full;
        let deterministic = self.program.numerics.determinism() == Determinism::Require;
        let (forced_engine, forced_knobs) = self.forced_engine_arguments(engine.as_ref());
        emit!(
            create,
            "        cudnnBackendDescriptor_t operations[] = {{attention}};\n        for (cudnnBackendDescriptor_t retained : {{{retained}}}) s->cudnn_retained.push_back(retained);\n        SINGE_CUDNN_PLAN(singe_cudnn_plan(s->cudnn, operations, 1, {tensor_core}, {deterministic}, false, {forced_engine}, {forced_knobs}, &s->cudnn_retained, &s->dnn_plan_{index}, &s->dnn_workspace_bytes_{index}));\n        if (s->dnn_workspace_bytes_{index} > 0) SINGE_CHECK(cudaMalloc(&s->dnn_workspace_{index}, s->dnn_workspace_bytes_{index}));\n    }}",
            retained = names.join(", ")
        );
        let mut code = plan_code(index, create, &pointers, bound_count);
        code.members.push_str(&members);
        Ok(code)
    }

    pub fn convolution(
        &self,
        index: usize,
        operands: (&BufferView, &BufferView, Option<&BufferView>, &BufferView),
        activation: Option<Activation>,
        stride: &[i64],
        padding: &[(i64, i64)],
        dilation: &[i64],
    ) -> Result<LibraryCode> {
        let (input, weight, bias, output) = operands;
        let rank = stride.len();
        // A bias holds a value per channel, broadcast over the batch and the spatial axes, and the sum rounds once.
        let bias = bias.map(|bias| {
            let channels = output.layout.dimensions()[1];
            let stride = bias.layout.strides().first().copied().unwrap_or(1);
            let span = channels * stride.max(1);
            let dimensions = [1, channels]
                .into_iter()
                .chain(repeat_n(1, rank))
                .collect::<Vec<_>>();
            let strides = [span, stride]
                .into_iter()
                .chain(repeat_n(span, rank))
                .collect::<Vec<_>>();
            (bias, dimensions, strides)
        });
        let (before, after): (Vec<_>, Vec<_>) = padding.iter().copied().unzip();
        let (stride, before, after, dilation) = (
            integers(stride.iter().copied()),
            integers(before),
            integers(after),
            integers(dilation.iter().copied()),
        );
        let (scale, scale_type) = if output.data_type == DataType::F64 {
            ("double", "CUDNN_TYPE_DOUBLE")
        } else {
            ("float", "CUDNN_TYPE_FLOAT")
        };
        self.cudnn_call(
            index,
            &[("x", input), ("w", weight)],
            bias.as_ref(),
            activation,
            output,
            false,
            None,
            0,
            |first, _| {
                Ok((
                    format!(
                        "        const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;\n        const int64_t spatial = {rank};\n        const int64_t stride[] = {{{stride}}};\n        const int64_t before[] = {{{before}}};\n        const int64_t after[] = {{{after}}};\n        const int64_t dilation[] = {{{dilation}}};\n        cudnnBackendDescriptor_t convolution;\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &convolution));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, {rank}, stride));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, {rank}, before));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, {rank}, after));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(convolution, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, {rank}, dilation));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendFinalize(convolution));\n        const {scale} one = 1, zero = 0;\n        cudnnBackendDescriptor_t forward;\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, &forward));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, {scale_type}, 1, &one));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, {scale_type}, 1, &zero));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_x));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_w));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &{first}));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendFinalize(forward));\n"
                    ),
                    vec!["convolution"],
                ))
            },
        )
    }

    pub fn cudnn_matmul(
        &self,
        index: usize,
        operands: (&BufferView, &BufferView, Option<&BufferView>, &BufferView),
        activation: Option<Activation>,
        engine: Option<&EngineChoice>,
    ) -> Result<LibraryCode> {
        let (lhs, rhs, addend, output) = operands;
        let addend = addend.map(|addend| {
            let (dimensions, strides) = promoted(
                addend.layout.dimensions(),
                addend.layout.strides(),
                MATMUL_RANK,
            );
            (addend, dimensions, strides)
        });
        self.cudnn_call(
            index,
            &[("a", lhs), ("c", rhs)],
            addend.as_ref(),
            activation,
            output,
            // cuDNN's default matmul engine can round `F32` inputs to `TF32` without noting it.
            output.data_type == DataType::F32
                && self.program.numerics.precision() == ComputePrecision::Full,
            engine,
            MATMUL_RANK,
            |first, _| {
                Ok((
                    format!(
                        "        cudnnBackendDescriptor_t matmul;\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &matmul));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(matmul, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendFinalize(matmul));\n        cudnnBackendDescriptor_t forward;\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &forward));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &matmul));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_a));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor_c));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendSetAttribute(forward, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &{first}));\n        SINGE_LIBRARY({CUDNN_BASE}, cudnnBackendFinalize(forward));\n"
                    ),
                    vec!["matmul"],
                ))
            },
        )
    }
}

// cuDNN plans matrix products of at least this rank.
const MATMUL_RANK: usize = 3;

// A shape with leading unit axes up to `rank`, which leave every element where it was.
fn promoted(dimensions: &[i64], strides: &[i64], rank: usize) -> (Vec<i64>, Vec<i64>) {
    let leading = rank.saturating_sub(dimensions.len());
    let span = dimensions
        .first()
        .zip(strides.first())
        .map_or(1, |(dimension, stride)| dimension.saturating_mul(*stride));
    (
        repeat_n(1, leading)
            .chain(dimensions.iter().copied())
            .collect(),
        repeat_n(span, leading)
            .chain(strides.iter().copied())
            .collect(),
    )
}
