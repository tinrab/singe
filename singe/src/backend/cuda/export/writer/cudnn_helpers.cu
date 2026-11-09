static cudnnStatus_t singe_cudnn_tensor(cudnnBackendDescriptor_t* tensor, int64_t uid, cudnnDataType_t type, int rank, const int64_t* dimensions, const int64_t* strides, int64_t alignment, bool is_virtual, cudnnBackendDescriptor_t ragged = nullptr, bool by_value = false) {
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, tensor);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &type);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, rank, dimensions);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, rank, strides);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, 1, &by_value);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (ragged != nullptr) {
        status = cudnnBackendSetAttribute(*tensor, CUDNN_ATTR_TENSOR_RAGGED_OFFSET_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &ragged);
        if (status != CUDNN_STATUS_SUCCESS) return status;
    }
    return cudnnBackendFinalize(*tensor);
}

static cudnnStatus_t singe_cudnn_pointwise(std::vector<cudnnBackendDescriptor_t>* retained, cudnnPointwiseMode_t mode, cudnnDataType_t compute, cudnnBackendDescriptor_t x, cudnnBackendDescriptor_t b, cudnnBackendDescriptor_t y, cudnnBackendDescriptor_t* operation) {
    const cudnnNanPropagation_t propagate = CUDNN_PROPAGATE_NAN;
    cudnnBackendDescriptor_t pointwise;
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &pointwise);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(pointwise);
    status = cudnnBackendSetAttribute(pointwise, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(pointwise, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &compute);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(pointwise, CUDNN_ATTR_POINTWISE_NAN_PROPAGATION, CUDNN_TYPE_NAN_PROPOGATION, 1, &propagate);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendFinalize(pointwise);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, operation);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(*operation);
    status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pointwise);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (b != nullptr) {
        status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_POINTWISE_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &b);
        if (status != CUDNN_STATUS_SUCCESS) return status;
    }
    status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    return cudnnBackendFinalize(*operation);
}

// Hides scores outside a diagonal band: `left` is a left bound, and without one the mask is the right bound on the diagonal itself.
static cudnnStatus_t singe_cudnn_band(std::vector<cudnnBackendDescriptor_t>* retained, cudnnPointwiseMode_t mode, cudnnBackendDescriptor_t x, cudnnBackendDescriptor_t b, cudnnBackendDescriptor_t left, cudnnBackendDescriptor_t query_lengths, cudnnBackendDescriptor_t key_lengths, cudnnBackendDescriptor_t y, cudnnBackendDescriptor_t* operation) {
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_DIAGONAL_BAND_MASK_DESCRIPTOR, operation);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(*operation);
    const auto set = [&](cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t value) {
        if (status == CUDNN_STATUS_SUCCESS && value != nullptr) status = cudnnBackendSetAttribute(*operation, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &value);
    };
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_XDESC, x);
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_BDESC, b);
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_QDESC, query_lengths);
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_SEQ_LEN_KVDESC, key_lengths);
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_LEFT_BOUND_DESC, left);
    set(CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_YDESC, y);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_DIAGONAL_BAND_MASK_COMPARISON_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &mode);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    return cudnnBackendFinalize(*operation);
}

// Attention, whose paged operands and score graph may be null.
static cudnnStatus_t singe_cudnn_attention(std::vector<cudnnBackendDescriptor_t>* retained, cudnnBackendDescriptor_t q, cudnnBackendDescriptor_t k, cudnnBackendDescriptor_t v, cudnnBackendDescriptor_t o, cudnnBackendDescriptor_t scale, cudnnBackendDescriptor_t stats, cudnnBackendDescriptor_t query_lengths, cudnnBackendDescriptor_t key_lengths, cudnnBackendDescriptor_t key_pages, cudnnBackendDescriptor_t value_pages, cudnnBackendDescriptor_t scores, int64_t scores_in, int64_t scores_out, cudnnBackendDescriptor_t* operation) {
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR, operation);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(*operation);
    const auto set = [&](cudnnBackendAttributeName_t name, cudnnBackendDescriptor_t value) {
        if (status == CUDNN_STATUS_SUCCESS && value != nullptr) status = cudnnBackendSetAttribute(*operation, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &value);
    };
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_QDESC, q);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_KDESC, k);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_VDESC, v);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_ODESC, o);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC, scale);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC, stats);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC, query_lengths);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC, key_lengths);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC, key_pages);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC, value_pages);
    set(CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH, scores);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (scores != nullptr) {
        status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_INPUT_UID, CUDNN_TYPE_INT64, 1, &scores_in);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        status = cudnnBackendSetAttribute(*operation, CUDNN_ATTR_OPERATION_SDPA_FWD_SUBGRAPH_OUTPUT_UID, CUDNN_TYPE_INT64, 1, &scores_out);
        if (status != CUDNN_STATUS_SUCCESS) return status;
    }
    return cudnnBackendFinalize(*operation);
}

static cudnnStatus_t singe_cudnn_operation_graph(cudnnHandle_t handle, cudnnBackendDescriptor_t* operations, unsigned long long count, std::vector<cudnnBackendDescriptor_t>* retained, cudnnBackendDescriptor_t* graph) {
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, graph);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(*graph);
    status = cudnnBackendSetAttribute(*graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, count, operations);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    return cudnnBackendFinalize(*graph);
}

static bool singe_cudnn_accepts(cudnnBackendDescriptor_t config, bool tensor_core, bool deterministic, bool no_default_engine) {
    cudnnBackendDescriptor_t engine;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine) != CUDNN_STATUS_SUCCESS) return false;
    cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
    int64_t count = 0, index = -1, written = 0;
    cudnnStatus_t status = cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &written, &index);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE, CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT, &count, notes);
    cudnnBackendDestroyDescriptor(engine);
    if (status != CUDNN_STATUS_SUCCESS) return false;
    if (no_default_engine && index == 0) return false;
    for (int64_t note = 0; note < count; ++note) {
        if (notes[note] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS || notes[note] == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION) return false;
        if (!tensor_core && notes[note] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE) return false;
        if (deterministic && notes[note] == CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC) return false;
    }
    return true;
}

// Plans the engine and knobs a record states, if cuDNN takes them for this graph and they keep the program's numerical policy.
static cudnnStatus_t singe_cudnn_forced_plan(cudnnHandle_t handle, cudnnBackendDescriptor_t graph, int64_t forced_engine, const std::vector<std::pair<int64_t, int64_t>>& forced_knobs, bool tensor_core, bool deterministic, bool no_default_engine, std::vector<cudnnBackendDescriptor_t>* retained, cudnnBackendDescriptor_t* plan, int64_t* workspace) {
    cudnnBackendDescriptor_t engine;
    cudnnStatus_t status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(engine);
    status = cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &forced_engine);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendFinalize(engine);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    cudnnBackendDescriptor_t config;
    status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &config);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    retained->push_back(config);
    status = cudnnBackendSetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    std::vector<cudnnBackendDescriptor_t> knobs;
    for (const std::pair<int64_t, int64_t>& forced : forced_knobs) {
        cudnnBackendDescriptor_t knob;
        status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, &knob);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        retained->push_back(knob);
        const cudnnBackendKnobType_t type = static_cast<cudnnBackendKnobType_t>(forced.first);
        status = cudnnBackendSetAttribute(knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &type);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendSetAttribute(knob, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &forced.second);
        if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendFinalize(knob);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        knobs.push_back(knob);
    }
    if (!knobs.empty()) {
        status = cudnnBackendSetAttribute(config, CUDNN_ATTR_ENGINECFG_KNOB_CHOICES, CUDNN_TYPE_BACKEND_DESCRIPTOR, static_cast<int64_t>(knobs.size()), knobs.data());
        if (status != CUDNN_STATUS_SUCCESS) return status;
    }
    status = cudnnBackendFinalize(config);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (!singe_cudnn_accepts(config, tensor_core, deterministic, no_default_engine)) return CUDNN_STATUS_NOT_SUPPORTED;
    status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, plan);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    status = cudnnBackendSetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendSetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &config);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendFinalize(*plan);
    if (status == CUDNN_STATUS_SUCCESS) {
        int64_t written = 0;
        status = cudnnBackendGetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &written, workspace);
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnBackendDestroyDescriptor(*plan);
        *plan = nullptr;
    }
    return status;
}

// Whether a heuristic's result runs `only_engine`, or any engine when that is negative.
static bool singe_cudnn_runs_engine(cudnnBackendDescriptor_t config, int64_t only_engine) {
    if (only_engine < 0) return true;
    cudnnBackendDescriptor_t engine;
    if (cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine) != CUDNN_STATUS_SUCCESS) return false;
    int64_t index = -1, written = 0;
    cudnnStatus_t status = cudnnBackendGetAttribute(config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);
    if (status == CUDNN_STATUS_SUCCESS) status = cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &written, &index);
    cudnnBackendDestroyDescriptor(engine);
    return status == CUDNN_STATUS_SUCCESS && index == only_engine;
}

// Takes the first engine that plans and keeps the program's numerical policy, trying the heuristics from fastest to choose to most general.
static cudnnStatus_t singe_cudnn_heuristic_plan(cudnnHandle_t handle, cudnnBackendDescriptor_t graph, int64_t only_engine, bool tensor_core, bool deterministic, bool no_default_engine, std::vector<cudnnBackendDescriptor_t>* retained, cudnnBackendDescriptor_t* plan, int64_t* workspace) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    const cudnnBackendHeurMode_t modes[] = {CUDNN_HEUR_MODE_INSTANT, CUDNN_HEUR_MODE_B, CUDNN_HEUR_MODE_FALLBACK};
    cudnnStatus_t last = CUDNN_STATUS_NOT_SUPPORTED;
    for (const cudnnBackendHeurMode_t mode : modes) {
        cudnnBackendDescriptor_t heuristics;
        status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &heuristics);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        status = cudnnBackendSetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        status = cudnnBackendSetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &mode);
        if (status != CUDNN_STATUS_SUCCESS) return status;
        status = cudnnBackendFinalize(heuristics);
        if (status != CUDNN_STATUS_SUCCESS) {
            last = status;
            cudnnBackendDestroyDescriptor(heuristics);
            continue;
        }
        cudnnBackendDescriptor_t configs[16];
        int created = 0;
        for (; created < 16; ++created) {
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &configs[created]);
            if (status != CUDNN_STATUS_SUCCESS) return status;
        }
        int64_t found = 0;
        status = cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 16, &found, configs);
        if (status != CUDNN_STATUS_SUCCESS) last = status;
        bool planned = false;
        for (int64_t candidate = 0; status == CUDNN_STATUS_SUCCESS && candidate < found && !planned; ++candidate) {
            if (!singe_cudnn_runs_engine(configs[candidate], only_engine)) continue;
            if (!singe_cudnn_accepts(configs[candidate], tensor_core, deterministic, no_default_engine)) continue;
            cudnnStatus_t created_plan = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, plan);
            if (created_plan != CUDNN_STATUS_SUCCESS) return created_plan;
            created_plan = cudnnBackendSetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &handle);
            if (created_plan == CUDNN_STATUS_SUCCESS) created_plan = cudnnBackendSetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configs[candidate]);
            if (created_plan == CUDNN_STATUS_SUCCESS) created_plan = cudnnBackendFinalize(*plan);
            if (created_plan == CUDNN_STATUS_SUCCESS) {
                int64_t written = 0;
                created_plan = cudnnBackendGetAttribute(*plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, &written, workspace);
                planned = created_plan == CUDNN_STATUS_SUCCESS;
                if (planned) {
                    retained->push_back(configs[candidate]);
                    configs[candidate] = nullptr;
                }
            } else {
                cudnnBackendDestroyDescriptor(*plan);
                *plan = nullptr;
            }
            last = created_plan;
        }
        for (int index = 0; index < created; ++index) {
            if (configs[index] != nullptr) cudnnBackendDestroyDescriptor(configs[index]);
        }
        cudnnBackendDestroyDescriptor(heuristics);
        if (planned) return CUDNN_STATUS_SUCCESS;
    }
    return last;
}

// What `singe_cudnn_plan` returns when cuDNN refuses the engine and knobs a record chose.
static const int singe_cudnn_refused = -1;

#define SINGE_CUDNN_PLAN(expression) do { const int singe_planned = (expression); if (singe_planned == singe_cudnn_refused) return singe_failed(SINGE_CONFIGURATION_REFUSED, "singe_cudnn_plan, with the engine a record chose"); if (singe_planned != 0) return singe_failed(SINGE_CUDNN_BASE + singe_planned, #expression); } while (false)

// Plans the engine and knobs a record states, or `singe_cudnn_refused` if cuDNN refuses them, or else the first engine the heuristics offer that keeps the numerical policy.
static int singe_cudnn_plan(cudnnHandle_t handle, cudnnBackendDescriptor_t* operations, unsigned long long count, bool tensor_core, bool deterministic, bool no_default_engine, int64_t forced_engine, const std::vector<std::pair<int64_t, int64_t>>& forced_knobs, std::vector<cudnnBackendDescriptor_t>* retained, cudnnBackendDescriptor_t* plan, int64_t* workspace) {
    cudnnBackendDescriptor_t graph;
    cudnnStatus_t status = singe_cudnn_operation_graph(handle, operations, count, retained, &graph);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    if (forced_engine >= 0) {
        // An engine with no knobs stated runs as the heuristics configure it, since an engine such as attention's has no default of its own to build with.
        if (forced_knobs.empty() && singe_cudnn_heuristic_plan(handle, graph, forced_engine, tensor_core, deterministic, no_default_engine, retained, plan, workspace) == CUDNN_STATUS_SUCCESS) return 0;
        if (singe_cudnn_forced_plan(handle, graph, forced_engine, forced_knobs, tensor_core, deterministic, no_default_engine, retained, plan, workspace) == CUDNN_STATUS_SUCCESS) return 0;
        return singe_cudnn_refused;
    }
    return singe_cudnn_heuristic_plan(handle, graph, -1, tensor_core, deterministic, no_default_engine, retained, plan, workspace);
}
