use crate::{
    backend::cuda::{
        export::{
            program::{Activation, BufferView, GemmOperand, LibraryCall},
            writer::{
                CONFIGURATION_REFUSED, CUBLAS_BASE, CUFFT_BASE, CUTENSOR_BASE, LibraryCode,
                Renderer, cuda_type, fill_template, integers,
            },
        },
        runtime::CublasLtMatmulAlgorithmFingerprint,
    },
    data_type::DataType,
    error::Result,
    graph::Determinism,
    utility::{emit, to_i64},
};

impl Renderer<'_> {
    pub fn gemm(
        &self,
        index: usize,
        lhs: (&BufferView, &GemmOperand),
        rhs: (&BufferView, &GemmOperand),
        output: &BufferView,
        sizes: [usize; 4],
    ) -> Result<LibraryCode> {
        let [rows, columns, inner, batch] = sizes;
        let transpose = |operand: &GemmOperand| {
            if operand.transposed {
                "CUBLAS_OP_T"
            } else {
                "CUBLAS_OP_N"
            }
        };
        let ty = cuda_type(lhs.0.data_type)?;
        let (scale, compute) = if lhs.0.data_type == DataType::F64 {
            ("double", "CUDA_R_64F")
        } else {
            ("float", "CUDA_R_32F")
        };
        let run = format!(
            "    {{\n        // Row-major C = A · B is column-major C^T = B^T · A^T, so cuBLAS multiplies the right operand by the left.\n        const {scale} alpha_{index} = 1, beta_{index} = 0;\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasSetStream(s->cublas, stream));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasGemmStridedBatchedEx(s->cublas, {rhs_op}, {lhs_op}, {columns}, {rows}, {inner}, &alpha_{index}, {rhs}, {ty}, {rhs_ld}, {rhs_stride}LL, {lhs}, {ty}, {lhs_ld}, {lhs_stride}LL, &beta_{index}, {out}, {ty}, {columns}, {output_stride}LL, {batch}, {compute}, CUBLAS_GEMM_DEFAULT));\n    }}\n",
            rhs_op = transpose(rhs.1),
            lhs_op = transpose(lhs.1),
            rhs = self.address(rhs.0)?,
            rhs_ld = rhs.1.leading_dimension,
            rhs_stride = rhs.1.batch_stride,
            lhs = self.address(lhs.0)?,
            lhs_ld = lhs.1.leading_dimension,
            lhs_stride = lhs.1.batch_stride,
            out = self.address(output)?,
            output_stride = rows * columns,
        );
        Ok(LibraryCode {
            run,
            ..LibraryCode::default()
        })
    }

    pub fn fft(
        &self,
        index: usize,
        inverse: bool,
        double: bool,
        sizes: (i64, usize),
        input: &BufferView,
        output: &BufferView,
    ) -> Result<LibraryCode> {
        let (length, batch) = sizes;
        let (kind, execute, real, complex) = match (inverse, double) {
            (false, false) => ("CUFFT_R2C", "cufftExecR2C", "cufftReal", "cufftComplex"),
            (false, true) => (
                "CUFFT_D2Z",
                "cufftExecD2Z",
                "cufftDoubleReal",
                "cufftDoubleComplex",
            ),
            (true, false) => ("CUFFT_C2R", "cufftExecC2R", "cufftReal", "cufftComplex"),
            (true, true) => (
                "CUFFT_Z2D",
                "cufftExecZ2D",
                "cufftDoubleReal",
                "cufftDoubleComplex",
            ),
        };
        let (from, to) = if inverse {
            (complex, real)
        } else {
            (real, complex)
        };
        Ok(LibraryCode {
            members: format!("    cufftHandle fft_{index};\n"),
            create: format!(
                "    {{\n        int length_{index} = {length};\n        SINGE_LIBRARY({CUFFT_BASE}, cufftPlanMany(&s->fft_{index}, 1, &length_{index}, nullptr, 1, 0, nullptr, 1, 0, {kind}, {batch}));\n    }}\n"
            ),
            run: format!(
                "    SINGE_LIBRARY({CUFFT_BASE}, cufftSetStream(s->fft_{index}, stream));\n    SINGE_LIBRARY({CUFFT_BASE}, {execute}(s->fft_{index}, reinterpret_cast<{from}*>({}), reinterpret_cast<{to}*>({})));\n",
                self.address(input)?,
                self.address(output)?
            ),
            destroy: format!("    if (s->fft_{index} != 0) cufftDestroy(s->fft_{index});\n"),
        })
    }

    pub fn scaled_matmul(
        &self,
        index: usize,
        sizes: [i64; 3],
        operands: [&BufferView; 5],
    ) -> Result<LibraryCode> {
        let [rows, depth, columns] = sizes;
        let [lhs, rhs, lhs_scale, rhs_scale, output] = operands;
        let choose = self.cublaslt_algorithm(index, output.data_type, None);
        let (rhs_type, lhs_type, output_type) = (
            cuda_type(rhs.data_type)?,
            cuda_type(lhs.data_type)?,
            cuda_type(output.data_type)?,
        );
        Ok(LibraryCode {
            members: format!(
                "    cublasLtMatmulDesc_t lt_desc_{index};\n    cublasLtMatrixLayout_t lt_a_{index};\n    cublasLtMatrixLayout_t lt_b_{index};\n    cublasLtMatrixLayout_t lt_c_{index};\n    cublasLtMatmulHeuristicResult_t lt_algorithm_{index};\n"
            ),
            create: format!(
                "    {{\n        // FP8 tensor cores take the `TN` layout, so the column-major right operand is cuBLASLt's A.\n        const cublasOperation_t transpose_a = CUBLAS_OP_T, transpose_b = CUBLAS_OP_N;\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescCreate(&s->lt_desc_{index}, CUBLAS_COMPUTE_32F, CUDA_R_32F));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)));\n        // Choosing an algorithm reads whether the scales are set, not what they hold, and each run sets them.\n        const void* placeholder = s;\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &placeholder, sizeof(placeholder)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &placeholder, sizeof(placeholder)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_a_{index}, {rhs_type}, {depth}, {columns}, {depth}));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_b_{index}, {lhs_type}, {depth}, {rows}, {depth}));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_c_{index}, {output_type}, {columns}, {rows}, {columns}));\n{choose}    }}\n"
            ),
            run: format!(
                "    {{\n        const float alpha_{index} = 1.0f, beta_{index} = 0.0f;\n        const void* a_scale = {rhs_scale};\n        const void* b_scale = {lhs_scale};\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmul(s->cublaslt, s->lt_desc_{index}, &alpha_{index}, {rhs}, s->lt_a_{index}, {lhs}, s->lt_b_{index}, &beta_{index}, {output}, s->lt_c_{index}, {output}, s->lt_c_{index}, &s->lt_algorithm_{index}.algo, nullptr, 0, stream));\n    }}\n",
                rhs_scale = self.address(rhs_scale)?,
                lhs_scale = self.address(lhs_scale)?,
                rhs = self.address(rhs)?,
                lhs = self.address(lhs)?,
                output = self.address(output)?,
            ),
            destroy: format!(
                "    if (s->lt_c_{index} != nullptr) cublasLtMatrixLayoutDestroy(s->lt_c_{index});\n    if (s->lt_b_{index} != nullptr) cublasLtMatrixLayoutDestroy(s->lt_b_{index});\n    if (s->lt_a_{index} != nullptr) cublasLtMatrixLayoutDestroy(s->lt_a_{index});\n    if (s->lt_desc_{index} != nullptr) cublasLtMatmulDescDestroy(s->lt_desc_{index});\n"
            ),
        })
    }

    pub fn contraction(
        &self,
        index: usize,
        operands: [&(BufferView, Vec<i64>); 4],
        alpha: f32,
        beta: f32,
    ) -> Result<LibraryCode> {
        let mut create = "    {\n".to_owned();
        let mut descriptors = Vec::with_capacity(4);
        for (position, (view, modes)) in operands.iter().enumerate() {
            let rank = view.layout.dimensions().len();
            emit!(
                create,
                "        const int64_t extent_{position}[] = {{{extents}}};\n        const int64_t stride_{position}[] = {{{strides}}};\n        const int32_t modes_{position}[] = {{{modes}}};\n        cutensorTensorDescriptor_t descriptor_{position};\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorCreateTensorDescriptor(s->cutensor, &descriptor_{position}, {rank}, extent_{position}, stride_{position}, CUDA_R_32F, 4));",
                extents = integers(view.layout.dimensions().iter().copied()),
                strides = integers(view.layout.strides().iter().copied()),
                modes = integers(modes.iter().copied())
            );
            descriptors.push(format!("descriptor_{position}, modes_{position}"));
        }

        emit!(
            create,
            "        cutensorOperationDescriptor_t operation;\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorCreateContraction(s->cutensor, &operation, {a}, CUTENSOR_OP_IDENTITY, {b}, CUTENSOR_OP_IDENTITY, {c}, CUTENSOR_OP_IDENTITY, {d}, CUTENSOR_COMPUTE_DESC_32F));\n        cutensorPlanPreference_t preference;\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorCreatePlanPreference(s->cutensor, &preference, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE));\n        uint64_t estimate = 0;\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorEstimateWorkspaceSize(s->cutensor, operation, preference, CUTENSOR_WORKSPACE_DEFAULT, &estimate));\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorCreatePlan(s->cutensor, &s->tensor_plan_{index}, operation, preference, estimate));\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorPlanGetAttribute(s->cutensor, s->tensor_plan_{index}, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &s->tensor_workspace_bytes_{index}, sizeof(s->tensor_workspace_bytes_{index})));\n        if (s->tensor_workspace_bytes_{index} > 0) SINGE_CHECK(cudaMalloc(&s->tensor_workspace_{index}, s->tensor_workspace_bytes_{index}));\n        cutensorDestroyPlanPreference(preference);\n        cutensorDestroyOperationDescriptor(operation);\n        cutensorDestroyTensorDescriptor(descriptor_0);\n        cutensorDestroyTensorDescriptor(descriptor_1);\n        cutensorDestroyTensorDescriptor(descriptor_2);\n        cutensorDestroyTensorDescriptor(descriptor_3);\n    }}",
            a = descriptors[0],
            b = descriptors[1],
            c = descriptors[2],
            d = descriptors[3]
        );
        Ok(LibraryCode {
            members: format!(
                "    cutensorPlan_t tensor_plan_{index};\n    void* tensor_workspace_{index};\n    uint64_t tensor_workspace_bytes_{index};\n"
            ),
            create,
            run: format!(
                "    {{\n        const float alpha_{index} = {alpha:?}f, beta_{index} = {beta:?}f;\n        SINGE_LIBRARY({CUTENSOR_BASE}, cutensorContract(s->cutensor, s->tensor_plan_{index}, &alpha_{index}, {a}, {b}, &beta_{index}, {c}, {d}, s->tensor_workspace_{index}, s->tensor_workspace_bytes_{index}, stream));\n    }}\n",
                a = self.address(&operands[0].0)?,
                b = self.address(&operands[1].0)?,
                c = self.address(&operands[2].0)?,
                d = self.address(&operands[3].0)?,
            ),
            destroy: format!(
                "    if (s->tensor_workspace_{index} != nullptr) cudaFree(s->tensor_workspace_{index});\n    if (s->tensor_plan_{index} != nullptr) cutensorDestroyPlan(s->tensor_plan_{index});\n"
            ),
        })
    }

    // `cublasLtMatmul` of `rhs · lhs`, column-major as a GEMM's, with `beta · addend` and an activation in the same call.
    pub fn cublaslt_matmul(
        &self,
        index: usize,
        lhs: (&BufferView, &GemmOperand),
        rhs: (&BufferView, &GemmOperand),
        addend: Option<&BufferView>,
        activation: Option<Activation>,
        output: &BufferView,
        sizes: [usize; 4],
        config: Option<&CublasLtMatmulAlgorithmFingerprint>,
    ) -> Result<LibraryCode> {
        let [rows, columns, inner, batch] = sizes;
        let (compute, scale_type, scale) = if output.data_type == DataType::F64 {
            ("CUBLAS_COMPUTE_64F", "CUDA_R_64F", "double")
        } else {
            ("CUBLAS_COMPUTE_32F", "CUDA_R_32F", "float")
        };
        let ty = cuda_type(output.data_type)?;
        let transpose = |operand: &GemmOperand| {
            if operand.transposed {
                "CUBLAS_OP_T"
            } else {
                "CUBLAS_OP_N"
            }
        };
        let stored = |operand: &GemmOperand, other: usize| {
            if operand.transposed {
                (inner, other)
            } else {
                (other, inner)
            }
        };
        let (rhs_rows, rhs_columns) = stored(rhs.1, columns);
        let (lhs_rows, lhs_columns) = if lhs.1.transposed {
            (rows, inner)
        } else {
            (inner, rows)
        };
        let batch_attributes = |layout: &str, stride: i64| {
            format!(
                "        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutSetAttribute(s->lt_{layout}_{index}, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));\n        {{\n            const int64_t stride = {stride}LL;\n            SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutSetAttribute(s->lt_{layout}_{index}, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));\n        }}\n"
            )
        };
        let output_stride = to_i64(rows * columns, "program matmul output stride")?;
        let batched = if batch > 1 {
            format!(
                "        const int batch_count = {batch};\n{}{}{}",
                batch_attributes("a", rhs.1.batch_stride),
                batch_attributes("b", lhs.1.batch_stride),
                batch_attributes("c", output_stride),
            )
        } else {
            String::new()
        };
        let epilogue = activation.map_or_else(String::new, |activation| {
            let name = match activation {
                Activation::Relu => "CUBLASLT_EPILOGUE_RELU",
                Activation::Gelu => "CUBLASLT_EPILOGUE_GELU",
            };
            format!(
                "        const cublasLtEpilogue_t epilogue = {name};\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));\n"
            )
        });
        let forced = config.map(|config| (config, compute, scale_type, ty));
        let choose = self.cublaslt_algorithm(index, output.data_type, forced);
        let (rhs_op, lhs_op) = (transpose(rhs.1), transpose(lhs.1));
        let (rhs_ld, lhs_ld) = (rhs.1.leading_dimension, lhs.1.leading_dimension);
        let beta = i32::from(addend.is_some());
        let accumulate =
            addend.map_or_else(|| self.address(output), |addend| self.address(addend))?;
        Ok(LibraryCode {
            members: format!(
                "    cublasLtMatmulDesc_t lt_desc_{index};\n    cublasLtMatrixLayout_t lt_a_{index};\n    cublasLtMatrixLayout_t lt_b_{index};\n    cublasLtMatrixLayout_t lt_c_{index};\n    cublasLtMatmulHeuristicResult_t lt_algorithm_{index};\n"
            ),
            create: format!(
                "    {{\n        const cublasOperation_t transpose_a = {rhs_op}, transpose_b = {lhs_op};\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescCreate(&s->lt_desc_{index}, {compute}, {scale_type}));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_a, sizeof(transpose_a)));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmulDescSetAttribute(s->lt_desc_{index}, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_b, sizeof(transpose_b)));\n{epilogue}        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_a_{index}, {ty}, {rhs_rows}, {rhs_columns}, {rhs_ld}));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_b_{index}, {ty}, {lhs_rows}, {lhs_columns}, {lhs_ld}));\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatrixLayoutCreate(&s->lt_c_{index}, {ty}, {columns}, {rows}, {columns}));\n{batched}{choose}    }}\n"
            ),
            run: format!(
                "    {{\n        const {scale} alpha = 1, beta = {beta};\n        SINGE_LIBRARY({CUBLAS_BASE}, cublasLtMatmul(s->cublaslt, s->lt_desc_{index}, &alpha, {rhs}, s->lt_a_{index}, {lhs}, s->lt_b_{index}, &beta, {accumulate}, s->lt_c_{index}, {out}, s->lt_c_{index}, &s->lt_algorithm_{index}.algo, nullptr, 0, stream));\n    }}\n",
                rhs = self.address(rhs.0)?,
                lhs = self.address(lhs.0)?,
                out = self.address(output)?,
            ),
            destroy: format!(
                "    cublasLtMatrixLayoutDestroy(s->lt_c_{index});\n    cublasLtMatrixLayoutDestroy(s->lt_b_{index});\n    cublasLtMatrixLayoutDestroy(s->lt_a_{index});\n    cublasLtMatmulDescDestroy(s->lt_desc_{index});\n"
            ),
        })
    }

    pub fn library(&self, index: usize, call: &LibraryCall) -> Result<LibraryCode> {
        match call {
            LibraryCall::Gemm {
                rows,
                columns,
                inner,
                batch_count,
                lhs_operand,
                rhs_operand,
                lhs,
                rhs,
                output,
            } => self.gemm(
                index,
                (lhs, lhs_operand),
                (rhs, rhs_operand),
                output,
                [*rows, *columns, *inner, *batch_count],
            ),
            LibraryCall::Fft {
                inverse,
                double,
                length,
                batch,
                input,
                output,
            } => self.fft(index, *inverse, *double, (*length, *batch), input, output),
            LibraryCall::ScaledMatmul {
                rows,
                depth,
                columns,
                lhs,
                rhs,
                lhs_scale,
                rhs_scale,
                output,
            } => self.scaled_matmul(
                index,
                [*rows, *depth, *columns],
                [lhs, rhs, lhs_scale, rhs_scale, output],
            ),
            LibraryCall::Contraction {
                a,
                b,
                c,
                output,
                alpha,
                beta,
            } => self.contraction(index, [a, b, c, output], *alpha, *beta),
            LibraryCall::CudnnAttention(call) => self.cudnn_attention(index, call),
            LibraryCall::CudnnMatmul {
                lhs,
                rhs,
                addend,
                activation,
                output,
                engine,
            } => self.cudnn_matmul(
                index,
                (lhs, rhs, addend.as_ref(), output),
                *activation,
                engine.as_ref(),
            ),
            LibraryCall::CublasLtMatmul {
                rows,
                columns,
                inner,
                batch_count,
                lhs_operand,
                rhs_operand,
                lhs,
                rhs,
                addend,
                activation,
                output,
                config,
            } => self.cublaslt_matmul(
                index,
                (lhs, lhs_operand),
                (rhs, rhs_operand),
                addend.as_ref(),
                *activation,
                output,
                [*rows, *columns, *inner, *batch_count],
                config.as_ref(),
            ),
            LibraryCall::Convolution {
                input,
                weight,
                bias,
                activation,
                output,
                stride,
                padding,
                dilation,
            } => self.convolution(
                index,
                (input, weight, bias.as_ref(), output),
                *activation,
                stride,
                padding,
                dilation,
            ),
        }
    }

    // Chooses the algorithm of a cuBLASLt call: what `forced` states, failing the program if cuBLASLt refuses it here, or else a zero-workspace algorithm its heuristic suggests that keeps the numerical policy.
    pub fn cublaslt_algorithm(
        &self,
        index: usize,
        data_type: DataType,
        forced: Option<(&CublasLtMatmulAlgorithmFingerprint, &str, &str, &str)>,
    ) -> String {
        let half = matches!(data_type, DataType::F16 | DataType::Bf16);
        let deterministic = self.program.numerics.determinism() == Determinism::Require;
        let forced = forced.map_or_else(String::new, |(config, compute, scale, ty)| {
            self.forced.borrow_mut().count += 1;
            let id = self.forced_value(i64::from(config.id));
            let value_of = |value| self.forced_value(value);
            format!(
                "        {{\n            cublasLtMatmulAlgo_t algo;\n            bool usable = cublasLtMatmulAlgoInit(s->cublaslt, {compute}, {scale}, {ty}, {ty}, {ty}, {ty}, {id}, &algo) == CUBLAS_STATUS_SUCCESS;\n{settings}            cublasLtMatmulHeuristicResult_t checked;\n            usable = usable && cublasLtMatmulAlgoCheck(s->cublaslt, s->lt_desc_{index}, s->lt_a_{index}, s->lt_b_{index}, s->lt_c_{index}, s->lt_c_{index}, &algo, &checked) == CUBLAS_STATUS_SUCCESS && checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize == 0;\n            if (!usable) return singe_failed({CONFIGURATION_REFUSED}, \"cublasLtMatmulAlgoCheck\");\n            checked.algo = algo;\n            s->lt_algorithm_{index} = checked;\n            forced_chosen = true;\n        }}\n",
                settings = config.cpp_settings("algo", "usable", &value_of),
            )
        });
        fill_template(
            include_str!("cublaslt_algorithm.cu.in"),
            &[
                ("CUBLAS_BASE", &CUBLAS_BASE.to_string()),
                ("index", &index.to_string()),
                ("half", &half.to_string()),
                ("deterministic", &deterministic.to_string()),
                ("forced", &forced),
            ],
        )
    }
}
