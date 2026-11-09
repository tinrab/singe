use crate::{
    backend::cuda::{
        export::{
            program::{ProgramState, ProgramValue, StateKind},
            writer::{
                CONFIGURATION_REFUSED, CUBLAS_BASE, CUDNN_BASE, CUFFT_BASE, CUTENSOR_BASE,
                LIBRARY_VERSION, Renderer, fill_template,
            },
        },
        kernel_plan::value_type,
    },
    error::Result,
    utility::emit,
};

impl Renderer<'_> {
    const fn has_functions(&self, state: &ProgramState) -> bool {
        self.program.state_functions || matches!(state.kind, StateKind::Generator { .. })
    }

    // Declarations of the functions that read and replace each state tensor, stating what each holds.
    pub fn state_declarations(&self) -> Result<String> {
        let prefix = self.prefix;
        let mut out = String::new();
        for state in &self.program.state {
            if !self.has_functions(state) {
                continue;
            }

            let (name, ty) = (&state.name, value_type(state.data_type)?);
            let what = match state.kind {
                StateKind::Generator { blocks } => format!(
                    "the seed and next block of `{}`, in host memory, which each run advances by {blocks}",
                    state.graph_name
                ),
                StateKind::Initializer | StateKind::Constant => format!(
                    "`{}`: {} {:?}, strides {:?}, from element {}, in host or device memory, with no run pending",
                    state.graph_name,
                    state.data_type,
                    state.layout.dimensions(),
                    state.layout.strides(),
                    state.layout.storage_offset(),
                ),
            };
            emit!(
                out,
                "// Copies out or replaces {what}.\n{prefix}_status {prefix}_get_{name}(const {prefix}_state* state, {ty}* values);\n{prefix}_status {prefix}_set_{name}({prefix}_state* state, const {ty}* values);"
            );
        }
        Ok(out)
    }

    pub fn state_definitions(&self) -> Result<String> {
        let prefix = self.prefix;
        let mut out = String::new();
        for (index, state) in self.program.state.iter().enumerate() {
            if !self.has_functions(state) {
                continue;
            }

            let (name, ty) = (&state.name, value_type(state.data_type)?);
            let buffer = self.state_buffer(index)?;
            let bytes = self.program.buffers[buffer].bytes;
            let (get, set) = if matches!(state.kind, StateKind::Generator { .. }) {
                (
                    format!(
                        "    values[0] = s->generator_{name}[0];\n    values[1] = s->generator_{name}[1];\n"
                    ),
                    format!(
                        "    s->generator_{name}[0] = values[0];\n    s->generator_{name}[1] = values[1];\n"
                    ),
                )
            } else if bytes == 0 {
                (
                    "    (void)s;\n    (void)values;\n".to_owned(),
                    "    (void)s;\n    (void)values;\n".to_owned(),
                )
            } else {
                (
                    format!(
                        "    SINGE_CHECK(cudaMemcpy(values, s->buffers[{buffer}], {bytes}ULL, cudaMemcpyDefault));\n"
                    ),
                    format!(
                        "    SINGE_CHECK(cudaMemcpy(s->buffers[{buffer}], values, {bytes}ULL, cudaMemcpyDefault));\n"
                    ),
                )
            };
            emit!(
                out,
                "static int singe_get_{name}(const {prefix}_state* s, {ty}* values) {{\n    singe_step = \"reading {name}\";\n{get}    return 0;\n}}\n\nextern \"C\" {prefix}_status {prefix}_get_{name}(const {prefix}_state* s, {ty}* values) {{\n    return static_cast<{prefix}_status>(singe_get_{name}(s, values));\n}}\n\nstatic int singe_set_{name}({prefix}_state* s, const {ty}* values) {{\n    singe_step = \"replacing {name}\";\n{set}    return 0;\n}}\n\nextern \"C\" {prefix}_status {prefix}_set_{name}({prefix}_state* s, const {ty}* values) {{\n    return static_cast<{prefix}_status>(singe_set_{name}(s, values));\n}}\n"
            );
        }
        Ok(out)
    }

    pub fn value_struct(
        &self,
        kind: &str,
        values: &[ProgramValue],
        constant: bool,
    ) -> Result<String> {
        let mut out = format!("struct {}_{kind} {{\n", self.prefix);
        for value in values {
            let rows = value.count.map_or_else(String::new, |count| {
                format!(
                    " A run writes as many leading rows as `{}` holds.",
                    values[count].name
                )
            });
            let qualifier = if constant && !value.written {
                "const "
            } else {
                ""
            };
            emit!(
                out,
                "    // {} {:?}, strides {:?}, from element {}.{rows}\n    {qualifier}{}* {};",
                value.data_type,
                value.layout.dimensions(),
                value.layout.strides(),
                value.layout.storage_offset(),
                value_type(value.data_type)?,
                value.name
            );
        }

        out.push_str("};\n");
        Ok(out)
    }

    pub fn header(&self) -> Result<String> {
        let prefix = self.prefix;
        let guard = prefix.to_uppercase();
        let inputs = self.value_struct("inputs", &self.program.inputs, true)?;
        let outputs = self.value_struct("outputs", &self.program.outputs, false)?;
        let state = self.state_declarations()?;
        Ok(fill_template(
            include_str!("program.h.in"),
            &[
                ("guard", &guard),
                ("prefix", prefix),
                ("inputs", &inputs),
                ("outputs", &outputs),
                ("state", &state),
                ("CUBLAS_BASE", &CUBLAS_BASE.to_string()),
                ("CUFFT_BASE", &CUFFT_BASE.to_string()),
                ("CUDNN_BASE", &CUDNN_BASE.to_string()),
                ("CUTENSOR_BASE", &CUTENSOR_BASE.to_string()),
                ("LIBRARY_VERSION", &LIBRARY_VERSION.to_string()),
                ("CONFIGURATION_REFUSED", &CONFIGURATION_REFUSED.to_string()),
            ],
        ))
    }
}
