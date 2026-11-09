use crate::emit::{
    EntryPoint, KernelSource, SourceLanguage, SourceParameter, cuda_c::prelude::PRELUDE,
};

// Builds one CUDA C++ kernel: its entry point's parameters in launch-argument order, the helper functions it declares before the entry point, and the entry point's body, one statement per line.
// The IR emitter and every hand-written generator write through it, so both spell a kernel the same way.
#[derive(Debug, Default)]
pub struct SourceWriter {
    parameters: Vec<SourceParameter>,
    declarations: String,
    pub(crate) body: String,
}

impl SourceWriter {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // Declares the entry point's next parameter, and returns its name for the body to use.
    pub fn parameter(&mut self, ty: impl Into<String>, name: impl Into<String>) -> String {
        let name = name.into();
        self.parameters.push(SourceParameter::new(ty, name.clone()));
        name
    }

    // Adds a helper, such as a `__device__` function, that the entry point calls.
    pub fn declare(&mut self, code: &str) {
        self.declarations.push_str(code.trim_matches('\n'));
        self.declarations.push('\n');
    }

    pub fn line(&mut self, line: &str) {
        self.body.push_str(line);
        self.body.push('\n');
    }

    // Adds each line of `code`, so a block of statements can be written as one string.
    pub fn lines(&mut self, code: &str) {
        for line in code.trim_matches('\n').lines() {
            self.line(line);
        }
    }

    #[must_use]
    pub fn finish(self, name: impl Into<String>) -> KernelSource {
        KernelSource::new(
            SourceLanguage::CudaC,
            PRELUDE,
            self.declarations,
            EntryPoint::new(name, self.parameters, self.body),
        )
    }
}
