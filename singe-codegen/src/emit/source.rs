use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SourceLanguage {
    CudaC,
}

impl Display for SourceLanguage {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::CudaC => "cuda c++",
        })
    }
}

// One parameter of a kernel's entry point, as its source declares it; `Display` writes the declaration.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SourceParameter {
    ty: String,
    name: String,
}

impl SourceParameter {
    #[must_use]
    pub fn new(ty: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            ty: ty.into(),
            name: name.into(),
        }
    }

    #[must_use]
    pub fn ty(&self) -> &str {
        &self.ty
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Display for SourceParameter {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.ty, self.name)
    }
}

// The function a launch runs: its parameters in launch-argument order, and the statements of its body, each line ending with a newline.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EntryPoint {
    name: String,
    parameters: Vec<SourceParameter>,
    body: String,
}

impl EntryPoint {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        parameters: Vec<SourceParameter>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            parameters,
            body: body.into(),
        }
    }

    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[must_use]
    pub fn parameters(&self) -> &[SourceParameter] {
        &self.parameters
    }

    #[must_use]
    pub fn body(&self) -> &str {
        &self.body
    }
}

// How an entry point is reached: by its plain name, as NVRTC's users look it up, or as a C++ name in its enclosing namespace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntryLinkage {
    C,
    Cpp,
}

// The prelude holds the helpers every kernel of a language shares, the declarations what one kernel alone needs, and the entry point the launch runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelSource {
    language: SourceLanguage,
    prelude: &'static str,
    declarations: String,
    entry: EntryPoint,
    // The threads per block the entry point launches with, and the blocks per SM it asks for, which let the compiler budget registers.
    launch_bounds: Option<(u32, Option<u32>)>,
}

impl KernelSource {
    #[must_use]
    pub fn new(
        language: SourceLanguage,
        prelude: &'static str,
        declarations: impl Into<String>,
        entry: EntryPoint,
    ) -> Self {
        Self {
            language,
            prelude,
            declarations: declarations.into(),
            entry,
            launch_bounds: None,
        }
    }

    // Declares the launch bounds of the entry point, in the language's spelling.
    #[must_use]
    pub const fn with_launch_bounds(mut self, threads: u32, min_blocks: Option<u32>) -> Self {
        self.launch_bounds = Some((threads, min_blocks));
        self
    }

    #[must_use]
    pub const fn language(&self) -> SourceLanguage {
        self.language
    }

    #[must_use]
    pub const fn prelude(&self) -> &'static str {
        self.prelude
    }

    #[must_use]
    pub fn declarations(&self) -> &str {
        &self.declarations
    }

    #[must_use]
    pub const fn entry(&self) -> &EntryPoint {
        &self.entry
    }

    // The entry point's definition as the language spells it, ending with a newline.
    #[must_use]
    pub fn render_definition(&self, linkage: EntryLinkage) -> String {
        self.definition(linkage, &self.entry.body)
    }

    // The entry point's definition as a program's dependent launch runs it, from `sm_90`.
    // It waits for what ran before it, which must be its first act, since those grids may still be writing what it reads.
    // Once it has done all it will do, it lets what depends on it start, and a block that returns early does the same by finishing.
    #[must_use]
    pub fn render_dependent_definition(&self, linkage: EntryLinkage) -> String {
        self.definition(
            linkage,
            &format!(
                "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n{}    asm volatile(\"griddepcontrol.launch_dependents;\" ::: \"memory\");\n",
                self.entry.body
            ),
        )
    }

    fn definition(&self, linkage: EntryLinkage, body: &str) -> String {
        let entry = &self.entry;
        let parameters = entry
            .parameters
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        match self.language {
            SourceLanguage::CudaC => format!(
                "{}__global__ {}void {}({parameters}) {{\n{}}}\n",
                match linkage {
                    EntryLinkage::C => "extern \"C\" ",
                    EntryLinkage::Cpp => "",
                },
                match self.launch_bounds {
                    Some((threads, Some(blocks))) =>
                        format!("__launch_bounds__({threads}, {blocks}) "),
                    Some((threads, None)) => format!("__launch_bounds__({threads}) "),
                    None => String::new(),
                },
                entry.name,
                body
            ),
        }
    }
}
