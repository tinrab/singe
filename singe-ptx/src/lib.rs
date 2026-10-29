//! PTX parsing, validation, and a CPU emulator for `sm_80` kernels.
//!
//! [`parser::parse_module`] builds an owned [`ast`] that keeps the source byte span of
//! every node. [`emulator::ParsedModule::compile`] checks that a kernel can execute, and an
//! [`emulator::Emulator`] runs it. Emulator commands return submissions that can be waited
//! on or awaited.
//!
//! # Example
//!
//! ```
//! use singe_ptx::emulator::ParsedModule;
//!
//! let ptx = r#"
//! .version 9.4
//! .target sm_80
//! .address_size 64
//!
//! .visible .entry add(.param .u64 data) {
//!     ret;
//! }
//! "#;
//!
//! let parsed = ParsedModule::parse(ptx)?;
//! let kernel = parsed.compile("add")?;
//! assert_eq!(kernel.parameters()[0].storage().byte_len(), 8);
//! kernel.launch_requirements().validate_block([128, 1, 1])?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// Owned PTX syntax tree. Every node records its source byte range.
pub mod ast;
pub mod diagnostic;
pub mod emulator;
/// PTX parser producing the [`ast`] syntax tree.
pub mod parser;
