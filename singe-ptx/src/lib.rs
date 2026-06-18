//! CUDA PTX parser, AST, and generated instruction metadata utilities.
//!
//! This crate parses PTX assembly into a Rust AST and exposes generated metadata
//! for known PTX instructions.
//!
//! # Example
//!
//! ```
//! let ptx = r#"
//! .version 9.1
//! .target sm_90
//! .address_size 64
//!
//! .visible .entry add(.param .u64 data) {
//!     ret;
//! }
//! "#;
//!
//! let module = singe_ptx::parser::parse_module(ptx)?;
//! assert_eq!(module.version.major, 9);
//! assert_eq!(module.items.len(), 1);
//! # Ok::<(), singe_ptx::error::ParseError>(())
//! ```

pub mod ast;
pub mod error;
pub mod parser;

mod generated;

pub use generated::instruction_set::*;
