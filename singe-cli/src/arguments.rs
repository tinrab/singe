// Values more than one command takes.

use clap::ValueEnum;
use singe_server::config::DataTypeChoice;

// The type weights load in, as `--type` names it.
#[derive(Clone, Copy, Default, ValueEnum)]
pub enum TypeChoice {
    #[default]
    Auto,
    F32,
    F16,
    Bf16,
}

impl TypeChoice {
    pub const fn choice(self) -> DataTypeChoice {
        match self {
            Self::Auto => DataTypeChoice::Auto,
            Self::F32 => DataTypeChoice::F32,
            Self::F16 => DataTypeChoice::F16,
            Self::Bf16 => DataTypeChoice::Bf16,
        }
    }
}
