use std::{
    fmt::{self, Display, Formatter},
    str::FromStr,
};

use crate::error::Error;

macro_rules! architectures {
    ($($variant:ident => $target:literal),* $(,)?) => {
        /// CUDA compilation target names accepted by Singe's compiler wrappers.
        ///
        /// `compute_` targets are virtual architectures for intermediate code;
        /// `sm_` targets select real GPU architectures for native device code.
        /// Display and string parsing use the exact CUDA target spelling.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum GpuArchitecture {
            $(#[doc = concat!("CUDA compilation target `", $target, "`.")]
            $variant,)*
        }

        impl GpuArchitecture {
            /// Returns whether this is a virtual `compute_` target rather than an `sm_` target.
            pub const fn is_virtual(self) -> bool {
                match self {
                    $(Self::$variant => { let target: &str = $target; target.as_bytes()[0] == b'c' },)*
                }
            }
        }

        impl Display for GpuArchitecture {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                f.write_str(match self {
                    $(Self::$variant => $target,)*
                })
            }
        }

        impl FromStr for GpuArchitecture {
            type Err = Error;

            fn from_str(target: &str) -> Result<Self, Self::Err> {
                match target {
                    $($target => Ok(Self::$variant),)*
                    _ => Err(Error::UnsupportedArchitecture { target: target.into() }),
                }
            }
        }
    };
}

architectures! {
    Compute75 => "compute_75",
    Compute80 => "compute_80",
    Compute86 => "compute_86",
    Compute87 => "compute_87",
    Compute89 => "compute_89",
    Compute90 => "compute_90",
    Compute90a => "compute_90a",
    Compute100 => "compute_100",
    Compute100f => "compute_100f",
    Compute100a => "compute_100a",
    Compute101 => "compute_101",
    Compute101f => "compute_101f",
    Compute101a => "compute_101a",
    Compute103 => "compute_103",
    Compute107 => "compute_107",
    Compute103f => "compute_103f",
    Compute107f => "compute_107f",
    Compute103a => "compute_103a",
    Compute107a => "compute_107a",
    Compute110 => "compute_110",
    Compute110f => "compute_110f",
    Compute110a => "compute_110a",
    Compute120 => "compute_120",
    Compute120f => "compute_120f",
    Compute120a => "compute_120a",
    Compute121 => "compute_121",
    Compute121f => "compute_121f",
    Compute121a => "compute_121a",
    Sm75 => "sm_75",
    Sm80 => "sm_80",
    Sm86 => "sm_86",
    Sm87 => "sm_87",
    Sm89 => "sm_89",
    Sm90 => "sm_90",
    Sm90a => "sm_90a",
    Sm100 => "sm_100",
    Sm100f => "sm_100f",
    Sm100a => "sm_100a",
    Sm101 => "sm_101",
    Sm101f => "sm_101f",
    Sm101a => "sm_101a",
    Sm103 => "sm_103",
    Sm107 => "sm_107",
    Sm103f => "sm_103f",
    Sm107f => "sm_107f",
    Sm103a => "sm_103a",
    Sm107a => "sm_107a",
    Sm110 => "sm_110",
    Sm110f => "sm_110f",
    Sm110a => "sm_110a",
    Sm120 => "sm_120",
    Sm120f => "sm_120f",
    Sm120a => "sm_120a",
    Sm121 => "sm_121",
    Sm121f => "sm_121f",
    Sm121a => "sm_121a",
}
