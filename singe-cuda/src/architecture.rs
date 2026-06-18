use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum GpuArchitecture {
    Compute75,
    Compute80,
    Compute86,
    Compute87,
    Compute89,
    Compute90,
    Compute90a,
    Compute100,
    Compute100f,
    Compute100a,
    Compute101,
    Compute101f,
    Compute101a,
    Compute103,
    Compute103f,
    Compute103a,
    Compute110,
    Compute110f,
    Compute110a,
    Compute120,
    Compute120f,
    Compute120a,
    Compute121,
    Compute121f,
    Compute121a,
    Sm75,
    Sm80,
    Sm86,
    Sm87,
    Sm89,
    Sm90,
    Sm90a,
    Sm100,
    Sm100f,
    Sm100a,
    Sm101,
    Sm101f,
    Sm101a,
    Sm103,
    Sm103f,
    Sm103a,
    Sm110,
    Sm110f,
    Sm110a,
    Sm120,
    Sm120f,
    Sm120a,
    Sm121,
    Sm121f,
    Sm121a,
}

impl GpuArchitecture {
    pub const fn is_virtual(self) -> bool {
        matches!(
            self,
            Self::Compute75
                | Self::Compute80
                | Self::Compute86
                | Self::Compute87
                | Self::Compute89
                | Self::Compute90
                | Self::Compute90a
                | Self::Compute100
                | Self::Compute100f
                | Self::Compute100a
                | Self::Compute101
                | Self::Compute101f
                | Self::Compute101a
                | Self::Compute103
                | Self::Compute103f
                | Self::Compute103a
                | Self::Compute110
                | Self::Compute110f
                | Self::Compute110a
                | Self::Compute120
                | Self::Compute120f
                | Self::Compute120a
                | Self::Compute121
                | Self::Compute121f
                | Self::Compute121a
        )
    }
}

impl Display for GpuArchitecture {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let value = match self {
            Self::Compute75 => "compute_75",
            Self::Compute80 => "compute_80",
            Self::Compute86 => "compute_86",
            Self::Compute87 => "compute_87",
            Self::Compute89 => "compute_89",
            Self::Compute90 => "compute_90",
            Self::Compute90a => "compute_90a",
            Self::Compute100 => "compute_100",
            Self::Compute100f => "compute_100f",
            Self::Compute100a => "compute_100a",
            Self::Compute101 => "compute_101",
            Self::Compute101f => "compute_101f",
            Self::Compute101a => "compute_101a",
            Self::Compute103 => "compute_103",
            Self::Compute103f => "compute_103f",
            Self::Compute103a => "compute_103a",
            Self::Compute110 => "compute_110",
            Self::Compute110f => "compute_110f",
            Self::Compute110a => "compute_110a",
            Self::Compute120 => "compute_120",
            Self::Compute120f => "compute_120f",
            Self::Compute120a => "compute_120a",
            Self::Compute121 => "compute_121",
            Self::Compute121f => "compute_121f",
            Self::Compute121a => "compute_121a",
            Self::Sm75 => "sm_75",
            Self::Sm80 => "sm_80",
            Self::Sm86 => "sm_86",
            Self::Sm87 => "sm_87",
            Self::Sm89 => "sm_89",
            Self::Sm90 => "sm_90",
            Self::Sm90a => "sm_90a",
            Self::Sm100 => "sm_100",
            Self::Sm100f => "sm_100f",
            Self::Sm100a => "sm_100a",
            Self::Sm101 => "sm_101",
            Self::Sm101f => "sm_101f",
            Self::Sm101a => "sm_101a",
            Self::Sm103 => "sm_103",
            Self::Sm103f => "sm_103f",
            Self::Sm103a => "sm_103a",
            Self::Sm110 => "sm_110",
            Self::Sm110f => "sm_110f",
            Self::Sm110a => "sm_110a",
            Self::Sm120 => "sm_120",
            Self::Sm120f => "sm_120f",
            Self::Sm120a => "sm_120a",
            Self::Sm121 => "sm_121",
            Self::Sm121f => "sm_121f",
            Self::Sm121a => "sm_121a",
        };
        write!(f, "{value}")
    }
}
