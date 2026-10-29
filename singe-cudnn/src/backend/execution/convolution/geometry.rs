use crate::{
    error::{Error, Result},
    frontend::error::Error as FrontendError,
    utility::{check_range, to_i64},
};

/// Validated convolution padding, strides and dilations with one nonzero spatial rank.
/// Padding retains its supplied sign and supports asymmetric requests.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConvolutionGeometry {
    pre_paddings: Vec<i64>,
    post_paddings: Vec<i64>,
    strides: Vec<i64>,
    dilations: Vec<i64>,
}

impl ConvolutionGeometry {
    /// Creates spatial geometry without a native handle.
    ///
    /// # Errors
    /// Returns an error for empty or mismatched arrays, or nonpositive strides and dilations.
    pub fn create(
        pre_paddings: impl Into<Vec<i64>>,
        post_paddings: impl Into<Vec<i64>>,
        strides: impl Into<Vec<i64>>,
        dilations: impl Into<Vec<i64>>,
    ) -> Result<Self> {
        let pre_paddings = pre_paddings.into();
        let post_paddings = post_paddings.into();
        let strides = strides.into();
        let dilations = dilations.into();
        let spatial_dim_count = pre_paddings.len();
        if spatial_dim_count == 0 {
            return Err(Error::InvalidDataShape);
        }
        if post_paddings.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "post_paddings".into(),
                expected: spatial_dim_count,
                actual: post_paddings.len(),
            });
        }
        if dilations.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "dilations".into(),
                expected: spatial_dim_count,
                actual: dilations.len(),
            });
        }
        if strides.len() != spatial_dim_count {
            return Err(Error::LengthMismatch {
                name: "strides".into(),
                expected: spatial_dim_count,
                actual: strides.len(),
            });
        }

        check_range!(
            "convolution strides",
            strides.iter().all(|&value| value > 0)
        )?;
        check_range!(
            "convolution dilations",
            dilations.iter().all(|&value| value > 0)
        )?;
        Ok(Self {
            pre_paddings,
            post_paddings,
            strides,
            dilations,
        })
    }

    /// Returns the nonzero spatial rank shared by all geometry arrays.
    #[must_use]
    pub fn spatial_dims(&self) -> usize {
        self.pre_paddings.len()
    }

    /// Borrows padding before each spatial input axis, preserving signed values.
    #[must_use]
    pub fn pre_paddings(&self) -> &[i64] {
        &self.pre_paddings
    }

    /// Borrows padding after each spatial input axis, preserving signed values.
    #[must_use]
    pub fn post_paddings(&self) -> &[i64] {
        &self.post_paddings
    }

    /// Borrows the positive spatial stride values.
    #[must_use]
    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Borrows the positive spatial filter-dilation factors.
    #[must_use]
    pub fn dilations(&self) -> &[i64] {
        &self.dilations
    }

    /// Infers logical `[N, K, spatial...]` response dimensions without constructing tensor layouts.
    /// Input and filter dimensions must include any vector lanes in their logical channel counts.
    /// Groups follow the input/filter channel ratio and must divide the output channels.
    /// Uses wide intermediate arithmetic for asymmetric padding and dilated filters.
    /// The result describes geometry; native support selection still determines executable combinations.
    ///
    /// # Errors
    /// Returns an error for invalid ranks, nonpositive dimensions, incompatible groups,
    /// or response dimensions that are nonpositive or exceed INT64.
    pub fn infer_forward_dimensions(&self, input: &[i64], filter: &[i64]) -> Result<Vec<i64>> {
        if input.len() < 3 {
            return Err(Error::LengthMismatch {
                name: "convolution rank".into(),
                expected: 3,
                actual: input.len(),
            });
        }
        if input.len() != filter.len() {
            return Err(Error::LengthMismatch {
                name: "convolution rank".into(),
                expected: input.len(),
                actual: filter.len(),
            });
        }
        let spatial_dims = input.len() - 2;
        if self.spatial_dims() != spatial_dims {
            return Err(Error::LengthMismatch {
                name: "convolution spatial dims".into(),
                expected: spatial_dims,
                actual: self.spatial_dims(),
            });
        }
        let input_channels = input[1];
        let filter_channels = filter[1];
        let output_channels = filter[0];
        check_range!(
            "convolution channels",
            input_channels > 0 && filter_channels > 0 && output_channels > 0
        )?;
        if input_channels % filter_channels != 0
            || output_channels % (input_channels / filter_channels) != 0
        {
            return Err(FrontendError::ConvolutionGroupChannelsMismatch {
                input_channels,
                filter_channels,
                output_channels,
            }
            .into());
        }
        check_range!("convolution shape", input[0] > 0)?;
        let mut output = Vec::with_capacity(input.len());
        output.extend([input[0], output_channels]);
        for index in 0..spatial_dims {
            let input_dimension = input[index + 2];
            let kernel = filter[index + 2];
            check_range!("convolution shape", input_dimension > 0 && kernel > 0)?;
            let stride = i128::from(self.strides()[index]);
            let dilation = i128::from(self.dilations()[index]);
            let numerator = i128::from(input_dimension)
                + i128::from(self.pre_paddings()[index])
                + i128::from(self.post_paddings()[index])
                - dilation * i128::from(kernel - 1)
                - 1;
            check_range!("convolution output dimension", numerator >= 0)?;
            output.push(to_i64(
                numerator / stride + 1,
                "convolution output dimension",
            )?);
        }
        Ok(output)
    }
}
