use singe_cuda::view::DeviceOutputView;

use crate::{
    error::Result,
    image::{filtering, view::C1},
    pipeline::image::{HoughLines, HoughLinesOutput, ImagePipeline},
    types::PointPolar,
};

impl ImagePipeline<'_, u8, C1> {
    /// Detects lines in the C1 `u8` image and returns their polar parameters.
    ///
    /// `delta` gives radial and angular discretization increments; `threshold` is the
    /// minimum number of points needed to accept a line. Allocates capacity for at most
    /// `max_line_count` lines, a device count, and temporary device scratch. After native
    /// completion, reads the count to the host and exposes only the initialized line prefix
    /// in `HoughLines::lines`, retaining the full underlying allocation. `line_count` retains
    /// the one-element device count.
    /// Prepares source and output regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if source view construction, capture checks, region preparation,
    /// producer waits, workspace queries, allocation, native detection, synchronization,
    /// count readback, or constructing the initialized prefix fail. Source dimensions and
    /// `threshold` must be positive, and `max_line_count` must be nonzero and fit `i32`.
    /// `delta.rho` must be strictly between zero and three, and `delta.theta` strictly
    /// between 0.25 and three. A negative count or count above capacity is rejected.
    pub fn filter_hough_lines(
        self,
        delta: PointPolar,
        threshold: i32,
        max_line_count: usize,
    ) -> Result<HoughLines> {
        self.ensure_eager()?;
        let mut lines =
            HoughLinesOutput::create(self.stream_context.cuda_context(), max_line_count)?;

        {
            let source = self.view()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&lines.lines)?;
            self.stream_context.prepare_region(&lines.line_count)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: checked native dispatch retains all borrows through completion.
                // Line storage remains explicitly uninitialized outside the produced prefix.
                unsafe {
                    filtering::filter_hough_line_to_polar(
                        self.stream_context,
                        &source,
                        delta,
                        threshold,
                        &mut lines.lines,
                        &mut DeviceOutputView::from_output(&mut lines.line_count),
                    )
                }
            })?;
        }

        // SAFETY: completed Hough dispatch writes the count and exactly that initialized line prefix.
        unsafe { lines.assume_init() }
    }

    /// Detects lines in the C1 `u8` image and returns their polar parameters.
    ///
    /// `delta` gives radial and angular discretization increments; `threshold` is the
    /// minimum number of points needed to accept a line. Allocates capacity for at most
    /// `max_line_count` lines, a device count, and temporary device scratch. After native
    /// completion, reads the count to the host and exposes only the initialized line prefix
    /// in `HoughLines::lines`, retaining the full underlying allocation. `line_count` retains
    /// the one-element device count.
    /// `destination_roi` contains the lower and upper polar bounds of the search region;
    /// each lower component must not exceed its corresponding upper component.
    /// Prepares source and output regions, waits for producers on other streams, rejects
    /// capture, and returns after successful stream completion.
    ///
    /// # Errors
    ///
    /// Returns an error if source view construction, capture checks, region preparation,
    /// producer waits, workspace queries, allocation, native detection, synchronization,
    /// count readback, or constructing the initialized prefix fail. Source dimensions and
    /// `threshold` must be positive, and `max_line_count` must be nonzero and fit `i32`.
    /// `delta.rho` must be strictly between zero and three, and `delta.theta` strictly
    /// between 0.25 and three. A negative count or count above capacity is rejected.
    /// Reversed polar bounds are rejected.
    pub fn filter_hough_lines_region(
        self,
        delta: PointPolar,
        threshold: i32,
        destination_roi: [PointPolar; 2],
        max_line_count: usize,
    ) -> Result<HoughLines> {
        self.ensure_eager()?;
        let mut lines =
            HoughLinesOutput::create(self.stream_context.cuda_context(), max_line_count)?;

        {
            let source = self.view()?;
            self.stream_context.stream().ensure_not_capturing()?;
            self.stream_context.prepare_region(&source)?;
            self.stream_context.prepare_region(&lines.lines)?;
            self.stream_context.prepare_region(&lines.line_count)?;
            self.stream_context.sync_scope(|| {
                // SAFETY: checked native dispatch retains all borrows through completion.
                // Line storage remains explicitly uninitialized outside the produced prefix.
                unsafe {
                    filtering::filter_hough_line_region_to_polar(
                        self.stream_context,
                        &source,
                        delta,
                        threshold,
                        &mut lines.lines,
                        destination_roi,
                        &mut DeviceOutputView::from_output(&mut lines.line_count),
                    )
                }
            })?;
        }

        // SAFETY: completed Hough dispatch writes the count and exactly that initialized line prefix.
        unsafe { lines.assume_init() }
    }
}
