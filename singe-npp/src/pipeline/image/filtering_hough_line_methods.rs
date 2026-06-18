use crate::{
    error::Result,
    image::{filtering, view::C1},
    types::PointPolar,
};

use super::{HoughLines, ImagePipeline};

impl<'a> ImagePipeline<'a, u8, C1> {
    pub fn filter_hough_lines(
        self,
        delta: PointPolar,
        threshold: i32,
        max_line_count: usize,
    ) -> Result<HoughLines> {
        let mut lines = HoughLines::create(max_line_count)?;

        {
            let source = self.view()?;
            filtering::filter_hough_line_to_polar(
                self.stream_context,
                &source,
                delta,
                threshold,
                &mut lines.lines,
                &mut lines.line_count,
            )?;
        }

        Ok(lines)
    }

    pub fn filter_hough_lines_region(
        self,
        delta: PointPolar,
        threshold: i32,
        destination_roi: [PointPolar; 2],
        max_line_count: usize,
    ) -> Result<HoughLines> {
        let mut lines = HoughLines::create(max_line_count)?;

        {
            let source = self.view()?;
            filtering::filter_hough_line_region_to_polar(
                self.stream_context,
                &source,
                delta,
                threshold,
                &mut lines.lines,
                destination_roi,
                &mut lines.line_count,
            )?;
        }

        Ok(lines)
    }
}
