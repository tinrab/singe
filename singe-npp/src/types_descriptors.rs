use super::*;

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct HistogramOfGradientsConfig {
    pub cell_size: i32,
    pub histogram_block_size: i32,
    pub histogram_bins: i32,
    pub detection_window_size: Size,
}

impl Default for HistogramOfGradientsConfig {
    fn default() -> Self {
        Self {
            cell_size: 0,
            histogram_block_size: 0,
            histogram_bins: 0,
            detection_window_size: Size::from_raw_parts(0, 0),
        }
    }
}

impl From<HistogramOfGradientsConfig> for sys::NppiHOGConfig {
    fn from(value: HistogramOfGradientsConfig) -> Self {
        Self {
            cellSize: value.cell_size,
            histogramBlockSize: value.histogram_block_size,
            nHistogramBins: value.histogram_bins,
            detectionWindowSize: value.detection_window_size.into(),
        }
    }
}

impl From<sys::NppiHOGConfig> for HistogramOfGradientsConfig {
    fn from(value: sys::NppiHOGConfig) -> Self {
        Self {
            cell_size: value.cellSize,
            histogram_block_size: value.histogramBlockSize,
            histogram_bins: value.nHistogramBins,
            detection_window_size: value.detectionWindowSize.into(),
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP Haar classifier descriptor.
///
/// Raw pointer fields must reference device memory that remains valid for the
/// full NPP call that consumes this descriptor. The descriptor does not own or
/// free those pointers.
pub struct HaarClassifier32f {
    pub classifier_count: i32,
    pub classifiers: *mut i32,
    pub classifier_step: usize,
    pub classifier_size: Size,
    pub device_counter: *mut i32,
}

impl From<HaarClassifier32f> for sys::NppiHaarClassifier_32f {
    fn from(value: HaarClassifier32f) -> Self {
        Self {
            numClassifiers: value.classifier_count,
            classifiers: value.classifiers,
            classifierStep: value.classifier_step as _,
            classifierSize: value.classifier_size.into(),
            counterDevice: value.device_counter,
        }
    }
}

impl From<sys::NppiHaarClassifier_32f> for HaarClassifier32f {
    fn from(value: sys::NppiHaarClassifier_32f) -> Self {
        Self {
            classifier_count: value.numClassifiers,
            classifiers: value.classifiers,
            classifier_step: value.classifierStep as _,
            classifier_size: value.classifierSize.into(),
            device_counter: value.counterDevice,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP Haar buffer descriptor.
///
/// `buffer` must reference device memory that remains valid for the full NPP
/// call that consumes this descriptor. The descriptor does not own or free the
/// pointer.
pub struct HaarBuffer {
    pub size: i32,
    pub buffer: *mut i32,
}

impl From<HaarBuffer> for sys::NppiHaarBuffer {
    fn from(value: HaarBuffer) -> Self {
        Self {
            haarBufferSize: value.size,
            haarBuffer: value.buffer,
        }
    }
}

impl From<sys::NppiHaarBuffer> for HaarBuffer {
    fn from(value: sys::NppiHaarBuffer) -> Self {
        Self {
            size: value.haarBufferSize,
            buffer: value.haarBuffer,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ConnectedRegion {
    pub bounding_box: Rectangle,
    pub connected_pixel_count: u32,
    pub seed_pixel_value: [u32; 3],
}

impl From<ConnectedRegion> for sys::NppiConnectedRegion {
    fn from(value: ConnectedRegion) -> Self {
        Self {
            oBoundingBox: value.bounding_box.into(),
            nConnectedPixelCount: value.connected_pixel_count,
            aSeedPixelValue: value.seed_pixel_value,
        }
    }
}

impl From<sys::NppiConnectedRegion> for ConnectedRegion {
    fn from(value: sys::NppiConnectedRegion) -> Self {
        Self {
            bounding_box: value.oBoundingBox.into(),
            connected_pixel_count: value.nConnectedPixelCount,
            seed_pixel_value: value.aSeedPixelValue,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP image descriptor.
///
/// `data` must reference image memory that remains valid for the full NPP call
/// that consumes this descriptor. The descriptor does not own or free the
/// pointer.
pub struct ImageDescriptor {
    pub data: *mut (),
    pub step: i32,
    pub size: Size,
}

impl From<ImageDescriptor> for sys::NppiImageDescriptor {
    fn from(value: ImageDescriptor) -> Self {
        Self {
            pData: value.data as _,
            nStep: value.step,
            oSize: value.size.into(),
        }
    }
}

impl From<sys::NppiImageDescriptor> for ImageDescriptor {
    fn from(value: sys::NppiImageDescriptor) -> Self {
        Self {
            data: value.pData as _,
            step: value.nStep,
            size: value.oSize.into(),
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP scratch/data buffer descriptor.
///
/// `data` must reference memory that remains valid for the full NPP call that
/// consumes this descriptor. The descriptor does not own or free the pointer.
pub struct BufferDescriptor {
    pub data: *mut (),
    pub size: i32,
}

impl From<BufferDescriptor> for sys::NppiBufferDescriptor {
    fn from(value: BufferDescriptor) -> Self {
        Self {
            pData: value.data as _,
            nBufferSize: value.size,
        }
    }
}

impl From<sys::NppiBufferDescriptor> for BufferDescriptor {
    fn from(value: sys::NppiBufferDescriptor) -> Self {
        Self {
            data: value.pData as _,
            size: value.nBufferSize,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct CompressedMarkerLabelsInfo {
    pub marker_label_pixel_count: u32,
    pub contour_pixel_count: u32,
    pub contour_pixels_found: u32,
    pub contour_first_pixel_location: Point,
    pub marker_label_bounding_box: Rectangle,
}

impl From<CompressedMarkerLabelsInfo> for sys::NppiCompressedMarkerLabelsInfo {
    fn from(value: CompressedMarkerLabelsInfo) -> Self {
        Self {
            nMarkerLabelPixelCount: value.marker_label_pixel_count,
            nContourPixelCount: value.contour_pixel_count,
            nContourPixelsFound: value.contour_pixels_found,
            oContourFirstPixelLocation: value.contour_first_pixel_location.into(),
            oMarkerLabelBoundingBox: value.marker_label_bounding_box.into(),
        }
    }
}

impl From<sys::NppiCompressedMarkerLabelsInfo> for CompressedMarkerLabelsInfo {
    fn from(value: sys::NppiCompressedMarkerLabelsInfo) -> Self {
        Self {
            marker_label_pixel_count: value.nMarkerLabelPixelCount,
            contour_pixel_count: value.nContourPixelCount,
            contour_pixels_found: value.nContourPixelsFound,
            contour_first_pixel_location: value.oContourFirstPixelLocation.into(),
            marker_label_bounding_box: value.oMarkerLabelBoundingBox.into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ContourBlockSegment {
    pub marker_label_id: u32,
    pub contour_pixel_count: u32,
    pub contour_starting_pixel_offset: u32,
    pub segment_number: u32,
}

impl From<ContourBlockSegment> for sys::NppiContourBlockSegment {
    fn from(value: ContourBlockSegment) -> Self {
        Self {
            nMarkerLabelID: value.marker_label_id,
            nContourPixelCount: value.contour_pixel_count,
            nContourStartingPixelOffset: value.contour_starting_pixel_offset,
            nSegmentNum: value.segment_number,
        }
    }
}

impl From<sys::NppiContourBlockSegment> for ContourBlockSegment {
    fn from(value: sys::NppiContourBlockSegment) -> Self {
        Self {
            marker_label_id: value.nMarkerLabelID,
            contour_pixel_count: value.nContourPixelCount,
            contour_starting_pixel_offset: value.nContourStartingPixelOffset,
            segment_number: value.nSegmentNum,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ContourPixelGeometryInfo {
    pub contour_ordered_geometry_location: Point,
    pub previous_contour_pixel_location: Point,
    pub contour_center_pixel_location: Point,
    pub next_contour_pixel_location: Point,
    pub order_index: i32,
    pub reverse_order_index: i32,
    pub first_index: u32,
    pub last_index: u32,
    pub next_contour_pixel_index: u32,
    pub previous_contour_pixel_index: u32,
    pub pixel_already_used: u8,
    pub already_linked: u8,
    pub already_output: u8,
    pub contour_interior_direction: u8,
}

impl From<ContourPixelGeometryInfo> for sys::NppiContourPixelGeometryInfo {
    fn from(value: ContourPixelGeometryInfo) -> Self {
        Self {
            oContourOrderedGeometryLocation: value.contour_ordered_geometry_location.into(),
            oContourPrevPixelLocation: value.previous_contour_pixel_location.into(),
            oContourCenterPixelLocation: value.contour_center_pixel_location.into(),
            oContourNextPixelLocation: value.next_contour_pixel_location.into(),
            nOrderIndex: value.order_index,
            nReverseOrderIndex: value.reverse_order_index,
            nFirstIndex: value.first_index,
            nLastIndex: value.last_index,
            nNextContourPixelIndex: value.next_contour_pixel_index,
            nPrevContourPixelIndex: value.previous_contour_pixel_index,
            nPixelAlreadyUsed: value.pixel_already_used,
            nAlreadyLinked: value.already_linked,
            nAlreadyOutput: value.already_output,
            nContourInteriorDirection: value.contour_interior_direction,
        }
    }
}

impl From<sys::NppiContourPixelGeometryInfo> for ContourPixelGeometryInfo {
    fn from(value: sys::NppiContourPixelGeometryInfo) -> Self {
        Self {
            contour_ordered_geometry_location: value.oContourOrderedGeometryLocation.into(),
            previous_contour_pixel_location: value.oContourPrevPixelLocation.into(),
            contour_center_pixel_location: value.oContourCenterPixelLocation.into(),
            next_contour_pixel_location: value.oContourNextPixelLocation.into(),
            order_index: value.nOrderIndex,
            reverse_order_index: value.nReverseOrderIndex,
            first_index: value.nFirstIndex,
            last_index: value.nLastIndex,
            next_contour_pixel_index: value.nNextContourPixelIndex,
            previous_contour_pixel_index: value.nPrevContourPixelIndex,
            pixel_already_used: value.nPixelAlreadyUsed,
            already_linked: value.nAlreadyLinked,
            already_output: value.nAlreadyOutput,
            contour_interior_direction: value.nContourInteriorDirection,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ContourPixelDirectionInfo {
    pub marker_label_id: u32,
    pub contour_direction_center_pixel: u8,
    pub contour_interior_direction_center_pixel: u8,
    pub connected: u8,
    pub geometry_info_is_valid: u8,
    pub contour_pixel_geometry_info: ContourPixelGeometryInfo,
    pub east: Point,
    pub north_east: Point,
    pub north: Point,
    pub north_west: Point,
    pub west: Point,
    pub south_west: Point,
    pub south: Point,
    pub south_east: Point,
    pub east_connected: u8,
    pub north_east_connected: u8,
    pub north_connected: u8,
    pub north_west_connected: u8,
    pub west_connected: u8,
    pub south_west_connected: u8,
    pub south_connected: u8,
    pub south_east_connected: u8,
}

impl From<ContourPixelDirectionInfo> for sys::NppiContourPixelDirectionInfo {
    fn from(value: ContourPixelDirectionInfo) -> Self {
        Self {
            nMarkerLabelID: value.marker_label_id,
            nContourDirectionCenterPixel: value.contour_direction_center_pixel,
            nContourInteriorDirectionCenterPixel: value.contour_interior_direction_center_pixel,
            nConnected: value.connected,
            nGeometryInfoIsValid: value.geometry_info_is_valid,
            oContourPixelGeometryInfo: value.contour_pixel_geometry_info.into(),
            nEast1: value.east.into(),
            nNorthEast1: value.north_east.into(),
            nNorth1: value.north.into(),
            nNorthWest1: value.north_west.into(),
            nWest1: value.west.into(),
            nSouthWest1: value.south_west.into(),
            nSouth1: value.south.into(),
            nSouthEast1: value.south_east.into(),
            nTest1EastConnected: value.east_connected,
            nTest1NorthEastConnected: value.north_east_connected,
            nTest1NorthConnected: value.north_connected,
            nTest1NorthWestConnected: value.north_west_connected,
            nTest1WestConnected: value.west_connected,
            nTest1SouthWestConnected: value.south_west_connected,
            nTest1SouthConnected: value.south_connected,
            nTest1SouthEastConnected: value.south_east_connected,
        }
    }
}

impl From<sys::NppiContourPixelDirectionInfo> for ContourPixelDirectionInfo {
    fn from(value: sys::NppiContourPixelDirectionInfo) -> Self {
        Self {
            marker_label_id: value.nMarkerLabelID,
            contour_direction_center_pixel: value.nContourDirectionCenterPixel,
            contour_interior_direction_center_pixel: value.nContourInteriorDirectionCenterPixel,
            connected: value.nConnected,
            geometry_info_is_valid: value.nGeometryInfoIsValid,
            contour_pixel_geometry_info: value.oContourPixelGeometryInfo.into(),
            east: value.nEast1.into(),
            north_east: value.nNorthEast1.into(),
            north: value.nNorth1.into(),
            north_west: value.nNorthWest1.into(),
            west: value.nWest1.into(),
            south_west: value.nSouthWest1.into(),
            south: value.nSouth1.into(),
            south_east: value.nSouthEast1.into(),
            east_connected: value.nTest1EastConnected,
            north_east_connected: value.nTest1NorthEastConnected,
            north_connected: value.nTest1NorthConnected,
            north_west_connected: value.nTest1NorthWestConnected,
            west_connected: value.nTest1WestConnected,
            south_west_connected: value.nTest1SouthWestConnected,
            south_connected: value.nTest1SouthConnected,
            south_east_connected: value.nTest1SouthEastConnected,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ContourTotalsInfo {
    pub total_image_pixel_contour_count: u32,
    pub longest_image_contour_pixel_count: u32,
}

impl From<ContourTotalsInfo> for sys::NppiContourTotalsInfo {
    fn from(value: ContourTotalsInfo) -> Self {
        Self {
            nTotalImagePixelContourCount: value.total_image_pixel_contour_count,
            nLongestImageContourPixelCount: value.longest_image_contour_pixel_count,
        }
    }
}

impl From<sys::NppiContourTotalsInfo> for ContourTotalsInfo {
    fn from(value: sys::NppiContourTotalsInfo) -> Self {
        Self {
            total_image_pixel_contour_count: value.nTotalImagePixelContourCount,
            longest_image_contour_pixel_count: value.nLongestImageContourPixelCount,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP resize batch descriptor.
///
/// `source` and `destination` must remain valid for the full NPP call that
/// consumes this descriptor. The descriptor does not own or free either
/// pointer.
pub struct ResizeBatchDescriptor {
    pub source: *const (),
    pub source_step: i32,
    pub destination: *mut (),
    pub destination_step: i32,
}

impl From<ResizeBatchDescriptor> for sys::NppiResizeBatchCXR {
    fn from(value: ResizeBatchDescriptor) -> Self {
        Self {
            pSrc: value.source as _,
            nSrcStep: value.source_step,
            pDst: value.destination as _,
            nDstStep: value.destination_step,
        }
    }
}

impl From<sys::NppiResizeBatchCXR> for ResizeBatchDescriptor {
    fn from(value: sys::NppiResizeBatchCXR) -> Self {
        Self {
            source: value.pSrc as _,
            source_step: value.nSrcStep,
            destination: value.pDst as _,
            destination_step: value.nDstStep,
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialOrd, Ord, PartialEq, Eq, Default)]
pub struct ResizeBatchRoiAdvanced {
    pub source_roi: Rectangle,
    pub destination_roi: Rectangle,
}

impl From<ResizeBatchRoiAdvanced> for sys::NppiResizeBatchROI_Advanced {
    fn from(value: ResizeBatchRoiAdvanced) -> Self {
        Self {
            oSrcRectROI: value.source_roi.into(),
            oDstRectROI: value.destination_roi.into(),
        }
    }
}

impl From<sys::NppiResizeBatchROI_Advanced> for ResizeBatchRoiAdvanced {
    fn from(value: sys::NppiResizeBatchROI_Advanced) -> Self {
        Self {
            source_roi: value.oSrcRectROI.into(),
            destination_roi: value.oDstRectROI.into(),
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP mirror batch descriptor.
///
/// `source` and `destination` must remain valid for the full NPP call that
/// consumes this descriptor. The descriptor does not own or free either
/// pointer.
pub struct MirrorBatchDescriptor {
    pub source: *const (),
    pub source_step: i32,
    pub destination: *mut (),
    pub destination_step: i32,
}

impl From<MirrorBatchDescriptor> for sys::NppiMirrorBatchCXR {
    fn from(value: MirrorBatchDescriptor) -> Self {
        Self {
            pSrc: value.source as _,
            nSrcStep: value.source_step,
            pDst: value.destination as _,
            nDstStep: value.destination_step,
        }
    }
}

impl From<sys::NppiMirrorBatchCXR> for MirrorBatchDescriptor {
    fn from(value: sys::NppiMirrorBatchCXR) -> Self {
        Self {
            source: value.pSrc as _,
            source_step: value.nSrcStep,
            destination: value.pDst as _,
            destination_step: value.nDstStep,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, PartialOrd, PartialEq)]
/// Non-owning NPP affine-warp batch descriptor.
///
/// Raw pointer fields must remain valid for the full NPP call that consumes
/// this descriptor. The descriptor does not own or free those pointers.
pub struct WarpAffineBatchDescriptor {
    pub source: *const (),
    pub source_step: i32,
    pub destination: *mut (),
    pub destination_step: i32,
    pub coefficients: *mut f64,
    pub transformed_coefficients: [[f64; 3]; 2],
}

impl From<WarpAffineBatchDescriptor> for sys::NppiWarpAffineBatchCXR {
    fn from(value: WarpAffineBatchDescriptor) -> Self {
        Self {
            pSrc: value.source as _,
            nSrcStep: value.source_step,
            pDst: value.destination as _,
            nDstStep: value.destination_step,
            pCoeffs: value.coefficients,
            aTransformedCoeffs: value.transformed_coefficients,
        }
    }
}

impl From<sys::NppiWarpAffineBatchCXR> for WarpAffineBatchDescriptor {
    fn from(value: sys::NppiWarpAffineBatchCXR) -> Self {
        Self {
            source: value.pSrc as _,
            source_step: value.nSrcStep,
            destination: value.pDst as _,
            destination_step: value.nDstStep,
            coefficients: value.pCoeffs,
            transformed_coefficients: value.aTransformedCoeffs,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, PartialOrd, PartialEq)]
/// Non-owning NPP perspective-warp batch descriptor.
///
/// Raw pointer fields must remain valid for the full NPP call that consumes
/// this descriptor. The descriptor does not own or free those pointers.
pub struct WarpPerspectiveBatchDescriptor {
    pub source: *const (),
    pub source_step: i32,
    pub destination: *mut (),
    pub destination_step: i32,
    pub coefficients: *mut f64,
    pub transformed_coefficients: [[f64; 3]; 3],
}

impl From<WarpPerspectiveBatchDescriptor> for sys::NppiWarpPerspectiveBatchCXR {
    fn from(value: WarpPerspectiveBatchDescriptor) -> Self {
        Self {
            pSrc: value.source as _,
            nSrcStep: value.source_step,
            pDst: value.destination as _,
            nDstStep: value.destination_step,
            pCoeffs: value.coefficients,
            aTransformedCoeffs: value.transformed_coefficients,
        }
    }
}

impl From<sys::NppiWarpPerspectiveBatchCXR> for WarpPerspectiveBatchDescriptor {
    fn from(value: sys::NppiWarpPerspectiveBatchCXR) -> Self {
        Self {
            source: value.pSrc as _,
            source_step: value.nSrcStep,
            destination: value.pDst as _,
            destination_step: value.nDstStep,
            coefficients: value.pCoeffs,
            transformed_coefficients: value.aTransformedCoeffs,
        }
    }
}

#[non_exhaustive]
#[repr(C)]
#[derive(Debug, Hash, PartialOrd, Ord, PartialEq, Eq)]
/// Non-owning NPP color-twist batch descriptor.
///
/// Raw pointer fields must remain valid for the full NPP call that consumes
/// this descriptor. The descriptor does not own or free those pointers.
pub struct ColorTwistBatchDescriptor {
    pub source: *const (),
    pub source_step: i32,
    pub destination: *mut (),
    pub destination_step: i32,
    pub twist: *mut f32,
}

impl From<ColorTwistBatchDescriptor> for sys::NppiColorTwistBatchCXR {
    fn from(value: ColorTwistBatchDescriptor) -> Self {
        Self {
            pSrc: value.source as _,
            nSrcStep: value.source_step,
            pDst: value.destination as _,
            nDstStep: value.destination_step,
            pTwist: value.twist,
        }
    }
}

impl From<sys::NppiColorTwistBatchCXR> for ColorTwistBatchDescriptor {
    fn from(value: sys::NppiColorTwistBatchCXR) -> Self {
        Self {
            source: value.pSrc as _,
            source_step: value.nSrcStep,
            destination: value.pDst as _,
            destination_step: value.nDstStep,
            twist: value.pTwist,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, PartialEq, Default)]
pub struct ProfileData {
    pub pixels: i32,
    pub mean_intensity: f32,
    pub standard_deviation_intensity: f32,
}

impl From<ProfileData> for sys::NppiProfileData {
    fn from(value: ProfileData) -> Self {
        Self {
            nPixels: value.pixels,
            nMeanIntensity: value.mean_intensity,
            nStdDevIntensity: value.standard_deviation_intensity,
        }
    }
}

impl From<sys::NppiProfileData> for ProfileData {
    fn from(value: sys::NppiProfileData) -> Self {
        Self {
            pixels: value.nPixels,
            mean_intensity: value.nMeanIntensity,
            standard_deviation_intensity: value.nStdDevIntensity,
        }
    }
}
