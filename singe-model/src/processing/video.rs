// Clips as models see them: frames, read from a directory of images in name order, as cases store a clip's decoded frames.
// A model's processor samples frames and transforms each as an image.

use std::{fs, path::Path, result::Result as StdResult};

use serde::{Deserialize, Deserializer, de::Error as DeError};

use crate::{
    error::{Error, Result},
    processing::image::Image,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Video {
    pub frames: Vec<Image>,
}

impl Video {
    // Reads every image in `directory`, in name order, as the clip's frames.
    pub fn open(directory: impl AsRef<Path>) -> Result<Self> {
        let directory = directory.as_ref();
        let mut paths = fs::read_dir(directory)
            .map_err(Error::io(directory))?
            .map(|entry| {
                entry
                    .map(|entry| entry.path())
                    .map_err(Error::io(directory))
            })
            .collect::<Result<Vec<_>>>()?;
        paths.retain(|path| {
            path.extension()
                .is_some_and(|extension| extension == "png" || extension == "jpg")
        });
        paths.sort();
        if paths.is_empty() {
            return Err(Error::Media(format!(
                "{}: a clip without frames",
                directory.display()
            )));
        }

        Ok(Self {
            frames: paths.iter().map(Image::open).collect::<Result<_>>()?,
        })
    }

    // Samples `count` frames from the first `count · rate`, evenly, as `transformers`' video pipeline does: `numpy.linspace(0, count · rate - 1, count)` truncated to indices.
    pub fn sample(&self, count: usize, rate: usize) -> Result<Vec<&Image>> {
        let last = count
            .checked_mul(rate)
            .and_then(|span| span.checked_sub(1))
            .ok_or_else(|| {
                Error::UnsupportedConfig(format!("a sample of {count} frames at a rate of {rate}"))
            })?;
        (0..count)
            .map(|index| {
                #[expect(
                    clippy::cast_possible_truncation,
                    clippy::cast_precision_loss,
                    clippy::cast_sign_loss,
                    reason = "frame indices are small, and `numpy` truncates them"
                )]
                let frame = if count == 1 {
                    0
                } else {
                    (index as f64 * last as f64 / (count - 1) as f64) as usize
                };
                self.frames.get(frame).ok_or_else(|| {
                    Error::Media(format!(
                        "frame {frame} of a clip of {} frames",
                        self.frames.len()
                    ))
                })
            })
            .collect()
    }
}

// A clip in JSON names the directory its frames are read from.
impl<'de> Deserialize<'de> for Video {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> StdResult<Self, D::Error> {
        let path = String::deserialize(deserializer)?;
        Self::open(&path).map_err(DeError::custom)
    }
}
