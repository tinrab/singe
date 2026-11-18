// AnimateDiff, redrawing a clip after a prompt, as `verify.py` redraws it with `diffusers`.

use singe::DataType;
use singe_model::task::vision::video_to_video::VideoToVideoPipeline;

use crate::support::Session;

pub fn verify(session: &mut Session) {
    session.check::<VideoToVideoPipeline>("redraw", DataType::F16);
}
