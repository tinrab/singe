// Verifications of the models under `model::diffusion`.

mod animatediff;
mod animatediff_video;
mod autoencoder_tiny;
mod ddpm;
mod instruct_pix2pix;
mod pia;
mod sd_turbo;
mod shap_e;
mod shap_e_img2img;

use crate::support::verifications;

verifications!(
    "diffusion",
    [
        animatediff,
        animatediff_video,
        autoencoder_tiny,
        ddpm,
        instruct_pix2pix,
        pia,
        sd_turbo,
        shap_e,
        shap_e_img2img,
    ]
);
