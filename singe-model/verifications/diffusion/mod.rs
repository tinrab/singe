// Verifications of the models under `model::diffusion`.

#[path = "animatediff/verify.rs"]
mod animatediff;
#[path = "animatediff_video/verify.rs"]
mod animatediff_video;
#[path = "autoencoder_tiny/verify.rs"]
mod autoencoder_tiny;
#[path = "ddpm/verify.rs"]
mod ddpm;
#[path = "instruct_pix2pix/verify.rs"]
mod instruct_pix2pix;
#[path = "pia/verify.rs"]
mod pia;
#[path = "sd_turbo/verify.rs"]
mod sd_turbo;
#[path = "shap_e/verify.rs"]
mod shap_e;
#[path = "shap_e_img2img/verify.rs"]
mod shap_e_img2img;

use crate::support::Verification;

pub const VERIFICATIONS: &[Verification] = &[
    Verification {
        name: "diffusion/animatediff",
        verify: animatediff::verify,
    },
    Verification {
        name: "diffusion/animatediff_video",
        verify: animatediff_video::verify,
    },
    Verification {
        name: "diffusion/autoencoder_tiny",
        verify: autoencoder_tiny::verify,
    },
    Verification {
        name: "diffusion/ddpm",
        verify: ddpm::verify,
    },
    Verification {
        name: "diffusion/instruct_pix2pix",
        verify: instruct_pix2pix::verify,
    },
    Verification {
        name: "diffusion/pia",
        verify: pia::verify,
    },
    Verification {
        name: "diffusion/sd_turbo",
        verify: sd_turbo::verify,
    },
    Verification {
        name: "diffusion/shap_e",
        verify: shap_e::verify,
    },
    Verification {
        name: "diffusion/shap_e_img2img",
        verify: shap_e_img2img::verify,
    },
];
