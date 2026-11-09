use crate::crc32::Spec;

/// Computes the CRC32 of `input` for `spec` on the CPU.
///
/// Use it to check results from the GPU.
#[must_use]
pub fn cpu_crc32(spec: Spec, input: &[u8]) -> u32 {
    let mut crc = spec.init;

    for byte in input {
        crc ^= if spec.ref_in {
            reverse(u32::from(*byte))
        } else {
            u32::from(*byte) << 24
        };

        for _ in 0..8 {
            crc = (crc << 1) ^ if crc & 0x8000_0000 != 0 { spec.poly } else { 0 };
        }
    }

    if spec.ref_out {
        crc = reverse(crc);
    }

    crc ^ spec.xorout
}

fn reverse(mut x: u32) -> u32 {
    x = ((x & 0xaaaa_aaaa) >> 1) | ((x & 0x5555_5555) << 1);
    x = ((x & 0xcccc_cccc) >> 2) | ((x & 0x3333_3333) << 2);
    x = ((x & 0xf0f0_f0f0) >> 4) | ((x & 0x0f0f_0f0f) << 4);
    x = ((x & 0xff00_ff00) >> 8) | ((x & 0x00ff_00ff) << 8);

    x.rotate_left(16)
}
