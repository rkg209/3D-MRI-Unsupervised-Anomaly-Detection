# Architecture x loss matrix (Spec 005)

| model | loss | role | Dice | IoU | PSNR (dB) — context only | SSIM — context only | source |
|---|---|---|---|---|---|---|---|
| unetr | mse | headline | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | unetr__mse |
| unetr | ssim | headline | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | unetr__ssim |
| unetr | mse_ssim | headline | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | unetr__mse_ssim |
| unetr | multiscale_mse | headline | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | n/a (no checkpoint) | unetr__multiscale_mse |
| unetr | perceptual | headline | n/a (no training code, no checkpoint) | n/a (no training code, no checkpoint) | n/a (no training code, no checkpoint) | n/a (no training code, no checkpoint) | unetr__perceptual |
| unet | mse | reference | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | unet__mse |
| attention_unet | mse_ssim | reference | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | n/a (not evaluated) | attention_unet__mse_ssim |
| diffusion | ddpm | paradigm | n/a (untrained) | n/a (untrained) | n/a (untrained) | n/a (untrained) | diffusion__ddpm |
| unetr_synth | mse_ssim | novelty | n/a (not trained) | n/a (not trained) | n/a (not trained) | n/a (not trained) | unetr_synth__mse_ssim |
| msa_unetr | mse_ssim | stretch | n/a (stretch — not trained (Spec 012 gate closed)) | n/a (stretch — not trained (Spec 012 gate closed)) | n/a (stretch — not trained (Spec 012 gate closed)) | n/a (stretch — not trained (Spec 012 gate closed)) | msa_unetr__mse_ssim |

PSNR/SSIM are explanatory context only (NFR-6/NFR-22) — never an optimization target and never a headline number.

Spearman rho: n/a (fewer than 3 evaluated cells, n_pairs=0)

0/10 cells evaluated; 10 cell(s) n/a.
