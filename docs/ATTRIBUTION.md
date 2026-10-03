# Sources and contribution

## Data

- [VITON-HD](https://github.com/shadow2496/VITON-HD): original person photographs and catalog garment images. See `code/LICENSE-VITON-HD.txt` for the included Attribution-NonCommercial 4.0 notice, copyright NeStyle Inc.
- [VITON-HD-edit](https://huggingface.co/datasets/NXN-Labs/VITON-HD-edit): edited inputs and masks. Pinned revision: `9de94ee10e15f5069fd650ce4c914215f124eb6f`.

The repository contains selected research images and generated derivatives as evidence, with original and generated images labeled. Presentation overlays mark evaluation regions. These materials are for noncommercial academic use. No new unrestricted license is granted over third-party material.

## Models and libraries

- [CatVTON code](https://github.com/Zheng-Chong/CatVTON), [paper](https://arxiv.org/abs/2407.15886) and [MaskFree checkpoint](https://huggingface.co/zhengchong/CatVTON-MaskFree).
- [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix), the pretrained diffusion base used by this pipeline.
- [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse).
- PyTorch, Diffusers, Transformers, NumPy, pandas, Pillow, scikit-image and Matplotlib.

Included vendor folders retain their licenses and pinned revision records. CatVTON's model design and upstream implementations belong to their authors. The standard denoising adaptation here does not reproduce the researchers' DREAM training procedure.

## Project contribution

I chose and directed the project, operated the environment, reviewed examples and studied the model and evaluation methods. The contribution is an experiment on fine-tuning an existing model and evaluating the outcome.

The report uses qualitative observations to explain the saved examples. These are not independent human ratings or evidence of customer preferences. The original review records are retained with the experimental evidence.
