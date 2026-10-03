# Klara: when a convincing preview gets the clothing wrong

Anas Mhana | DS#055 | Final evidence: 3 October 2026

## The question

A virtual try-on image can look believable while changing the very item someone wants to buy. In this project, I investigated whether fine-tuning CatVTON-MaskFree's existing self-attention weights improves garment reconstruction while preserving the person and background. The audience is developers evaluating online clothing-preview tools.

The distinction matters: generating a plausible image is different from faithfully showing the requested garment. I therefore treated the application as a demonstration of an experiment, rather than as evidence of model quality by itself.

## What the experiment found

Average garment MAE fell from 0.13355 to 0.11270, a 15.6% reduction, on 27 complete paired test cases. Twenty-one cases improved and six worsened. Garment SSIM rose from 0.40831 to 0.47388, with improvement in 23 cases.

Preservation was less consistent. Mean outside-clothing MAE relative to the input fell from 0.04769 to 0.03534, but 14 of 27 cases became worse. Larger improvements in some cases outweighed smaller regressions elsewhere. Eleven cases improved both garment MAE and outside-input MAE, ten improved only garment MAE, two improved only outside-input MAE, and four improved neither.

My conclusion is therefore limited: attention fine-tuning improved average reference reconstruction under the tested settings, but it did not reliably preserve every person and background.

## Data and experiment design

VITON-HD provides the original person photographs and catalog garments. VITON-HD-edit provides edited person images and masks. For each case, the edited person and garment are the inputs, and the original photograph is the reference target. This is an aligned reconstruction task, not a measurement of garment sizing or physical fit.

The frozen custom split contains 1,000 training cases, 64 validation cases and 32 final-test cases. It reuses the original VITON-HD test partition, so these are not official benchmark scores. Source-ID groups were separate. All 160 final-test source files passed technical checks, and byte/pixel hashes showed no exact overlap with 3,192 development RGB files. These checks do not establish person-identity separation or absence from upstream model training.

The dataset revision is 9de94ee10e15f5069fd650ce4c914215f124eb6f. Technical checks covered all training files, while visual review covered a predetermined sample. I do not claim that every training image received human review. The final test is now consumed evaluation data and cannot remain an untouched test if future choices use its results.

## What changed in the network

The starting point was the pinned CatVTON-MaskFree checkpoint, which uses an InstructPix2Pix diffusion U-Net and a separately pretrained VAE. The adaptation trained 49,574,080 parameters in 16 existing self-attention modules: query, key, value and output projections. No attention layers were added. The VAE and other U-Net parameters stayed frozen, with unchanged hashes after training.

Training used 384 by 512 images, batch size one, three shuffled epochs and 3,000 optimizer updates. AdamW used learning rate 0.00001 and weight decay 0.01, with gradient clipping at 1.0 and garment-condition dropout of 0.1. The training seed was 9026. The runner sampled and cached latents once, then sampled a timestep and diffusion noise at each update.

The loss was ordinary noise-prediction MSE over the target-plus-garment latent canvas. The optional preservation-region loss had weight zero. This is a standard denoising adaptation, not a reproduction of the researchers' DREAM training procedure. The training log contains every training ID once per epoch, and the saved checkpoint passed its reload check.

## How I compared the models

Both models used the same 32 final-test inputs, full-frame resize, seeds 9026 and 9027, 20 DDIM steps, guidance 2.5 and eta 1.0. Random states were reset before each attempt. The checkpoint and protocol were fixed before final generation, and the test results did not guide another training run or checkpoint choice.

There were 128 attempts and 122 generated images. The safety checker excluded two pretrained outputs and four fine-tuned outputs across five cases. Excluded outputs were not rerolled or counted as successful images. The primary paired analysis uses 27 cases with both seeds valid for both models.

The primary measure is garment-region MAE against the reference target, normalized to 0-1. Lower is better. Secondary measures are garment-region SSIM, where higher is better, and outside-clothing MAE against the input and target. Outside-target MAE fell from 0.04812 to 0.02938 and improved in 22 cases.

The garment region is the target clothing mask eroded by three pixels, excluding a three-pixel border. The outside region lies beyond the union of input and target masks dilated by eight pixels. Regions smaller than 64 pixels are unavailable rather than zero. The generator itself receives no mask. SSIM uses scikit-image 0.26.0, window size seven, data range 255, uniform weights and sample covariance.

I averaged the two seeds within each case and weighted cases equally. The descriptive paired bootstrap uses 10,000 case resamples and seed 41026. The garment MAE difference, fine-tuned minus pretrained, is -0.02085, with a descriptive 95% interval of [-0.03222, -0.01032]. This interval describes case-sampling uncertainty under the recorded experiment. It does not establish independent people or stability across training seeds.

## What the pictures add

Case 08931_00 has the largest case-average garment-MAE reduction. At seed 9026, fine-tuning removes large clothing artifacts, but the output still raises an arm and changes the trousers. Case 07703_00 shows a pink striped shirt whose color and hem differ from the reference. Other examples lose fabric texture or extend the shirt into a skirt region. The second seed can change the outcome.

The full gallery includes every case and both seeds, with safety-exclusion placeholders. Assistant inspection covered all 64 case-seed rows. These observations help interpret the metrics, but they are not blinded human ratings or a survey. The presentation examples have explicit case IDs, seeds and selection rules.

## Limits and recommendation

The data contains edited, aligned inputs and a restricted range of garments, poses and people. The final analysis has only 27 complete pairs and one training seed. Safety exclusions can bias complete-case results. Exact duplicate checks do not prove identity-disjoint data or exclude upstream exposure. Pixel and structural metrics cannot establish realism, exact logos, identity retention, physical fit or customer preference.

The live website uses the same adapted attention weights at 768 by 1024 pixels and 50 steps. The 384 by 512, 20-step research scores do not directly measure that website configuration. I recommend presenting Klara as a research demonstration with visible failures. Broader independent data and repeated training seeds would be needed before making stronger quality claims.

## Reproducibility and contribution

The repository includes the frozen selections, raw attempt records, image hashes, generated outputs, training log, executable analysis notebook and source code. The large checkpoint is provided separately in the course evidence bundle. Its SHA256 is e4f332704879fd3929120c4c038cbb65837c8a9c694bdb9fc75bd5a18924de38. The executed training source is preserved alongside later commented teaching copies.

CatVTON's architecture and pretrained components belong to their authors. I directed this project, operated the environment, reviewed examples and studied the methods. Code, analysis tooling and presentation writing were developed with AI assistance. This work is a noncommercial academic adaptation and evaluation of an existing model.

## Sources

- VITON-HD: https://github.com/shadow2496/VITON-HD
- VITON-HD-edit: https://huggingface.co/datasets/NXN-Labs/VITON-HD-edit
- CatVTON code: https://github.com/Zheng-Chong/CatVTON
- CatVTON paper: https://arxiv.org/abs/2407.15886
- MaskFree weights: https://huggingface.co/zhengchong/CatVTON-MaskFree
- Diffusion base: https://huggingface.co/timbrooks/instruct-pix2pix
- VAE: https://huggingface.co/stabilityai/sd-vae-ft-mse
