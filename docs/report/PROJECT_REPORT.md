# Klara: when a convincing preview gets the clothing wrong

Anas Mhana, DS#055
Results recorded on 3 October 2026

## The question

When I tried the clothing examples in Klara, I noticed that a believable image could still show the wrong color or extra clothing. That made me question how useful the preview would be to someone choosing what to buy. In this project, I investigated whether fine-tuning CatVTON-MaskFree's existing self-attention weights improves garment reconstruction while preserving the person and background. The project is intended for developers evaluating online clothing-preview tools.

I wanted to judge the generated images against a reference photograph, rather than rely on how convincing the app looked. The website gives people a way to try the model. The experiment provides the evidence for its strengths and weaknesses.

## What the experiment found

Average garment MAE fell from 0.13355 to 0.11270, a 15.6% reduction, on 27 complete paired test cases. Twenty-one cases improved and six worsened. Garment SSIM rose from 0.40831 to 0.47388, with improvement in 23 cases.

Preservation was less consistent. Mean outside-clothing MAE relative to the input fell from 0.04769 to 0.03534, but 14 of 27 cases became worse. Larger improvements in some cases outweighed smaller regressions elsewhere. Eleven cases improved both garment MAE and outside-input MAE, ten improved only garment MAE, two improved only outside-input MAE, and four improved neither.

These results give a mixed answer to my question. Fine-tuning improved the average clothing match under the tested settings, but unwanted changes to the person and background remained.

## Data and experiment design

VITON-HD provides the original person photographs and catalog garments. VITON-HD-edit provides edited person images and masks. For each case, the edited person and garment are the inputs, and the original photograph is the reference target. This is an aligned reconstruction task, not a measurement of garment sizing or physical fit.

I fixed a custom split containing 1,000 training cases, 64 validation cases and 32 final-test cases. It reuses the original VITON-HD test partition, so these are not official benchmark scores. The splits used separate source-ID groups. All 160 final-test source files passed technical checks, and byte/pixel hashes showed no exact overlap with 3,192 development RGB files. These checks do not establish person-identity separation or absence from upstream model training.

The dataset revision is 9de94ee10e15f5069fd650ce4c914215f124eb6f. Automated checks covered all training files. Visual review covered a sample chosen in advance, not every image. Now that I have seen the final results, I would need a new test set if I used these findings to guide further model changes.

## What changed in the network

The starting point was the pinned CatVTON-MaskFree checkpoint, which uses an InstructPix2Pix diffusion U-Net and a separately pretrained VAE. I fine-tuned 49,574,080 parameters in 16 existing self-attention modules: the query, key, value and output projections. The experiment uses the existing architecture. The VAE and other U-Net parameters stayed frozen, with unchanged hashes after training.

Training used 384 by 512 images, batch size one, three shuffled epochs and 3,000 optimizer updates. AdamW used learning rate 0.00001 and weight decay 0.01, with gradient clipping at 1.0 and garment-condition dropout of 0.1. The training seed was 9026. The runner sampled and cached latents once, then sampled a timestep and diffusion noise at each update.

The loss was ordinary noise-prediction MSE over the target-plus-garment latent canvas. The optional preservation-region loss had weight zero. This is a standard denoising adaptation, not a reproduction of the researchers' DREAM training procedure. The training log contains every training ID once per epoch, and the saved checkpoint passed its reload check.

## How I compared the models

Both models used the same 32 final-test inputs, full-frame resize, seeds 9026 and 9027, 20 DDIM steps, guidance 2.5 and eta 1.0. The runner reset the random state before each attempt. I fixed the checkpoint and evaluation settings before generating the final outputs. I didn't use the test results to choose another checkpoint or run more training.

There were 128 attempts and 122 generated images. The safety checker excluded two pretrained outputs and four fine-tuned outputs across five cases. I kept those exclusions in the record without retrying them or counting them as successful images. The main comparison therefore uses the 27 cases with valid outputs from both seeds and both models.

The primary measure is garment-region MAE against the reference target, normalized to 0-1. Lower is better. Secondary measures are garment-region SSIM, where higher is better, and outside-clothing MAE against the input and target. Outside-target MAE fell from 0.04812 to 0.02938 and improved in 22 cases.

The garment region is the target clothing mask eroded by three pixels, excluding a three-pixel border. The outside region lies beyond the union of input and target masks dilated by eight pixels. Regions smaller than 64 pixels are unavailable rather than zero. The generator itself receives no mask. SSIM uses scikit-image 0.26.0, window size seven, data range 255, uniform weights and sample covariance.

I averaged the two seeds within each case and weighted cases equally. The descriptive paired bootstrap uses 10,000 case resamples and seed 41026. The garment MAE difference, fine-tuned minus pretrained, is -0.02085, with a descriptive 95% interval of [-0.03222, -0.01032]. This interval describes case-sampling uncertainty under the recorded experiment. It does not establish independent people or stability across training seeds.

## What the pictures add

Case 08931_00 has the largest case-average garment-MAE reduction. At seed 9026, fine-tuning removes large clothing artifacts, but the output still raises an arm and changes the trousers. Case 07703_00 shows a pink striped shirt whose color and hem differ from the reference. Other examples lose fabric texture or extend the shirt into a skirt region. The second seed can change the outcome.

The evidence folder records every case and both seeds, covering all 64 combinations of case and seed. Excluded attempts remain in the result records. The qualitative observations help explain the scores, but are not independent human ratings or survey results. Each presentation example identifies its case, seed and reason for selection.

## Limits and recommendation

The data contains edited, aligned inputs and a restricted range of garments, poses and people. The final analysis has only 27 complete pairs and one training seed. Safety exclusions can bias complete-case results. Exact duplicate checks do not prove identity-disjoint data or exclude upstream exposure. Pixel and structural metrics cannot establish realism, exact logos, identity retention, physical fit or customer preference.

The live website uses the same adapted attention weights at 768 by 1024 pixels and 50 steps. The 384 by 512, 20-step research scores do not directly measure that website configuration. I would use Klara as a research demonstration and show the mistakes alongside the improvements. To make stronger claims, I would need more varied independent data and several training runs with different seeds.

## Reproducibility and contribution

The repository includes the frozen selections, raw attempt records, image hashes, generated outputs, training log, executable analysis notebook and source code. The large checkpoint is available separately on request. Its SHA256 is e4f332704879fd3929120c4c038cbb65837c8a9c694bdb9fc75bd5a18924de38. The executed training source is preserved alongside the experiment records.

The CatVTON researchers created the architecture and pretrained components. I directed this project, operated the environment, reviewed examples and studied the methods. This is a noncommercial academic study of how an existing model responds to fine-tuning.

## Sources

- VITON-HD: https://github.com/shadow2496/VITON-HD
- VITON-HD-edit: https://huggingface.co/datasets/NXN-Labs/VITON-HD-edit
- CatVTON code: https://github.com/Zheng-Chong/CatVTON
- CatVTON paper: https://arxiv.org/abs/2407.15886
- MaskFree weights: https://huggingface.co/zhengchong/CatVTON-MaskFree
- Diffusion base: https://huggingface.co/timbrooks/instruct-pix2pix
- VAE: https://huggingface.co/stabilityai/sd-vae-ft-mse
