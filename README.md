# Klara: can we trust the clothing preview?

**A data-science project by Anas Mhana, WBS Coding School DS#055.**

An AI clothing preview can look convincing while changing the shirt's color, inventing a pattern or altering the person's trousers. That gap between a plausible picture and the requested garment is the reason for this project.

I investigated one question: **does fine-tuning CatVTON-MaskFree's existing self-attention weights improve garment reconstruction while preserving the person and background?** The intended audience is developers evaluating virtual clothing previews for online shoppers.

## What I found

On the final test, average garment error fell **15.6%**, with improvement in **21 of 27 complete paired cases**. The preservation result was mixed: average outside-clothing error fell, but **14 of 27 cases became worse** on that measure. Better average reconstruction did not make every preview dependable.

| Measure | Pretrained | Fine-tuned | Cases improved |
|---|---:|---:|---:|
| Garment MAE, lower is better | 0.13355 | 0.11270 | 21 / 27 |
| Garment SSIM, higher is better | 0.40831 | 0.47388 | 23 / 27 |
| Outside-clothing MAE to input, lower is better | 0.04769 | 0.03534 | 13 / 27 |

![Each point compares garment error with preservation error for one final-test case.](docs/final/preservation-tradeoff.svg)

The experiment attempted 32 cases with two models and two fixed seeds: 128 attempts. Six safety exclusions across five cases left 27 complete pairs. I kept those exclusions, used no rerolls, and averaged seeds within each case. These are reference-image measurements, not an accuracy percentage or a test of physical fit.

## The experiment

- **Data:** VITON-HD and VITON-HD-edit. Each case has an edited person image, catalog garment, original reference photograph and evaluation masks.
- **Split:** 1,000 training cases, 64 validation cases and 32 final-test cases. This custom split reuses the original VITON-HD test partition; it is not an official benchmark result.
- **Adaptation:** 49,574,080 parameters in 16 existing self-attention modules. The VAE and other U-Net weights stayed frozen.
- **Training:** three epochs, 3,000 AdamW updates, batch size 1, learning rate 0.00001, noise-prediction MSE. The optional preservation loss had weight zero.
- **Final evaluation:** 384×512 images, 20 DDIM steps, guidance 2.5, eta 1.0, seeds 9026 and 9027. Both models used the same settings.

## Start here

1. [Final presentation](docs/final/Klara_Final.pdf) and [speaker notes](docs/final/SPEAKER_NOTES.md).
2. [Research report](docs/report/PROJECT_REPORT.md), with the full method and limitations.
3. [Executed analysis notebook](notebooks/Klara_Final_Study.ipynb).
4. [MVP presentation](docs/mvp/Klara_Mvp.pdf), focused on the question and experimental design.
5. [Code map](docs/CODE_MAP.md) and [reproduction guide](REPRODUCE.md).

After cloning or downloading the repository, open `demo/index.html` for the complete recorded gallery. It works offline. GitHub's normal file view does not run the HTML application.

The live research demo is [klara-app.de](https://klara-app.de). It needs the GPU server and an access code provided separately. The site uses the trained checkpoint at 768×1024 and 50 steps, so the controlled research scores do not directly measure website quality.

## Reproduce the analysis

Use Python 3.10 or newer in an activated virtual environment, from the repository root:

```bash
python -m pip install -r notebooks/requirements.txt
python code/analyze_final_1000.py --output evidence/final-test-1000-01 --plan data/final-evaluation-plan.json --evaluator code/training-feasibility/evaluate_final_1000.py
```

This recomputes the image metrics and paired analysis on a CPU. It makes no model, cloud or paid API calls. The notebook also checks the recorded training epochs. The large attention checkpoint is supplied separately in the course evidence bundle; its hash and loading instructions are in [CHECKPOINT.md](CHECKPOINT.md).

## Limits I would keep in mind

The inputs are edited versions of aligned reference photos. They do not represent every shopper, pose or garment. Source-ID and exact-image checks passed, but person-identity separation and upstream pretraining exposure remain unverified. The test is small, complete-case exclusion can bias the results, and only one training seed was used. MAE and SSIM cannot establish realism, exact logos, identity retention, sizing or user preference.

## Attribution and contribution

The model and underlying implementations come from the CatVTON researchers, PyTorch and Hugging Face Diffusers. My project applies and evaluates attention fine-tuning, records the experimental evidence and presents the results through a web demonstration. I directed the project, reviewed examples, operated the environment and studied the architecture. Code, analysis tooling and writing were developed with AI assistance. This is not a claim that I invented CatVTON or wrote every line unaided.

[Sources, licenses and data attribution](docs/ATTRIBUTION.md). Noncommercial academic research. Third-party code and images retain their original terms.
