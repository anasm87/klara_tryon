# Klara FINAL speaker notes

Use these as a rehearsal script. Speak naturally and adjust wording to your voice. Planned duration: five minutes.

## 1. Would you trust this preview? (25 seconds)

Imagine choosing this pink striped shirt online. You ask an AI tool for a preview, and the result looks believable. But the color has changed. Would that help you decide what to buy? Klara started with this problem: a convincing image can still show the wrong clothing.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 2. Question and data (40 seconds)

My question was whether attention fine-tuning could improve garment reconstruction while preserving the person and background. I used VITON-HD and VITON-HD-edit. The edited person and catalog garment are inputs. The original photo is the reference. I separated one thousand training cases, sixty-four validation cases and thirty-two final-test cases. This is a reconstruction study for clothing-preview developers. It does not test sizing or physical fit, and identity separation remains unverified.

Source: Data: NXN-Labs/VITON-HD-edit revision 9de94ee10e15f5069fd650ce4c914215f124eb6f. Frozen custom splits. No official benchmark claim.

## 3. What changed in the model (35 seconds)

The starting point was CatVTON-MaskFree. I adapted the query, key, value and output projections in sixteen existing self-attention modules. The VAE and other U-Net weights stayed frozen. Training covered all one thousand cases in three epochs, with three thousand optimizer updates. The training loss compares predicted noise with the noise added to the target. The image metrics come later, during evaluation. Checks confirmed that frozen weights stayed unchanged and the checkpoint reloaded correctly.

Source: run.json and steps.jsonl. 49,574,080 selected parameters. Standard epsilon-MSE adaptation; not a reproduction of DREAM training.

## 4. A fair comparison (40 seconds)

I fixed the checkpoint before the final test. Both models saw the same inputs and two fixed seeds, with identical generation settings. Garment MAE is the primary metric. SSIM checks structure, and outside-clothing error checks preservation. The masks define measurement regions; the generator receives no mask. There were one hundred twenty-eight attempts. Six safety exclusions left twenty-seven complete paired cases. I kept those exclusions and averaged the two seeds within each case.

Source: Frozen final-evaluation-plan.json. results.json: 128 attempts, 122 saved outputs. Six exclusions affect five cases.

## 5. Garment reconstruction improved on average (45 seconds)

The fine-tuned model reduced average garment error by fifteen point six percent. Each line here is a test case. Green lines improve, and red lines worsen. Twenty-one of the twenty-seven complete cases improved. The mean fell from about zero point one three four to zero point one one three. The paired bootstrap interval was below zero, but it describes uncertainty across these cases. It does not establish performance across new populations or different training seeds.

Source: analysis.json, garment_mae. Equal case weights after averaging two seeds. 10,000 paired case-bootstrap resamples, seed 41026.

## 6. Preservation remained inconsistent (40 seconds)

Preservation tells a more complicated story. Each dot is one case. Moving left means better garment reconstruction, and moving down means better preservation. Eleven cases improved on both. But fourteen had more outside-clothing error after fine-tuning, even though average outside error fell. Some larger improvements outweighed smaller regressions. This is why I would not describe the model as reliably preserving every person based on the average alone.

Source: analysis.json, case_means and joint_outcomes. Outside-input MAE: 13 improvements, 14 regressions; relative mean change −25.899%.

## 7. A metric gain can still contain visible errors (45 seconds)

This case had the largest reduction in garment error, so it is a useful test of what the number means. The fine-tuned image removes much of the severe clothing distortion. But compare it with the input and reference: the arm is raised and the trousers still change. The improvement is real under the metric, while the image remains imperfect. I kept all cases and both seeds in the gallery so the audience can inspect more than a few selected successes.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use. Selection rule: smallest case-average fine-tuned minus pretrained garment MAE. Assistant observations are qualitative, not human-rating scores.

## 8. Answer to the research question (30 seconds)

My conclusion is that attention fine-tuning improved average reconstruction on this dataset, while preservation remained inconsistent. I would use Klara as a research demonstration with visible failures. The small complete-case test, edited inputs and one training seed limit broader claims. The website uses the trained checkpoint, and the offline gallery preserves the experiment. The next research step would be broader independent data and repeated training seeds.

Source: Final analysis and limitations. Website configuration differs: 768×1024, 50 steps versus research 384×512, 20 steps.