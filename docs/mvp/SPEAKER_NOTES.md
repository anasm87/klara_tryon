# Klara MVP speaker notes

Use these as a rehearsal script. Speak naturally and adjust wording to your voice. Method-focused MVP. Confirm the allotted time with the instructor.

## 1. Would you trust this preview? (35 seconds)

Imagine choosing this pink striped shirt online. You ask an AI tool to show it on a person, and the preview looks believable. But the color is different. If the preview changes the item you are thinking of buying, how useful is it? That is the problem I am investigating with Klara.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 2. The research question (40 seconds)

My question is whether fine-tuning the model's attention weights can improve garment reconstruction while preserving the person and background. This is useful to developers building clothing previews. The data gives me an edited person photo, a requested garment and an original reference photo. That lets me compare the generated result with something concrete. It does not tell me whether the garment would physically fit a customer.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 3. Separate data for learning and evaluation (40 seconds)

The experiment uses one thousand training cases, sixty-four validation cases and thirty-two final-test cases. The roles are separate: training changes the weights, validation helps development, and the final test checks the fixed model. I check pairings, file quality and exact overlap. A limitation is that different source IDs do not guarantee different people. These are custom splits, so I will not present them as official benchmark results.

Source: Frozen experiment-plan.json and final-evaluation-plan.json. Split checks and provenance receipts.

## 4. Fine-tuning existing attention (40 seconds)

I use the pretrained CatVTON-MaskFree model. I am focusing the experiment on existing self-attention projections inside its U-Net. Attention lets spatial features interact, including information from the person and garment. The VAE and the remaining U-Net weights stay frozen. The comparison is therefore between the original checkpoint and the attention-adapted version, rather than between unrelated applications.

Source: CatVTON-MaskFree pinned vendor source. configure_attention in training_core.py.

## 5. Evaluation plan (45 seconds)

A realistic-looking image is not enough. I measure garment pixel error with MAE and structural agreement with SSIM. Outside the clothing region, I measure changes relative to the input. The colored areas show actual evaluation regions. Both models use the same cases, seeds and generation settings. I compare results case by case and inspect failures, because one average can hide very different outcomes.

Source: Exact regional masks and metrics in analyze_edit_evaluation.py. Green: eroded target clothing. Gold: outside dilated source/target union.

## 6. The feedback I need (30 seconds)

For this MVP, I want feedback on the research question and how I explain the evaluation. Do these measurements cover the important part of the problem? And what would you simplify for the final jury? I will use that feedback to refine the scientific story and choose clear examples for the final presentation.

Source: DS#055 schedule and KLARA_DAILY_PROGRESS.md. Method-focused presentation scope, not a claim of unfinished execution.