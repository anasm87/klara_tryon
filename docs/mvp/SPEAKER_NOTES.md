# Klara MVP speaker notes

These notes are a speaking guide. Pause to point at the images and charts, and use your own wording where it feels more natural. The source lines are for reference, not for reading aloud. Method-focused MVP. Confirm the allotted time with the instructor.

## 1. Would you trust this preview? (35 seconds)

When I started trying clothing previews, some results looked convincing at first. Then I noticed the wrong colors and extra clothing. This example shows why that matters: I chose a pink striped shirt, but the preview changes its color. If I were shopping, I'd want to see the item I'd actually receive. That's the problem behind Klara.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 2. The research question (40 seconds)

My question is whether fine-tuning the model's attention weights can help it reproduce the requested clothing while keeping the person and background unchanged. Here you can see the data: an edited person photo, a catalog garment and the original photograph. That original gives me something concrete to compare the output with. I'm studying image quality for clothing-preview tools. I'm not measuring whether a garment would physically fit someone.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 3. Separate data for learning and evaluation (40 seconds)

I've separated the data into one thousand training cases, sixty-four for validation and thirty-two for the final test. Each group has a different job. Training updates the model, validation helps with development, and the test checks the chosen model. The checks cover missing files, image sizes, pairings and exact duplicates. One limitation remains: different image IDs don't guarantee different people. This is my own study split, so I won't call the results an official benchmark.

Source: Frozen experiment-plan.json and final-evaluation-plan.json. Split checks and provenance receipts.

## 4. Fine-tuning existing attention (40 seconds)

I'm using CatVTON-MaskFree as the starting point. It already has attention layers, which let different parts of the image exchange information. I focused the training on sixteen of those existing self-attention modules. The VAE and the other U-Net weights stay frozen. That gives me a specific change to investigate: how the model behaves before and after updating its attention weights.

Source: CatVTON-MaskFree pinned vendor source. configure_attention in training_core.py.

## 5. Evaluation plan (45 seconds)

I want to check two things: does the clothing match the reference, and does the rest of the photo stay the same? MAE measures pixel differences in the clothing, while SSIM looks at structure. Outside the clothing, I measure changes from the input photo. These colored regions show where the measurements happen. I'll explain the results case by case as well as with averages, because one overall score can hide some very poor examples.

Source: Exact regional masks and metrics in analyze_edit_evaluation.py. Green: eroded target clothing. Gold: outside dilated source/target union.

## 6. The feedback I need (30 seconds)

The feedback I'd find most useful today is whether the question is clear and whether these measurements answer it. I'd also like to know which parts need a simpler explanation for the jury. I want the final presentation to make sense to someone who hasn't studied neural networks, while still giving enough evidence for the technical audience.

Source: DS#055 schedule and KLARA_DAILY_PROGRESS.md. Method-focused presentation scope, not a claim of unfinished execution.