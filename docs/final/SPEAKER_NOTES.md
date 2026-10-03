# Klara FINAL speaker notes

These notes are a speaking guide. Pause to point at the images and charts, and use your own wording where it feels more natural. The source lines are for reference, not for reading aloud. Planned duration: five minutes.

## 1. Would you trust this preview? (25 seconds)

At first, I was pleased that Klara could generate a person wearing a different shirt. Then I tried more examples. Some looked convincing, but the color was wrong or extra clothing appeared. This pink striped shirt shows the problem. I wanted to find out whether I could improve that.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use.

## 2. Question and data (40 seconds)

My question was: can attention fine-tuning help the model reproduce the right clothing, while keeping the person and background unchanged? Here are the inputs: an edited person photo and the requested garment. The original photograph gives me a reference to compare with. I used VITON-HD and VITON-HD-edit, with one thousand training cases, sixty-four for validation and thirty-two for the final test. This matters to developers building clothing previews. It doesn't tell us whether the clothes will physically fit.

Source: Data: NXN-Labs/VITON-HD-edit revision 9de94ee10e15f5069fd650ce4c914215f124eb6f. Frozen custom splits. No official benchmark claim.

## 3. What changed in the model (35 seconds)

I started with CatVTON-MaskFree, a pretrained model. Inside its U-Net, attention lets different parts of the image exchange information. I fine-tuned sixteen existing self-attention modules and kept the rest of the model frozen. The model saw all one thousand training cases three times, making three thousand weight updates. During training it learns to predict added noise. Later, I checked the images it generated to see whether that training actually helped.

Source: run.json and steps.jsonl. 49,574,080 selected parameters. Standard epsilon-MSE adaptation; not a reproduction of DREAM training.

## 4. A fair comparison (40 seconds)

To make the comparison fair, I gave both models the same inputs and generation settings. I ran each case with two fixed random seeds. In the clothing region, I measured pixel error and structural similarity. Outside it, I measured changes from the input photo. The colored masks define where I measure these errors. The model itself doesn't need a mask. Of one hundred twenty-eight attempts, the safety checker excluded six. That left twenty-seven cases with complete results for both models.

Source: Frozen final-evaluation-plan.json. results.json: 128 attempts, 122 saved outputs. Six exclusions affect five cases.

## 5. Garment reconstruction improved on average (45 seconds)

The main result was encouraging: average garment error fell by fifteen point six percent. Each line is one test case. The green lines show improvement, and the red lines show things getting worse. Twenty-one of the twenty-seven cases improved. So there is a clear average gain here, but six examples still went the other way. The uncertainty interval also favors fine-tuning. That supports the result for this small test, but it doesn't tell us how well the model will work for everyone.

Source: analysis.json, garment_mae. Equal case weights after averaging two seeds. 10,000 paired case-bootstrap resamples, seed 41026.

## 6. Preservation remained inconsistent (40 seconds)

Keeping the rest of the photo unchanged was harder. Each dot here is one case. Left means a better clothing match, and down means fewer changes outside the clothing. Eleven cases improved on both. But fourteen had more unwanted changes after fine-tuning. You might wonder how the average could still improve. The gains in some examples were simply larger than the mistakes added in others. Looking only at the average would have hidden that.

Source: analysis.json, case_means and joint_outcomes. Outside-input MAE: 13 improvements, 14 regressions; relative mean change −25.899%.

## 7. A metric gain can still contain visible errors (45 seconds)

This is the case with the largest drop in garment error. The pretrained model badly distorts the clothing. After fine-tuning, the result is much closer to the reference. But look at the arm and trousers. They still change, even though I only asked for different clothing. That's why I wanted to show the images alongside the scores. I've included every test case and both seeds in the gallery, so you can look beyond the examples I've selected for these slides.

Source: VITON-HD / VITON-HD-edit. Recorded final-test outputs, seed 9026. Academic use. Selection rule: smallest case-average fine-tuned minus pretrained garment MAE. Assistant observations are qualitative, not human-rating scores.

## 8. Answer to the research question (30 seconds)

My answer is that fine-tuning helped the clothing match the reference on average, but keeping the rest of the photo unchanged is still a problem. Klara demonstrates both outcomes. You can try the trained model on the website or explore the saved results offline. Before making broader claims, I'd test more varied, independent data and repeat the training with different seeds. For now, this is a promising result with clear limits.

Source: Final analysis and limitations. Website configuration differs: 768×1024, 50 steps versus research 384×512, 20 steps.