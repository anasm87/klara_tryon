# Read the experiment in code

Start with [training_core.py](../code/training-feasibility/training_core.py), then [train_1000.py](../code/training-feasibility/train_1000.py). The first defines one noise-prediction loss; the second runs the optimizer and saves the checkpoint. For the results without running a GPU, open [the executed notebook](../notebooks/Klara_Final_Study.ipynb).

## Data and training

| File | Its role |
|---|---|
| [experiment-plan.json](../data/experiment-plan.json) | Fixed development cases, split roles and provenance |
| [data_1000.py](../code/training-feasibility/data_1000.py) | Validate the development data and define seeded epoch order |
| [edit_data.py](../code/training-feasibility/edit_data.py) | Load the input person, garment and target in the correct roles |
| [models.py](../code/training-feasibility/models.py) | Assemble pinned pretrained components |
| [training_core.py](../code/training-feasibility/training_core.py) | Select `attn1` weights, add known noise, predict it and calculate MSE |
| [train_1000.py](../code/training-feasibility/train_1000.py) | Cache latents, run three epochs, update attention and verify the saved checkpoint |

## Generation and attention

| File | Its role |
|---|---|
| [pipeline.py](../code/training-feasibility/vendor/catvton-maskfree/model/pipeline.py) | `CatVTONPix2PixPipeline`: two input images, CFG, DDIM and VAE decoding |
| [attn_processor.py](../code/training-feasibility/vendor/catvton-maskfree/model/attn_processor.py) | Q/K/V projections, multi-head attention and output projection |
| [model/utils.py](../code/training-feasibility/vendor/catvton-maskfree/model/utils.py) | Install attention processors and collect existing self-attention modules |

These files are bundled upstream CatVTON code, with the provenance and license retained. The VAE and U-Net layer implementations come from Diffusers; this project adapts their pretrained attention weights rather than implementing a new architecture.

## Final test and analysis

| File | Its role |
|---|---|
| [evaluate_final_1000.py](../code/training-feasibility/evaluate_final_1000.py) | Generate both models' outputs with fixed cases, seeds and settings |
| [analyze_edit_evaluation.py](../code/analyze_edit_evaluation.py) | Exact regional MAE and SSIM definitions |
| [analyze_1000_evaluation.py](../code/analyze_1000_evaluation.py) | Shared paired aggregation and bootstrap calculations |
| [analyze_final_1000.py](../code/analyze_final_1000.py) | Verify the final output files and run the paired analysis |
| [Klara_Final_Study.ipynb](../notebooks/Klara_Final_Study.ipynb) | Executed analysis with recorded findings |

[Reproduction instructions](../REPRODUCE.md) explain how to recompute the results. [Executed training source](../evidence/executed-training-source) preserves the code actually hashed in the training record. The optional website and personal teaching copies are maintained separately. Earlier small-data experiments are outside this submission.
