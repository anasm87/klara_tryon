# A practical reading order

Start with the experiment and follow one example through it.

| File | What to look for |
|---|---|
| `data/experiment-plan.json` | Which cases belong to each split and where they came from |
| `code/training-feasibility/data_1000.py` | Checks before data enters training |
| `code/training-feasibility/edit_data.py` | Person, garment, target and mask roles |
| `code/training-feasibility/training_core.py` | Freeze weights, enable attention updates, build noisy input, compute loss |
| `code/training-feasibility/train_1000.py` | Cache latents, call backward and optimizer.step, save and verify checkpoint |
| `code/training-feasibility/evaluate_final_1000.py` | Two-model generation with fixed cases and seeds |
| `code/analyze_edit_evaluation.py` | Exact regional MAE and SSIM definitions |
| `code/analyze_final_1000.py` | Output verification and the final paired analysis |
| `code/website/studio_engine.py` | Load pretrained components, replace attention weights, generate on demand |
| `code/website/studio.py` | HTTP requests, sessions and generation jobs |
| `notebooks/Klara_Final_Study.ipynb` | Reproduce the scientific results in one place |

`evidence/executed-training-source/` preserves the source actually hashed in the training record. `code/` also includes teaching comments added later. Earlier small-data experiments remain outside this focused submission.
