# Reproducing Klara

There are two useful levels of reproduction: checking the recorded experiment on a CPU, and running the model again on a GPU. The first is enough to inspect the reported findings.

## CPU analysis

Clone/download the complete repository and create a virtual environment with Python 3.10+. Activate it and run the command in README.md. The analyzer checks all 128 attempt records, output and reference hashes, the frozen protocol and the metric implementation. It then recalculates the regional metrics and paired bootstrap summaries. It rewrites the derived analysis.json and CSV, leaving raw outputs unchanged.

Open notebooks/Klara_Final_Study.ipynb in Jupyter or VS Code. Its executed outputs are already included. Run from the repository root or the notebooks directory. The notebook checks three complete training epochs and recomputes final metrics. If the optional attention checkpoint is present, it checks its SHA256 too; otherwise it explicitly reports that this local weight check was skipped.

scikit-image 0.26.0 is required because its metric behavior forms part of the frozen protocol. requirements.txt lists the minimal CPU analysis dependencies; environment-analysis.txt records the versions used for verification. No GPU, website access code or cloud account is needed for analysis.

## Saved model and generation

See CHECKPOINT.md for the trained attention weights. The pretrained VAE and diffusion weights are separate Hugging Face downloads, pinned in code/training-feasibility/vendor/catvton-maskfree/revisions.json. The saved adapter alone is not a full image generator.

The original training runner expects the data and evidence layout described in code/training-feasibility/data_1000.py. Frozen selections and download receipts live in data/. The pinned upstream dataset is NXN-Labs/VITON-HD-edit, revision 9de94ee10e15f5069fd650ce4c914215f124eb6f. Full training photographs are not bundled. prepare_1000_data.py documents their retrieval and verification. It expects the original experiment directory layout; copying it to a new machine requires arranging that layout first.

The GPU environment used PyTorch 2.8.0+cu128 on an NVIDIA L4. Additional pinned dependencies are in code/training-feasibility/requirements.txt. Source snapshots of train_1000.py and training_core.py actually used for training are in evidence/executed-training-source/. Commented teaching copies in code/ have the same intended behavior but do not necessarily have the same source hash.

The included run_final_1000.sh records how the final evaluation ran on the original VM. It references that VM's paths and systemd studio service, so it is not a portable one-command setup script. The test cases have now been evaluated; tuning against them would require a new held-out test for future claims.

## Website

code/website/ contains the server, authentication module, model engine and static interface. Authentication configuration and user uploads are deliberately absent. The engine expects the verified adapter in models/attention-adapter.safetensors or at KLARA_ADAPTER_PATH. The README in that directory explains the source snapshot and preview command.

## Integrity

MANIFEST-SHA256.json records the hand-in files. The checkpoint has its own recorded digest. Source-data receipts and raw attempt hashes provide the link from inputs to outputs. The original training run.json retains its historical “evaluation pending” status; final-test-1000-01/results.json records the later completed evaluation. Historical run records have not been rewritten to change that chronology.
