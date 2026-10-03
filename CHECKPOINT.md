# Trained attention checkpoint

The website uses the 1,000-case checkpoint, not the older small-data experiment.

- File: `attention-adapter.safetensors`
- SHA256: `e4f332704879fd3929120c4c038cbb65837c8a9c694bdb9fc75bd5a18924de38`
- Training: 1,000 cases, three epochs, 3,000 updates.
- Scope: selected existing U-Net self-attention weights, 49,574,080 parameters.

The large binary is excluded from ordinary Git history and the compact presentation hand-in. It is available separately from Anas on request. Place it at `evidence/training-1000-01/attention-adapter.safetensors` to enable the notebook's optional weight-file hash check. The model still needs the pinned pretrained components listed in the reproduction guide.

All recorded image measurements and outputs remain available to reproduce the final analysis without this binary. Contact the repository owner for the research checkpoint, subject to the upstream model terms.
