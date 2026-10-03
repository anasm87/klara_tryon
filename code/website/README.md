# Klara studio source

`studio.py` serves the interface and handles authenticated requests. `studio_auth.py` manages access-code sessions. `studio_engine.py` loads the verified 1,000-case attention checkpoint into the pinned CatVTON-MaskFree pipeline. The UI calls generation after the Generate action; selecting a sample does not itself run inference.

This is a source snapshot. Production secrets, access codes, TLS configuration and user uploads are not included. Read the command-line arguments at the bottom of studio.py before starting it in a new environment. Use the offline gallery in the repository's demo/ directory when you only need to inspect recorded results.

The GPU model dependencies are in requirements-catvton.txt. The model must run on a suitable CUDA environment. The live site uses 768x1024 resolution and 50 steps; the research evaluation uses 384x512 and 20 steps.
