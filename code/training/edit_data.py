"""Check source mappings and load person, garment and reference images."""

import hashlib
from itertools import combinations
from pathlib import Path
from PIL import Image
from common import SIZE, sha

SOURCE_REVISION = "9de94ee10e15f5069fd650ce4c914215f124eb6f"


def safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("Dataset path outside bundle")
    return path


def validated_cases(
    root, manifest, receipt, lookup, expected, order, expected_counts, quality_status
):
    """Shared file/mapping checks; callers authenticate their immutable protocol first."""
    for collection in (manifest, receipt):
        if len(collection) != len(expected) or {r["id"] for r in collection} != set(
            expected
        ):
            raise ValueError("Case coverage differs from the frozen selection")
    if len(order) != len(expected) or set(order) != set(expected):
        raise ValueError("Frozen case order differs from selection")
    receipts = {r["id"]: r for r in receipt}
    hashes = {s: set() for s in expected_counts}
    pixel_hashes = {s: set() for s in expected_counts}
    groups = {s: set() for s in expected_counts}
    cases = []
    for row in manifest:
        sid, split = (row["id"], row["split"])
        if (
            split not in hashes
            or split != expected[sid]
            or split != lookup[sid]["split"]
        ):
            raise ValueError("Wrong split or test/excluded data in development batch")
        if receipts[sid]["split"] != split:
            raise ValueError("Receipt split mismatch")
        if row["source_files"] != receipts[sid]["files"]:
            raise ValueError("Source receipt changed")
        if row["quality_status"] != quality_status:
            raise ValueError("Missing documented inspection")
        groups[split].add(lookup[sid]["group"])
        specs = {
            "person": ("person.jpg", f"test/image-edit/{sid}.jpg", "hf"),
            "garment": ("garment.jpg", f"test/cloth/{sid}.jpg", "zip"),
            "target": ("target.jpg", f"test/image/{sid}.jpg", "zip"),
            "edited_mask": (
                "edited-mask.png",
                f"test/image-edit_cloth-mask/{sid}.png",
                "hf",
            ),
            "target_mask": (
                "target-mask.png",
                f"test/gt_cloth_warped_mask/{sid}.png",
                "hf",
            ),
        }
        files = {f["file"]: f for f in row["source_files"]}
        if len(files) != 5 or len(row["source_files"]) != 5:
            raise ValueError("Expected five source files per case")
        paths = {}
        for role, (name, source, provider) in specs.items():
            relative = f"samples/{sid}/{name}"
            if role in ("person", "garment", "target") and row[role] != relative:
                raise ValueError("Person/garment/target mapping changed")
            path = safe_path(root, relative)
            entry = files.get(relative)
            if not entry or entry["source"] != source or entry["provider"] != provider:
                raise ValueError("Unexpected source mapping")
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError("Dataset file changed: " + relative)
            with Image.open(path) as im:
                if im.size != (768, 1024):
                    raise ValueError("Unexpected source dimensions")
                im.load()
                if role in ("person", "garment", "target"):
                    hashes[split].add(entry["sha256"])
                    pixel_hashes[split].add(
                        hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()
                    )
            paths[role] = str(path)
        cases.append(
            {
                "id": sid,
                "split": split,
                "group": lookup[sid]["group"],
                "edit_paths": paths,
            }
        )
    for a, b in combinations(expected_counts, 2):
        if (
            hashes[a] & hashes[b]
            or pixel_hashes[a] & pixel_hashes[b]
            or groups[a] & groups[b]
        ):
            raise ValueError("Cross-split duplicate or group overlap")
    counts = {s: sum((c["split"] == s for c in cases)) for s in hashes}
    if counts != expected_counts:
        raise ValueError("Unexpected training/validation counts")
    case_lookup = {c["id"]: c for c in cases}
    return [case_lookup[sid] for sid in order]


def training_images(case):
    """Return source, requested garment, real target in the denoising-loss order."""
    images = []
    for role in ("person", "garment", "target"):
        with Image.open(case["edit_paths"][role]) as im:
            images.append(im.convert("RGB").resize(SIZE, Image.Resampling.LANCZOS))
    return images
