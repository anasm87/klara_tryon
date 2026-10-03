"""Final-test loader: validates immutable IDs, receipts and the fixed checkpoint."""

import json
from pathlib import Path
from common import ROOT, DATA, PROJECT, sha
from edit_data import validated_cases

PLAN_SHA = "e189f29e940dd09c245f3d4632d34065818b663cc5be0a860689a36f53de9d96"


def checked_setup():
    plan_file = DATA / "final-evaluation-plan.json"
    if sha(plan_file) != PLAN_SHA:
        raise ValueError("Final evaluation plan changed")
    plan = json.loads(plan_file.read_text())
    parent_file = DATA / "experiment-plan.json"
    if sha(parent_file) != plan["parent_plan_sha256"]:
        raise ValueError("Parent plan changed")
    parent = json.loads(parent_file.read_text())
    rows = parent["candidates"]["reserved_test"]
    if [r["id"] for r in rows] != plan["case_ids"]:
        raise ValueError("Reserved IDs changed")
    root = DATA / "final-test-data"
    verification = json.loads((root / "verification.json").read_text())
    if (
        verification["plan_sha256"] != PLAN_SHA
        or verification["status"] != "passed"
        or verification["receipt_sha256"] != sha(root / "download-receipt.json")
    ):
        raise ValueError("Final input verification missing or changed")
    receipt = json.loads((root / "download-receipt.json").read_text())
    quality = "technical-audit-all-reserved-cases-no-outcome-selection"
    manifest = [
        {
            **r,
            "person": "samples/" + r["id"] + "/person.jpg",
            "garment": "samples/" + r["id"] + "/garment.jpg",
            "target": "samples/" + r["id"] + "/target.jpg",
            "source_files": r["files"],
            "quality_status": quality,
        }
        for r in receipt
    ]
    cases = validated_cases(
        root,
        manifest,
        receipt,
        {r["id"]: r for r in rows},
        {r["id"]: r["split"] for r in rows},
        plan["case_ids"],
        {rows[0]["split"]: 32},
        quality_status=quality,
    )
    folders = {"adapted-1000": PROJECT / "evidence/training-1000-01"}
    folder = folders["adapted-1000"]
    spec = plan["checkpoints"]["adapted-1000"]
    if (
        sha(folder / "run.json") != spec["run_sha256"]
        or sha(folder / "attention-adapter.safetensors") != spec["adapter_sha256"]
    ):
        raise ValueError("Frozen checkpoint changed")
    run = json.loads((folder / "run.json").read_text())
    pins = json.loads((ROOT / "vendor/catvton-maskfree/revisions.json").read_text())
    if (
        not run["frozen_weights_unchanged"]
        or not run["adapter_reload_verified"]
        or run["steps_completed"] != 3000
        or run["revisions"] != pins
    ):
        raise ValueError("Training evidence mismatch")
    return (
        plan,
        {
            "expected_revisions": pins,
            "data_provenance": {
                "dataset": parent["dataset"],
                "revision": parent["source_revision"],
                "test_receipt_sha256": sha(root / "download-receipt.json"),
                "verification_sha256": sha(root / "verification.json"),
                "identity_disjoint": "unverified",
                "official_source_split": "test; custom adaptation split",
                "selection": "all previously reserved 32 cases",
            },
        },
        cases,
        folders,
    )
