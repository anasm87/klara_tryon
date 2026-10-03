"""Reproduce the final paired image analysis on a CPU; no model or cloud calls."""

import collections
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import skimage
from PIL import Image
from metrics import metrics, load_image
from paired_statistics import summarize, NAMES

PROJECT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(folder=None, plan_file=None):
    """Verify the recorded experiment and recompute scores without changing evidence."""
    folder = folder or PROJECT / "evidence/final-test-1000-01"
    plan_file = plan_file or PROJECT / "data/final-evaluation-plan.json"
    folder = Path(folder).resolve()
    plan_file = Path(plan_file)
    plan = json.loads(plan_file.read_text())
    record = json.loads((folder / "results.json").read_text())
    with zipfile.ZipFile(PROJECT / "evidence/source-at-run.zip") as archive:
        evaluator_hash = hashlib.sha256(
            archive.read("evaluate_final_1000.py")
        ).hexdigest()
        recorded_metric_hash = hashlib.sha256(
            archive.read("analyze_edit_evaluation.py")
        ).hexdigest()
    if (
        record["status"] != "completed"
        or record["plan"] != plan
        or record["plan_sha256"] != sha(plan_file)
        or (record["evaluator_sha256"] != evaluator_hash)
        or (not record["safety_checker"])
    ):
        raise ValueError("Final protocol/result/evaluator mismatch")
    if recorded_metric_hash != plan["metric_source_sha256"]:
        raise ValueError("Metric implementation changed")
    if skimage.__version__ != plan["ssim"]["version"]:
        raise ValueError("SSIM version mismatch")
    variants = plan["variants"]
    case_ids = plan["case_ids"]
    seeds = plan["seeds"]
    expected = {
        (c, variant, s) for c in case_ids for variant in variants for s in seeds
    }
    actual = [(x["case_id"], x["variant"], x["seed"]) for x in record["records"]]
    if (
        len(actual) != plan["expected_attempts"]
        or len(set(actual)) != len(actual)
        or set(actual) != expected
    ):
        raise ValueError("Missing, duplicate or unexpected final attempts")
    references = {}
    for case_id in case_ids:
        references[case_id] = {}
        for role in ("person", "garment", "target", "edited_mask", "target_mask"):
            image_path = folder / "references" / case_id / (role + ".png")
            if sha(image_path) != record["reference_hashes"][case_id][role]:
                raise ValueError("Reference changed")
            references[case_id][role] = load_image(image_path, role.endswith("mask"))
    rows = []
    for row in record["records"]:
        row = dict(row)
        if row["status"] == "ok":
            image_path = (folder / row["file"]).resolve()
            if (
                not image_path.is_relative_to(folder)
                or sha(image_path) != row["sha256"]
            ):
                raise ValueError("Output changed")
            with Image.open(image_path) as image:
                image.load()
                if image.size != tuple(plan["size"]):
                    raise ValueError("Output resolution mismatch")
            reference = references[row["case_id"]]
            row["metrics"] = metrics(
                reference["person"],
                reference["target"],
                load_image(image_path),
                reference["edited_mask"],
                reference["target_mask"],
            )
        elif row["status"] != "safety-excluded":
            raise ValueError("Unresolved generation error")
        rows.append(row)
    case_means = {case_id: {} for case_id in case_ids}
    for case_id in case_ids:
        for variant in variants:
            selected = [
                row
                for row in rows
                if row["case_id"] == case_id and row["variant"] == variant
            ]
            if all((row["status"] == "ok" for row in selected)):
                case_means[case_id][variant] = {
                    metric: (
                        float(np.mean([x["metrics"][metric] for x in selected]))
                        if all((x["metrics"][metric] is not None for x in selected))
                        else None
                    )
                    for metric in NAMES
                }
    complete = [
        case_id
        for case_id in case_ids
        if all((variant in case_means[case_id] for variant in variants))
    ]
    joint = collections.Counter()
    for case_id in complete:
        adapted = case_means[case_id]["adapted-1000"]
        pretrained = case_means[case_id]["pretrained"]
        if any(
            (
                x is None
                for x in (
                    adapted["garment_mae"],
                    pretrained["garment_mae"],
                    adapted["outside_mae_input"],
                    pretrained["outside_mae_input"],
                )
            )
        ):
            joint["metric-unavailable"] += 1
        else:
            joint[
                (
                    "garment-better"
                    if adapted["garment_mae"] < pretrained["garment_mae"]
                    else "garment-not-better"
                )
                + " / "
                + (
                    "preservation-better"
                    if adapted["outside_mae_input"] < pretrained["outside_mae_input"]
                    else "preservation-not-better"
                )
            ] += 1
    analysis = {
        "scope": plan["scope"],
        "plan_sha256": sha(plan_file),
        "results_sha256": sha(folder / "results.json"),
        "analyzer_sha256": sha(__file__),
        "recorded_metric_source_sha256": recorded_metric_hash,
        "metric_source_sha256": sha(Path(__file__).with_name("metrics.py")),
        "attempts": len(rows),
        "images_verified": sum((x["status"] == "ok" for x in rows)),
        "status_by_variant": {
            variant: dict(
                collections.Counter(
                    (x["status"] for x in rows if x["variant"] == variant)
                )
            )
            for variant in variants
        },
        "complete_cases": complete,
        "excluded_cases": [case_id for case_id in case_ids if case_id not in complete],
        "case_means": case_means,
        "summary": summarize(case_means, case_ids, variants),
        "joint_outcomes": dict(joint),
        "rows": rows,
        "scikit_image_version": skimage.__version__,
        "numpy_version": np.__version__,
        "limitations": plan["limits"],
        "generation_seconds": record["generation_seconds"],
    }
    return analysis


if __name__ == "__main__":
    result = analyze()
    fields = (
        "attempts",
        "images_verified",
        "excluded_cases",
        "status_by_variant",
        "joint_outcomes",
        "summary",
    )
    print(json.dumps({key: result[key] for key in fields}, indent=2))
