"""Compare models within each case, after averaging its two generation seeds."""

import numpy as np

NAMES = ["garment_mae", "garment_ssim", "outside_mae_input", "outside_mae_target"]
VARIANTS = ["pretrained", "adapted-1000"]


def summarize(case_means, ids, variants=VARIANTS):
    summary = {}
    for metric in NAMES:
        eligible = [
            c
            for c in ids
            if c in case_means
            and all(
                (
                    v in case_means[c] and case_means[c][v][metric] is not None
                    for v in variants
                )
            )
        ]
        values = {
            v: np.array([case_means[c][v][metric] for c in eligible]) for v in variants
        }
        result = {
            "n": len(eligible),
            "cases": eligible,
            "models": {},
            "comparisons": {},
        }
        for v, a in values.items():
            result["models"][v] = {
                "mean": float(a.mean()) if len(a) else None,
                "median": float(np.median(a)) if len(a) else None,
            }
        for newer, older in [("adapted-1000", "pretrained")]:
            if newer not in variants or older not in variants or (not eligible):
                continue
            delta = values[newer] - values[older]
            improvement = delta if metric == "garment_ssim" else -delta
            ci = None
            if len(eligible) >= 8:
                # Resample CASE differences, not individual images or pixels.
                rng = np.random.default_rng(41026)
                samples = delta[
                    rng.integers(0, len(delta), size=(10000, len(delta)))
                ].mean(axis=1)
                ci = [float(x) for x in np.quantile(samples, [0.025, 0.975])]
            result["comparisons"][newer + "_vs_" + older] = {
                "mean_delta_new_minus_old": float(delta.mean()),
                "median_paired_delta": float(np.median(delta)),
                "relative_mean_change_percent": (
                    float(100 * delta.mean() / values[older].mean())
                    if values[older].mean() != 0
                    else None
                ),
                "wins": int((improvement > 0).sum()),
                "ties": int((improvement == 0).sum()),
                "losses": int((improvement < 0).sum()),
                "descriptive_paired_bootstrap_95_interval": ci,
            }
        summary[metric] = result
    return summary
