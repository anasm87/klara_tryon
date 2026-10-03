"""Isolated 1000-case adaptation: three shuffled epochs, unchanged denoising loss."""

import argparse
import json
import time
import traceback
from common import ROOT, DATA, RUNS, SIZE, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--output", default="training-1000-01")
    parser.add_argument(
        "--edit-data",
        default=str(DATA / "edit-data"),
        help="Folder containing the verified development images",
    )
    args = parser.parse_args()
    if args.steps != 3000:
        parser.error(
            "Frozen 1000-case experiment requires exactly 3000 updates and standard denoising"
        )
    output = (RUNS / args.output).resolve()
    if not output.is_relative_to(RUNS.resolve()) or output == RUNS.resolve():
        parser.error("Output must be a new folder inside runs/")
    from edit_data import training_images
    from data_1000 import load_1000_batch, epoch_indices

    value = load_1000_batch(args.edit_data)
    if args.preflight:
        print(
            json.dumps(
                {
                    "reviewed_cases": len(value["cases"]),
                    "train": sum((r["split"] == "train" for r in value["cases"])),
                    "validation": sum(
                        (r["split"] == "validation" for r in value["cases"])
                    ),
                    "excluded_cases": value["excluded_cases"],
                    "steps": args.steps,
                    "preservation_weight": 0.0,
                    "scope": "1000-case attention adaptation",
                    "data_provenance": value["data_provenance"],
                    "downloads": False,
                    "gpu_started": False,
                }
            )
        )
        return
    if output.exists():
        raise ValueError(
            "Output exists. Preserve previous results and choose a new name."
        )
    import torch
    import diffusers, transformers, accelerate, platform
    from diffusers import DDPMScheduler
    from safetensors.torch import save_file, load_file
    from models import load_pipeline
    from training_core import (
        configure_attention,
        frozen_digest,
        denoising_loss,
        adapter_state,
        apply_adapter,
    )

    torch.manual_seed(9026)
    load_started = time.monotonic()
    pipe, revisions = load_pipeline()
    if revisions != value["expected_revisions"]:
        raise ValueError("Loaded model revisions differ from training protocol")
    load_seconds = time.monotonic() - load_started
    from utils import prepare_image

    output.mkdir(parents=True)
    record = {
        "status": "started",
        "scope": "1000-case exploratory attention adaptation",
        "plan_sha256": value["data_provenance"]["plan_sha256"],
        "data_provenance": value.get("data_provenance"),
        "case_ids": [r["id"] for r in value["cases"]],
        "excluded_cases": value["excluded_cases"],
        "revisions": revisions,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "model_load_seconds": load_seconds,
        "python": platform.python_version(),
        "packages": {
            module.__name__: module.__version__
            for module in (diffusers, transformers, accelerate)
        },
        "resolution": SIZE,
        "steps_requested": args.steps,
        "seed": 9026,
        "optimizer": "AdamW",
        "learning_rate": 1e-05,
        "weight_decay": 0.01,
        "condition_dropout": 0.1,
        "preservation_weight": 0.0,
        "method": "Proposed standard denoising objective; not a reproduction of DREAM training.",
        "frozen_weights_unchanged": False,
        "adapter_reload_verified": False,
        "quality_improvement_established": False,
    }
    (output / "run.json").write_text(json.dumps(record, indent=2))
    started = time.monotonic()
    try:
        pipe.vae.requires_grad_(False).eval()
        pipe.safety_checker.requires_grad_(False).eval()
        pipe.unet.float().train()
        selected = configure_attention(pipe.unet)
        pipe.unet.enable_gradient_checkpointing()
        scheduler = DDPMScheduler.from_config(pipe.noise_scheduler.config)
        if pipe.unet.config.in_channels != 8:
            raise ValueError("Expected the eight-channel mask-free U-Net")
        record["prediction_type"] = scheduler.config.prediction_type
        record["epochs"] = 3
        record["initialization"] = "pinned pretrained"
        record["sampling"] = "seeded shuffled full passes"
        record["runner_sha256"] = sha(__file__)
        record["training_core_sha256"] = sha(ROOT / "training_core.py")
        record["trainable_parameters"] = sum((p.numel() for p in selected.values()))
        if record["trainable_parameters"] != 49574080:
            raise ValueError("Training protocol trainable parameter count differs")
        record["total_unet_parameters"] = sum(
            (p.numel() for p in pipe.unet.parameters())
        )
        record["frozen_before_sha256"] = frozen_digest(pipe.unet)
        initial = adapter_state(selected)
        cached = {"train": [], "validation": []}
        for case_index, case in enumerate(value["cases"]):
            if time.monotonic() - started > 40 * 60:
                raise TimeoutError("40-minute processing limit during caching")
            encoded = []
            images = training_images(case)
            for image in images:
                pixels = prepare_image(image).to("cuda", dtype=pipe.vae.dtype)
                with torch.no_grad():
                    latent = (
                        pipe.vae.encode(pixels).latent_dist.sample()
                        * pipe.vae.config.scaling_factor
                    )
                encoded.append(latent.detach())
            cached[case["split"]].append([tensor.cpu() for tensor in encoded])
            if (case_index + 1) % 50 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "latent-cache",
                            "cases": case_index + 1,
                            "total": len(value["cases"]),
                        }
                    ),
                    flush=True,
                )
        optimizer = torch.optim.AdamW(
            list(selected.values()), lr=1e-05, weight_decay=0.01, foreach=False
        )

        def validation_loss():
            values = []
            pipe.unet.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for index, item in enumerate(cached["validation"]):
                    generator = torch.Generator("cuda").manual_seed(8100 + index)
                    values.append(
                        float(
                            denoising_loss(
                                pipe.unet,
                                scheduler,
                                *[x.to("cuda") for x in item],
                                generator,
                                dropout=0,
                            )
                        )
                    )
            pipe.unet.train()
            return sum(values) / len(values)

        record["validation_denoising_mse_before"] = validation_loss()
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator("cuda").manual_seed(9026)
        order = epoch_indices(len(cached["train"]))
        if len(order) != args.steps:
            raise ValueError("Training order length differs from protocol")
        training_ids = [c["id"] for c in value["cases"] if c["split"] == "train"]
        for step in range(args.steps):
            if time.monotonic() - started > 40 * 60:
                raise TimeoutError(
                    "40-minute processing limit reached before another training update"
                )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            before = time.monotonic()
            item = [x.to("cuda") for x in cached["train"][order[step]]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = denoising_loss(pipe.unet, scheduler, *item, generator)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            loss.backward()
            grads = [p.grad for p in selected.values() if p.grad is not None]
            if not grads or not all((bool(torch.isfinite(g).all()) for g in grads)):
                raise ValueError("Missing or non-finite attention gradients")
            norm = torch.nn.utils.clip_grad_norm_(list(selected.values()), 1.0)
            if not torch.isfinite(norm) or norm.item() <= 0:
                raise ValueError("Zero or non-finite gradient norm")
            optimizer.step()
            torch.cuda.synchronize()
            event = {
                "step": step + 1,
                "loss": loss.item(),
                "grad_norm_before_clip": norm.item(),
                "training_case_id": training_ids[order[step]],
                "epoch": step // len(cached["train"]) + 1,
                "seconds": time.monotonic() - before,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
            with (output / "steps.jsonl").open("a") as log:
                log.write(json.dumps(event) + "\n")
            if (step + 1) % 25 == 0:
                print(json.dumps(event), flush=True)
            if (step + 1) % 1000 == 0:
                epoch_file = output / (
                    "epoch-" + str((step + 1) // 1000) + ".safetensors"
                )
                save_file(
                    adapter_state(selected),
                    str(epoch_file),
                    metadata={
                        "scope": "intermediate recovery only; final checkpoint prespecified"
                    },
                )
        record["validation_denoising_mse_after"] = validation_loss()
        record["frozen_after_sha256"] = frozen_digest(pipe.unet)
        if record["frozen_before_sha256"] != record["frozen_after_sha256"]:
            raise ValueError("Frozen U-Net weights changed")
        record["frozen_weights_unchanged"] = True
        state = adapter_state(selected)
        change = sum(((state[k] - initial[k]).abs().sum().item() for k in state))
        record["mean_absolute_attention_update"] = (
            change / record["trainable_parameters"]
        )
        if change <= 0:
            raise ValueError("No attention weight update occurred")
        adapter = output / "attention-adapter.safetensors"
        save_file(
            state,
            str(adapter),
            metadata={"scope": record["scope"], "plan_sha256": record["plan_sha256"]},
        )
        apply_adapter(pipe.unet, initial)
        reloaded = load_file(str(adapter))
        apply_adapter(pipe.unet, reloaded)
        if not all(
            (
                torch.equal(p.detach().cpu(), reloaded[name])
                for name, p in selected.items()
            )
        ):
            raise ValueError("Adapter reload mismatch")
        record["adapter_reload_verified"] = True
        record["adapter_sha256"] = sha(adapter)
        record["steps_completed"] = args.steps
        record["status"] = "adaptation-complete-evaluation-pending"
        print(
            json.dumps(
                {
                    "status": record["status"],
                    "steps_completed": args.steps,
                    "adapter_sha256": record["adapter_sha256"],
                }
            ),
            flush=True,
        )
        return
    except Exception as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (output / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        record["processing_seconds"] = time.monotonic() - started
        (output / "run.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
