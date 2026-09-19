#!/usr/bin/env python3
"""Run OmniVoice's official trainer inside a session budget, and nothing more.

The official adaptation path is ``accelerate launch -m omnivoice.cli.train``
with the example configs of ``k2-fsa/OmniVoice`` (pinned at 08be0b4c). This
wrapper builds the same model, tokenizer, data loaders and trainer, with the
same functions in the same order, and runs the official training loop. What it
adds is what a Kaggle session needs and a record of what happened:

``--stop-after-steps N``
    Stop after N optimizer steps: the probe. Nothing is saved.
``--session-seconds S``
    After each official checkpoint save, stop if the next save would not fit
    in the session. The next session resumes from that checkpoint through the
    official ``resume_from_checkpoint``. All processes take the same decision.
``--export-every K``
    At official saves whose step is a multiple of K, also write the weights as
    float16 safetensors, which is how inference loads them.
``--memory``
    ``official`` by default. ``checkpointing`` recomputes each LLM layer's
    activations in the backward pass; ``bucket-view`` lets DDP's gradient
    buckets share storage with the gradients; ``single-tensor`` has AdamW update
    one parameter at a time instead of all at once, which avoids a
    model-sized temporary. Joined with ``+``. All three change memory, not
    what is computed; they are used only when the official configuration does
    not fit, and the record says which ran. The first optimizer step's
    gradients are fingerprinted (their norm after clipping, and the sums of a
    few tensors) so two runs can be checked to have computed the same thing.
``--set key=value``
    Only for settings that do not change training: the checkpoint to resume
    from, the local path of the pinned base model, and how many checkpoints to
    keep on disk. Anything else is refused.

Every optimizer step is timed and its peak memory read through an optimizer
post-hook and a forward hook that read and never write. ``--record`` gets the
lot as JSON from the main process.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import shutil
import sys
import time
from pathlib import Path

#: Settings a session may change because they do not touch what is computed.
SAFE_OVERRIDES = {"resume_from_checkpoint", "init_from_checkpoint", "keep_last_n_checkpoints"}
MEMORY_TECHNIQUES = {"checkpointing", "bucket-view", "single-tensor"}


class StopSession(Exception):
    """Raised on every process at the same point to end the session cleanly."""


def parse_overrides(items: list[str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if key not in SAFE_OVERRIDES:
            raise SystemExit(f"--set {key}: refused - only {sorted(SAFE_OVERRIDES)} may change")
        if raw.lower() in {"none", "null", ""}:
            parsed[key] = None
        else:
            try:
                parsed[key] = int(raw)
            except ValueError:
                parsed[key] = raw
    return parsed


class Meter:
    """Step times, peak memory and losses, read from hooks."""

    def __init__(self, device) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.steps = 0
        self.started = time.time()
        self.last = self.started
        self.step_seconds: list[float] = []
        self.step_peaks_gib: list[float] = []
        self.losses: list[float] = []
        self.eval_losses: list[float] = []
        self.stop_after: int | None = None
        self.first_gradients: dict | None = None

    def forward_hook(self, module, args, output) -> None:
        loss = getattr(output, "loss", None)
        if loss is None and isinstance(output, dict):
            loss = output.get("loss")
        if loss is not None:
            value = float(loss.detach().float().item())
            (self.losses if module.training else self.eval_losses).append(value)

    def gradient_hook(self, optimizer, args, kwargs) -> None:
        """Before the first update: the gradients the step is about to apply."""
        if self.first_gradients is not None:
            return
        grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not grads:
            return
        norm = self.torch.linalg.vector_norm(
            self.torch.stack([self.torch.linalg.vector_norm(g.float()) for g in grads])
        )
        picked = grads[:3] + grads[-3:]
        self.first_gradients = {
            "tensors": len(grads),
            "norm_after_clipping": float(norm),
            "sums_of_first_and_last_three": [float(g.float().sum()) for g in picked],
        }

    def step_hook(self, optimizer, args, kwargs) -> None:
        now = time.time()
        self.steps += 1
        self.step_seconds.append(now - self.last)
        self.last = now
        self.step_peaks_gib.append(self.torch.cuda.max_memory_allocated(self.device) / 2**30)
        if self.stop_after is not None and self.steps >= self.stop_after:
            raise StopSession(f"stopped after {self.steps} optimizer steps, as asked")

    def summary(self) -> dict:
        timed = self.step_seconds[2:] if len(self.step_seconds) > 3 else self.step_seconds
        return {
            "optimizer_steps": self.steps,
            "step_seconds": [round(s, 3) for s in self.step_seconds],
            "mean_step_seconds_after_two": round(sum(timed) / len(timed), 3) if timed else None,
            "peak_allocated_gib": round(max(self.step_peaks_gib), 3) if self.step_peaks_gib else None,
            "peak_reserved_gib": round(self.torch.cuda.max_memory_reserved(self.device) / 2**30, 3),
            "train_losses": [round(v, 5) for v in self.losses],
            "eval_losses": [round(v, 5) for v in self.eval_losses],
            "first_step_gradients": self.first_gradients,
        }


def export_float16(model, source_checkpoint: Path, destination: Path, audio_tokenizer: Path | None) -> None:
    """Write the weights as float16 beside the files inference reads.

    The live model is not touched: a float16 copy of its state goes through the
    model's own ``save_pretrained``, so the directory loads like the published
    one. The tokenizer files come from the official checkpoint just written,
    and the audio tokenizer from the pinned base snapshot.
    """
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    state = {k: (v.detach().to(torch.float16) if v.is_floating_point() else v.detach()).cpu()
             for k, v in model.state_dict().items()}
    model.save_pretrained(destination, state_dict=state, safe_serialization=True)
    for name in os.listdir(source_checkpoint):
        path = source_checkpoint / name
        if path.is_file() and name.startswith(("tokenizer", "special_tokens", "chat_template", "added_tokens")):
            shutil.copy2(path, destination / name)
    if audio_tokenizer is not None and audio_tokenizer.is_dir():
        shutil.copytree(audio_tokenizer, destination / "audio_tokenizer", dirs_exist_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--data_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--session-seconds", type=float)
    parser.add_argument("--export-every", type=int)
    parser.add_argument("--audio-tokenizer", type=Path, help="Copied beside each export")
    parser.add_argument("--memory", default="official",
                        help=f"official, or any of {sorted(MEMORY_TECHNIQUES)} joined with +")
    parser.add_argument("--evaluate-once", action="store_true",
                        help="After stopping, run the official evaluation pass once and time it")
    parser.add_argument("--record", type=Path, help="JSON record; one file per process, suffixed .rankN")
    args = parser.parse_args(argv)
    techniques = set() if args.memory == "official" else set(args.memory.split("+"))
    if techniques - MEMORY_TECHNIQUES:
        raise SystemExit(f"--memory: unknown {sorted(techniques - MEMORY_TECHNIQUES)}")

    import omnivoice.training.trainer as official_trainer
    import torch
    from accelerate import DistributedDataParallelKwargs
    from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
    from omnivoice.training.config import TrainingConfig
    from omnivoice.training.trainer import OmniTrainer

    started = time.time()
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config
    for key, value in parse_overrides(args.set).items():
        setattr(config, key, value)

    if "bucket-view" in techniques:
        official_trainer.DistributedDataParallelKwargs = functools.partial(
            DistributedDataParallelKwargs, gradient_as_bucket_view=True
        )

    model, tokenizer = build_model_and_tokenizer(config)
    if "checkpointing" in techniques:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    deadline = started + args.session_seconds if args.session_seconds else None
    exported: list[int] = []
    saves: list[dict] = []

    class Budgeted(OmniTrainer):
        def save_checkpoint(self, step):
            begun = time.time()
            super().save_checkpoint(step)
            checkpoint = Path(self.config.output_dir) / f"checkpoint-{step}"
            if args.export_every and step % args.export_every == 0 and self.accelerator.is_main_process:
                export_float16(self.accelerator.unwrap_model(self.model), checkpoint,
                               Path(self.config.output_dir) / f"export-{step}-float16", args.audio_tokenizer)
                exported.append(step)
            self.accelerator.wait_for_everyone()
            saves.append({"step": step, "seconds": round(time.time() - begun, 1)})
            if deadline is None or step >= self.config.steps:
                return
            spent = meter.step_seconds[2:] or meter.step_seconds
            per_step = sum(spent) / max(len(spent), 1)
            needed = per_step * self.config.save_steps + 2 * (time.time() - begun) + 300
            stop = torch.tensor([1 if time.time() + needed > deadline else 0], device=self.accelerator.device)
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.broadcast(stop, src=0)
            if int(stop.item()):
                raise StopSession(f"stopped at checkpoint-{step}: the next save would not fit the session")

    official_adamw = torch.optim.AdamW
    if "single-tensor" in techniques:
        # The trainer builds its optimizer through this name; the update rule is
        # the same, applied one tensor at a time.
        torch.optim.AdamW = functools.partial(official_adamw, foreach=False)
    try:
        trainer = Budgeted(model=model, config=config, train_dataloader=train_loader,
                           eval_dataloader=eval_loader, tokenizer=tokenizer)
    finally:
        torch.optim.AdamW = official_adamw
    meter = Meter(trainer.accelerator.device)
    meter.stop_after = args.stop_after_steps
    trainer.accelerator.unwrap_model(trainer.model).register_forward_hook(meter.forward_hook)
    trainer.optimizer.optimizer.register_step_pre_hook(meter.gradient_hook)
    trainer.optimizer.optimizer.register_step_post_hook(meter.step_hook)
    torch.cuda.reset_peak_memory_stats(trainer.accelerator.device)

    outcome = "finished"
    evaluation = None
    try:
        trainer.train()
    except StopSession as stop:
        outcome = str(stop)
    except torch.cuda.OutOfMemoryError as error:
        outcome = f"out of memory: {str(error).splitlines()[0]}"
        if args.record:
            write_record(args, config, meter, outcome, started, exported, saves, trainer.global_step, None)
        raise

    if args.evaluate_once and trainer.eval_dataloader is not None:
        begun = time.time()
        metrics = trainer.evaluate()
        evaluation = {"seconds": round(time.time() - begun, 1), **metrics}

    if args.record:
        write_record(args, config, meter, outcome, started, exported, saves, trainer.global_step, evaluation)
    print(f"omnivoice_adapt: {outcome} at global step {trainer.global_step}", flush=True)
    return 0


def write_record(args, config, meter, outcome, started, exported, saves, global_step, evaluation) -> None:
    from importlib import metadata

    import torch

    versions = {}
    for name in ("torch", "torchaudio", "transformers", "accelerate", "webdataset", "safetensors",
                 "tokenizers", "huggingface-hub", "numpy", "omnivoice", "triton"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "outcome": outcome,
        "global_step": global_step,
        "wall_seconds": round(time.time() - started, 1),
        "memory_mode": args.memory,
        "config": dataclasses.asdict(config),
        "versions": versions,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
        "capacity_gib": round(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30, 2)
        if torch.cuda.is_available() else None,
        "rank": int(os.environ.get("RANK", "0")),
        "exports": exported,
        "saves": saves,
        "evaluation_once": evaluation,
        **meter.summary(),
    }
    target = args.record.with_name(f"{args.record.stem}.rank{record['rank']}{args.record.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
