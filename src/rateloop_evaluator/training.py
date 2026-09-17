"""Opt-in local training with the same question semantics as inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .backends import (GLiNERBackend, MODEL_ID, MODEL_REVISION, offline_environment,
                       question_schema, render_input, validate_scores, write_model_manifest, model_token_limit, file_hash, MANIFEST_NAME, validate_local_model)


REVIEWED_BASE_WEIGHTS_SHA256 = "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395"


def assert_public_training_base(manifest: dict[str, Any]) -> None:
    """Retrain from the reviewed public base, never chain private adaptations.

    This keeps each model's lineage complete in its own authorized snapshot.
    Warm-starting from private/adapted weights would also require recursively
    tracking and retiring every ancestor; that workflow is not supported yet.
    """
    source = manifest.get("source", {})
    if (manifest.get("training") is not None or source.get("repository") != MODEL_ID
            or source.get("revision") != MODEL_REVISION
            or manifest.get("files", {}).get("model.safetensors") != REVIEWED_BASE_WEIGHTS_SHA256):
        raise PermissionError("Training must start from the reviewed public GLiNER base; include all currently authorized examples in a new snapshot")


@dataclass(frozen=True)
class TrainOptions:
    method: str = "lora"
    device: str = "cpu"
    epochs: int = 3
    max_steps: int = -1
    batch_size: int = 1
    learning_rate: float = 1e-5
    lora_rank: int = 8
    seed: int = 42

    def validate(self) -> None:
        if self.method not in {"full", "lora"}:
            raise ValueError("Training method must be full or lora")
        if self.device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Training device must be cpu, mps or cuda")
        if self.epochs < 1 or self.batch_size < 1 or self.lora_rank < 1:
            raise ValueError("Epochs, batch size and LoRA rank must be positive")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError("max_steps must be -1 or a positive number")
        if not math.isfinite(self.learning_rate) or not 0 < self.learning_rate < 1:
            raise ValueError("Invalid learning rate")


def training_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce upstream training records without changing the label vocabulary."""
    if not examples:
        raise ValueError("Snapshot has no eligible training examples")
    records = []
    for example in examples:
        questions = example["template"]["questions"]
        if set(example["labels"]) != {question["id"] for question in questions}:
            raise ValueError("Training example lacks a complete human judgment")
        classifications = []
        for question in questions:
            labels = {item["id"]: item["description"] for item in question["labels"]}
            label = example["labels"][question["id"]]
            if label not in labels:
                raise ValueError("Human label does not belong to this template")
            classifications.append({
                "task": question["id"], "labels": list(labels),
                "true_label": [label], "multi_label": False,
                "prompt": question["text"], "label_descriptions": labels,
            })
        records.append({"input": render_input(example["input"]),
                        "output": {"classifications": classifications}})
    return records


def make_trainer(model: Any, output_dir: Path, options: TrainOptions) -> Any:
    """Upstream trainer with explicit device selection and conservative FP32.

    Upstream 2.0.0 selects CUDA/CPU automatically and does not select MPS.
    This small device override makes the requested choice explicit without
    changing the upstream loss, optimizer, batching or checkpoint logic.
    """
    import torch
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    class LocalDeviceTrainer(GLiNER2Trainer):
        def _setup_device(self) -> None:
            if options.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable for training")
            if options.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for training")
            self.device = torch.device(options.device)
            self.is_distributed = False
            self.model.to(self.device).float()

    config = TrainingConfig(
        output_dir=str(output_dir), experiment_name="rateloop-local",
        num_epochs=options.epochs, max_steps=options.max_steps,
        batch_size=options.batch_size, eval_batch_size=options.batch_size,
        encoder_lr=options.learning_rate, task_lr=options.learning_rate,
        fp16=False, bf16=False, num_workers=0, pin_memory=False,
        fused_optimizer=False, report_to_wandb=False, eval_strategy="no",
        save_best=False, save_total_limit=1, seed=options.seed,
        use_lora=options.method == "lora", lora_r=options.lora_rank,
        lora_alpha=2.0 * options.lora_rank,
        lora_target_modules=["encoder", "classifier"], save_adapter_only=True,
        logging_steps=1, strict_training=True, skip_step_errors=False,
        ignore_nonfinite_losses=False,
    )
    return LocalDeviceTrainer(model, config)


def parameter_fingerprint(model: Any) -> str:
    """Compact check that optimizer execution changed trainable parameters."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().flatten()[:128].float().cpu().numpy().tobytes())
    return digest.hexdigest()


def sanitized_training_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Keep numeric upstream metrics only; Infinity for unevaluated best is null."""
    def numeric(value):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("Unexpected non-numeric training metric")
        return value if math.isfinite(value) else None
    summary = {key: numeric(result[key]) for key in (
        "total_steps", "total_epochs", "total_time_seconds", "samples_per_second", "best_metric"
    ) if key in result}
    fields = {"loss", "classification_loss", "structure_loss", "count_loss", "learning_rate",
              "epoch", "step", "samples_seen", "throughput"}
    for key in ("train_metrics_history", "eval_metrics_history"):
        summary[key] = [{name: numeric(value) for name, value in row.items() if name in fields}
                        for row in result.get(key, [])]
    return summary


def train_snapshot(store: Any, snapshot_id: str, workspace_id: str,
                   model_dir: str | Path, output_dir: str | Path, *, bundle_id: str,
                   options: TrainOptions | None = None) -> dict[str, Any]:
    """Train only the authorized train split, save, reload and compare predictions.

    Calibration/test partitions never enter the upstream optimizer. The store
    rechecks current grants at job start, immediately before training, and before
    registering lineage. A revoked job cannot publish a usable model bundle.
    """
    options = options or TrainOptions()
    options.validate()
    snapshot = store.load_snapshot(snapshot_id, workspace_id)
    examples = snapshot["train"]
    records = training_records(examples)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output directory must be empty")
    offline_environment()
    assert_public_training_base(validate_local_model(model_dir))
    backend = GLiNERBackend(model_dir, options.device)
    model = backend.load()
    for example, record in zip(examples, records):
        count = backend.count_tokens(record["input"], example["template"]["questions"])
        limit = min(example["template"]["maxTokens"], model_token_limit(model))
        if count > limit:
            raise ValueError("Training example exceeds the template or model token limit")
    store.load_snapshot(snapshot_id, workspace_id)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    trainer = make_trainer(model, output_dir / "checkpoints", options)
    before_fingerprint = parameter_fingerprint(trainer.model)
    result = trainer.train(train_data=records)
    after_fingerprint = parameter_fingerprint(trainer.model)
    if before_fingerprint == after_fingerprint:
        raise RuntimeError("Optimizer steps did not change sampled trainable parameters")
    if getattr(trainer, "global_step", 0) < 1:
        raise RuntimeError("Trainer did not complete an optimizer step")
    trained = trainer.model
    if options.method == "lora":
        trained = trained.merge_and_unload()
    trained.eval()
    probe = examples[0]
    text = records[0]["input"]
    questions = probe["template"]["questions"]
    expected = validate_scores(trained.extract(text, question_schema(questions),
                                              include_confidence=True), questions)
    artifact_dir = output_dir / "model"
    trained.save_pretrained(str(artifact_dir))
    metadata = {
        "bundleId": bundle_id, "workspaceId": workspace_id, "snapshotId": snapshot_id,
        "trainedAt": datetime.now(timezone.utc).isoformat(), "options": asdict(options),
        "optimizerSteps": trainer.global_step, "trainExamples": len(examples),
        "trainableParametersChanged": before_fingerprint != after_fingerprint,
        "trainingGroupIds": sorted({example["group_id"] for example in examples}),
        "trainingExampleIds": sorted({example["evaluation_id"] for example in examples}),
        "templateId": snapshot["template_id"], "templateVersion": snapshot["template_version"],
        "metrics": sanitized_training_metrics(result),
    }
    source = dict((backend.manifest or {}).get("source", {
        "repository": MODEL_ID, "revision": MODEL_REVISION,
    }))
    source.setdefault("baseWeightsSha256", (backend.manifest or {})["files"]["model.safetensors"])
    source["parentModelManifestSha256"] = file_hash(Path(model_dir) / MANIFEST_NAME)
    manifest = write_model_manifest(artifact_dir, source=source, training=metadata)
    # Release training state before loading the checkpoint again. This verifies
    # a portable, merged checkpoint rather than relying on a live adapter.
    del trainer, trained, model
    backend._model = None
    gc.collect()
    import torch
    if options.device == "mps":
        torch.mps.empty_cache()
    elif options.device == "cuda":
        torch.cuda.empty_cache()
    reloaded = GLiNERBackend(artifact_dir, options.device)
    actual = reloaded.predict(text, questions)
    max_error = max(abs(actual[qid][label] - score)
                    for qid, scores in expected.items() for label, score in scores.items())
    if max_error > 1e-4:
        raise RuntimeError("Trained checkpoint prediction round trip failed")
    metadata["reloadMaxAbsoluteError"] = max_error
    store.register_model_lineage(bundle_id, snapshot_id, workspace_id)
    # Record only after current rights are checked by lineage registration.
    manifest = write_model_manifest(artifact_dir, source=source, training=metadata)
    report = {"modelDir": str(artifact_dir), "manifest": manifest, "training": metadata}
    (output_dir / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
