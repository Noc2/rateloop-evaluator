"""Opt-in local training with the same question semantics as inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any
from types import MethodType
from .execution import serialized_training
from .protocol import validate_no_demonstration_overlap
from .quality import (balanced_optimizer_examples, freeze_validation_partition, group_representatives, score_predictions,
                      validation_improved)

from .backends import (GLiNERBackend, MODEL_ID, MODEL_REVISION, offline_environment,
                       question_schema, question_examples, render_input, validate_scores, write_model_manifest, model_token_limit, file_hash, MANIFEST_NAME, validate_local_model, artifact_inventory, tokenizer_commitment)


REVIEWED_BASE_WEIGHTS_SHA256 = "c1ff4ec0bc00031c15530b8f3c33d3677f27949e6a0cb52e1247a6224b6c5395"


class ValidationQualityError(ValueError):
    """A functional optimizer run did not establish an acceptable validation gain."""


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
    validation_fraction: float = 0
    validation_interval: int = 25
    early_stopping_patience: int = 3
    min_validation_per_label: int = 5

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
        if self.validation_fraction:
            if (self.method != "lora" or self.validation_fraction != .2
                    or not 1 <= self.max_steps <= 2000
                    or not 1 <= self.validation_interval <= self.max_steps
                    or not 1 <= self.early_stopping_patience <= 10
                    or not 2 <= self.min_validation_per_label <= 100):
                raise ValueError("Validation training needs LoRA, bounded steps, and valid validation limits")


def training_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Produce upstream training records without changing the label vocabulary."""
    if not examples:
        raise ValueError("Snapshot has no eligible training examples")
    records = []
    for example in examples:
        questions = example["template"]["questions"]
        validate_no_demonstration_overlap(example["input"]["text"], questions)
        if set(example["labels"]) != {question["id"] for question in questions}:
            raise ValueError("Training example lacks a complete declared judgment")
        classifications = []
        for question in questions:
            labels = {item["id"]: item["description"] for item in question["labels"]}
            label = example["labels"][question["id"]]
            if label not in labels:
                raise ValueError("Training label does not belong to this template")
            classifications.append({
                "task": question["id"], "labels": list(labels),
                "true_label": [label], "multi_label": False,
                "prompt": question["text"], "label_descriptions": labels,
                **question_examples(question),
            })
        records.append({"input": render_input(example["input"]),
                        "output": {"classifications": classifications}})
    return records


def exact_classification_processor(processor: Any) -> Any:
    """Keep optimizer classification inputs identical to the immutable rubric.

    Upstream 2.0.0's default augmentation can replace declared label IDs with
    aliases and then reinsert the original true ID as a *negative* choice. It
    can also omit the descriptions/examples that define the question. Neither
    transformation is valid for a workspace's fixed evaluation contract.

    Copy only the processor shell: the immutable tokenizer remains shared. The
    model's inference processor is not patched. Training collation and boundary
    target construction retain their normal upstream training behavior, with
    exact task binding instead of upstream's ambiguous task-ID prefix lookup.
    """
    frozen = copy(processor)
    infer = type(processor)._infer_from_json
    transform = type(processor)._transform_schema
    build = type(processor)._build_outputs

    def immutable_schema(self, schema):
        was_training = self.is_training
        self.is_training = False
        try:
            # Evaluation-mode classification transformation retains the exact
            # IDs, question, ordered descriptions and demonstrations, while the
            # original true_label remains available for supervised targets.
            return infer(self, schema)
        finally:
            self.is_training = was_training

    def exact_outputs(self, processed, schema, text_tokens, len_prefix):
        # Upstream startswith(task) can bind `quality` to `q` and silently
        # apply the wrong one-hot targets. Match the complete immutable schema
        # before handing each group to the original target builder.
        was_training = self.is_training
        self.is_training = False
        try:
            tasks = {tuple(transform(self, item['task'], item['labels'], self.L_TOKEN,
                prompt=item.get('prompt'), examples=item.get('examples', []),
                label_descriptions=item.get('label_descriptions', {}))): item
                for item in schema.get('classifications', [])}
        finally:
            self.is_training = was_training
        outputs = []
        for tokens, kind, labels in zip(processed['schemas'], processed['task_types'],
                                        processed['structure_labels']):
            selected = schema
            if kind == 'classifications':
                item = tasks.get(tuple(tokens))
                if item is None:
                    raise ValueError('Training classification differs from the immutable rubric')
                selected = {**schema, 'classifications': [item]}
            outputs.extend(build(self, {'schemas': [tokens], 'task_types': [kind],
                'structure_labels': [labels]}, selected, text_tokens, len_prefix))
        return outputs

    frozen._infer_from_json = MethodType(immutable_schema, frozen)
    frozen._build_outputs = MethodType(exact_outputs, frozen)
    return frozen


def make_trainer(model: Any, output_dir: Path, options: TrainOptions,
                 validation_examples: list[dict] | None = None) -> Any:
    """Upstream trainer with explicit device selection and conservative FP32.

    Upstream 2.0.0 selects CUDA/CPU automatically and does not select MPS.
    This small device override makes the requested choice explicit without
    changing the upstream loss, optimizer, batching or checkpoint logic.
    """
    import torch
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    class LocalDeviceTrainer(GLiNER2Trainer):
        selection_history: list[dict]
        selected_parameters: dict | None = None
        selected_step: int = 0

        def _optimizer_step(self) -> bool:
            if getattr(self,"authorization_check",None):
                self.authorization_check()
            return super()._optimizer_step()

        def _evaluate(self, eval_dataset: Any) -> dict:
            if not validation_examples:
                return super()._evaluate(eval_dataset)
            self.model.eval()
            self.processor.change_mode(is_training=False)
            rows = group_representatives(validation_examples)
            predictions = []
            with torch.no_grad():
                for row in rows:
                    if getattr(self, "authorization_check", None):
                        self.authorization_check()
                    questions = row['template']['questions']
                    predictions.append(validate_scores(self.model.extract(render_input(row['input']),
                        question_schema(questions), include_confidence=True), questions))
            report = score_predictions(rows, predictions)
            score = report['balanced_agreement']
            if score is None:
                raise ValueError('Validation requires support for every declared label')
            self.selection_history.append({'step': self.global_step, **report})
            metrics = {'eval_loss': report['mean_brier_score'],
                'eval_balanced_agreement': score, 'step': self.global_step, 'epoch': self.epoch}
            self.eval_metrics_history.append(metrics)
            # Earliest checkpoint wins a tie; no test result influences selection.
            eligible = not getattr(self, 'baseline_validation', None) or validation_improved(self.baseline_validation, report)
            if score > self.best_metric and eligible:
                self.best_metric = score
                self.selected_step = self.global_step
                self.selected_parameters = {name: parameter.detach().cpu().clone()
                    for name, parameter in self.model.named_parameters() if parameter.requires_grad}
            return metrics

        def _save_checkpoint(self, checkpoint_name: str) -> None:
            # The selected LoRA parameters fit in memory. train_snapshot writes
            # and verifies the one portable final artifact, avoiding full copies
            # of a checkpoint on every validation interval.
            if not validation_examples:
                return super()._save_checkpoint(checkpoint_name)

        def _check_early_stopping(self, metrics: dict, prev_best: float | None = None) -> bool:
            if not validation_examples:
                return super()._check_early_stopping(metrics, prev_best)
            score = metrics['eval_balanced_agreement']
            best = getattr(self, 'validation_progress', self.baseline_validation['balanced_agreement'])
            if score > best:
                self.validation_progress = score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            return self.patience_counter >= options.early_stopping_patience

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
        fused_optimizer=False, report_to_wandb=False,
        eval_strategy="steps" if validation_examples else "no",
        eval_steps=options.validation_interval,
        metric_for_best="eval_balanced_agreement", greater_is_better=True,
        early_stopping=bool(validation_examples), early_stopping_patience=options.early_stopping_patience,
        save_best=False, save_total_limit=1, seed=options.seed,
        use_lora=options.method == "lora", lora_r=options.lora_rank,
        lora_alpha=2.0 * options.lora_rank,
        lora_target_modules=["encoder", "classifier"], save_adapter_only=True,
        logging_steps=1, strict_training=True, skip_step_errors=False,
        ignore_nonfinite_losses=False,
    )
    trainer = LocalDeviceTrainer(model, config, processor=exact_classification_processor(model.processor))
    trainer.selection_history = []
    return trainer


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


def tokenizer_semantics(model: Any) -> dict[str, Any]:
    """Capture the complete fast tokenizer, vocabulary and special-token rules."""
    tokenizer = model.processor.tokenizer
    return {"backend": json.loads(tokenizer.backend_tokenizer.to_str()),
            "specialTokens": tokenizer.special_tokens_map,
            "modelInputs": tokenizer.model_input_names,
            "maxLength": tokenizer.model_max_length,
            "paddingSide": tokenizer.padding_side,
            "truncationSide": tokenizer.truncation_side,
            "cleanupSpaces": tokenizer.clean_up_tokenization_spaces}


def preserve_tokenizer_assets(source_dir: Path, artifact_dir: Path, source_manifest: dict[str, Any]) -> None:
    """Keep exact base bytes; upstream save_pretrained reserializes these files.

    Remove generated tokenizer sidecars absent from the pinned source as they can
    override its settings on reload. Loaded semantics are checked separately.
    """
    sidecars = {"special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt",
                "merges.txt", "sentencepiece.bpe.model", "spiece.model"}
    def tokenizer_asset(name): return "tokenizer" in name or name in sidecars
    selected = {name: digest for name, digest in source_manifest["files"].items() if tokenizer_asset(name)}
    if not {"tokenizer.json", "tokenizer_config.json"} <= selected.keys():
        raise ValueError("Pinned source tokenizer assets are incomplete")
    source_files = artifact_inventory(source_dir)
    for name, digest in selected.items():
        if name not in source_files or file_hash(source_files[name]) != digest:
            raise ValueError("Pinned source tokenizer changed during training")
    for name, path in artifact_inventory(artifact_dir).items():
        if tokenizer_asset(name) and name not in selected:
            path.unlink()
    for name in selected:
        target = artifact_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_files[name], target)


@serialized_training
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
    validation_examples = []
    examples = snapshot["train"]
    if options.validation_fraction:
        examples, validation_examples = freeze_validation_partition(store, snapshot,
            minimum_per_label=options.min_validation_per_label)
        examples = group_representatives(examples)
    optimizer_examples = balanced_optimizer_examples(examples) if validation_examples else examples
    records = training_records(optimizer_examples)
    validation_records = training_records(validation_examples) if validation_examples else None
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Training output directory must be empty")
    offline_environment()
    assert_public_training_base(validate_local_model(model_dir))
    backend = GLiNERBackend(model_dir, options.device)
    model = backend.load()
    original_tokenizer = tokenizer_semantics(model)
    for example, record in zip(optimizer_examples + validation_examples, records + (validation_records or [])):
        count = backend.count_tokens(record["input"], example["template"]["questions"])
        limit = min(example["template"]["maxTokens"], model_token_limit(model))
        if count > limit:
            raise ValueError("Training example exceeds the template or model token limit")
    store.load_snapshot(snapshot_id, workspace_id)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    trainer = make_trainer(model, output_dir / "checkpoints", options, validation_examples)
    trainer.authorization_check = lambda: store.load_snapshot(snapshot_id,workspace_id)
    before_fingerprint = parameter_fingerprint(trainer.model)
    if validation_examples:
        trainer._evaluate(None)
        trainer.baseline_validation = trainer.selection_history[-1]
        trainer.best_metric = float('-inf')
        trainer.selected_parameters = None
    result = trainer.train(train_data=records, **({'eval_data': validation_records} if validation_records else {}))
    if validation_examples:
        if trainer.selected_parameters is None:
            store.load_snapshot(snapshot_id, workspace_id)
            (output_dir / 'selection-report.json').write_text(json.dumps({
                'kind': 'validation_rejected', 'optimizerSteps': trainer.global_step,
                'optimizerSampling': 'balanced_source_groups', 'history': trainer.selection_history,
                'qualityGate': False, 'registered': False,
                'reason': 'No checkpoint improved balanced agreement without reducing any observed class recall.'},
                indent=2, allow_nan=False) + '\n')
            raise ValidationQualityError('Validation did not improve without class-level regressions; keep the original evaluator and add representative examples')
        import torch
        with torch.no_grad():
            for name, parameter in trainer.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(trainer.selected_parameters[name].to(parameter.device))
    after_fingerprint = parameter_fingerprint(trainer.model)
    if before_fingerprint == after_fingerprint:
        raise RuntimeError("Optimizer steps did not change sampled trainable parameters")
    if getattr(trainer, "global_step", 0) < 1:
        raise RuntimeError("Trainer did not complete an optimizer step")
    trained = trainer.model
    if options.method == "lora":
        trained = trained.merge_and_unload()
    trained.eval()
    if tokenizer_semantics(trained) != original_tokenizer:
        raise RuntimeError("Training changed tokenizer semantics")
    probe = examples[0]
    text = records[0]["input"]
    questions = probe["template"]["questions"]
    expected = validate_scores(trained.extract(text, question_schema(questions),
                                              include_confidence=True), questions)
    artifact_dir = output_dir / "model"
    trained.save_pretrained(str(artifact_dir))
    preserve_tokenizer_assets(Path(model_dir), artifact_dir, backend.manifest)
    metadata = {
        "bundleId": bundle_id, "workspaceId": workspace_id, "snapshotId": snapshot_id,
        "trainedAt": datetime.now(timezone.utc).isoformat(), "options": asdict(options),
        "optimizerSteps": trainer.global_step, "trainExamples": len(examples),
        "optimizerRowsPerEpoch": len(optimizer_examples),
        "optimizerSampling": "balanced_source_groups" if validation_examples else "as_supplied",
        "classificationSchemaPolicy": "immutable-inference-schema-v1",
        "trainableParametersChanged": before_fingerprint != after_fingerprint,
        "trainingGroupIds": sorted({example["group_id"] for example in examples}),
        "trainingExampleIds": sorted({example["evaluation_id"] for example in examples}),
        "datasetVersionIds": sorted({example["dataset_version_id"] for example in examples if example.get("dataset_version_id")}),
        "labelProvenanceCounts": {kind: sum(example.get("label_provenance", "blind_human") == kind for example in examples)
                                   for kind in sorted({example.get("label_provenance", "blind_human") for example in examples})},
        "templateId": snapshot["template_id"], "templateVersion": snapshot["template_version"],
        "metrics": sanitized_training_metrics(result),
        "selection": {"kind": "validation_selected" if validation_examples else "fixed_budget_smoke",
            "selectedStep": trainer.selected_step if validation_examples else trainer.global_step,
            "validationExamples": len(validation_examples),
            "validationGroupIds": sorted({row['group_id'] for row in validation_examples}),
            "validationExampleIds": sorted({row['evaluation_id'] for row in validation_examples}),
            "history": trainer.selection_history,
            "qualityGate": False,
            "limits": "Selection uses only frozen training groups. Calibration and final test remain untouched. Validation scores are not independent qualification."},
    }
    source = dict((backend.manifest or {}).get("source", {
        "repository": MODEL_ID, "revision": MODEL_REVISION,
    }))
    source.setdefault("baseWeightsSha256", (backend.manifest or {})["files"]["model.safetensors"])
    source["parentModelManifestSha256"] = file_hash(Path(model_dir) / MANIFEST_NAME)
    manifest = write_model_manifest(artifact_dir, source=source, training=metadata)
    if tokenizer_commitment(manifest) != tokenizer_commitment(backend.manifest):
        raise RuntimeError("Trained checkpoint changed the pinned tokenizer identity")
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
    if tokenizer_semantics(reloaded.load()) != original_tokenizer:
        raise RuntimeError("Trained checkpoint tokenizer round trip failed")
    actual = reloaded.predict(text, questions)
    max_error = max(abs(actual[qid][label] - score)
                    for qid, scores in expected.items() for label, score in scores.items())
    if max_error > 1e-4:
        raise RuntimeError("Trained checkpoint prediction round trip failed")
    reloaded._model = None
    del reloaded
    gc.collect()
    if options.device == "mps":
        torch.mps.empty_cache()
    elif options.device == "cuda":
        torch.cuda.empty_cache()
    metadata["reloadMaxAbsoluteError"] = max_error
    # Write the complete verified artifact before its immutable lineage. A
    # restart can safely finish registration using the same authorized snapshot;
    # an initial manifest without the reload receipt is never publishable.
    store.load_snapshot(snapshot_id, workspace_id)
    manifest = write_model_manifest(artifact_dir, source=source, training=metadata)
    report = {"modelDir": str(artifact_dir), "manifest": manifest, "training": metadata}
    (output_dir / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    store.register_model_lineage(bundle_id, snapshot_id, workspace_id)
    return report
