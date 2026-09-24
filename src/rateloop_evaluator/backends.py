"""Explicitly provisioned local GLiNER classification; never a cloud fallback."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

MODEL_ID = "fastino/gliner2.5-multi-v1"
MODEL_REVISION = "235cf92d6d4318da9bfca0d08975c8fa7250d13b"
MODEL_FILES = (
    "config.json", "encoder_config/config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json", "README.md",
)
MANIFEST_NAME = "rateloop-model.json"

# Version this declaration whenever the adapter score semantics change.
GLINER_SCORE_CAPABILITY = {
    "schemaVersion": "rateloop.evaluator-score-capability.v1",
    "adapter": "rateloop-evaluator/gliner2",
    "adapterVersion": 1,
    "scoreType": "mutually_exclusive_softmax",
}


def offline_environment() -> None:
    """Set library offline/telemetry controls before importing the ML libraries.

    These controls are not an operating-system network firewall. Air-gapped
    installations should also deny process egress at their network boundary.
    """
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1",
        "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ[key] = value


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_inventory(directory: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Model artifacts and directories must not be symlinks")
        relative = path.relative_to(directory)
        if relative.parts[0] == ".cache" or relative.as_posix() == MANIFEST_NAME:
            continue
        if path.is_file():
            files[relative.as_posix()] = path
    return files


def write_model_manifest(directory: Path, *, source: dict[str, Any],
                         training: dict[str, Any] | None = None) -> dict[str, Any]:
    directory = Path(directory).resolve()
    files = {relative: file_hash(path) for relative, path in artifact_inventory(directory).items()}
    manifest = {"schemaVersion": "rateloop.model-files.v1", "source": source, "files": files}
    if training is not None:
        manifest["training"] = training
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def validate_artifact_manifest(directory: str | Path, required_files: tuple[str, ...]) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Model directory is not a local directory")
    inventory = artifact_inventory(directory)
    manifest = json.loads((directory / MANIFEST_NAME).read_text())
    if manifest.get("schemaVersion") != "rateloop.model-files.v1" or not manifest.get("files"):
        raise ValueError("Missing provisioned model integrity manifest")
    for relative, expected in manifest["files"].items():
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError("Model manifest references an external artifact")
        if not re.fullmatch(r"[a-f0-9]{64}", expected) or file_hash(path) != expected:
            raise ValueError("Model artifact integrity check failed")
    if set(inventory) != set(manifest["files"]):
        raise ValueError("Model directory contains undeclared artifacts")
    for required in required_files:
        if required not in manifest["files"]:
            raise ValueError(f"Missing offline model asset: {required}")
    return manifest


def validate_local_model(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = validate_artifact_manifest(directory, MODEL_FILES[:-1])
    for relative in ("config.json", "encoder_config/config.json", "tokenizer_config.json"):
        config = json.loads((directory / relative).read_text())
        if config.get("auto_map"):
            raise ValueError("Remote custom model code is not supported")
    if json.loads((directory / "config.json").read_text()).get("architecture") != "boundary":
        raise ValueError("This backend requires the GLiNER2.5 boundary architecture")
    return manifest


def verify_checkpoint_license(directory: Path) -> None:
    card = (directory / "README.md").read_text()
    if not card.startswith("---\n"):
        raise ValueError("Checkpoint requires an explicit Apache-2.0 model-card license")
    metadata = card.split("---", 2)[1]
    if not re.search(r"(?m)^license:\s*apache-2\.0\s*$", metadata):
        raise ValueError("Checkpoint license differs from the reviewed Apache-2.0 release")


def provision_model(destination: str | Path, *, revision: str = MODEL_REVISION) -> dict[str, Any]:
    """Download the reviewed upstream checkpoint only during this explicit step.

    A full commit SHA is required; floating branches/tags are rejected. This
    function must run in a separate CLI process before offline workers start.
    """
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Provisioning requires a full upstream commit SHA")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Provisioning destination must be empty")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("Provision in a separate online process, then start offline workers")
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot_download(MODEL_ID, revision=revision, local_dir=str(destination),
                      allow_patterns=list(MODEL_FILES), token=False)
    verify_checkpoint_license(destination)
    manifest = write_model_manifest(destination, source={
        "repository": MODEL_ID, "revision": revision, "license": "Apache-2.0",
        "library": "gliner2==2.0.0",
    })
    validate_local_model(destination)
    return manifest


def question_schema(questions: list[dict[str, Any]]) -> Any:
    from gliner2 import Schema
    schema = Schema()
    seen = set()
    for question in questions:
        qid = question["id"]
        if qid in seen:
            raise ValueError("Question IDs must be unique")
        seen.add(qid)
        labels = {label["id"]: label["description"] for label in question["labels"]}
        if len(labels) != len(question["labels"]) or len(labels) < 2:
            raise ValueError("Questions require at least two distinct labels")
        # multi_label requests every score from upstream's decoder. Explicit
        # softmax preserves mutually-exclusive semantics and the training loss.
        schema.classification(qid, labels, prompt=question["text"],
                              multi_label=True, class_act="softmax", cls_threshold=0.0)
    return schema


def render_input(value: dict[str, Any]) -> str:
    """One stable representation shared by service inference and training."""
    from .protocol import CaseInput
    return CaseInput.model_validate(value).render()


def validate_scores(result: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    scores = {}
    for question in questions:
        expected = {label["id"] for label in question["labels"]}
        rows = result.get(question["id"])
        if not isinstance(rows, list):
            raise ValueError("GLiNER did not return all classification scores")
        values = {row["label"]: float(row["confidence"]) for row in rows}
        if set(values) != expected or len(rows) != len(expected):
            raise ValueError("GLiNER dropped or duplicated classification labels")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError("GLiNER returned invalid classification scores")
        if not math.isclose(sum(values.values()), 1.0, abs_tol=1e-4):
            raise ValueError("GLiNER did not return mutually exclusive softmax scores")
        scores[question["id"]] = values
    return scores


def model_token_limit(model: Any) -> int:
    """Respect absolute positions without mistaking DeBERTa relative buckets for a limit."""
    config = model.encoder.config
    limits = [getattr(model.config, "max_len", 4096)]
    if getattr(config, "position_biased_input", True):
        limits.append(getattr(config, "max_position_embeddings", None))
    return min(value for value in limits if isinstance(value, int) and value > 0)


class GLiNERBackend:
    """Lazy local model. One worker owns one instance; no implicit downloads."""
    question_execution = "joint_schema"

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu, mps or cuda")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = device
        self._model = None
        self.manifest: dict[str, Any] | None = None

    def load(self) -> Any:
        if self._model is None:
            self.manifest = validate_local_model(self.model_dir)
            offline_environment()
            import torch
            from gliner2 import AutoExtractor
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is not available in this process")
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available in this process")
            model = AutoExtractor.from_pretrained(str(self.model_dir), local_files_only=True)
            model.to(self.device).float().eval()
            self._model = model
        return self._model

    def count_tokens(self, text: str, questions: list[dict[str, Any]]) -> int:
        model = self.load()
        schema = question_schema(questions)
        batch = model.processor.collate_fn_inference(
            [(text, schema.schema)], error_policy="raise", max_len=None,
            architecture=model.architecture,
        )
        return int(batch.attention_mask.sum().item())

    def predict(self, text: str, questions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        model = self.load()
        schema = question_schema(questions)
        token_count = self.count_tokens(text, questions)
        if token_count > model_token_limit(model):
            raise ValueError("Input and question schema exceed the model context limit")
        result = model.extract(text, schema, include_confidence=True, max_len=None)
        return validate_scores(result, questions)


GLICLASS_MODEL_ID = "knowledgator/gliclass-modern-base-v3.0"
GLICLASS_MODEL_REVISION = "ac369222ca4375ca66ebaf7fb5220f223514c035"
GLICLASS_FILES = ("config.json", "model.safetensors", "special_tokens_map.json",
                  "tokenizer.json", "tokenizer_config.json", "README.md")


def validate_gliclass_model(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    manifest = validate_artifact_manifest(directory, GLICLASS_FILES[:-1])
    config = json.loads((directory / "config.json").read_text())
    encoder = config.get("encoder_config", {})
    if config.get("architecture_type") != "uni-encoder" or encoder.get("model_type") != "modernbert":
        raise ValueError("Comparison backend requires the pinned ModernBERT GLiClass architecture")
    if config.get("auto_map") or encoder.get("auto_map"):
        raise ValueError("Remote custom model code is not supported")
    tokenizer = json.loads((directory / "tokenizer_config.json").read_text())
    if tokenizer.get("auto_map"):
        raise ValueError("Remote custom tokenizer code is not supported")
    return manifest


def provision_gliclass_model(destination: str | Path, *, revision: str = GLICLASS_MODEL_REVISION) -> dict[str, Any]:
    """Explicit optional challenger provisioning; use its separate environment."""
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Provisioning requires a full upstream commit SHA")
    destination = Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Provisioning destination must be empty")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError("Provision in a separate online process, then start offline workers")
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    snapshot_download(GLICLASS_MODEL_ID, revision=revision, local_dir=str(destination),
                      allow_patterns=list(GLICLASS_FILES), token=False)
    verify_checkpoint_license(destination)
    manifest = write_model_manifest(destination, source={
        "repository": GLICLASS_MODEL_ID, "revision": revision, "license": "Apache-2.0",
        "library": "gliclass==0.1.20",
    })
    validate_gliclass_model(destination)
    return manifest


class GLiClassBackend:
    """Optional question-by-question batched challenger, separately calibrated.

    Requires the compare extra in an environment separate from GLiNER. Every
    question retains its own prompt and mutually exclusive label softmax.
    Input text is repeated per question: latency comparisons must state this.
    """
    question_execution = "batched_question_prompts"

    def __init__(self, model_dir: str | Path, device: str = "cpu") -> None:
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError("Device must be cpu, mps or cuda")
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = device
        self._model = None
        self.pipeline = None
        self.manifest = None

    def load(self) -> Any:
        if self._model is None:
            self.manifest = validate_gliclass_model(self.model_dir)
            offline_environment()
            import torch
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
            if self.device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS is unavailable")
            if self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            model = GLiClassModel.from_pretrained(str(self.model_dir), local_files_only=True)
            model.to(self.device).float().eval()
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), local_files_only=True,
                                                       trust_remote_code=False, add_prefix_space=True)
            self.pipeline = ZeroShotClassificationPipeline(model, tokenizer,
                classification_type="single-label", device=torch.device(self.device),
                progress_bar=False, max_length=model.config.encoder_config.max_position_embeddings)
            # v3 checkpoints use <<ENT>>, newer pipeline defaults use <<LABEL>>.
            # Resolve the *checkpoint's existing IDs*; never add random embeddings.
            self.pipeline.pipe.label_token = tokenizer.convert_ids_to_tokens(model.config.class_token_index)
            self.pipeline.pipe.sep_token = tokenizer.convert_ids_to_tokens(model.config.text_token_index)
            self._model = model
        return self._model

    @staticmethod
    def _labels(question: dict[str, Any]) -> list[str]:
        return [f'{label["id"]}: {label["description"]}' for label in question["labels"]]

    def _token_counts(self, text: str, questions: list[dict[str, Any]]) -> list[int]:
        self.load()
        pipe = self.pipeline.pipe
        counts = []
        for question in questions:
            encoded = pipe.prepare_input(text, self._labels(question), prompt=question["text"])
            counts.append(len(pipe.tokenizer(encoded, truncation=False)["input_ids"]))
        return counts

    def count_tokens(self, text: str, questions: list[dict[str, Any]]) -> int:
        return sum(self._token_counts(text, questions))

    def predict(self, text: str, questions: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        model = self.load()
        if not questions:
            raise ValueError("At least one question is required")
        if max(self._token_counts(text, questions)) > model.config.encoder_config.max_position_embeddings:
            raise ValueError("Input and question schema exceed the model context limit")
        results = self.pipeline([text] * len(questions), [self._labels(q) for q in questions],
                                prompt=[q["text"] for q in questions], return_hierarchical=True,
                                batch_size=len(questions))
        if len(results) != len(questions):
            raise ValueError("GLiClass dropped a question")
        formatted = {}
        for question, scores in zip(questions, results):
            names = self._labels(question)
            if set(scores) != set(names):
                raise ValueError("GLiClass dropped a label")
            formatted[question["id"]] = [{"label": label["id"], "confidence": scores[name]}
                                         for label, name in zip(question["labels"], names)]
        return validate_scores(formatted, questions)
