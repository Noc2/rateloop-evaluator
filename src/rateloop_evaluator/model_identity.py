"""Safe operational identity for a successfully loaded, verified checkpoint."""
from __future__ import annotations

import hashlib
import json
import re

from .backends import MODEL_ID, MODEL_REVISION


def loaded_model_identity(manifest: dict) -> dict:
    """Project verified loader metadata, never arbitrary source/training fields.

    Call only after the backend has successfully loaded the files validated by
    this manifest. The digest commits to file names and their verified hashes;
    neither file names, local paths nor private training metadata are emitted.
    """
    files = manifest["files"]
    if (not isinstance(files, dict) or not files
            or any(not isinstance(name, str) or not isinstance(digest, str)
                   or re.fullmatch(r"[a-f0-9]{64}", digest) is None
                   for name, digest in files.items())
            or "model.safetensors" not in files):
        raise ValueError("Missing verified checkpoint identity")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, allow_nan=False).encode("utf-8")
    source = manifest.get("source", {})
    reviewed_base = (source.get("repository"), source.get("revision")) == (MODEL_ID, MODEL_REVISION)
    return {"schemaVersion": "rateloop.loaded-model.v1",
            # Only the reviewed public identity can enter provider logs. A
            # custom source may contain a private repository or other secrets.
            "modelId": MODEL_ID if reviewed_base else "local",
            "sourceRevision": MODEL_REVISION if reviewed_base else None,
            "trained": "training" in manifest,
            "weightsSha256": files["model.safetensors"],
            "filesSha256": hashlib.sha256(encoded).hexdigest()}


def emit_loaded_model(identity: dict) -> None:
    print(json.dumps({"event": "evaluator_model_loaded", **identity}), flush=True)
