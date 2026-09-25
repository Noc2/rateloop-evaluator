"""Warm, allowlisted inference shared by local and hosted outbound workers."""
from __future__ import annotations

from .backends import GLiNERBackend
from .service import Principal, create_app
from .templates import overall_approval


def prepare_evaluator(connector, registry, bundle_ids: list[str], *, device: str = "cpu"):
    """Validate and warm every bundle before any presence heartbeat is possible.

    Language-specific registrations pointing at identical local artifacts share
    one backend. No downloads, grants, customer content or training are involved.
    """
    workspace=connector.workspace_id
    identity=Principal(workspace,frozenset({"evaluate"}))
    apps={}; backends={}
    for bundle_id in bundle_ids:
        record=registry.get(bundle_id,workspace)
        manifest=record["manifest"]
        # Registry verification binds the complete immutable artifact inventory.
        key=(record["artifact_root"],tuple(sorted(manifest["files"].items())))
        if key not in backends:
            backend=GLiNERBackend(record["artifact_root"],device)
            backend.load()
            backends[key]=backend
        backend=backends[key]
        for language in manifest["languages"]:
            template=overall_approval(language)
            backend.predict("Ready.",[question.model_dump() for question in template.questions])
        def validate(value, expected_id=bundle_id):
            registry.get(expected_id,workspace,verify_artifacts=False)
            active=registry.active(workspace,value.template_commitment(),value.template.language,verify_artifacts=False)
            if active["bundle_id"] != expected_id: raise PermissionError("Queued model is no longer active")
            return active
        def allow_retention(value):
            with connector.learning.transaction() as database:
                return connector._state(database).get("collections",{}).get(value.input_commitment(),{}).get("trainingAllowed") is True
        apps[bundle_id]=create_app(backend=backend,bundle=manifest,learning=connector.learning,runtime=connector.runtime,
            tokens={"0"*64:identity},validate_bundle=validate,allow_training_retention=allow_retention)
    def evaluate(request):
        if request.modelBundleId not in apps: raise PermissionError("Unconfigured worker bundle")
        return apps[request.modelBundleId].state.evaluate(request,identity)
    return evaluate
