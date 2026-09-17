"""Generate or verify committed JSON schemas; no models or private data needed."""
import argparse
import json
from pathlib import Path
from rateloop_evaluator.protocol import EvaluationRequest, EvaluationResult, Template

parser = argparse.ArgumentParser(); parser.add_argument("--write", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
for name, model in [("request", EvaluationRequest), ("result", EvaluationResult), ("template", Template)]:
    path = root / "contracts" / f"{name}.schema.json"
    content = json.dumps(model.model_json_schema(), indent=2, ensure_ascii=False) + "\n"
    if args.write: path.write_text(content)
    elif path.read_text() != content: raise SystemExit(f"Schema drift: {path.name}; run scripts/check_contracts.py --write")
