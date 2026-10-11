"""Re-tokenize a frozen local cohort; never publish raw benchmark content."""
import argparse
import json
from pathlib import Path
from rateloop_evaluator.backends import GLiNERBackend
from rateloop_evaluator.coverage_diagnostic import token_coverage_diagnostic
from rateloop_evaluator.general_experiment import write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'rows', 'point', 'model-dir', 'output'):
        parser.add_argument('--'+field, required=True)
    args = parser.parse_args()
    def read(path): return json.loads(Path(path).read_text())
    report = token_coverage_diagnostic(read(args.manifest), read(args.rows), read(args.point), GLiNERBackend(args.model_dir,'cpu'))
    write_private(Path(args.output), report)
    print(json.dumps({'output':str(Path(args.output).resolve()),'total_groups':report['total_groups'],
        'current_supported_groups':report['current_template_supported_groups'],
        'budgets':report['budgets'],'inference_performed':False}))


if __name__ == '__main__': main()
