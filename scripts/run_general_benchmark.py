"""Offline general-request benchmark; outputs are private and never activate models."""
import argparse
import json
from pathlib import Path

from rateloop_evaluator.benchmark_reporting import matched_cohort_report
from rateloop_evaluator.judge_benchmark import run_local_judge
from rateloop_evaluator.general_benchmark import blind_review_packet, freeze_benchmark
from rateloop_evaluator.general_experiment import prepare_public, run_local, write_private
from rateloop_evaluator.general_qualification import freeze_operating_point, score_general_benchmark


def read(path):
    content = Path(path).read_bytes()
    if len(content) > 32 * 1024 * 1024:
        raise ValueError('Benchmark files are limited to 32 MiB')
    return json.loads(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    public = commands.add_parser('prepare-public')
    public.add_argument('--source-gzip', required=True)
    public.add_argument('--criterion', choices=['helpfulness', 'correctness'], default='helpfulness')
    public.add_argument('--maximum-groups', type=int, default=600)
    public.add_argument('--output', required=True)
    freeze = commands.add_parser('freeze'); freeze.add_argument('--sources', required=True)
    freeze.add_argument('--previous'); freeze.add_argument('--sampling', choices=['diagnostic_balanced', 'representative_traffic'], default='diagnostic_balanced')
    for name in ('packet', 'run', 'run-judge', 'matched', 'report', 'point'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', required=True)
        if name != 'point': command.add_argument('--rows', required=True)
        command.add_argument('--output', required=True)
        if name == 'run':
            command.add_argument('--model-dir', required=True)
            command.add_argument('--backend', choices=['gliner', 'gliclass'], default='gliner')
            command.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
            command.add_argument('--threshold', type=float, default=.9)
        if name == 'run-judge':
            command.add_argument('--model-dir', required=True)
            command.add_argument('--timeout-seconds', type=float, default=60)
        if name == 'matched': command.add_argument('--observations-by-model', required=True)
        if name == 'report':
            command.add_argument('--point', required=True); command.add_argument('--observations', required=True)
        if name == 'point': command.add_argument('--configuration', required=True)
    freeze.add_argument('--rows', required=True); freeze.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare-public':
        rows, sources = prepare_public(args.source_gzip, criterion=args.criterion, maximum_groups=args.maximum_groups)
        output = Path(args.output); output.mkdir(mode=0o700, parents=True, exist_ok=False)
        manifest = freeze_benchmark(rows, sources)
        for name, content in [('rows.json', rows), ('sources.json', sources), ('manifest.json', manifest)]:
            write_private(output/name, content)
        result = {'rows': len(rows), 'coverage': manifest['coverage'], 'benchmark_commitment': manifest['commitment']}
    elif args.command == 'freeze':
        result = freeze_benchmark(read(args.rows), read(args.sources),
            previous=read(args.previous) if args.previous else None, sampling=args.sampling)
        write_private(Path(args.output), result)
    elif args.command == 'packet':
        result = blind_review_packet(read(args.manifest), read(args.rows)); write_private(Path(args.output), result)
    elif args.command == 'run-judge':
        result = run_local_judge(read(args.manifest), read(args.rows), model_dir=args.model_dir,
            output=args.output, timeout_seconds=args.timeout_seconds)
    elif args.command == 'matched':
        result = matched_cohort_report(read(args.manifest), read(args.rows), read(args.observations_by_model))
        write_private(Path(args.output), result)
    elif args.command == 'point':
        result = freeze_operating_point(read(args.manifest), **read(args.configuration)); write_private(Path(args.output), result)
    elif args.command == 'report':
        result = score_general_benchmark(read(args.manifest), read(args.rows), read(args.point), read(args.observations))
        write_private(Path(args.output), result)
    else:
        result = run_local(read(args.manifest), read(args.rows), model_dir=args.model_dir, output=args.output,
            backend_name=args.backend, device=args.device, threshold=args.threshold)
    print(json.dumps({'output': str(Path(args.output).resolve()), 'qualified': False,
                      'activation_changed': False, 'schema_version': result.get('schema_version')}))


if __name__ == '__main__':
    main()
